"""
VivaSense Genetics – Enhanced Word Report Export
================================================
POST /genetics/download-results
POST /genetics/export-word  (alias)

Accepts DownloadReportRequest (UploadAnalysisResponse + optional correlation)
and generates a publication-ready .docx report containing:

  • Title & metadata
  • Trait Summary Table (all traits side-by-side)
  • Trait Correlations (if correlation data supplied)
  • Per-trait sections (new page each):
      – Executive summary
      – Descriptive statistics
      – ANOVA table (significance-starred, narrative)
      – Mean separation (table + bar chart, 300 DPI)
      – Genetic parameters (formulas, GCV/PCV commentary)
      – Interpretation & breeding recommendations
  • Footer
"""

import io
import logging
import math
import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")                        # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from genetics_schemas import AnovaTable, MeanSeparation, GeneticsResult, GeneticsResponse
from multitrait_upload_schemas import UploadAnalysisResponse, SummaryTableRow, TraitResult
from trait_relationships_schemas import CorrelationResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Export"])


# ============================================================================
# REQUEST MODEL
# ============================================================================

class DownloadReportRequest(UploadAnalysisResponse):
    """
    Extends UploadAnalysisResponse with optional correlation data.
    The frontend can POST the raw UploadAnalysisResponse and correlation
    will default to None — fully backwards compatible.
    """
    correlation: Optional[CorrelationResponse] = None


# ─── Tukey-group colour palette ───────────────────────────────────────────────
_GROUP_COLOURS: Dict[str, str] = {
    "a": "#2ecc71",
    "b": "#f39c12",
    "c": "#e74c3c",
    "d": "#95a5a6",
    "e": "#34495e",
}
_DEFAULT_BAR_COLOUR = "#7f8c8d"

# ─── ANOVA source → human label ───────────────────────────────────────────────
_ANOVA_LABELS: Dict[str, str] = {
    "rep": "Replication",
    "genotype": "Genotype",
    "environment": "Environment",
    "environment:rep": "Rep(Environment)",
    "genotype:environment": "G×E Interaction",
    "Residuals": "Error",
}

_HEADER_BG = "F2F2F2"


# ============================================================================
# NUMBER FORMATTING
# ============================================================================

def _fmt(value: Optional[float], decimals: int = 2, thousands: bool = True) -> str:
    if value is None:
        return "—"
    return f"{value:,.{decimals}f}" if thousands else f"{value:.{decimals}f}"


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return "< 0.001 ***"
    if p < 0.01:
        return f"{p:.4f} **"
    if p < 0.05:
        return f"{p:.4f} *"
    return f"{p:.4f} ns"


def _sig_label(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ============================================================================
# TABLE HELPERS (python-docx XML)
# ============================================================================

def _set_cell_bg(cell, hex_colour: str) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_colour)
    tcPr.append(shd)


def _set_cell_border(cell) -> None:
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), "000000")
        tcBorders.append(border)
    tcPr.append(tcBorders)


def _cell_font(cell, size_pt: int = 11, bold: bool = False,
               right_align: bool = False) -> None:
    for para in cell.paragraphs:
        para.paragraph_format.space_before = Pt(1)
        para.paragraph_format.space_after = Pt(1)
        if right_align:
            para.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for run in para.runs:
            run.font.name = "Calibri"
            run.font.size = Pt(size_pt)
            run.font.bold = bold


def _bold_row(row, size_pt: int = 12, bg: Optional[str] = None) -> None:
    for cell in row.cells:
        if bg:
            _set_cell_bg(cell, bg)
        _set_cell_border(cell)
        _cell_font(cell, size_pt=size_pt, bold=True)


def _style_data_row(row, numeric_cols: Optional[set] = None) -> None:
    numeric_cols = numeric_cols or set()
    for i, cell in enumerate(row.cells):
        _set_cell_border(cell)
        _cell_font(cell, size_pt=11, right_align=(i in numeric_cols))


def _add_stat_table(
    doc: Document,
    headers: List[str],
    rows: List[List[str]],
    numeric_cols: Optional[set] = None,
) -> None:
    n_cols = len(headers)
    table = doc.add_table(rows=1, cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr = table.rows[0]
    for i, h in enumerate(headers):
        hdr.cells[i].text = h
    _bold_row(hdr, size_pt=12, bg=_HEADER_BG)

    for row_data in rows:
        r = table.add_row()
        for i, val in enumerate(row_data):
            r.cells[i].text = str(val)
        _style_data_row(r, numeric_cols=numeric_cols)


# ============================================================================
# VISUALIZATION
# ============================================================================

def _bar_colour(group_letter: str) -> str:
    first = group_letter[0].lower() if group_letter else ""
    return _GROUP_COLOURS.get(first, _DEFAULT_BAR_COLOUR)


def _generate_mean_separation_chart(
    trait_name: str,
    genotypes: List[str],
    means: List[float],
    ses: List[Optional[float]],
    groups: List[str],
) -> bytes:
    try:
        for style in ("seaborn-v0_8", "ggplot", "default"):
            try:
                plt.style.use(style)
                break
            except OSError:
                continue

        fig, ax = plt.subplots(figsize=(6, 4))
        n = len(genotypes)
        x = np.arange(n)
        colours = [_bar_colour(g) for g in groups]
        err = [s if (s is not None and not math.isnan(s)) else 0.0 for s in ses]

        ax.bar(
            x, means,
            color=colours,
            edgecolor="white",
            linewidth=0.5,
            yerr=err,
            capsize=4,
            error_kw={"elinewidth": 1, "ecolor": "#555555"},
            zorder=3,
        )

        max_val = max(means) if means else 1.0
        label_offset = max_val * 0.03
        for xi, (m, s, g) in enumerate(zip(means, err, groups)):
            ax.text(
                xi, m + s + label_offset,
                g,
                ha="center", va="bottom",
                fontweight="bold", fontsize=10,
                color="#222222",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            genotypes,
            rotation=45 if n > 8 else 0,
            ha="right" if n > 8 else "center",
            fontsize=9,
        )
        ax.set_ylabel(trait_name, fontsize=11)
        ax.set_xlabel("Genotype", fontsize=11)
        ax.set_title(f"Mean Separation — {trait_name}", fontsize=12, fontweight="bold")
        ax.yaxis.grid(True, color="#cccccc", alpha=0.5, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        present_groups = sorted({g[0].lower() for g in groups if g})
        patches = [
            mpatches.Patch(
                color=_GROUP_COLOURS.get(g, _DEFAULT_BAR_COLOUR),
                label=f"Group {g}",
            )
            for g in present_groups
        ]
        if len(patches) > 1:
            ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.7)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as exc:
        logger.error("Chart generation failed for '%s': %s", trait_name, exc, exc_info=True)
        plt.close("all")
        return b""


# ============================================================================
# DOCUMENT HELPERS
# ============================================================================

def _add_heading(doc: Document, text: str, level: int) -> None:
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(6 if level > 1 else 12)
    h.paragraph_format.space_after = Pt(3)


def _add_body(doc: Document, text: str, italic: bool = False) -> None:
    p = doc.add_paragraph(text)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    if italic:
        for run in p.runs:
            run.italic = True


def _add_kv(doc: Document, key: str, value: str) -> None:
    p = doc.add_paragraph()
    rk = p.add_run(f"{key}: ")
    rk.bold = True
    rk.font.name = "Calibri"
    rk.font.size = Pt(11)
    rv = p.add_run(value)
    rv.font.name = "Calibri"
    rv.font.size = Pt(11)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)


# ============================================================================
# BREEDING RECOMMENDATION
# ============================================================================

def _breeding_recommendation(h2: Optional[float], gam_pct: Optional[float]) -> str:
    if h2 is None:
        return "Insufficient data for a breeding recommendation."
    if h2 >= 0.80 and gam_pct is not None and gam_pct >= 5.0:
        return (
            "Direct selection is recommended. High heritability combined with "
            "high genetic advance (GAM ≥ 5 %) indicates strong additive genetic "
            "control — phenotypic selection will be effective and rapid."
        )
    if h2 >= 0.80:
        return (
            "High heritability but narrow genetic advance suggests strong genetic "
            "control with a limited phenotypic range. Consider hybridisation or "
            "crossing programmes to broaden the genetic base before selection."
        )
    if h2 >= 0.50:
        return (
            "Moderate heritability. Both additive genetic and environmental effects "
            "contribute to phenotypic variation. Use replicated multi-environment "
            "trials; family-based or progeny selection may improve efficiency."
        )
    return (
        "Low heritability indicates dominant environmental influence. "
        "Direct phenotypic selection is unlikely to be efficient. "
        "Increase replication, use controlled environments, or apply "
        "marker-assisted selection."
    )


# ============================================================================
# SECTION: CROSS-TRAIT SUMMARY TABLE
# ============================================================================

def _add_summary_table(doc: Document, data: UploadAnalysisResponse) -> None:
    headers = ["Trait", "Mean", "H²", "GCV %", "PCV %", "GAM %", "Class", "Status"]
    rows_data = [
        [
            row.trait,
            _fmt(row.grand_mean),
            _fmt(row.h2, 3),
            _fmt(row.gcv, 2),
            _fmt(row.pcv, 2),
            _fmt(row.gam_percent, 2),
            row.heritability_class or "—",
            row.status,
        ]
        for row in data.summary_table
    ]
    _add_stat_table(doc, headers, rows_data, numeric_cols={1, 2, 3, 4, 5})


# ============================================================================
# SECTION: TRAIT CORRELATIONS
# ============================================================================

def _add_correlation_section(doc: Document, corr: CorrelationResponse) -> None:
    doc.add_page_break()
    _add_heading(doc, "Trait Correlations", level=1)

    traits = corr.trait_names
    n = len(traits)
    method_label = corr.method.capitalize()

    _add_kv(doc, "Method", f"{method_label} correlation")
    _add_kv(doc, "No. Genotype Means Used", str(corr.n_observations))
    doc.add_paragraph()

    # Pairwise table (upper triangle, excluding diagonal)
    headers = ["Trait 1", "Trait 2", "r-value", "p-value", "Sig."]
    rows_data = []
    for i in range(n):
        for j in range(i + 1, n):
            r_val = corr.r_matrix[i][j] if corr.r_matrix else None
            p_val = corr.p_matrix[i][j] if corr.p_matrix else None
            sig = _sig_label(p_val)
            rows_data.append([
                traits[i],
                traits[j],
                _fmt(r_val, 3, thousands=False),
                _fmt_p(p_val),
                sig if sig else "ns",
            ])

    if rows_data:
        _add_stat_table(doc, headers, rows_data, numeric_cols={2})
        doc.add_paragraph()

    # Auto-interpretation
    _add_heading(doc, "Correlation Interpretation", level=2)
    if corr.interpretation:
        _add_body(doc, corr.interpretation)
        doc.add_paragraph()

    # Count strong positive correlations for co-selection advice
    strong_pos = [
        (traits[i], traits[j])
        for i in range(n)
        for j in range(i + 1, n)
        if corr.r_matrix
        and corr.r_matrix[i][j] is not None
        and corr.r_matrix[i][j] >= 0.70
        and corr.p_matrix
        and corr.p_matrix[i][j] is not None
        and corr.p_matrix[i][j] < 0.05
    ]
    if strong_pos:
        pairs_str = ", ".join(f"{a} & {b}" for a, b in strong_pos[:5])
        _add_body(
            doc,
            f"Strong positive correlations (r ≥ 0.70, p < 0.05) were detected "
            f"between: {pairs_str}. "
            "A co-selection strategy is recommended — improving one of these "
            "traits through selection will likely produce concurrent gains in "
            "the correlated traits, improving breeding efficiency.",
        )

    strong_neg = [
        (traits[i], traits[j])
        for i in range(n)
        for j in range(i + 1, n)
        if corr.r_matrix
        and corr.r_matrix[i][j] is not None
        and corr.r_matrix[i][j] <= -0.70
        and corr.p_matrix
        and corr.p_matrix[i][j] is not None
        and corr.p_matrix[i][j] < 0.05
    ]
    if strong_neg:
        pairs_str = ", ".join(f"{a} & {b}" for a, b in strong_neg[:5])
        _add_body(
            doc,
            f"Strong negative correlations were found between: {pairs_str}. "
            "Simultaneous improvement of these traits may require special "
            "crossing strategies or recurrent selection.",
        )

    if corr.statistical_note:
        _add_body(doc, corr.statistical_note, italic=True)


# ============================================================================
# SECTION: EXECUTIVE SUMMARY (per trait)
# ============================================================================

def _add_executive_summary(
    doc: Document,
    trait: str,
    result: GeneticsResult,
) -> None:
    _add_heading(doc, "Executive Summary", level=2)

    hp = result.heritability if isinstance(result.heritability, dict) else {}
    gp = result.genetic_parameters if isinstance(result.genetic_parameters, dict) else {}
    h2 = hp.get("h2_broad_sense")
    gcv = gp.get("GCV")
    pcv = gp.get("PCV")
    gam_pct = gp.get("GAM_percent")

    h2_class = (
        "High" if h2 is not None and h2 >= 0.6
        else "Moderate" if h2 is not None and h2 >= 0.3
        else "Low" if h2 is not None
        else "—"
    )

    for key, val in [
        ("Trait", trait),
        ("Grand Mean", _fmt(result.grand_mean)),
        ("Heritability (H²)", f"{_fmt(h2, 3)} [{h2_class}]"),
        ("GCV (%)", _fmt(gcv, 2)),
        ("PCV (%)", _fmt(pcv, 2)),
        ("GAM (%)", _fmt(gam_pct, 2)),
    ]:
        _add_kv(doc, key, val)

    doc.add_paragraph()
    _add_body(doc, _breeding_recommendation(h2, gam_pct))


# ============================================================================
# SECTION: DESCRIPTIVE STATISTICS (per trait)
# ============================================================================

def _add_descriptive_stats(doc: Document, result: GeneticsResult) -> None:
    _add_heading(doc, "Descriptive Statistics", level=2)

    # Use only real, directly-available fields — do not fabricate derived stats
    rows: List[tuple] = [
        ("Grand Mean", _fmt(result.grand_mean)),
        ("No. Genotypes", str(result.n_genotypes)),
        ("No. Replications", str(result.n_reps)),
    ]
    if result.n_environments:
        rows.append(("No. Environments", str(result.n_environments)))

    # Append extra fields from descriptive_stats dict if the R engine provided it
    if result.descriptive_stats and isinstance(result.descriptive_stats, dict):
        for key, val in result.descriptive_stats.items():
            label = key.replace("_", " ").title()
            if isinstance(val, float):
                rows.append((label, _fmt(val)))
            elif val is not None:
                rows.append((label, str(val)))

    _add_stat_table(doc, ["Parameter", "Value"], rows, numeric_cols={1})


# ============================================================================
# SECTION: ANOVA (per trait)
# ============================================================================

def _add_anova_section(doc: Document, at: AnovaTable) -> None:
    _add_heading(doc, "Analysis of Variance (ANOVA)", level=2)

    headers = ["Source", "DF", "SS", "MS", "F-value", "p-value"]
    rows_data = []
    genotype_idx = ge_idx = None

    for i, src in enumerate(at.source):
        label = _ANOVA_LABELS.get(src, src)
        df_val = at.df[i] if i < len(at.df) else None
        ss_val = at.ss[i] if i < len(at.ss) else None
        ms_val = at.ms[i] if i < len(at.ms) else None
        f_val  = at.f_value[i] if i < len(at.f_value) else None
        p_val  = at.p_value[i] if i < len(at.p_value) else None

        rows_data.append([
            label,
            str(int(df_val)) if df_val is not None else "—",
            _fmt(ss_val),
            _fmt(ms_val),
            _fmt(f_val, 3) if f_val is not None else "—",
            _fmt_p(p_val),
        ])
        if src == "genotype":
            genotype_idx = i
        if src == "genotype:environment":
            ge_idx = i

    _add_stat_table(doc, headers, rows_data, numeric_cols={1, 2, 3, 4})
    doc.add_paragraph()

    if genotype_idx is not None:
        f_val = at.f_value[genotype_idx] if genotype_idx < len(at.f_value) else None
        p_val = at.p_value[genotype_idx] if genotype_idx < len(at.p_value) else None
        sig = _sig_label(p_val)
        sig_word = (
            "highly significant" if sig in ("***", "**")
            else "significant" if sig == "*"
            else "not significant"
        )
        _add_body(
            doc,
            f"The genotype effect was {sig_word} "
            f"(F = {_fmt(f_val, 3)}, {_fmt_p(p_val)}), indicating "
            + ("substantial genetic variation among genotypes."
               if sig_word != "not significant"
               else "limited genetic variation among genotypes."),
        )

    if ge_idx is not None:
        p_val = at.p_value[ge_idx] if ge_idx < len(at.p_value) else None
        sig = _sig_label(p_val)
        if sig in ("***", "**", "*"):
            _add_body(
                doc,
                f"The G×E interaction was significant ({_fmt_p(p_val)}), "
                "suggesting genotype performance differs across environments.",
            )
        else:
            _add_body(
                doc,
                "The G×E interaction was not significant — genotype rankings "
                "are stable across environments.",
            )


# ============================================================================
# SECTION: MEAN SEPARATION (per trait)
# ============================================================================

def _add_mean_separation_section(
    doc: Document,
    trait_name: str,
    ms: MeanSeparation,
) -> None:
    _add_heading(doc, f"Mean Separation — {ms.test} (α = {ms.alpha})", level=2)

    headers = ["Rank", "Genotype", "Mean", "SE", "Group"]
    rows_data = []
    for i, geno in enumerate(ms.genotype):
        se_val = ms.se[i] if i < len(ms.se) else None
        rows_data.append([
            str(i + 1),
            geno,
            _fmt(ms.mean[i]),
            _fmt(se_val),
            ms.group[i] if i < len(ms.group) else "—",
        ])
    _add_stat_table(doc, headers, rows_data, numeric_cols={0, 2, 3})
    doc.add_paragraph()

    top_letter = ms.group[0] if ms.group else "a"
    top_genos = [ms.genotype[i] for i, g in enumerate(ms.group) if g == top_letter]
    _add_body(
        doc,
        "Means followed by the same letter are not significantly different "
        f"at α = {ms.alpha} ({ms.test}). "
        f"Top-performing genotype(s) in group '{top_letter}': "
        + ", ".join(top_genos) + ".",
        italic=True,
    )

    chart_bytes = _generate_mean_separation_chart(
        trait_name=trait_name,
        genotypes=ms.genotype,
        means=ms.mean,
        ses=ms.se,
        groups=ms.group,
    )
    if chart_bytes:
        doc.add_paragraph()
        _add_heading(doc, "Mean Separation Chart", level=3)
        doc.add_picture(io.BytesIO(chart_bytes), width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = doc.add_paragraph(
            f"Figure: Mean ± SE for {trait_name}. "
            "Bars sharing the same letter are not significantly different "
            f"({ms.test}, α = {ms.alpha})."
        )
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if cap.runs:
            cap.runs[0].italic = True
            cap.runs[0].font.size = Pt(9)
    else:
        _add_body(doc, "(Chart unavailable for this trait.)", italic=True)


# ============================================================================
# SECTION: GENETIC PARAMETERS (per trait)
# ============================================================================

def _add_genetic_parameters_section(doc: Document, result: GeneticsResult) -> None:
    _add_heading(doc, "Genetic Parameters", level=2)

    vc = result.variance_components if isinstance(result.variance_components, dict) else {}
    hp = result.heritability if isinstance(result.heritability, dict) else {}
    gp = result.genetic_parameters if isinstance(result.genetic_parameters, dict) else {}

    sigma2_g  = vc.get("sigma2_genotype")
    sigma2_e  = vc.get("sigma2_error")
    sigma2_ge = vc.get("sigma2_ge")
    sigma2_p  = vc.get("sigma2_phenotypic")
    h2        = hp.get("h2_broad_sense")
    gcv       = gp.get("GCV")
    pcv       = gp.get("PCV")
    gam       = gp.get("GAM")
    gam_pct   = gp.get("GAM_percent")
    sel_i     = gp.get("selection_intensity", 1.4)

    # Variance components table
    vc_rows = []
    if sigma2_g  is not None: vc_rows.append(["Genotypic Variance",   "σ²g",  _fmt(sigma2_g,  4)])
    if sigma2_e  is not None: vc_rows.append(["Error Variance",       "σ²e",  _fmt(sigma2_e,  4)])
    if sigma2_ge is not None: vc_rows.append(["G×E Variance",         "σ²ge", _fmt(sigma2_ge, 4)])
    if sigma2_p  is not None: vc_rows.append(["Phenotypic Variance",  "σ²p",  _fmt(sigma2_p,  4)])
    if h2        is not None:
        h2_cls = "High" if h2 >= 0.6 else "Moderate" if h2 >= 0.3 else "Low"
        vc_rows.append([f"Heritability (H²) [{h2_cls}]", "h²", _fmt(h2, 4)])

    if vc_rows:
        _add_stat_table(doc, ["Component", "Symbol", "Value"], vc_rows, numeric_cols={2})
        doc.add_paragraph()

    # Formulas
    _add_heading(doc, "Formulas", level=3)
    for fml in [
        "h² = σ²g / σ²p",
        "GA = h² × i × σp",
        "GAM (%) = (GA / Grand Mean) × 100",
        f"Where: i = {_fmt(sel_i, 2, thousands=False)} (selection intensity), σp = √σ²p",
    ]:
        p = doc.add_paragraph(fml, style="No Spacing")
        if p.runs:
            p.runs[0].font.name = "Courier New"
            p.runs[0].font.size = Pt(10)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
    doc.add_paragraph()

    # Genetic advance table
    ga_rows = []
    if gcv     is not None: ga_rows.append(["GCV (%)",                          _fmt(gcv,     2)])
    if pcv     is not None: ga_rows.append(["PCV (%)",                          _fmt(pcv,     2)])
    if gam     is not None: ga_rows.append(["Genetic Advance (GA)",             _fmt(gam,     4)])
    if gam_pct is not None: ga_rows.append(["Genetic Advance as % of Mean (GAM%)", _fmt(gam_pct, 2)])

    if ga_rows:
        _add_heading(doc, "Genetic Advance Estimates", level=3)
        _add_stat_table(doc, ["Parameter", "Value"], ga_rows, numeric_cols={1})
        doc.add_paragraph()

    # GCV vs PCV text
    if gcv is not None and pcv is not None:
        diff = abs(gcv - pcv)
        if diff < 1.0:
            comment = (
                f"GCV ({_fmt(gcv, 2)}%) ≈ PCV ({_fmt(pcv, 2)}%) — "
                "genetic and phenotypic variances are nearly identical, "
                "indicating minimal environmental influence on this trait."
            )
        elif gcv < pcv:
            comment = (
                f"GCV ({_fmt(gcv, 2)}%) < PCV ({_fmt(pcv, 2)}%) — "
                "environmental effects contribute to observed phenotypic variation."
            )
        else:
            comment = (
                f"GCV ({_fmt(gcv, 2)}%) > PCV ({_fmt(pcv, 2)}%) — "
                "verify variance component estimates."
            )
        _add_body(doc, comment)


# ============================================================================
# SECTION: INTERPRETATION (per trait)
# ============================================================================

def _add_interpretation_section(
    doc: Document, ar: GeneticsResponse, result: GeneticsResult
) -> None:
    _add_heading(doc, "Interpretation & Breeding Recommendations", level=2)

    hp = result.heritability if isinstance(result.heritability, dict) else {}
    gp = result.genetic_parameters if isinstance(result.genetic_parameters, dict) else {}
    h2      = hp.get("h2_broad_sense")
    gam_pct = gp.get("GAM_percent")

    if ar.interpretation:
        _add_heading(doc, "Statistical Interpretation", level=3)
        _add_body(doc, ar.interpretation)
        doc.add_paragraph()

    # Breeding implication from the R engine (if present in payload)
    if result.breeding_implication:
        _add_heading(doc, "Breeding Implication", level=3)
        _add_body(doc, result.breeding_implication)
        doc.add_paragraph()

    _add_heading(doc, "Breeding Recommendation", level=3)
    _add_body(doc, _breeding_recommendation(h2, gam_pct))


# ============================================================================
# SECTION: ASSUMPTION TESTS (per trait, optional)
# ============================================================================

def _add_assumption_tests_section(doc: Document, assumption_tests: Dict[str, Any]) -> None:
    """Render assumption test results (Shapiro-Wilk, Levene, etc.) if available."""
    _add_heading(doc, "Assumption Tests", level=2)

    rows_data: List[List[str]] = []
    for test_name, test_result in assumption_tests.items():
        label = test_name.replace("_", " ").title()
        if isinstance(test_result, dict):
            # Flexible key lookup — R may use p_value or p.value
            p_val = (
                test_result.get("p_value")
                or test_result.get("p.value")
                or test_result.get("p")
            )
            stat = (
                test_result.get("statistic")
                or test_result.get("test_stat")
                or test_result.get("W")
            )
            conclusion = test_result.get("conclusion") or test_result.get("verdict")
            if conclusion is None and p_val is not None:
                conclusion = "Passed (p ≥ 0.05)" if p_val >= 0.05 else "Failed (p < 0.05)"
            rows_data.append([
                label,
                _fmt(stat, 3, thousands=False) if stat is not None else "—",
                _fmt_p(p_val) if p_val is not None else "—",
                conclusion or "—",
            ])
        else:
            rows_data.append([label, "—", "—", str(test_result)])

    if rows_data:
        _add_stat_table(
            doc,
            ["Test", "Statistic", "p-value", "Result"],
            rows_data,
            numeric_cols={1},
        )
    else:
        _add_body(doc, "Assumption test data present but could not be parsed.", italic=True)


# ============================================================================
# PER-TRAIT SECTION ORCHESTRATOR
# ============================================================================

def _add_trait_section(
    doc: Document,
    trait_name: str,
    tr: TraitResult,
    row: Optional[SummaryTableRow],
) -> None:
    doc.add_page_break()
    _add_heading(doc, f"Trait: {trait_name}", level=1)

    # Gate only on analysis_result — do NOT trust status field, which can be
    # inferred as "failed" in Pydantic v1 when status is absent in the payload
    # and the validator runs before analysis_result is populated.
    if tr.analysis_result is None:
        error_msg = tr.error or (row.error if row else None) or "No analysis result available"
        _add_body(doc, f"Analysis failed: {error_msg}")
        logger.warning("Trait '%s' skipped — analysis_result is None, error=%s", trait_name, error_msg)
        return

    ar = tr.analysis_result
    result = ar.result

    if result is None:
        _add_body(doc, "Analysis result object is empty.")
        logger.warning("Trait '%s': analysis_result.result is None", trait_name)
        return

    logger.info(
        "Rendering trait '%s' | anova=%s | mean_sep=%s | h2=%s",
        trait_name,
        result.anova_table is not None,
        result.mean_separation is not None,
        (result.heritability or {}).get("h2_broad_sense"),
    )

    # Data warnings
    if tr.data_warnings:
        _add_heading(doc, "Data Warnings", level=3)
        for w in tr.data_warnings:
            doc.add_paragraph(f"• {w}", style="List Bullet")
        doc.add_paragraph()

    _add_executive_summary(doc, trait_name, result)
    doc.add_paragraph()

    _add_descriptive_stats(doc, result)
    doc.add_paragraph()

    if result.anova_table:
        _add_anova_section(doc, result.anova_table)
    else:
        _add_heading(doc, "Analysis of Variance (ANOVA)", level=2)
        _add_body(doc, "ANOVA table not available for this trait.", italic=True)
    doc.add_paragraph()

    if result.mean_separation:
        _add_mean_separation_section(doc, trait_name, result.mean_separation)
    else:
        _add_heading(doc, "Mean Separation", level=2)
        _add_body(
            doc,
            "Mean separation (Tukey HSD) not available — "
            "insufficient degrees of freedom or singular model.",
            italic=True,
        )
    doc.add_paragraph()

    _add_genetic_parameters_section(doc, result)
    doc.add_paragraph()

    _add_interpretation_section(doc, ar, result)


# ============================================================================
# GENERIC TABLE HELPER  (DataFrame or list-of-dicts → Word table)
# ============================================================================

def _add_table_to_doc(doc: Document, data: Any) -> None:
    """
    Dynamically build a Word table from a Pandas DataFrame, a list of dicts,
    or a pre-built list of header + row pairs.

    Applies the same grey-header / bordered styling as _add_stat_table so
    the two helpers produce visually consistent output.
    """
    if isinstance(data, pd.DataFrame):
        headers = data.columns.tolist()
        rows = [[str(v) for v in row] for row in data.values.tolist()]
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        headers = list(data[0].keys())
        rows = [[str(row.get(h, "")) for h in headers] for row in data]
    else:
        # Fallback: unsupported format — render as plain text
        doc.add_paragraph(str(data))
        return

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = str(h)
    _bold_row(table.rows[0], size_pt=12, bg=_HEADER_BG)

    # Data rows
    for row_vals in rows:
        r = table.add_row()
        for i, val in enumerate(row_vals):
            r.cells[i].text = val
        _style_data_row(r)


# ============================================================================
# PYDANTIC MODEL → list-of-dicts CONVERTERS
# ============================================================================

def _anova_to_records(at: AnovaTable) -> List[Dict[str, str]]:
    """Convert AnovaTable (parallel arrays) to a list of dicts for _add_table_to_doc."""
    records = []
    for i, src in enumerate(at.source):
        df_val = at.df[i] if i < len(at.df) else None
        ss_val = at.ss[i] if i < len(at.ss) else None
        ms_val = at.ms[i] if i < len(at.ms) else None
        f_val  = at.f_value[i] if i < len(at.f_value) else None
        p_val  = at.p_value[i] if i < len(at.p_value) else None
        records.append({
            "Source":  _ANOVA_LABELS.get(src, src),
            "DF":      str(int(df_val)) if df_val is not None else "—",
            "SS":      _fmt(ss_val),
            "MS":      _fmt(ms_val),
            "F-value": _fmt(f_val, 3) if f_val is not None else "—",
            "p-value": _fmt_p(p_val),
        })
    return records


def _mean_sep_to_records(ms: MeanSeparation) -> List[Dict[str, str]]:
    """Convert MeanSeparation (parallel arrays) to a list of dicts for _add_table_to_doc."""
    records = []
    for i, geno in enumerate(ms.genotype):
        mean_v = ms.mean[i] if i < len(ms.mean) else None
        se_v   = ms.se[i]   if i < len(ms.se)   else None
        grp    = ms.group[i] if i < len(ms.group) else "—"
        records.append({
            "Rank":     str(i + 1),
            "Genotype": geno,
            "Mean":     _fmt(mean_v),
            "SE":       _fmt(se_v),
            "Group":    grp,
        })
    return records


# ============================================================================
# EXPORT TRAITS TO WORD  (public entry point for per-trait sections)
# ============================================================================

def export_traits_to_word(
    results: DownloadReportRequest,
    doc: Document,
) -> Document:
    """
    Iterate through trait_results and append a complete per-trait section to doc.

    Uses trait_results[trait].status as the primary gate:
      • status == "success" AND analysis_result is not None → full export
      • otherwise → error section with reason

    Per-trait section order:
      1. Executive Summary (grand mean, H², GCV, PCV, GAM)
      2. Descriptive Statistics (real fields only — no fabrication)
      3. ANOVA Table (if anova_table present in result)
      4. Mean Separation table + bar chart (if mean_separation present)
      5. Assumption Tests (if assumption_tests present in result)
      6. Genetic Parameters (variance components, formulas, GA estimates)
      7. Interpretation, Breeding Implication, Breeding Recommendation
    """
    trait_results = results.trait_results or {}

    if not trait_results:
        doc.add_paragraph("No trait data found in the results.")
        return doc

    for trait, tr in trait_results.items():
        doc.add_page_break()
        _add_heading(doc, f"Trait: {trait}", level=1)

        # ── Primary gate: status field (inferred from analysis_result when absent)
        if tr.status != "success" or tr.analysis_result is None:
            _add_kv(doc, "Status", "Failed")
            error_msg = tr.error or "No analysis result available."
            _add_kv(doc, "Reason", error_msg)
            logger.warning(
                "export_traits_to_word: trait '%s' failed — status=%s error=%s",
                trait, tr.status, tr.error,
            )
            continue

        ar     = tr.analysis_result          # GeneticsResponse
        result = ar.result                   # GeneticsResult

        if result is None:
            # analysis_result parsed but result sub-object is absent — log and skip
            _add_body(doc, "Analysis result structure is incomplete (result object missing).")
            logger.warning(
                "export_traits_to_word: trait '%s' — analysis_result.result is None "
                "despite success status",
                trait,
            )
            continue

        logger.info(
            "export_traits_to_word: rendering '%s' | "
            "anova=%s | mean_sep=%s | assumption_tests=%s | h2=%s",
            trait,
            result.anova_table is not None,
            result.mean_separation is not None,
            result.assumption_tests is not None,
            (result.heritability or {}).get("h2_broad_sense"),
        )

        # ── Data warnings (non-fatal structural notes from the engine) ───────
        if tr.data_warnings:
            _add_heading(doc, "Data Warnings", level=3)
            for w in tr.data_warnings:
                doc.add_paragraph(f"• {w}", style="List Bullet")
            doc.add_paragraph()

        try:
            # ── 1. Executive Summary ─────────────────────────────────────────
            _add_executive_summary(doc, trait, result)
            doc.add_paragraph()

            # ── 2. Descriptive Statistics ────────────────────────────────────
            _add_descriptive_stats(doc, result)
            doc.add_paragraph()

            # ── 3. ANOVA ─────────────────────────────────────────────────────
            if result.anova_table:
                _add_anova_section(doc, result.anova_table)
            else:
                _add_heading(doc, "Analysis of Variance (ANOVA)", level=2)
                _add_body(doc, "ANOVA table not available for this trait.", italic=True)
            doc.add_paragraph()

            # ── 4. Mean Separation (table + bar chart) ────────────────────────
            if result.mean_separation:
                _add_mean_separation_section(doc, trait, result.mean_separation)
            else:
                _add_heading(doc, "Mean Separation", level=2)
                _add_body(
                    doc,
                    "Mean separation not available — insufficient degrees of "
                    "freedom or singular model.",
                    italic=True,
                )
            doc.add_paragraph()

            # ── 5. Assumption Tests (optional) ────────────────────────────────
            if result.assumption_tests:
                _add_assumption_tests_section(doc, result.assumption_tests)
                doc.add_paragraph()

            # ── 6. Genetic Parameters ─────────────────────────────────────────
            _add_genetic_parameters_section(doc, result)
            doc.add_paragraph()

            # ── 7. Interpretation & Breeding Recommendations ──────────────────
            _add_interpretation_section(doc, ar, result)

        except Exception as exc:
            logger.error(
                "export_traits_to_word: error in section for trait '%s': %s",
                trait, exc, exc_info=True,
            )
            doc.add_paragraph(
                f"Section rendering error: {type(exc).__name__}: {exc}"
            )

    return doc


# ============================================================================
# FOOTER
# ============================================================================

def _add_footer(doc: Document) -> None:
    for section in doc.sections:
        footer = section.footer
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.clear()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(
            f"VivaSense Genetics Engine v1.0  ·  "
            f"Generated {datetime.date.today().strftime('%d %B %Y')}  ·  "
            "Statistical analysis powered by R (ANOVA + Tukey HSD)"
        )
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
        run.font.name = "Calibri"


# ============================================================================
# DOCUMENT BUILDER
# ============================================================================

def _build_document(data: DownloadReportRequest) -> Document:
    doc = Document()

    for sec in doc.sections:
        sec.top_margin    = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin   = Inches(1.25)
        sec.right_margin  = Inches(1.25)

    # ── Title ─────────────────────────────────────────────────────────────────
    title = doc.add_heading("VivaSense Genetics Analysis Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    ds = data.dataset_summary
    mode_label = "Multi-environment" if ds.mode == "multi" else "Single-environment"
    env_part = f"  ·  {ds.n_environments} environments" if ds.n_environments else ""

    sub = doc.add_paragraph(
        f"{mode_label}  ·  {ds.n_genotypes} genotypes  ·  "
        f"{ds.n_reps} replications{env_part}  ·  {ds.n_traits} traits"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if sub.runs:
        sub.runs[0].font.size = Pt(11)

    date_p = doc.add_paragraph(
        f"Report generated: {datetime.date.today().strftime('%d %B %Y')}"
    )
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if date_p.runs:
        date_p.runs[0].font.size = Pt(10)
        date_p.runs[0].font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    doc.add_paragraph()

    # ── Cross-trait summary ───────────────────────────────────────────────────
    _add_heading(doc, "Trait Summary", level=1)
    _add_summary_table(doc, data)
    doc.add_paragraph()

    if data.failed_traits:
        _add_body(
            doc,
            "Failed traits (excluded from detailed sections): "
            + ", ".join(data.failed_traits),
            italic=True,
        )

    # ── Correlation section (optional) ───────────────────────────────────────
    if data.correlation is not None:
        _add_correlation_section(doc, data.correlation)

    # ── Per-trait sections ────────────────────────────────────────────────────
    logger.info(
        "Building per-trait sections. summary_table=%s | trait_results keys=%s",
        [r.trait for r in data.summary_table],
        list((data.trait_results or {}).keys()),
    )
    export_traits_to_word(data, doc)

    _add_footer(doc)
    return doc


# ============================================================================
# ENDPOINT
# ============================================================================

async def export_word_report(data: DownloadReportRequest) -> Response:
    """
    Generate a publication-ready Word (.docx) report from genetics analysis results.

    The request body is the JSON object returned by POST /genetics/analyze-upload,
    optionally extended with a `correlation` field from POST /genetics/correlation.

    Registered at both /genetics/download-results and /genetics/export-word.
    """
    trait_result_keys = list(data.trait_results.keys()) if data.trait_results else []
    summary_traits    = [r.trait for r in data.summary_table]

    logger.info(
        "Download request | summary_traits=%s | trait_result_keys=%s | "
        "failed=%s | has_correlation=%s",
        summary_traits,
        trait_result_keys,
        data.failed_traits,
        data.correlation is not None,
    )

    # Diagnose key-mismatch upfront
    missing_keys = [t for t in summary_traits if t not in trait_result_keys]
    if missing_keys:
        logger.warning(
            "trait_results missing keys for: %s  (present keys: %s)",
            missing_keys,
            trait_result_keys,
        )

    # Log each trait's analysis_result presence to aid debugging
    for trait, tr in (data.trait_results or {}).items():
        has_ar  = tr.analysis_result is not None
        has_res = has_ar and tr.analysis_result.result is not None
        logger.info(
            "  trait='%s' has_analysis_result=%s has_result=%s error='%s'",
            trait,
            has_ar,
            has_res,
            tr.error,
        )

    try:
        doc = _build_document(data)
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type=(
                "application/vnd.openxmlformats-officedocument"
                ".wordprocessingml.document"
            ),
            headers={
                "Content-Disposition": (
                    "attachment; filename=vivasense_genetics_report.docx"
                )
            },
        )
    except Exception as exc:
        logger.error("Report generation failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Report generation failed: {exc}"},
        )


# Register both URL paths
router.add_api_route(
    "/genetics/download-results",
    export_word_report,
    methods=["POST"],
    summary="Download genetics analysis report as Word document",
    tags=["Export"],
)
router.add_api_route(
    "/genetics/export-word",
    export_word_report,
    methods=["POST"],
    summary="Download genetics analysis report as Word document (alias)",
    tags=["Export"],
)
