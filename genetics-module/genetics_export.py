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

    grand_mean = result.grand_mean
    vc = result.variance_components if isinstance(result.variance_components, dict) else {}
    sigma2_p = vc.get("sigma2_phenotypic")
    sd = math.sqrt(sigma2_p) if sigma2_p and sigma2_p > 0 else None
    cv = (sd / grand_mean * 100) if (sd and grand_mean) else None

    rows = [
        ("Grand Mean", _fmt(grand_mean)),
        ("Phenotypic SD (σp)", _fmt(sd)),
        ("Coefficient of Variation (CV%)", _fmt(cv, 2)),
        ("No. Genotypes", str(result.n_genotypes)),
        ("No. Replications", str(result.n_reps)),
    ]
    if result.n_environments:
        rows.append(("No. Environments", str(result.n_environments)))

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

    _add_heading(doc, "Breeding Recommendation", level=3)
    _add_body(doc, _breeding_recommendation(h2, gam_pct))


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

    # Check for failed / missing analysis
    status = getattr(tr, "status", None) or (
        "success" if tr.analysis_result is not None else "failed"
    )
    if status == "failed" or tr.analysis_result is None:
        error_msg = (
            (row.error if row and row.error else None)
            or tr.error
            or "No analysis result available"
        )
        _add_body(doc, f"Analysis failed: {error_msg}")
        logger.warning("Trait '%s' skipped — status=%s error=%s", trait_name, status, error_msg)
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
    row_by_trait = {r.trait: r for r in data.summary_table}
    trait_results = data.trait_results or {}

    logger.info(
        "Building per-trait sections. summary_table traits=%s | trait_results keys=%s",
        [r.trait for r in data.summary_table],
        list(trait_results.keys()),
    )

    for row in data.summary_table:
        trait = row.trait
        tr = trait_results.get(trait)

        if tr is None:
            # Key mismatch — log and write a placeholder
            logger.warning(
                "Trait '%s' in summary_table but NOT in trait_results. "
                "Available keys: %s",
                trait,
                list(trait_results.keys()),
            )
            doc.add_page_break()
            _add_heading(doc, f"Trait: {trait}", level=1)
            _add_body(
                doc,
                f"No result data found for '{trait}'. "
                "Check that the trait name in summary_table matches the key in trait_results.",
                italic=True,
            )
            continue

        try:
            _add_trait_section(doc, trait, tr, row_by_trait.get(trait))
        except Exception as exc:
            logger.error(
                "Error rendering section for trait '%s': %s",
                trait, exc, exc_info=True,
            )
            # Write a diagnostic placeholder so other traits continue
            doc.add_page_break()
            _add_heading(doc, f"Trait: {trait}", level=1)
            _add_body(
                doc,
                f"Section rendering error: {type(exc).__name__}: {exc}",
                italic=True,
            )

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

    # Log first trait's analysis_result structure
    for trait, tr in (data.trait_results or {}).items():
        has_ar  = tr.analysis_result is not None
        has_res = has_ar and tr.analysis_result.result is not None
        logger.info(
            "  trait='%s' status='%s' has_analysis_result=%s has_result=%s",
            trait,
            getattr(tr, "status", "?"),
            has_ar,
            has_res,
        )
        break   # only log first trait — enough to diagnose

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
