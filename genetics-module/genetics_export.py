"""
VivaSense Genetics – Enhanced Word Report Export
================================================
POST /genetics/download-results
POST /genetics/export-word  (alias)

Accepts the full UploadAnalysisResponse returned by /genetics/analyze-upload
and generates a publication-ready .docx report per trait containing:

  • Title & metadata
  • Executive summary
  • Descriptive statistics
  • ANOVA table (formatted, significance-starred)
  • Mean separation table + bar chart (300 DPI, Tukey-group colours)
  • Genetic parameters with formulas
  • Interpretation & breeding recommendations
  • Footer
"""

import io
import logging
import math
import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")                        # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Cm
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from genetics_schemas import AnovaTable, MeanSeparation, GeneticsResult, GeneticsResponse
from multitrait_upload_schemas import UploadAnalysisResponse, SummaryTableRow, TraitResult

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Export"])

# ─── Tukey-group colour palette (group letter → hex) ─────────────────────────
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

# ─── Header background (light grey) ──────────────────────────────────────────
_HEADER_BG = "F2F2F2"


# ============================================================================
# NUMBER FORMATTING
# ============================================================================

def _fmt(value: Optional[float], decimals: int = 2, thousands: bool = True) -> str:
    """Format a float with optional thousands separator."""
    if value is None:
        return "—"
    fmt_str = f"{value:,.{decimals}f}" if thousands else f"{value:.{decimals}f}"
    return fmt_str


def _fmt_p(p: Optional[float]) -> str:
    """Format a p-value with significance stars."""
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
    """Return only the significance stars."""
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
# CELL / TABLE HELPERS (python-docx XML)
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
    """Apply thin black borders to a single cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")        # half-points
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
    """
    Insert a formatted table with grey header, borders, Calibri font.
    numeric_cols = set of column indices to right-align.
    """
    n_cols = len(headers)
    table = doc.add_table(rows=1, cols=n_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        hdr.cells[i].text = h
    _bold_row(hdr, size_pt=12, bg=_HEADER_BG)

    # Data rows
    for row_data in rows:
        r = table.add_row()
        for i, val in enumerate(row_data):
            r.cells[i].text = val
        _style_data_row(r, numeric_cols=numeric_cols)


# ============================================================================
# VISUALIZATION
# ============================================================================

def _bar_colour(group_letter: str) -> str:
    # Use only the first letter (some groups are "ab", "bc", etc.)
    first = group_letter[0].lower() if group_letter else ""
    return _GROUP_COLOURS.get(first, _DEFAULT_BAR_COLOUR)


def _generate_mean_separation_chart(
    trait_name: str,
    genotypes: List[str],
    means: List[float],
    ses: List[Optional[float]],
    groups: List[str],
) -> bytes:
    """
    Return PNG bytes (6" × 4", 300 DPI) for the mean-separation bar chart.
    Falls back gracefully on any matplotlib error.
    """
    try:
        # Use a safe style
        try:
            plt.style.use("seaborn-v0_8")
        except OSError:
            try:
                plt.style.use("ggplot")
            except OSError:
                pass

        fig, ax = plt.subplots(figsize=(6, 4))

        n = len(genotypes)
        x = np.arange(n)
        colours = [_bar_colour(g) for g in groups]
        err = [s if s is not None else 0.0 for s in ses]

        bars = ax.bar(
            x, means,
            color=colours,
            edgecolor="white",
            linewidth=0.5,
            yerr=err,
            capsize=4,
            error_kw={"elinewidth": 1, "ecolor": "#555555"},
            zorder=3,
        )

        # Grouping letters above error bars
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

        # Axis labels & formatting
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

        # Light horizontal grid
        ax.yaxis.grid(True, color="#cccccc", alpha=0.5, linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend for Tukey groups present in this chart
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
        logger.error("Chart generation failed for %s: %s", trait_name, exc, exc_info=True)
        plt.close("all")
        return b""


# ============================================================================
# DOCUMENT SECTIONS
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
    run_key = p.add_run(f"{key}: ")
    run_key.bold = True
    run_key.font.name = "Calibri"
    run_key.font.size = Pt(11)
    run_val = p.add_run(value)
    run_val.font.name = "Calibri"
    run_val.font.size = Pt(11)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)


def _breeding_recommendation(h2: Optional[float], gam_pct: Optional[float]) -> str:
    if h2 is None:
        return "Insufficient data for breeding recommendation."
    if h2 >= 0.80 and gam_pct is not None and gam_pct >= 5.0:
        return (
            "Direct selection is recommended. High heritability combined with "
            "high genetic advance indicates strong additive genetic control — "
            "phenotypic selection will be effective."
        )
    if h2 >= 0.80:
        return (
            "High heritability but narrow genetic advance suggests strong genetic "
            "control with limited phenotypic range. Consider hybridisation or "
            "crossing programmes to broaden the genetic base before selection."
        )
    if h2 >= 0.50:
        return (
            "Moderate heritability. Both additive and environmental effects "
            "contribute. Use replicated trials across environments; family-based "
            "or progeny selection may improve efficiency."
        )
    return (
        "Low heritability indicates dominant environmental influence. "
        "Direct phenotypic selection may not be efficient. "
        "Increase replication, use controlled environments, or apply marker-assisted selection."
    )


def _add_executive_summary(
    doc: Document,
    trait: str,
    result: GeneticsResult,
    row: Optional[SummaryTableRow],
) -> None:
    _add_heading(doc, "Executive Summary", level=2)

    hp = result.heritability if isinstance(result.heritability, dict) else {}
    gp = result.genetic_parameters if isinstance(result.genetic_parameters, dict) else {}
    h2 = hp.get("h2_broad_sense")
    gcv = gp.get("GCV")
    pcv = gp.get("PCV")
    gam_pct = gp.get("GAM_percent")
    grand_mean = result.grand_mean

    lines = [
        ("Trait", trait),
        ("Grand Mean", _fmt(grand_mean)),
        ("Heritability (H²)", f"{_fmt(h2, 3)} — {'High' if h2 and h2 >= 0.6 else 'Moderate' if h2 and h2 >= 0.3 else 'Low' if h2 is not None else '—'}"),
        ("GCV (%)", _fmt(gcv, 2)),
        ("PCV (%)", _fmt(pcv, 2)),
        ("GAM (%)", _fmt(gam_pct, 2)),
    ]
    for k, v in lines:
        _add_kv(doc, k, v)

    doc.add_paragraph()
    _add_body(doc, _breeding_recommendation(h2, gam_pct))


def _add_descriptive_stats(
    doc: Document,
    result: GeneticsResult,
) -> None:
    _add_heading(doc, "Descriptive Statistics", level=2)

    grand_mean = result.grand_mean
    vc = result.variance_components if isinstance(result.variance_components, dict) else {}
    sigma2_p = vc.get("sigma2_phenotypic")
    sd = math.sqrt(sigma2_p) if sigma2_p and sigma2_p > 0 else None
    cv = (sd / grand_mean * 100) if (sd and grand_mean) else None

    _add_kv(doc, "Grand Mean", _fmt(grand_mean))
    _add_kv(doc, "Phenotypic SD (σp)", _fmt(sd))
    _add_kv(doc, "Coefficient of Variation (CV%)", _fmt(cv, 2))
    _add_kv(doc, "No. Genotypes", str(result.n_genotypes))
    _add_kv(doc, "No. Replications", str(result.n_reps))
    if result.n_environments:
        _add_kv(doc, "No. Environments", str(result.n_environments))


def _add_anova_section(doc: Document, at: AnovaTable) -> None:
    _add_heading(doc, "Analysis of Variance (ANOVA)", level=2)

    headers = ["Source", "DF", "SS", "MS", "F-value", "p-value"]
    rows_data = []
    genotype_idx = None

    for i, src in enumerate(at.source):
        label = _ANOVA_LABELS.get(src, src)
        rows_data.append([
            label,
            str(at.df[i]),
            _fmt(at.ss[i]),
            _fmt(at.ms[i]),
            _fmt(at.f_value[i], 3) if at.f_value[i] is not None else "—",
            _fmt_p(at.p_value[i]),
        ])
        if src == "genotype":
            genotype_idx = i

    _add_stat_table(doc, headers, rows_data, numeric_cols={1, 2, 3, 4})
    doc.add_paragraph()

    # Narrative sentence for genotype effect
    if genotype_idx is not None:
        f_val = at.f_value[genotype_idx]
        p_val = at.p_value[genotype_idx]
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

    # G×E narrative
    ge_idx = next(
        (i for i, s in enumerate(at.source) if s == "genotype:environment"),
        None,
    )
    if ge_idx is not None:
        p_val = at.p_value[ge_idx]
        sig = _sig_label(p_val)
        if sig in ("***", "**", "*"):
            _add_body(
                doc,
                f"The Genotype × Environment interaction was significant "
                f"({_fmt_p(p_val)}), suggesting genotype performance is "
                "influenced by the growing environment.",
            )
        else:
            _add_body(
                doc,
                "The Genotype × Environment interaction was not significant, "
                "suggesting stable genotype performance across environments.",
            )


def _add_mean_separation_section(
    doc: Document,
    trait_name: str,
    ms: MeanSeparation,
) -> None:
    _add_heading(doc, f"Mean Separation — {ms.test} (α = {ms.alpha})", level=2)

    # Table
    headers = ["Rank", "Genotype", "Mean", "SE", "Group"]
    rows_data = []
    for i, geno in enumerate(ms.genotype):
        se_val = ms.se[i] if i < len(ms.se) else None
        rows_data.append([
            str(i + 1),
            geno,
            _fmt(ms.mean[i]),
            _fmt(se_val),
            ms.group[i],
        ])
    _add_stat_table(doc, headers, rows_data, numeric_cols={0, 2, 3})
    doc.add_paragraph()

    # Interpretation note
    top_group_letter = ms.group[0] if ms.group else "a"
    top_genotypes = [
        ms.genotype[i]
        for i, g in enumerate(ms.group)
        if g == top_group_letter
    ]
    _add_body(
        doc,
        "Means followed by the same letter are not significantly different "
        f"at α = {ms.alpha} ({ms.test}). "
        f"Top-performing genotype(s) in group '{top_group_letter}': "
        + ", ".join(top_genotypes) + ".",
        italic=True,
    )

    # Bar chart
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
        last_para = doc.paragraphs[-1]
        last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption = doc.add_paragraph(
            f"Figure: Mean ± SE for {trait_name}. "
            "Bars sharing the same letter are not significantly different "
            f"({ms.test}, α = {ms.alpha})."
        )
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.runs[0].italic = True
        caption.runs[0].font.size = Pt(9)
    else:
        _add_body(doc, "(Chart could not be generated for this trait.)", italic=True)


def _add_genetic_parameters_section(
    doc: Document,
    result: GeneticsResult,
) -> None:
    _add_heading(doc, "Genetic Parameters", level=2)

    vc = result.variance_components if isinstance(result.variance_components, dict) else {}
    hp = result.heritability if isinstance(result.heritability, dict) else {}
    gp = result.genetic_parameters if isinstance(result.genetic_parameters, dict) else {}

    sigma2_g = vc.get("sigma2_genotype")
    sigma2_e = vc.get("sigma2_error")
    sigma2_ge = vc.get("sigma2_ge")
    sigma2_p = vc.get("sigma2_phenotypic")
    h2 = hp.get("h2_broad_sense")
    h2_basis = hp.get("interpretation_basis", "")
    gcv = gp.get("GCV")
    pcv = gp.get("PCV")
    gam = gp.get("GAM")
    gam_pct = gp.get("GAM_percent")
    sel_i = gp.get("selection_intensity", 1.4)

    # Variance components table
    vc_headers = ["Component", "Symbol", "Value"]
    vc_rows = []
    if sigma2_g is not None:
        vc_rows.append(["Genotypic Variance", "σ²g", _fmt(sigma2_g, 4)])
    if sigma2_e is not None:
        vc_rows.append(["Error Variance", "σ²e", _fmt(sigma2_e, 4)])
    if sigma2_ge is not None:
        vc_rows.append(["G×E Variance", "σ²ge", _fmt(sigma2_ge, 4)])
    if sigma2_p is not None:
        vc_rows.append(["Phenotypic Variance", "σ²p", _fmt(sigma2_p, 4)])
    if h2 is not None:
        h2_class = "High" if h2 >= 0.6 else "Moderate" if h2 >= 0.3 else "Low"
        vc_rows.append([f"Heritability (H²) [{h2_class}]", "h²", _fmt(h2, 4)])

    if vc_rows:
        _add_stat_table(doc, vc_headers, vc_rows, numeric_cols={2})
        doc.add_paragraph()

    # Formulas as formatted text
    _add_heading(doc, "Formulas Used", level=3)
    for formula_text in [
        "h² = σ²g / σ²p",
        "GA = h² × i × σp",
        "GAM (%) = (GA / Grand Mean) × 100",
        f"Where: i = {_fmt(sel_i, 2)} (selection intensity), σp = phenotypic SD",
    ]:
        p = doc.add_paragraph(formula_text, style="No Spacing")
        p.runs[0].font.name = "Courier New"
        p.runs[0].font.size = Pt(10)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
    doc.add_paragraph()

    # Genetic advance table
    ga_rows = []
    if gcv is not None:
        ga_rows.append(["GCV (%)", _fmt(gcv, 2)])
    if pcv is not None:
        ga_rows.append(["PCV (%)", _fmt(pcv, 2)])
    if gam is not None:
        ga_rows.append(["Genetic Advance (GA)", _fmt(gam, 4)])
    if gam_pct is not None:
        ga_rows.append(["Genetic Advance as % of Mean (GAM%)", _fmt(gam_pct, 2)])

    if ga_rows:
        _add_heading(doc, "Genetic Advance Estimates", level=3)
        _add_stat_table(doc, ["Parameter", "Value"], ga_rows, numeric_cols={1})
        doc.add_paragraph()

    # GCV vs PCV interpretation
    if gcv is not None and pcv is not None:
        ratio = abs(gcv - pcv)
        if ratio < 1.0:
            env_comment = (
                f"GCV ({_fmt(gcv, 2)}%) ≈ PCV ({_fmt(pcv, 2)}%) — "
                "the genetic and phenotypic variances are nearly identical, "
                "indicating minimal environmental influence on this trait."
            )
        elif gcv < pcv:
            env_comment = (
                f"GCV ({_fmt(gcv, 2)}%) < PCV ({_fmt(pcv, 2)}%) — "
                "environmental effects contribute to the observed phenotypic variation."
            )
        else:
            env_comment = (
                f"GCV ({_fmt(gcv, 2)}%) > PCV ({_fmt(pcv, 2)}%) — "
                "unexpected pattern; verify variance component estimates."
            )
        _add_body(doc, env_comment)


def _add_interpretation_section(
    doc: Document,
    ar: GeneticsResponse,
    result: GeneticsResult,
) -> None:
    _add_heading(doc, "Interpretation & Breeding Recommendations", level=2)

    gp = result.genetic_parameters if isinstance(result.genetic_parameters, dict) else {}
    hp = result.heritability if isinstance(result.heritability, dict) else {}
    h2 = hp.get("h2_broad_sense")
    gam_pct = gp.get("GAM_percent")

    if ar.interpretation:
        _add_heading(doc, "Statistical Interpretation", level=3)
        _add_body(doc, ar.interpretation)
        doc.add_paragraph()

    _add_heading(doc, "Breeding Recommendation", level=3)
    _add_body(doc, _breeding_recommendation(h2, gam_pct))


def _add_trait_section(
    doc: Document,
    trait_name: str,
    tr: TraitResult,
    row: Optional[SummaryTableRow],
) -> None:
    doc.add_page_break()
    _add_heading(doc, f"Trait: {trait_name}", level=1)

    if tr.status == "failed" or tr.analysis_result is None:
        error_msg = (row.error if row and row.error else None) or tr.error or "Unknown error"
        _add_body(doc, f"Analysis failed: {error_msg}")
        return

    ar = tr.analysis_result
    result = ar.result

    if result is None:
        _add_body(doc, "No analysis result available.")
        return

    # Data warnings
    if tr.data_warnings:
        _add_heading(doc, "Data Warnings", level=3)
        for w in tr.data_warnings:
            doc.add_paragraph(f"• {w}", style="List Bullet")
        doc.add_paragraph()

    _add_executive_summary(doc, trait_name, result, row)
    doc.add_paragraph()

    _add_descriptive_stats(doc, result)
    doc.add_paragraph()

    if result.anova_table:
        _add_anova_section(doc, result.anova_table)
        doc.add_paragraph()
    else:
        _add_heading(doc, "Analysis of Variance (ANOVA)", level=2)
        _add_body(doc, "ANOVA table not available for this trait.", italic=True)
        doc.add_paragraph()

    if result.mean_separation:
        _add_mean_separation_section(doc, trait_name, result.mean_separation)
        doc.add_paragraph()
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
# CROSS-TRAIT SUMMARY TABLE
# ============================================================================

def _add_summary_table(doc: Document, data: UploadAnalysisResponse) -> None:
    headers = ["Trait", "Mean", "H²", "GCV %", "PCV %", "GAM %", "Class", "Status"]
    rows_data = []
    for row in data.summary_table:
        rows_data.append([
            row.trait,
            _fmt(row.grand_mean),
            _fmt(row.h2, 3),
            _fmt(row.gcv, 2),
            _fmt(row.pcv, 2),
            _fmt(row.gam_percent, 2),
            row.heritability_class or "—",
            row.status,
        ])
    _add_stat_table(doc, headers, rows_data, numeric_cols={1, 2, 3, 4, 5})


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

def _build_document(data: UploadAnalysisResponse) -> Document:
    doc = Document()

    # Page margins
    for sec in doc.sections:
        sec.top_margin = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin = Inches(1.25)
        sec.right_margin = Inches(1.25)

    # ── Title ─────────────────────────────────────────────────────────────────
    title = doc.add_heading("VivaSense Genetics Analysis Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    ds = data.dataset_summary
    mode_label = "Multi-environment" if ds.mode == "multi" else "Single-environment"
    env_part = f"  ·  {ds.n_environments} environments" if ds.n_environments else ""
    subtitle = doc.add_paragraph(
        f"{mode_label}  ·  {ds.n_genotypes} genotypes  ·  "
        f"{ds.n_reps} replications{env_part}  ·  {ds.n_traits} traits"
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(11)

    date_p = doc.add_paragraph(
        f"Report generated: {datetime.date.today().strftime('%d %B %Y')}"
    )
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
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
            f"Failed traits (excluded from detailed sections): "
            + ", ".join(data.failed_traits),
            italic=True,
        )

    # ── Per-trait sections ────────────────────────────────────────────────────
    row_by_trait = {r.trait: r for r in data.summary_table}

    for row in data.summary_table:
        trait = row.trait
        tr = data.trait_results.get(trait)
        if tr is None:
            continue
        _add_trait_section(doc, trait, tr, row_by_trait.get(trait))

    _add_footer(doc)
    return doc


# ============================================================================
# ENDPOINT
# ============================================================================

async def export_word_report(data: UploadAnalysisResponse) -> Response:
    """
    Generate a publication-ready Word (.docx) report from genetics analysis results.

    Registered at both /genetics/download-results and /genetics/export-word.
    """
    logger.info(
        "Received payload: traits=%s, failed=%s, dataset=%s",
        [r.trait for r in data.summary_table],
        data.failed_traits,
        data.dataset_summary.dict() if data.dataset_summary else None,
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
