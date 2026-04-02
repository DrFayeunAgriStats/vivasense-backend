"""
VivaSense Genetics – Word Report Export
=======================================
POST /genetics/export-word

Accepts the full UploadAnalysisResponse (same JSON the UI already holds after
calling /genetics/analyze-upload) and returns a .docx binary.

Sections per trait:
  • ANOVA Table
  • Mean Separation (Tukey HSD)
  • Variance Components
  • Genetic Parameters
  • Interpretation
"""

import io
import logging
from typing import Optional

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from fastapi import APIRouter
from fastapi.responses import Response

from genetics_schemas import AnovaTable, MeanSeparation, GeneticsResult
from multitrait_upload_schemas import UploadAnalysisResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Export"])

# Human-readable labels for ANOVA source terms produced by R
_ANOVA_LABELS = {
    "rep": "Replication",
    "genotype": "Genotype",
    "environment": "Environment",
    "environment:rep": "Rep(Environment)",
    "genotype:environment": "G×E Interaction",
    "Residuals": "Error",
}


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/genetics/export-word",
    summary="Download genetics analysis report as Word document",
    tags=["Export"],
)
async def export_word_report(data: UploadAnalysisResponse) -> Response:
    """
    Generate a Word (.docx) report from a completed genetics analysis.

    The request body is the JSON object returned by POST /genetics/analyze-upload.
    The frontend passes that object directly — no second analysis is run.
    """
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
    env_part = (
        f"  ·  {ds.n_environments} environments" if ds.n_environments else ""
    )
    doc.add_paragraph(
        f"{mode_label}  ·  {ds.n_genotypes} genotypes  ·  {ds.n_reps} reps"
        f"{env_part}  ·  {ds.n_traits} traits"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    # ── Cross-trait summary table ─────────────────────────────────────────────
    doc.add_heading("Summary", level=1)
    _add_summary_table(doc, data)

    # ── Per-trait sections ────────────────────────────────────────────────────
    for row in data.summary_table:
        trait = row.trait
        doc.add_page_break()
        doc.add_heading(trait, level=1)

        tr = data.trait_results.get(trait)

        if row.status == "failed" or tr is None:
            doc.add_paragraph(
                f"Analysis failed: {row.error or 'unknown error'}"
            )
            continue

        ar = tr.analysis_result
        if ar is None or ar.result is None:
            doc.add_paragraph("No analysis result available.")
            continue

        result = ar.result

        # Data warnings
        if tr.data_warnings:
            p = doc.add_paragraph()
            run = p.add_run("Data warnings:")
            run.bold = True
            for w in tr.data_warnings:
                doc.add_paragraph(f"• {w}", style="List Bullet")

        # ANOVA Table
        if result.anova_table:
            doc.add_heading("ANOVA Table", level=2)
            _add_anova_table(doc, result.anova_table)
            doc.add_paragraph()

        # Mean Separation
        if result.mean_separation:
            ms = result.mean_separation
            doc.add_heading(
                f"Mean Separation — {ms.test} (α = {ms.alpha})", level=2
            )
            _add_mean_separation_table(doc, ms)
            doc.add_paragraph(
                "Means sharing the same letter are not significantly different."
            ).italic = True
            doc.add_paragraph()

        # Variance Components
        doc.add_heading("Variance Components", level=2)
        _add_variance_table(doc, result)
        doc.add_paragraph()

        # Genetic Parameters
        doc.add_heading("Genetic Parameters", level=2)
        _add_genetic_params_table(doc, result)
        doc.add_paragraph()

        # Interpretation
        if ar.interpretation:
            doc.add_heading("Interpretation", level=2)
            doc.add_paragraph(ar.interpretation)

    return doc


# ============================================================================
# TABLE HELPERS
# ============================================================================

def _fmt(val: Optional[float], decimals: int = 2) -> str:
    return f"{val:.{decimals}f}" if val is not None else "—"


def _p_fmt(p: Optional[float]) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return "< 0.001 ***"
    if p < 0.01:
        return f"{p:.4f} **"
    if p < 0.05:
        return f"{p:.4f} *"
    return f"{p:.4f} ns"


def _bold_row(row) -> None:
    """Make all cells in a table row bold."""
    for cell in row.cells:
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True


def _add_summary_table(doc: Document, data: UploadAnalysisResponse) -> None:
    headers = ["Trait", "Mean", "H²", "GCV %", "PCV %", "GAM %", "Class", "Status"]
    table = doc.add_table(rows=1, cols=len(headers), style="Table Grid")
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        hdr.cells[i].text = h
    _bold_row(hdr)

    for row in data.summary_table:
        r = table.add_row()
        r.cells[0].text = row.trait
        r.cells[1].text = _fmt(row.grand_mean)
        r.cells[2].text = _fmt(row.h2, 3)
        r.cells[3].text = _fmt(row.gcv, 1)
        r.cells[4].text = _fmt(row.pcv, 1)
        r.cells[5].text = _fmt(row.gam_percent, 1)
        r.cells[6].text = row.heritability_class or "—"
        r.cells[7].text = row.status


def _add_anova_table(doc: Document, at: AnovaTable) -> None:
    headers = ["Source", "df", "SS", "MS", "F-value", "P-value"]
    table = doc.add_table(rows=1, cols=len(headers), style="Table Grid")
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        hdr.cells[i].text = h
    _bold_row(hdr)

    for i, src in enumerate(at.source):
        r = table.add_row()
        r.cells[0].text = _ANOVA_LABELS.get(src, src)
        r.cells[1].text = str(at.df[i])
        r.cells[2].text = _fmt(at.ss[i])
        r.cells[3].text = _fmt(at.ms[i])
        r.cells[4].text = _fmt(at.f_value[i], 3)
        r.cells[5].text = _p_fmt(at.p_value[i])


def _add_mean_separation_table(doc: Document, ms: MeanSeparation) -> None:
    headers = ["Rank", "Genotype", "Mean", "SE", "Group"]
    table = doc.add_table(rows=1, cols=len(headers), style="Table Grid")
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        hdr.cells[i].text = h
    _bold_row(hdr)

    for i, geno in enumerate(ms.genotype):
        r = table.add_row()
        r.cells[0].text = str(i + 1)
        r.cells[1].text = geno
        r.cells[2].text = _fmt(ms.mean[i])
        se = ms.se[i] if i < len(ms.se) else None
        r.cells[3].text = _fmt(se)
        r.cells[4].text = ms.group[i]


def _add_variance_table(doc: Document, result: GeneticsResult) -> None:
    vc = result.variance_components
    _LABELS = {
        "sigma2_genotype": "σ²Genotype",
        "sigma2_error": "σ²Error",
        "sigma2_ge": "σ²G×E",
        "sigma2_phenotypic": "σ²Phenotypic",
        "heritability_basis": "Basis",
    }
    rows_data = [
        (_LABELS.get(k, k), str(round(v, 4)) if isinstance(v, float) else str(v))
        for k, v in vc.items()
        if v is not None
    ]
    h = result.heritability
    h2 = h.get("h2_broad_sense")
    rows_data.append(("H² (broad-sense)", _fmt(h2, 4) if h2 is not None else "—"))
    rows_data.append(("Basis", h.get("interpretation_basis", "—")))

    table = doc.add_table(rows=1, cols=2, style="Table Grid")
    hdr = table.rows[0]
    hdr.cells[0].text = "Component"
    hdr.cells[1].text = "Value"
    _bold_row(hdr)
    for label, val in rows_data:
        r = table.add_row()
        r.cells[0].text = label
        r.cells[1].text = val


def _add_genetic_params_table(doc: Document, result: GeneticsResult) -> None:
    gp = result.genetic_parameters
    _LABELS = {
        "GCV": "GCV (%)",
        "PCV": "PCV (%)",
        "GAM": "Genetic Advance (GAM)",
        "GAM_percent": "GAM as % of Mean",
        "selection_intensity": "Selection Intensity (i)",
    }
    table = doc.add_table(rows=1, cols=2, style="Table Grid")
    hdr = table.rows[0]
    hdr.cells[0].text = "Parameter"
    hdr.cells[1].text = "Value"
    _bold_row(hdr)
    for key, label in _LABELS.items():
        val = gp.get(key)
        if val is None:
            continue
        r = table.add_row()
        r.cells[0].text = label
        r.cells[1].text = _fmt(val, 4) if isinstance(val, float) else str(val)
