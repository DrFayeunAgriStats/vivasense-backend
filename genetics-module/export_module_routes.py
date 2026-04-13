"""
VivaSense – Module-specific Word export endpoints

POST /export/anova-word              → ANOVA + mean separation per trait
POST /export/genetic-parameters-word → Variance components + parameters per trait
POST /export/correlation-word        → Correlation matrices + pairwise table
POST /export/heatmap-report          → Heatmap matrix + interpretation

Each endpoint accepts the response from its matching /analysis/* endpoint
and generates a focused, publication-ready .docx file.

Document building reuses the section-builder helpers from genetics_export.py;
no genetics logic or table-styling code is duplicated here.
"""

import datetime
import io
import logging
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

# Section builders from the existing export module
from genetics_export import (
    _add_anova_section,
    _add_assumption_tests_section,
    _add_body,
    _add_correlation_section,
    _add_descriptive_stats,
    _add_footer,
    _add_genetic_parameters_section,
    _add_heading,
    _add_interpretation_section,
    _add_kv,
    _add_mean_separation_section,
    _add_stat_table,
    _fmt,
    _fmt_p,
    _sig_label,
    _HEADER_BG,
)
from genetics_schemas import GeneticsResult, GeneticsResponse
from trait_relationships_schemas import CorrelationResponse
from module_schemas import (
    AnovaExportRequest,
    AnovaTraitResult,
    CorrelationExportRequest,
    GeneticParametersExportRequest,
    GeneticParametersTraitResult,
    HeatmapExportRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Export"])

_DOCX_MIME = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


# ============================================================================
# DOCUMENT HELPERS
# ============================================================================

def _new_document(title: str, subtitle: str = "") -> Document:
    """Create a new Document with standard margins and a centred title."""
    doc = Document()
    for sec in doc.sections:
        sec.top_margin    = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin   = Inches(1.25)
        sec.right_margin  = Inches(1.25)

    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER

    if subtitle:
        sub = doc.add_paragraph(subtitle)
        sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if sub.runs:
            sub.runs[0].font.size = Pt(11)

    date_p = doc.add_paragraph(
        f"Generated: {datetime.date.today().strftime('%d %B %Y')}"
    )
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if date_p.runs:
        date_p.runs[0].font.size = Pt(10)
        date_p.runs[0].font.color.rgb = RGBColor(0x60, 0x60, 0x60)

    doc.add_paragraph()
    return doc


def _docx_response(doc: Document, filename: str) -> Response:
    """Serialise a Document to bytes and return a download Response."""
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type=_DOCX_MIME,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _reconstruct_genetics_result(
    tr: GeneticParametersTraitResult,
    mode: str,
) -> GeneticsResult:
    """
    Build a minimal GeneticsResult from a GeneticParametersTraitResult so
    the existing _add_genetic_parameters_section helper can be reused without
    modification.
    """
    gp_dict: Dict[str, Any] = {}
    if tr.gcv is not None:
        gp_dict["GCV"] = tr.gcv
    if tr.pcv is not None:
        gp_dict["PCV"] = tr.pcv
    if tr.ga is not None:
        gp_dict["GAM"] = tr.ga          # absolute GA stored in ga field
    if tr.gam is not None:
        gp_dict["GAM_percent"] = tr.gam

    return GeneticsResult(
        environment_mode=mode,
        n_genotypes=0,
        n_reps=0,
        grand_mean=tr.grand_mean or 0.0,
        variance_components=tr.variance_components or {},
        heritability=tr.heritability or {},
        genetic_parameters=gp_dict,
        breeding_implication=tr.breeding_implication,
    )


# ============================================================================
# POST /export/anova-word
# ============================================================================

@router.post(
    "/export/anova-word",
    summary="Export ANOVA results as a Word document",
    tags=["Export"],
)
async def export_anova_word(data: AnovaExportRequest):
    """
    Generate a focused Word report containing:
      • Per-trait: descriptive stats, ANOVA table, assumption tests, mean separation
      • Data warnings for each trait
    """
    try:
        n_traits    = len(data.trait_results)
        n_success   = sum(1 for tr in data.trait_results.values() if tr.status == "success")
        mode_label  = "Multi-environment" if data.mode == "multi" else "Single-environment"

        doc = _new_document(
            "VivaSense ANOVA Report",
            f"{mode_label}  ·  {n_traits} trait(s)  ·  {n_success} successful",
        )

        if data.failed_traits:
            _add_body(
                doc,
                "Failed traits (excluded): " + ", ".join(data.failed_traits),
                italic=True,
            )

        for trait, tr in data.trait_results.items():
            doc.add_page_break()
            _add_heading(doc, f"Trait: {trait}", level=1)

            if tr.status != "success" or tr.anova_table is None:
                _add_body(doc, f"Analysis failed: {tr.error or 'No ANOVA result available'}")
                continue

            if tr.data_warnings:
                _add_heading(doc, "Data Warnings", level=3)
                for w in tr.data_warnings:
                    doc.add_paragraph(f"• {w}", style="List Bullet")
                doc.add_paragraph()

            # Descriptive stats (scalar fields from the trait result)
            _add_heading(doc, "Descriptive Statistics", level=2)
            rows = [("Grand Mean", _fmt(tr.grand_mean))]
            if tr.n_genotypes:
                rows.append(("No. Genotypes", str(tr.n_genotypes)))
            if tr.n_reps:
                rows.append(("No. Replications", str(tr.n_reps)))
            if tr.n_environments:
                rows.append(("No. Environments", str(tr.n_environments)))
            if tr.descriptive_stats and isinstance(tr.descriptive_stats, dict):
                for k, v in tr.descriptive_stats.items():
                    label = k.replace("_", " ").title()
                    rows.append((label, _fmt(v) if isinstance(v, float) else str(v)))
            _add_stat_table(doc, ["Parameter", "Value"], rows, numeric_cols={1})
            doc.add_paragraph()

            # ANOVA table
            _add_anova_section(doc, tr.anova_table)
            doc.add_paragraph()

            # Assumption tests
            if tr.assumption_tests:
                _add_assumption_tests_section(doc, tr.assumption_tests)
                doc.add_paragraph()

            # Mean separation
            if tr.mean_separation:
                _add_mean_separation_section(doc, trait, tr.mean_separation)
            else:
                _add_heading(doc, "Mean Separation", level=2)
                _add_body(doc, "Mean separation not available for this trait.", italic=True)
            doc.add_paragraph()

            if tr.interpretation:
                _add_heading(doc, "Interpretation", level=2)
                _add_body(doc, tr.interpretation)

        _add_footer(doc)
        return _docx_response(doc, "vivasense_anova_report.docx")

    except Exception as exc:
        logger.error("ANOVA export failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"ANOVA export failed: {exc}"},
        )


# ============================================================================
# POST /export/genetic-parameters-word
# ============================================================================

@router.post(
    "/export/genetic-parameters-word",
    summary="Export Genetic Parameters results as a Word document",
    tags=["Export"],
)
async def export_genetic_parameters_word(data: GeneticParametersExportRequest):
    """
    Generate a focused Word report containing:
      • Per-trait: variance components, heritability, GCV/PCV, GA/GAM, breeding advice
    """
    try:
        n_traits   = len(data.trait_results)
        n_success  = sum(1 for tr in data.trait_results.values() if tr.status == "success")
        mode_label = "Multi-environment" if data.mode == "multi" else "Single-environment"

        doc = _new_document(
            "VivaSense Genetic Parameters Report",
            f"{mode_label}  ·  {n_traits} trait(s)  ·  {n_success} successful",
        )

        if data.failed_traits:
            _add_body(
                doc,
                "Failed traits (excluded): " + ", ".join(data.failed_traits),
                italic=True,
            )

        for trait, tr in data.trait_results.items():
            doc.add_page_break()
            _add_heading(doc, f"Trait: {trait}", level=1)

            if tr.status != "success":
                _add_body(
                    doc, f"Analysis failed: {tr.error or 'No genetic parameters available'}"
                )
                continue

            if tr.data_warnings:
                _add_heading(doc, "Data Warnings", level=3)
                for w in tr.data_warnings:
                    doc.add_paragraph(f"• {w}", style="List Bullet")
                doc.add_paragraph()

            # Reconstruct GeneticsResult so existing helper can be reused
            gr = _reconstruct_genetics_result(tr, data.mode)
            _add_genetic_parameters_section(doc, gr)
            doc.add_paragraph()

            if tr.breeding_implication:
                _add_heading(doc, "Breeding Implication", level=2)
                _add_body(doc, tr.breeding_implication)
                doc.add_paragraph()

            if tr.interpretation:
                _add_heading(doc, "Interpretation", level=2)
                _add_body(doc, tr.interpretation)

        _add_footer(doc)
        return _docx_response(doc, "vivasense_genetic_parameters_report.docx")

    except Exception as exc:
        logger.error("Genetic parameters export failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Genetic parameters export failed: {exc}"},
        )


# ============================================================================
# POST /export/correlation-word
# ============================================================================

@router.post(
    "/export/correlation-word",
    summary="Export Trait Correlations as a Word document",
    tags=["Export"],
)
async def export_correlation_word(data: CorrelationExportRequest):
    """
    Generate a focused Word report containing:
      • Pairwise correlation table (r-value, p-value, significance)
      • Correlation interpretation
      • Breeding co-selection advice
    """
    try:
        n_traits = len(data.trait_names)
        doc = _new_document(
            "VivaSense Trait Correlation Report",
            f"{data.method.capitalize()} correlation  ·  {n_traits} trait(s)  ·  "
            f"{data.n_observations} genotype mean(s)",
        )

        # Convert CorrelationModuleResponse → CorrelationResponse for the helper
        corr = CorrelationResponse(
            trait_names=data.trait_names,
            n_observations=data.n_observations,
            method=data.method,
            r_matrix=data.r_matrix,
            p_matrix=data.p_matrix,
            interpretation=data.interpretation or "",
            warnings=data.warnings,
            statistical_note=data.statistical_note or "",
        )

        _add_correlation_section(doc, corr)

        _add_footer(doc)
        return _docx_response(doc, "vivasense_correlation_report.docx")

    except Exception as exc:
        logger.error("Correlation export failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Correlation export failed: {exc}"},
        )


# ============================================================================
# POST /export/heatmap-report
# ============================================================================

def _generate_heatmap_image(
    matrix: List[List[Optional[float]]],
    labels: List[str],
    method: str,
    min_val: float,
    max_val: float,
) -> Optional[bytes]:
    """Render the correlation matrix as a colour heatmap PNG (300 DPI)."""
    try:
        n = len(labels)
        data = np.array(
            [[v if v is not None else float("nan") for v in row] for row in matrix],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.65)))
        cmap = plt.get_cmap("RdYlGn")
        im = ax.imshow(data, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")

        # Axes labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > 0.75 else "black"
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=7, color=text_color,
                    )

        plt.colorbar(im, ax=ax, shrink=0.8, label=f"{method.capitalize()} r")
        ax.set_title(
            f"Trait Correlation Heatmap ({method.capitalize()})",
            fontsize=11, fontweight="bold", pad=12,
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as exc:
        logger.warning("Heatmap image generation failed: %s", exc)
        return None


@router.post(
    "/export/heatmap-report",
    summary="Export Heatmap as a Word document",
    tags=["Export"],
)
async def export_heatmap_report(data: HeatmapExportRequest):
    """
    Generate a focused Word report containing:
      • Heatmap image (300 DPI PNG)
      • r-value matrix as a Word table
      • Interpretation
    """
    try:
        n_traits = len(data.labels)
        doc = _new_document(
            "VivaSense Heatmap Report",
            f"{data.method.capitalize()} correlation heatmap  ·  {n_traits} trait(s)",
        )

        # ── Heatmap image ─────────────────────────────────────────────────────
        doc.add_page_break()
        _add_heading(doc, "Trait Correlation Heatmap", level=1)

        img_bytes = _generate_heatmap_image(
            data.matrix, data.labels, data.method, data.min_val, data.max_val
        )
        if img_bytes:
            doc.add_picture(io.BytesIO(img_bytes), width=Inches(6.0))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(
                f"Figure: {data.method.capitalize()} correlation heatmap. "
                "Colour scale: green = positive, red = negative.  "
                "Values on diagonal = 1.0 (self-correlation)."
            )
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if cap.runs:
                cap.runs[0].italic = True
                cap.runs[0].font.size = Pt(9)
        else:
            _add_body(doc, "(Heatmap image could not be generated.)", italic=True)

        doc.add_paragraph()

        # ── Numeric matrix table ──────────────────────────────────────────────
        _add_heading(doc, "Correlation Matrix", level=2)
        n = len(data.labels)
        if n > 0 and data.matrix:
            headers = ["Trait"] + data.labels
            rows_data = []
            for i, label in enumerate(data.labels):
                row_vals = [label]
                for j in range(n):
                    val = data.matrix[i][j] if i < len(data.matrix) and j < len(data.matrix[i]) else None
                    row_vals.append(_fmt(val, 3, thousands=False) if val is not None else "—")
                rows_data.append(row_vals)
            numeric_cols = set(range(1, n + 1))
            _add_stat_table(doc, headers, rows_data, numeric_cols=numeric_cols)
        doc.add_paragraph()

        # ── Interpretation ────────────────────────────────────────────────────
        if data.interpretation:
            _add_heading(doc, "Interpretation", level=2)
            _add_body(doc, data.interpretation)

        if data.warnings:
            _add_heading(doc, "Warnings", level=3)
            for w in data.warnings:
                doc.add_paragraph(f"• {w}", style="List Bullet")

        _add_footer(doc)
        return _docx_response(doc, "vivasense_heatmap_report.docx")

    except Exception as exc:
        logger.error("Heatmap export failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Heatmap export failed: {exc}"},
        )
