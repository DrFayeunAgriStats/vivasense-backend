"""
VivaSense – Heatmap Analysis Module

POST /analysis/heatmap

Returns heatmap-ready data derived from correlation analysis:

  • matrix  — n×n correlation values (phenotypic or genotypic based on user objective)
  • labels  — ordered trait names (axis labels)
  • min_val, max_val — global range for colour-scale normalisation
  • method, interpretation, warnings

The heatmap mode is selected based on user objective:
- "Field understanding": Uses phenotypic correlations (all observations)
- "Genotype comparison"/"Breeding decision": Uses genotypic correlations (genotype means)

The frontend renders this directly as a correlation heatmap (e.g. via a
charting library).  No separate R computation is performed — the heatmap
is a reshaped view of the correlation result.

Requires a dataset_token from POST /upload/dataset and ≥ 2 trait columns.
"""

import base64
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import Field

from trait_relationships_routes import _build_wide_records
from multitrait_upload_routes import read_file
from module_schemas import HeatmapModuleResponse
from analysis_correlation_routes import CorrelationModuleRequest
import dataset_cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])

_STATISTICAL_NOTE = (
    "Heatmap values are Pearson/Spearman r coefficients. Mode selection "
    "follows agricultural biometrics principles: phenotypic correlations for "
    "field understanding, genotypic correlations for genotype comparison. "
    "Diagonal = 1.0 (self-correlation)."
)


@router.post(
    "/analysis/heatmap",
    response_model=HeatmapModuleResponse,
    summary="Generate heatmap matrix from trait correlations",
)
async def analysis_heatmap(request: CorrelationModuleRequest):
    """
    Compute the genotypic correlation matrix and reformat it as heatmap data.

    Genotypic correlations (computed on genotype means) are used here because
    they reduce environmental noise and are more appropriate for visualizing
    genotype-level trait relationships. For field-level co-variation patterns,
    use the full dual-mode correlation analysis at /analysis/correlation.
    """

    The returned `matrix` is the n×n r-value grid; `labels` gives the trait
    names for each row/column; `min_val`/`max_val` give the observed range
    (excluding diagonal 1.0 values) for colour-scale calibration.

    Requires ≥ 2 trait columns and a dataset_token from POST /upload/dataset.
    """
    import trait_relationships_routes as _tr_mod

    engine = _tr_mod.tr_engine
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Trait relationships engine not ready — "
                "vivasense_trait_relationships.R may be missing"
            ),
        )

    if len(request.trait_columns) < 2:
        raise HTTPException(
            status_code=400,
            detail="Heatmap requires at least 2 trait columns.",
        )

    ctx = dataset_cache.get_dataset(request.dataset_token)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Dataset token '{request.dataset_token}' not found. "
                "Re-upload via POST /upload/dataset to get a new token."
            ),
        )

    try:
        file_bytes = base64.b64decode(ctx["base64_content"])
        df = read_file(file_bytes, ctx["file_type"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}") from exc

    geno_col = ctx["genotype_column"]
    rep_col  = ctx["rep_column"]
    env_col  = ctx["environment_column"] if ctx["mode"] == "multi" else None

    missing = [c for c in request.trait_columns if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"Trait columns not found in dataset: {missing}"
        )

    try:
        records = _build_wide_records(df, geno_col, rep_col, request.trait_columns, env_col)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        raw = engine.run_correlation(records, request.trait_columns, request.method)
    except Exception as exc:
        logger.error("Heatmap (correlation) engine error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Heatmap analysis failed: {exc}"
        ) from exc

    labels    = raw.get("trait_names", request.trait_columns)
    
    # Choose correlation mode based on user objective
    if request.user_objective == "Field understanding":
        # Use phenotypic correlations for field-level understanding
        r_matrix  = raw.get("phenotypic", {}).get("r_matrix", [])
        mode_used = "phenotypic"
    else:
        # Use genotypic correlations for genotype comparison or breeding decisions
        r_matrix  = raw.get("genotypic", {}).get("r_matrix", [])
        mode_used = "genotypic"
    
    warnings  = raw.get("warnings", [])

    # Compute statistics over off-diagonal values for interpretation
    off_diag: List[float] = []
    for i, row in enumerate(r_matrix):
        for j, val in enumerate(row):
            if i != j and val is not None:
                try:
                    off_diag.append(float(val))
                except (TypeError, ValueError):
                    pass

    # Use fixed scale (-1 to +1) for all heatmaps to ensure color scale
    # always represents the full correlation range, making sign structure clear
    min_val = -1.0
    max_val = 1.0

    # Build a human-readable interpretation
    n_traits = len(labels)
    strong_pos = sum(1 for v in off_diag if v >= 0.7)
    strong_neg = sum(1 for v in off_diag if v <= -0.7)
    mode_desc = {
        "phenotypic": "all observations (reflecting field-level co-variation)",
        "genotypic": "genotype means (reflecting genotype-level relationships)"
    }
    interp_parts = [
        f"Heatmap shows {request.method.capitalize()} r-values for "
        f"{n_traits} trait(s) based on {mode_desc[mode_used]}."
    ]
    if strong_pos:
        interp_parts.append(
            f"{strong_pos} trait pair(s) show strong positive correlation (r ≥ 0.70)."
        )
    if strong_neg:
        interp_parts.append(
            f"{strong_neg} trait pair(s) show strong negative correlation (r ≤ −0.70)."
        )
    interpretation = "  ".join(interp_parts)

    return HeatmapModuleResponse(
        dataset_token=request.dataset_token,
        matrix=r_matrix,
        labels=labels,
        min_val=min_val,
        max_val=max_val,
        method=request.method,
        interpretation=interpretation,
        warnings=warnings,
    )
