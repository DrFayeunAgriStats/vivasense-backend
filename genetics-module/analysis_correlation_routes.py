"""
VivaSense – Correlation Analysis Module

POST /analysis/correlation

Returns:
  • Correlation matrix (r_matrix) — Pearson or Spearman r for every trait pair
  • p-value matrix (p_matrix)
  • n_observations — number of unique genotype means used
  • method, statistical_note, interpretation, warnings

Requires a dataset_token from POST /upload/dataset.
Wraps the existing TraitRelationshipsEngine (vivasense_trait_relationships.R)
and the _build_wide_records helper from trait_relationships_routes.
"""

import base64
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from trait_relationships_routes import (
    TraitRelationshipsEngine,
    _build_wide_records,
    tr_engine,
)
from multitrait_upload_routes import read_file
from trait_relationships_schemas import CorrelationResponse
from module_schemas import CorrelationModuleResponse, CorrelationModuleRequest
from trait_association_interpretation import generate_dual_mode_correlation_interpretation
import dataset_cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])

_STATISTICAL_NOTE = (
    "Dual-mode analysis evaluates both phenotypic (all observations) "
    "and genotypic (genotype means) correlations."
)


@router.post(
    "/analysis/correlation",
    response_model=CorrelationModuleResponse,
    summary="Compute phenotypic correlations between selected traits",
)
async def analysis_correlation(request: CorrelationModuleRequest):
    """
    Compute Pearson or Spearman phenotypic correlation coefficients and
    p-values for all pairs of the requested trait columns.

    Correlations are based on per-genotype means (averaged across reps/environments),
    which isolates genetic signal from replication noise.

    Requires ≥ 2 trait columns and a dataset_token from POST /upload/dataset.
    """
    import trait_relationships_routes as _tr_mod  # lazy: tr_engine set on startup

    engine: TraitRelationshipsEngine = _tr_mod.tr_engine
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
            detail="Correlation requires at least 2 trait columns.",
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

    # Validate trait columns exist
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
        logger.error("Correlation engine error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {exc}") from exc

    # Parse R response — discard R-generated interpretation text entirely
    trait_names = raw.get("trait_names", request.trait_columns)
    warnings = raw.get("warnings", [])
    phenotypic = raw.get("phenotypic", {})
    genotypic = raw.get("genotypic", {})

    python_interpretation = generate_dual_mode_correlation_interpretation(
        trait_names=trait_names,
        genotypic=genotypic,
        phenotypic=phenotypic,
        user_objective=request.user_objective
    )
    logger.info(
        "Correlation interpretation generated (Python): %d chars",
        len(python_interpretation)
    )

    corr = CorrelationResponse(
        trait_names=trait_names,
        method=request.method,
        phenotypic=phenotypic,
        genotypic=genotypic,
        interpretation=python_interpretation,
        warnings=warnings,
        statistical_note=_STATISTICAL_NOTE,
    )

    return CorrelationModuleResponse(
        dataset_token=request.dataset_token,
        trait_names=corr.trait_names,
        method=corr.method,
        phenotypic=corr.phenotypic,
        genotypic=corr.genotypic,
        statistical_note=corr.statistical_note,
        interpretation=corr.interpretation,
        warnings=corr.warnings,
    )
