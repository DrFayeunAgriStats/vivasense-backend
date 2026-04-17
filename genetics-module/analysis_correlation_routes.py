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
from module_schemas import CorrelationModuleResponse, ModuleRequest
from trait_association_interpretation import generate_trait_association_interpretation, _compute_risk_flags
import dataset_cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])

_STATISTICAL_NOTE = (
    "Correlations were computed using genotype-level means; "
    "significance levels are based on the number of genotypes."
)


class CorrelationModuleRequest(ModuleRequest):
    """Extends ModuleRequest with correlation-specific options."""
    method: str = Field(default="pearson", pattern="^(pearson|spearman)$")
    gxe_significant: bool = Field(default=False)
    environment_context: str = Field(default="single_environment", pattern="^(single_environment|multi_environment)$")


def _compute_significant_pairs_and_strongest(
    trait_names: List[str],
    r_matrix: List[List[Optional[float]]],
    p_matrix: List[List[Optional[float]]],
    alpha: float = 0.05
) -> tuple[int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Compute significant pairs count and strongest positive/negative pairs.
    """
    n_significant_pairs = 0
    strongest_positive = None
    strongest_negative = None
    max_positive_r = 0.0
    max_negative_r = 0.0
    
    n = len(trait_names)
    for i in range(n):
        for j in range(i + 1, n):  # Upper triangle only
            r_val = r_matrix[i][j] if r_matrix and i < len(r_matrix) and j < len(r_matrix[i]) else None
            p_val = p_matrix[i][j] if p_matrix and i < len(p_matrix) and j < len(p_matrix[i]) else None
            
            if r_val is not None and p_val is not None and p_val <= alpha:
                n_significant_pairs += 1
            
            # Track strongest pairs
            if r_val is not None:
                if r_val > 0 and r_val > max_positive_r:
                    max_positive_r = r_val
                    strongest_positive = {
                        "trait_1": trait_names[i],
                        "trait_2": trait_names[j],
                        "r": r_val
                    }
                elif r_val < 0 and r_val < max_negative_r:
                    max_negative_r = r_val
                    strongest_negative = {
                        "trait_1": trait_names[i],
                        "trait_2": trait_names[j],
                        "r": r_val
                    }
    
    return n_significant_pairs, strongest_positive, strongest_negative


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
    n_observations = raw.get("n_observations", 0)
    r_matrix = raw.get("r_matrix", [])
    p_matrix = raw.get("p_matrix", [])
    warnings = raw.get("warnings", [])

    # Compute significant pairs and strongest pairs for interpretation
    n_significant_pairs, strongest_positive, strongest_negative = (
        _compute_significant_pairs_and_strongest(trait_names, r_matrix, p_matrix)
    )

    # Build risk flags and generate Python interpretation (never raw R text)
    risk_flags = _compute_risk_flags(
        n=n_observations,
        analysis_unit="genotype_mean",
        gxe_significant=request.gxe_significant,
    )
    python_interpretation = generate_trait_association_interpretation(
        n_traits=len(trait_names),
        n_observations=n_observations,
        n_significant_pairs=n_significant_pairs,
        strongest_positive=strongest_positive,
        strongest_negative=strongest_negative,
        risk_flags=risk_flags,
        gxe_significant=request.gxe_significant,
        environment_context=request.environment_context,
    )
    logger.info(
        "Correlation interpretation generated (Python): %d chars, n_obs=%d, n_sig_pairs=%d",
        len(python_interpretation), n_observations, n_significant_pairs,
    )

    corr = CorrelationResponse(
        trait_names=trait_names,
        n_observations=n_observations,
        method=request.method,
        r_matrix=r_matrix,
        p_matrix=p_matrix,
        interpretation=python_interpretation,
        warnings=warnings,
        statistical_note=_STATISTICAL_NOTE,
    )

    return CorrelationModuleResponse(
        dataset_token=request.dataset_token,
        trait_names=corr.trait_names,
        r_matrix=corr.r_matrix,
        p_matrix=corr.p_matrix,
        n_observations=corr.n_observations,
        method=corr.method,
        statistical_note=corr.statistical_note,
        interpretation=corr.interpretation,
        warnings=corr.warnings,
    )
