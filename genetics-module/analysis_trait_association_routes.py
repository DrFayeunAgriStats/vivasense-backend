"""
VivaSense – Trait Association Intelligence Module

POST /genetics/trait-association/analyze

Unified backend module that computes Pearson correlation coefficients,
p-values, and prepares heatmap-ready data with statistical summaries,
strongest associations, and risk flags.

Replaces separate correlation and heatmap workflows with one backend.
"""

import base64
import logging
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import Field

from trait_relationships_routes import (
    TraitRelationshipsEngine,
    _build_wide_records,
    tr_engine,
)
from trait_association_interpretation import (
    generate_trait_association_interpretation,
    _compute_risk_flags,
)
from multitrait_upload_routes import read_file
from module_schemas import TraitAssociationModuleRequest, TraitAssociationModuleResponse, SignificantPair, StrongestPair, TraitAssociationSummary, TraitAssociationHeatmap, InterpretationPlaceholder
import dataset_cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


def _classify_strength(r: float) -> str:
    """Classify correlation strength based on absolute r value."""
    abs_r = abs(r)
    if abs_r < 0.20:
        return "very weak"
    elif abs_r < 0.40:
        return "weak"
    elif abs_r < 0.60:
        return "moderate"
    elif abs_r < 0.80:
        return "strong"
    else:
        return "very strong"


def _classify_direction(r: float) -> str:
    """Classify correlation direction."""
    if r > 0:
        return "positive"
    elif r < 0:
        return "negative"
    else:
        return "none"


def _classify_confidence(n: int, analysis_unit: str, gxe_significant: bool, risk_flags: List[str]) -> str:
    """
    Classify confidence status for trait association pairs.

    For v1 beta, if pairwise N is not tracked then all pairwise
    confidence is downgraded to "limited_by_pairwise_n".
    A future version will require a pairwise n_matrix for full
    confidence grading.
    """
    if "pairwise_n_not_tracked" in risk_flags:
        return "limited_by_pairwise_n"

    # Placeholder for future behavior when pairwise N tracking is available.
    return "limited_by_pairwise_n"


def _classify_selection_relevance(r: float, p_value: float, alpha: float, n: int) -> str:
    """Classify selection signal."""
    if p_value > alpha:
        return "exploratory only"

    abs_r = abs(r)

    if n < 10:
        return "potentially useful with validation"
    if abs_r >= 0.60 and p_value <= alpha and n >= 10:
        return "useful with validation"
    return "exploratory only"


def _build_matrix_dict(trait_names: List[str], matrix: List[List[Optional[float]]]) -> Dict[str, Dict[str, Optional[float]]]:
    """Convert a 2D array matrix into a dictionary of dictionaries keyed by trait names."""
    result = {}
    for i, t1 in enumerate(trait_names):
        result[t1] = {}
        for j, t2 in enumerate(trait_names):
            if i < len(matrix) and j < len(matrix[i]):
                result[t1][t2] = matrix[i][j]
            else:
                result[t1][t2] = None
    return result


def _process_trait_association_data(
    ctx: Dict[str, Any], request: TraitAssociationModuleRequest, engine: TraitRelationshipsEngine
) -> Dict[str, Any]:
    """Process trait association data (blocking operations)."""
    geno_col = ctx["genotype_column"]
    rep_col = ctx["rep_column"]
    env_col = ctx["environment_column"] if ctx["mode"] == "multi" else None

    # Decode and read file
    file_bytes = base64.b64decode(ctx["base64_content"])
    df = read_file(file_bytes, ctx["file_type"])

    # Validate trait columns exist
    missing = [c for c in request.trait_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Trait columns not found in dataset: {missing}")

    # Build records
    records = _build_wide_records(df, geno_col, rep_col, request.trait_columns, env_col)

    # Run correlation analysis
    raw = engine.run_correlation(records, request.trait_columns, "pearson")
    
    return raw


@router.post(
    "/genetics/trait-association/analyze",
    response_model=TraitAssociationModuleResponse,
    summary="Unified trait association analysis with correlations, p-values, and heatmap data",
)
async def analyze_trait_association(request: TraitAssociationModuleRequest):
    """
    Compute unified trait association analysis including:
    - Pearson correlation coefficients
    - Pairwise p-values
    - Significant trait pairs with classifications
    - Strongest associations
    - Risk flags
    - Heatmap-ready matrix data

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
            detail="Trait association requires at least 2 trait columns.",
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

    import asyncio
    try:
        # Run all blocking I/O and computation in a thread to avoid blocking the event loop
        raw = await asyncio.to_thread(
            _process_trait_association_data, ctx, request, engine
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Trait association analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Trait association analysis failed: {exc}") from exc

    # Extract data
    trait_names = raw.get("trait_names", request.trait_columns)
    n_observations = raw.get("n_observations", 0)
    r_matrix_list = raw.get("r_matrix", [])
    p_matrix_list = raw.get("p_matrix", [])
    warnings = raw.get("warnings", [])

    # Build dict matrices
    correlation_matrix = _build_matrix_dict(trait_names, r_matrix_list)
    pvalue_matrix = _build_matrix_dict(trait_names, p_matrix_list)

    # Compute risk flags (needed for confidence classification)
    risk_flags = _compute_risk_flags(n_observations, request.analysis_unit, request.gxe_significant)

    # Compute significant pairs
    significant_pairs = []
    strongest_positive = None
    strongest_negative = None
    max_positive_r = 0.0  # Start from 0, only track positive correlations
    max_negative_r = 0.0  # Start from 0, only track negative correlations

    for i, trait1 in enumerate(trait_names):
        for j, trait2 in enumerate(trait_names):
            if i >= j:  # Skip diagonal and lower triangle to avoid duplicates
                continue

            r_val = r_matrix_list[i][j]
            p_val = p_matrix_list[i][j]

            if r_val is None or p_val is None:
                continue

            if p_val <= request.alpha:
                direction = _classify_direction(r_val)
                strength = _classify_strength(r_val)
                confidence_status = _classify_confidence(n_observations, request.analysis_unit, request.gxe_significant, risk_flags)
                selection_signal = _classify_selection_relevance(r_val, p_val, request.alpha, n_observations)

                # For beta safety, cap selection signal if pairwise N is not tracked.
                if "pairwise_n_not_tracked" in risk_flags and selection_signal == "useful with validation":
                    selection_signal = "potentially useful with validation"

                pair = SignificantPair(
                    trait_1=trait1,
                    trait_2=trait2,
                    r=r_val,
                    p_value=p_val,
                    direction=direction,
                    strength=strength,
                    confidence_status=confidence_status,
                    selection_signal=selection_signal,
                )
                significant_pairs.append(pair)

            # Track strongest pairs (only off-diagonal)
            if r_val > 0 and r_val > max_positive_r:
                max_positive_r = r_val
                strongest_positive = StrongestPair(trait_1=trait1, trait_2=trait2, r=r_val)

            if r_val < 0 and r_val < max_negative_r:
                max_negative_r = r_val
                strongest_negative = StrongestPair(trait_1=trait1, trait_2=trait2, r=r_val)

    # Risk flags already computed above

    # Build summary
    summary = TraitAssociationSummary(
        num_traits=len(trait_names),
        num_significant_pairs=len(significant_pairs),
        strongest_positive_pair_label=f"{strongest_positive.trait_1} vs {strongest_positive.trait_2}" if strongest_positive else None,
        strongest_negative_pair_label=f"{strongest_negative.trait_1} vs {strongest_negative.trait_2}" if strongest_negative else None,
    )

    # Build heatmap data
    heatmap = TraitAssociationHeatmap(
        matrix=correlation_matrix,
        type="correlation_heatmap_ready"
    )

    # Derive strongest SIGNIFICANT pairs for interpretation.
    # strongest_positive / strongest_negative above track the overall strongest
    # (used for the response summary label) but may be non-significant.
    # The interpretation must only reference pairs whose p-value is significant.
    sig_strongest_positive = None
    sig_strongest_negative = None
    for sp in significant_pairs:
        if sp.r > 0 and (sig_strongest_positive is None or sp.r > sig_strongest_positive.r):
            sig_strongest_positive = sp
        elif sp.r < 0 and (sig_strongest_negative is None or sp.r < sig_strongest_negative.r):
            sig_strongest_negative = sp

    # Generate actual interpretation (not placeholder)
    interpretation_text = generate_trait_association_interpretation(
        n_traits=len(trait_names),
        n_observations=n_observations,
        n_significant_pairs=len(significant_pairs),
        strongest_positive=sig_strongest_positive.dict() if sig_strongest_positive else None,
        strongest_negative=sig_strongest_negative.dict() if sig_strongest_negative else None,
        risk_flags=risk_flags,
        gxe_significant=request.gxe_significant,
        environment_context=request.environment_context
    )

    return TraitAssociationModuleResponse(
        analysis_unit=request.analysis_unit,
        n_observations=n_observations,
        alpha=request.alpha,
        environment_context=request.environment_context,
        gxe_significant=request.gxe_significant,
        trait_names=trait_names,
        correlation_matrix=correlation_matrix,
        pvalue_matrix=pvalue_matrix,
        significant_pairs=significant_pairs,
        strongest_positive_pair=strongest_positive,
        strongest_negative_pair=strongest_negative,
        risk_flags=risk_flags,
        summary=summary,
        heatmap=heatmap,
        interpretation=interpretation_text,
        dataset_token=request.dataset_token,
        warnings=warnings,
    )