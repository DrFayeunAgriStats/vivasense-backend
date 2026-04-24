"""
VivaSense – Smith-Hazel Selection Index Module

POST /analysis/selection-index

Computes the Smith-Hazel selection index to optimise multi-trait selection
for maximum aggregate genetic merit. Combines multiple traits with economic
weights into a single linear selection criterion.

The index weights (b-values) are derived by solving:
    P b = G a
where:
    P = phenotypic (co)variance matrix (from genotype means)
    G = genetic (co)variance matrix (diagonal from P × h² when not provided)
    a = vector of economic weights
    b = vector of index weights (solved via P⁻¹ G a)

Expected genetic gain per generation:
    ΔGᵢ = i × r_IH × σ_Gᵢ  (Smith, 1936; Hazel, 1943)

References:
  Smith, H.F. (1936). A discriminant function for plant selection.
  Annals of Eugenics, 7(3), 240-250.

  Hazel, L.N. (1943). The genetic basis for constructing selection indices.
  Genetics, 28(6), 476-490.

  Falconer, D.S. & Mackay, T.F.C. (1996). Introduction to Quantitative
  Genetics (4th ed.). Longman.
"""

import base64
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from scipy import linalg

import dataset_cache
from module_schemas import (
    GenotypeIndex,
    SelectionIndexRequest,
    SelectionIndexResponse,
)
from multitrait_upload_routes import read_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


# ============================================================================
# COMPUTATION HELPERS
# ============================================================================

def _compute_selection_index(
    data: pd.DataFrame,
    traits: List[str],
    economic_weights: Dict[str, float],
    h2_values: Dict[str, float],
    gen_corr_matrix: Optional[np.ndarray],
    selection_intensity: float,
) -> Dict[str, Any]:
    """Smith-Hazel Selection Index computation."""

    # Aggregate to genotype means
    geno_col = None
    for candidate in ("genotype", "variety", "cultivar", "entry", "line", "accession"):
        if candidate in data.columns:
            geno_col = candidate
            break

    if geno_col is not None:
        geno_means = (
            data.groupby(geno_col)[traits]
            .mean()
            .dropna()
        )
    else:
        geno_means = data[traits].dropna()

    n_genotypes = len(geno_means)
    if n_genotypes < 5:
        raise ValueError(
            f"Selection index requires at least 5 genotypes. Found {n_genotypes}."
        )

    # Phenotypic covariance matrix (P)
    P = geno_means.cov().values.astype(float)

    # Validate economic weights
    missing_w = [t for t in traits if t not in economic_weights]
    if missing_w:
        raise ValueError(
            f"Economic weights missing for traits: {missing_w}. "
            "Provide a weight for every trait in trait_columns."
        )

    a = np.array([economic_weights[t] for t in traits], dtype=float)

    # Heritability values — default to 0.5 if not provided
    h2_array = np.array(
        [h2_values.get(t, 0.5) for t in traits], dtype=float
    )

    # Genetic covariance matrix (G)
    if gen_corr_matrix is not None:
        G = np.asarray(gen_corr_matrix, dtype=float)
        if G.shape != (len(traits), len(traits)):
            raise ValueError(
                f"Genetic correlation matrix shape {G.shape} does not match "
                f"the number of traits ({len(traits)})."
            )
        # Scale genetic correlations to genetic covariances
        pheno_var = np.diag(P)
        gen_var = h2_array * pheno_var
        gen_std = np.sqrt(gen_var)
        G = G * np.outer(gen_std, gen_std)
    else:
        # Simplified: use phenotypic correlations scaled by h²
        pheno_corr = geno_means.corr().values.astype(float)
        pheno_var = np.diag(P)
        gen_var = h2_array * pheno_var
        gen_std = np.sqrt(gen_var)
        G = pheno_corr * np.outer(gen_std, gen_std)

    # Solve P b = G a for index weights b
    try:
        b = linalg.solve(P, G @ a)
    except linalg.LinAlgError:
        logger.warning("Singular phenotypic covariance matrix — using pseudoinverse.")
        b = linalg.pinv(P) @ G @ a

    # Index scores for each genotype
    if isinstance(geno_means.index, pd.Index) and geno_col is not None:
        geno_labels = [str(g) for g in geno_means.index.tolist()]
    else:
        geno_labels = [str(i) for i in range(n_genotypes)]

    index_values = geno_means.values.astype(float) @ b

    # Build sorted list of scores
    genotype_scores: List[GenotypeIndex] = []
    for i, geno in enumerate(geno_labels):
        genotype_scores.append(
            GenotypeIndex(
                genotype=geno,
                index_value=float(index_values[i]),
                rank=0,  # filled below
            )
        )

    genotype_scores.sort(key=lambda x: x.index_value, reverse=True)
    for rank, gs in enumerate(genotype_scores, start=1):
        gs.rank = rank

    # Selection accuracy (correlation between index and true aggregate genotype)
    var_index = float(b.T @ P @ b)
    var_aggregate = float(a.T @ G @ a)

    if var_index > 0 and var_aggregate > 0:
        accuracy = float(np.sqrt(var_index / var_aggregate))
    else:
        accuracy = 0.0

    # Expected genetic gain per trait: ΔGᵢ = i × r_IH × σ_Gᵢ × (b_i / √var_I)
    expected_gain: Dict[str, float] = {}
    sqrt_var_index = float(np.sqrt(max(var_index, 1e-12)))

    for j, trait in enumerate(traits):
        sigma_g = float(np.sqrt(max(G[j, j], 0.0)))
        delta_g = (
            selection_intensity * accuracy * sigma_g * b[j] / sqrt_var_index
        )
        expected_gain[trait] = float(delta_g)

    # Total merit = Σ aᵢ × ΔGᵢ
    total_merit = float(sum(economic_weights[t] * expected_gain[t] for t in traits))

    # Relative efficiency vs. single-trait selection
    relative_efficiency: Dict[str, float] = {}
    for j, trait in enumerate(traits):
        sigma_p = float(np.sqrt(max(P[j, j], 1e-12)))
        sigma_g = float(np.sqrt(max(G[j, j], 0.0)))
        single_trait_gain = selection_intensity * h2_array[j] * sigma_g
        if abs(single_trait_gain) > 1e-10:
            relative_efficiency[trait] = float(
                abs(expected_gain[trait]) / abs(single_trait_gain)
            )
        else:
            relative_efficiency[trait] = 0.0

    # Top genotypes
    n_selected = max(1, int(n_genotypes * 0.1))
    selected_genotypes = [gs.genotype for gs in genotype_scores[:n_selected]]

    return {
        "n_genotypes": n_genotypes,
        "index_weights": {traits[i]: float(b[i]) for i in range(len(traits))},
        "genotype_scores": genotype_scores,
        "expected_gain": expected_gain,
        "total_merit": total_merit,
        "selection_accuracy": accuracy,
        "relative_efficiency": relative_efficiency,
        "selected_genotypes": selected_genotypes,
    }


def _build_interpretation(
    traits: List[str],
    index_weights: Dict[str, float],
    expected_gain: Dict[str, float],
    total_merit: float,
    accuracy: float,
    selected_genotypes: List[str],
    selection_intensity: float,
    economic_weights: Dict[str, float],
    n_genotypes: int,
) -> str:
    """Generate a plain-English thesis-quality selection index interpretation."""
    n_selected = max(1, int(n_genotypes * 0.1))
    top_pct = round(n_selected / n_genotypes * 100, 1)

    parts = [
        f"The Smith-Hazel selection index was computed for {n_genotypes} genotypes "
        f"across {len(traits)} traits (selection intensity i = {selection_intensity:.3f}, "
        f"top {top_pct}% selected). "
    ]

    # Accuracy interpretation
    if accuracy >= 0.80:
        acc_str = "high"
    elif accuracy >= 0.50:
        acc_str = "moderate"
    else:
        acc_str = "low"

    parts.append(
        f"The selection index showed {acc_str} accuracy (r_IH = {accuracy:.3f}), "
        f"indicating that the index is "
        + (
            "a reliable predictor of true aggregate breeding value."
            if accuracy >= 0.80
            else "a moderately reliable predictor of aggregate breeding value. "
            "Increase heritability estimates or expand the trait set to improve accuracy."
            if accuracy >= 0.50
            else "a limited predictor. Consider refining heritability estimates, "
            "increasing trait number, or adjusting economic weights."
        )
        + " "
    )

    # Gain summary
    gain_strs = []
    for t in traits:
        g = expected_gain[t]
        gain_strs.append(f"{t}: {g:+.4f}")
    parts.append(
        f"Expected genetic gain per generation: {'; '.join(gain_strs)}. "
        f"Total aggregate merit = {total_merit:.4f}. "
    )

    # Top weight
    top_weight_trait = max(index_weights, key=lambda t: abs(index_weights[t]))
    parts.append(
        f"The index assigns the largest weight to '{top_weight_trait}' "
        f"(b = {index_weights[top_weight_trait]:.4f}), reflecting its high "
        f"economic importance and genetic variance. "
    )

    if selected_genotypes:
        listed = ", ".join(selected_genotypes[:5])
        suffix = f" (and {len(selected_genotypes) - 5} more)" if len(selected_genotypes) > 5 else ""
        parts.append(
            f"Recommended genotypes for selection (top {top_pct}%): {listed}{suffix}. "
        )

    parts.append(
        "The Smith-Hazel index maximises the correlation between the selection "
        "criterion and aggregate genotypic value, providing the theoretically "
        "optimal linear combination of traits for simultaneous improvement."
    )

    return "".join(parts)


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/analysis/selection-index",
    response_model=SelectionIndexResponse,
    summary="Run Smith-Hazel Selection Index for multi-trait genotype selection",
)
async def analysis_selection_index(request: SelectionIndexRequest):
    """
    Compute the Smith-Hazel selection index to optimise simultaneous selection
    across multiple traits for maximum aggregate genetic merit.

    The index combines traits with economic weights into a single score
    (I = b₁X₁ + b₂X₂ + … + bₙXₙ) and ranks genotypes accordingly.

    Provides:
      • Optimal index weights (b-values) solving P b = G a
      • Per-genotype index scores and ranks
      • Expected genetic gain per trait per generation
      • Selection accuracy (r_IH)
      • Relative efficiency vs. single-trait selection

    Heritabilities default to 0.5 if not provided via genetic_parameters.
    Requires a dataset_token from POST /upload/dataset.
    """
    if len(request.trait_columns) < 2:
        raise HTTPException(
            status_code=400,
            detail="Selection index requires at least 2 trait columns.",
        )

    # Validate economic weights cover all traits
    missing_w = [t for t in request.trait_columns if t not in request.economic_weights]
    if missing_w:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Economic weights missing for: {missing_w}. "
                "Provide a weight for every trait in trait_columns."
            ),
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

    missing_cols = [c for c in request.trait_columns if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Trait columns not found in dataset: {missing_cols}",
        )

    for trait in request.trait_columns:
        if not pd.api.types.is_numeric_dtype(df[trait]):
            raise HTTPException(
                status_code=400,
                detail=f"Trait column '{trait}' must be numeric.",
            )

    # Extract h2 values from genetic_parameters if provided
    h2_values: Dict[str, float] = {}
    for trait in request.trait_columns:
        if trait in request.genetic_parameters:
            h2_values[trait] = float(request.genetic_parameters[trait].get("h2", 0.5))
        else:
            h2_values[trait] = 0.5

    # Build genetic correlation matrix if provided
    gen_corr_matrix: Optional[np.ndarray] = None
    if request.genetic_correlations is not None:
        try:
            gen_corr_matrix = np.array(
                [
                    [request.genetic_correlations[t1].get(t2, 1.0 if t1 == t2 else 0.0)
                     for t2 in request.trait_columns]
                    for t1 in request.trait_columns
                ],
                dtype=float,
            )
        except (KeyError, TypeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid genetic_correlations structure: {exc}",
            ) from exc

    try:
        result = _compute_selection_index(
            df,
            request.trait_columns,
            request.economic_weights,
            h2_values,
            gen_corr_matrix,
            request.selection_intensity,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Selection index computation error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Selection index failed: {exc}"
        ) from exc

    interpretation = _build_interpretation(
        traits=request.trait_columns,
        index_weights=result["index_weights"],
        expected_gain=result["expected_gain"],
        total_merit=result["total_merit"],
        accuracy=result["selection_accuracy"],
        selected_genotypes=result["selected_genotypes"],
        selection_intensity=request.selection_intensity,
        economic_weights=request.economic_weights,
        n_genotypes=result["n_genotypes"],
    )

    return SelectionIndexResponse(
        status="success",
        traits=request.trait_columns,
        n_genotypes=result["n_genotypes"],
        index_weights=result["index_weights"],
        genotype_scores=result["genotype_scores"],
        expected_gain=result["expected_gain"],
        total_merit=result["total_merit"],
        selection_accuracy=result["selection_accuracy"],
        relative_efficiency=result["relative_efficiency"],
        selected_genotypes=result["selected_genotypes"],
        interpretation=interpretation,
    )
