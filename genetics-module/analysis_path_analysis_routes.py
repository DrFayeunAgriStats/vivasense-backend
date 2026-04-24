"""
VivaSense – Path Analysis Module

POST /analysis/path-analysis

Partitions phenotypic correlations between causal (independent) traits and a
target (dependent) trait into direct and indirect path coefficients.

Algorithm (Wright 1921):
  p = R_xx⁻¹ · r_xy
  where R_xx is the correlation matrix of causal traits and r_xy is the
  vector of correlations between each causal trait and the target.

  Indirect effect of Xi via Xj = r(Xi, Xj) · p_j
  Residual effect              = sqrt(1 - R²)
  R²                           = Σ(p_i · r_iy)

All computations use per-genotype means to isolate genetic signal from
replication noise (standard agronomic practice for path analysis).
"""

import base64
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from scipy import stats as scipy_stats

import dataset_cache
from multitrait_upload_routes import read_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


# ============================================================================
# SCHEMAS
# ============================================================================

class PathAnalysisRequest(BaseModel):
    dataset_token: Optional[str] = Field(
        default=None,
        description="Token from POST /upload/dataset or /genetics/upload-preview",
    )
    trait_columns: List[str] = Field(..., min_length=2, description="Causal (independent) trait columns")
    target_trait: str = Field(..., description="Dependent variable column (e.g. yield)")
    method: str = Field(default="pearson", pattern="^(pearson|spearman)$")


class IndirectEffect(BaseModel):
    via_trait: str
    value: float


class PathCoefficientRow(BaseModel):
    trait: str
    direct_effect: float
    indirect_effects: List[IndirectEffect]
    total_indirect: float
    total_correlation_with_target: float


class PathAnalysisResponse(BaseModel):
    status: str
    dataset_token: str
    target_trait: str
    causal_traits: List[str]
    n_observations: int
    method: str
    r_squared: float
    residual_effect: float
    path_coefficients: List[PathCoefficientRow]
    correlation_with_target: Dict[str, float]
    interpretation: str
    warnings: List[str]


# ============================================================================
# HELPERS
# ============================================================================

def _compute_genotype_means(
    df: pd.DataFrame,
    genotype_col: str,
    all_traits: List[str],
) -> pd.DataFrame:
    """Average all trait columns by genotype, drop rows with all-NaN traits."""
    numeric_cols = [c for c in all_traits if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    means = df.groupby(genotype_col)[numeric_cols].mean()
    means = means.dropna(how="all")
    return means


def _corr_matrix(data: np.ndarray, method: str) -> np.ndarray:
    """Compute correlation matrix using Pearson or Spearman."""
    n_traits = data.shape[1]
    r = np.ones((n_traits, n_traits))
    for i in range(n_traits):
        for j in range(i + 1, n_traits):
            mask = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
            xi, xj = data[mask, i], data[mask, j]
            if len(xi) < 3:
                r[i, j] = r[j, i] = float("nan")
                continue
            if method == "spearman":
                rho, _ = scipy_stats.spearmanr(xi, xj)
            else:
                rho, _ = scipy_stats.pearsonr(xi, xj)
            r[i, j] = r[j, i] = float(rho)
    return r


def _generate_interpretation(
    path_rows: List[PathCoefficientRow],
    target_trait: str,
    r_squared: float,
    residual_effect: float,
    n: int,
) -> str:
    lines = [
        f"Path analysis was performed with {len(path_rows)} causal trait(s) "
        f"predicting {target_trait} (n = {n} genotype means). "
        f"The causal traits collectively explain {r_squared * 100:.1f}% of the "
        f"variation in {target_trait} (residual effect = {residual_effect:.4f})."
    ]

    # Rank by |direct_effect|
    ranked = sorted(path_rows, key=lambda x: abs(x.direct_effect), reverse=True)
    dominant = ranked[0]
    sign = "positive" if dominant.direct_effect > 0 else "negative"
    lines.append(
        f"The strongest direct effect on {target_trait} was exerted by "
        f"{dominant.trait} (p = {dominant.direct_effect:.4f}, {sign}), "
        f"which also had a total correlation of r = {dominant.total_correlation_with_target:.4f} "
        f"with {target_trait}."
    )

    # Note traits where indirect effects substantially modify the direct effect
    for row in ranked[1:]:
        if abs(row.total_indirect) > 0.1:
            lines.append(
                f"{row.trait} had a direct effect of {row.direct_effect:.4f} but "
                f"total indirect effects of {row.total_indirect:.4f}, indicating "
                f"that its correlation with {target_trait} (r = {row.total_correlation_with_target:.4f}) "
                f"is substantially mediated through other causal traits."
            )

    if r_squared < 0.30:
        lines.append(
            "The low R² suggests that important determinants of "
            f"{target_trait} were not included among the causal traits, or "
            "that relationships are non-linear."
        )

    return " ".join(lines)


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/analysis/path-analysis",
    response_model=PathAnalysisResponse,
    summary="Compute path coefficients (direct and indirect effects) for a target trait",
)
async def analysis_path_analysis(request: PathAnalysisRequest):
    """
    Partition phenotypic correlations into direct and indirect path coefficients.

    Each causal trait's total phenotypic correlation with the target trait is
    decomposed into:
      • Direct effect  — the path coefficient (p_i) solved from the normal equations
      • Indirect effects — contribution through each other causal trait

    Computations use per-genotype means to reduce environmental noise.

    Requires at least 2 causal trait columns, a distinct target_trait column,
    and a dataset_token from POST /upload/dataset.
    """
    if not request.dataset_token:
        raise HTTPException(
            status_code=400,
            detail="dataset_token is required. Upload your file first via POST /upload/dataset.",
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

    if request.target_trait in request.trait_columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"target_trait '{request.target_trait}' must not appear in trait_columns. "
                "It is the dependent variable."
            ),
        )

    all_traits = request.trait_columns + [request.target_trait]

    try:
        file_bytes = base64.b64decode(ctx["base64_content"])
        df = read_file(file_bytes, ctx["file_type"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}") from exc

    missing_cols = [c for c in all_traits if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Columns not found in dataset: {missing_cols}",
        )

    genotype_col = ctx.get("genotype_column")
    if not genotype_col or genotype_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "Genotype column not found in dataset context. "
                "Re-upload via POST /upload/dataset with genotype_column specified."
            ),
        )

    warnings: List[str] = []

    means_df = _compute_genotype_means(df, genotype_col, all_traits)

    # Drop genotypes with any missing value in the analysis traits
    means_df = means_df[all_traits].dropna()
    n = len(means_df)

    if n < 4:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient genotype means (n={n}). Minimum 4 required for path analysis.",
        )
    if n < 10:
        warnings.append(f"Small sample size (n={n} genotype means) — path coefficients may be unstable.")

    data = means_df[all_traits].values.astype(float)
    n_causal = len(request.trait_columns)

    # Full correlation matrix over [causal traits..., target]
    full_r = _corr_matrix(data, request.method)

    if np.any(np.isnan(full_r)):
        warnings.append(
            "Some trait pairs had fewer than 3 complete observations; "
            "NaN correlations were replaced with 0 for matrix inversion."
        )
        full_r = np.where(np.isnan(full_r), 0.0, full_r)
        np.fill_diagonal(full_r, 1.0)

    R_xx = full_r[:n_causal, :n_causal]   # causal × causal
    r_xy = full_r[:n_causal, n_causal]    # causal × target

    # Solve for path coefficients: p = R_xx⁻¹ · r_xy
    try:
        p = np.linalg.solve(R_xx, r_xy)
    except np.linalg.LinAlgError:
        # Singular matrix — try pseudoinverse
        warnings.append(
            "Correlation matrix of causal traits is singular (perfect multicollinearity). "
            "Path coefficients computed using pseudoinverse — interpret with caution."
        )
        p = np.linalg.lstsq(R_xx, r_xy, rcond=None)[0]

    # Coefficient of determination and residual
    r_squared = float(np.dot(p, r_xy))
    r_squared = max(0.0, min(1.0, r_squared))
    residual = math.sqrt(max(0.0, 1.0 - r_squared))

    # Build output rows
    path_rows: List[PathCoefficientRow] = []
    corr_with_target: Dict[str, float] = {}

    for i, trait in enumerate(request.trait_columns):
        direct = float(p[i])
        r_iy = float(r_xy[i])
        corr_with_target[trait] = r_iy

        indirect_effects: List[IndirectEffect] = []
        total_indirect = 0.0
        for j, other_trait in enumerate(request.trait_columns):
            if i == j:
                continue
            via_value = float(R_xx[i, j] * p[j])
            indirect_effects.append(IndirectEffect(via_trait=other_trait, value=via_value))
            total_indirect += via_value

        path_rows.append(
            PathCoefficientRow(
                trait=trait,
                direct_effect=direct,
                indirect_effects=indirect_effects,
                total_indirect=total_indirect,
                total_correlation_with_target=r_iy,
            )
        )

    interpretation = _generate_interpretation(
        path_rows, request.target_trait, r_squared, residual, n
    )

    return PathAnalysisResponse(
        status="success",
        dataset_token=request.dataset_token,
        target_trait=request.target_trait,
        causal_traits=request.trait_columns,
        n_observations=n,
        method=request.method,
        r_squared=r_squared,
        residual_effect=residual,
        path_coefficients=path_rows,
        correlation_with_target=corr_with_target,
        interpretation=interpretation,
        warnings=warnings,
    )
