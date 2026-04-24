"""
VivaSense – Path Analysis Module

POST /analysis/path-analysis

Decomposes phenotypic correlations between traits into direct and indirect
path coefficients. Identifies which traits exert independent direct effects
on the outcome trait versus those that influence it through other traits.
Essential for designing efficient indirect selection strategies.

Method:
  Multiple linear regression on standardised (or unstandardised) trait values.
  Path coefficients = standardised partial regression coefficients.
  Indirect effects computed via the correlation–path decomposition identity:
    r_xy = Σ p_ij × r_xj   (Wright, 1934)

References:
  Wright, S. (1934). The method of path coefficients. Annals of Mathematical
  Statistics, 5(3), 161-215.

  Dewey, D.R. & Lu, K.H. (1959). A correlation and path coefficient analysis
  of components of crested wheat grass seed production.
  Agronomy Journal, 51(9), 515-518.
"""

import base64
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from scipy import stats as scipy_stats

import dataset_cache
from module_schemas import (
    CorrelationDecomp,
    PathAnalysisRequest,
    PathAnalysisResponse,
    PathCoefficient,
)
from multitrait_upload_routes import read_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


# ============================================================================
# COMPUTATION HELPERS
# ============================================================================

def _compute_path_analysis(
    data: pd.DataFrame,
    outcome: str,
    predictors: List[str],
    standardize: bool,
) -> Dict[str, Any]:
    """Path analysis via multiple OLS regression (statsmodels for inference)."""
    import statsmodels.api as sm

    cols = [outcome] + predictors
    data_clean = data[cols].dropna()
    n = len(data_clean)

    min_n = max(len(predictors) * 3, len(predictors) + 3)
    if n < min_n:
        raise ValueError(
            f"Insufficient observations for path analysis. "
            f"Need at least {min_n} complete rows, found {n}."
        )

    # Standardise if requested
    if standardize:
        means = data_clean.mean()
        stds = data_clean.std()
        # Avoid division by zero for constant columns
        stds[stds == 0] = 1.0
        data_std = (data_clean - means) / stds
    else:
        data_std = data_clean.copy()

    X = data_std[predictors].values.astype(float)
    y = data_std[outcome].values.astype(float)

    # Fit OLS with statsmodels for proper inference
    X_with_const = sm.add_constant(X)
    sm_model = sm.OLS(y, X_with_const).fit()

    # Extract coefficients (index 0 is intercept)
    coefs = sm_model.params[1:]
    std_errors = sm_model.bse[1:]
    t_stats = sm_model.tvalues[1:]
    p_vals = sm_model.pvalues[1:]

    path_coefs: List[PathCoefficient] = []
    for i, pred in enumerate(predictors):
        path_coefs.append(
            PathCoefficient(
                predictor=pred,
                direct_effect=float(coefs[i]),
                std_error=float(std_errors[i]),
                t_statistic=float(t_stats[i]),
                p_value=float(p_vals[i]),
                significant=bool(p_vals[i] < 0.05),
            )
        )

    # Correlation matrix for decomposition
    all_std_df = data_std[cols]
    corr_matrix = all_std_df.corr()

    decomp: List[CorrelationDecomp] = []
    for i, pred in enumerate(predictors):
        total_r = float(corr_matrix.loc[pred, outcome])
        direct = float(coefs[i])
        indirect = total_r - direct
        pct_direct = (
            float(abs(direct) / abs(total_r) * 100) if total_r != 0 else 0.0
        )
        decomp.append(
            CorrelationDecomp(
                predictor=pred,
                total_correlation=total_r,
                direct_effect=direct,
                indirect_effect=indirect,
                percent_direct=pct_direct,
            )
        )

    # Indirect effects matrix: predictor_i → outcome via predictor_j
    indirect_matrix: Dict[str, Dict[str, float]] = {}
    for i, pred1 in enumerate(predictors):
        indirect_matrix[pred1] = {}
        for j, pred2 in enumerate(predictors):
            if i != j:
                r_12 = float(corr_matrix.loc[pred1, pred2])
                path_2_outcome = float(coefs[j])
                indirect_matrix[pred1][pred2] = float(r_12 * path_2_outcome)

    r_squared = float(sm_model.rsquared)
    residual = float(np.sqrt(max(0.0, 1.0 - r_squared)))

    # Path diagram data for frontend visualisation
    path_diagram_data: Dict[str, Any] = {
        "nodes": [{"id": outcome, "type": "outcome"}]
        + [{"id": p, "type": "predictor"} for p in predictors],
        "edges": [
            {
                "from": pc.predictor,
                "to": outcome,
                "weight": pc.direct_effect,
                "p_value": pc.p_value,
                "significant": pc.significant,
            }
            for pc in path_coefs
        ],
        "residual_path": residual,
    }

    return {
        "n_observations": n,
        "path_coefficients": path_coefs,
        "correlation_decomposition": decomp,
        "r_squared": r_squared,
        "residual_effect": residual,
        "indirect_effects": indirect_matrix,
        "path_diagram_data": path_diagram_data,
    }


def _build_interpretation(
    outcome: str,
    path_coefs: List[PathCoefficient],
    decomp: List[CorrelationDecomp],
    r_squared: float,
    residual: float,
    standardize: bool,
) -> str:
    """Generate a plain-English thesis-quality path analysis interpretation."""
    coef_type = "standardised" if standardize else "unstandardised"

    parts = [
        f"Path analysis ({coef_type} coefficients) explained {r_squared * 100:.1f}% "
        f"of the variation in {outcome} (R² = {r_squared:.4f}; "
        f"residual path = {residual:.4f}). "
    ]

    # Sort by absolute direct effect
    sorted_coefs = sorted(path_coefs, key=lambda x: abs(x.direct_effect), reverse=True)

    sig_direct = [c for c in sorted_coefs if c.significant]
    nonsig_direct = [c for c in sorted_coefs if not c.significant]

    if sig_direct:
        sig_strs = [
            f"{c.predictor} (p = {c.direct_effect:+.3f}, P = {c.p_value:.4f})"
            for c in sig_direct
        ]
        parts.append(
            f"Traits with significant direct effects on {outcome}: "
            + "; ".join(sig_strs) + ". "
        )

    if nonsig_direct:
        nonsig_names = [c.predictor for c in nonsig_direct]
        parts.append(
            f"Traits with non-significant direct effects: {', '.join(nonsig_names)}. "
        )

    # Highlight indirect effects
    for d in decomp:
        if abs(d.indirect_effect) > abs(d.direct_effect) and d.total_correlation != 0:
            parts.append(
                f"{d.predictor} influences {outcome} primarily through indirect pathways "
                f"(indirect = {d.indirect_effect:+.3f} vs. direct = {d.direct_effect:+.3f}); "
                f"consider selecting via correlated traits rather than {d.predictor} directly. "
            )

    parts.append(
        "Traits with large positive direct effects are the most efficient direct "
        "selection targets. Traits with large indirect effects suggest indirect "
        "selection strategies via correlated intermediary traits."
    )

    return "".join(parts)


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/analysis/path-analysis",
    response_model=PathAnalysisResponse,
    summary="Run path analysis — decompose correlations into direct and indirect effects",
)
async def analysis_path_analysis(request: PathAnalysisRequest):
    """
    Decompose phenotypic correlations between predictor traits and an outcome
    trait into direct path coefficients and indirect effects through other
    predictors.

    Useful for:
      • Identifying traits for direct selection (large direct effects)
      • Indirect selection strategy design (large indirect effects)
      • Understanding correlation structure in breeding populations

    Requires a dataset_token from POST /upload/dataset.
    """
    if not request.predictor_traits:
        raise HTTPException(
            status_code=400,
            detail="At least one predictor trait is required.",
        )

    if request.outcome_trait in request.predictor_traits:
        raise HTTPException(
            status_code=400,
            detail="outcome_trait must not appear in predictor_traits.",
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

    all_cols = [request.outcome_trait] + request.predictor_traits
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Columns not found in dataset: {missing}",
        )

    for col in all_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' must be numeric for path analysis.",
            )

    try:
        result = _compute_path_analysis(
            df,
            request.outcome_trait,
            request.predictor_traits,
            request.standardize,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Path analysis error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Path analysis failed: {exc}") from exc

    interpretation = _build_interpretation(
        outcome=request.outcome_trait,
        path_coefs=result["path_coefficients"],
        decomp=result["correlation_decomposition"],
        r_squared=result["r_squared"],
        residual=result["residual_effect"],
        standardize=request.standardize,
    )

    return PathAnalysisResponse(
        status="success",
        outcome_trait=request.outcome_trait,
        predictor_traits=request.predictor_traits,
        n_observations=result["n_observations"],
        path_coefficients=result["path_coefficients"],
        correlation_decomposition=result["correlation_decomposition"],
        r_squared=result["r_squared"],
        residual_effect=result["residual_effect"],
        indirect_effects=result["indirect_effects"],
        interpretation=interpretation,
        path_diagram_data=result["path_diagram_data"],
    )
