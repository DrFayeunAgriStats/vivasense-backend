# -*- coding: utf-8 -*-
"""
VivaSense – MANOVA (Multivariate Analysis of Variance) Module

POST /analysis/manova

Tests the overall effect of a grouping factor on multiple correlated traits
simultaneously. More powerful than separate ANOVAs when traits are correlated.
Suitable for overall genotype performance assessment across trait panels.

Supported test statistics:
  • Wilks' Lambda       — most commonly reported; sensitive to small samples
  • Pillai's Trace      — most robust to violations; recommended default
  • Hotelling-Lawley    — powerful when group differences are concentrated
  • Roy's Largest Root  — powerful for a single canonical dimension

Computation uses statsmodels MANOVA.

References:
  Wilks, S.S. (1932). Certain generalizations in the analysis of variance.
  Biometrika, 24(3/4), 471-494.

  Morrison, D.F. (1976). Multivariate Statistical Methods (2nd ed.).
  McGraw-Hill.

  Rencher, A.C. (2002). Methods of Multivariate Analysis (2nd ed.). Wiley.
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
    MANOVARequest,
    MANOVAResponse,
    UnivariateResult,
)
from multitrait_upload_routes import read_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])

# Map user-friendly statistic names to statsmodels keys
_STAT_MAP = {
    "wilks": "Wilks' lambda",
    "pillai": "Pillai's trace",
    "hotelling-lawley": "Hotelling-Lawley trace",
    "roy": "Roy's greatest root",
}

_STAT_DISPLAY = {
    "wilks": "Wilks' Lambda",
    "pillai": "Pillai's Trace",
    "hotelling-lawley": "Hotelling-Lawley Trace",
    "roy": "Roy's Greatest Root",
}


# ============================================================================
# COMPUTATION HELPERS
# ============================================================================

def _compute_manova(
    data: pd.DataFrame,
    traits: List[str],
    factor: str,
    test_stat: str,
    alpha: float,
) -> Dict[str, Any]:
    """Fit MANOVA using statsmodels and extract results."""
    from statsmodels.multivariate.manova import MANOVA

    data_clean = data[traits + [factor]].dropna()
    n_groups = data_clean[factor].nunique()

    if n_groups < 2:
        raise ValueError("MANOVA requires at least 2 groups in the factor column.")

    min_obs_required = n_groups * len(traits)
    if len(data_clean) < min_obs_required:
        raise ValueError(
            f"Insufficient observations for MANOVA. "
            f"Need at least {min_obs_required} rows (n_groups x n_traits), "
            f"but only {len(data_clean)} complete rows found."
        )

    # Sanitise column names for use in patsy formulae — reserved keywords and
    # special characters are replaced with safe identifiers.
    import keyword
    import re

    def _safe_name(name: str, idx: int) -> str:
        safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
        if not safe or safe[0].isdigit() or keyword.iskeyword(safe):
            safe = f"trait_{idx}"
        return safe

    safe_traits = [_safe_name(t, i) for i, t in enumerate(traits)]
    safe_factor = _safe_name(factor, len(traits))

    rename_map = {t: s for t, s in zip(traits, safe_traits)}
    rename_map[factor] = safe_factor
    data_safe = data_clean.rename(columns=rename_map)

    # Build formula: "safe_trait1 + safe_trait2 ~ C(safe_factor)"
    formula = " + ".join(safe_traits) + f" ~ C({safe_factor})"

    try:
        manova = MANOVA.from_formula(formula, data=data_safe)
        results = manova.mv_test()
    except Exception as exc:
        raise ValueError(f"MANOVA computation failed: {exc}") from exc

    # Locate factor results key (statsmodels uses the formula term name)
    factor_key = f"C({safe_factor})"
    if factor_key not in results.results:
        available = list(results.results.keys())
        raise ValueError(
            f"Factor key '{factor_key}' not found in MANOVA results. "
            f"Available keys: {available}"
        )

    factor_results = results.results[factor_key]
    stat_key = _STAT_MAP.get(test_stat)
    if stat_key not in factor_results:
        # Try first available statistic
        stat_key = next(iter(factor_results.keys()))

    stat_table = factor_results[stat_key]
    stat_value = float(stat_table["Value"].values[0])
    f_stat = float(stat_table["F Value"].values[0])
    num_df = int(round(float(stat_table["Num DF"].values[0])))
    den_df = int(round(float(stat_table["Den DF"].values[0])))
    p_value = float(stat_table["Pr > F"].values[0])

    # Univariate follow-up ANOVAs + effect sizes (use original data_clean)
    univariate_results: List[UnivariateResult] = []
    effect_sizes: Dict[str, float] = {}

    for trait in traits:
        groups = [
            data_clean[data_clean[factor] == g][trait].dropna().values
            for g in data_clean[factor].unique()
        ]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            univariate_results.append(
                UnivariateResult(
                    trait=trait,
                    f_statistic=float("nan"),
                    p_value=float("nan"),
                    significant=False,
                    eta_squared=None,
                )
            )
            effect_sizes[trait] = 0.0
            continue

        f, p = scipy_stats.f_oneway(*groups)

        # Partial eta-squared = SS_between / SS_total
        grand_mean = float(data_clean[trait].mean())
        ss_between = sum(
            len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups
        )
        ss_total = float(((data_clean[trait] - grand_mean) ** 2).sum())
        eta2 = ss_between / ss_total if ss_total > 0 else 0.0

        univariate_results.append(
            UnivariateResult(
                trait=trait,
                f_statistic=float(f),
                p_value=float(p),
                significant=bool(p < alpha),
                eta_squared=float(eta2),
            )
        )
        effect_sizes[trait] = float(eta2)

    return {
        "n_groups": n_groups,
        "n_observations": len(data_clean),
        "test_statistic_name": _STAT_DISPLAY.get(test_stat, test_stat),
        "test_statistic_value": stat_value,
        "f_statistic": f_stat,
        "df_hypothesis": num_df,
        "df_error": den_df,
        "p_value": p_value,
        "significant": p_value < alpha,
        "univariate_results": univariate_results,
        "effect_sizes": effect_sizes,
    }


def _build_interpretation(
    test_stat_name: str,
    stat_value: float,
    f_stat: float,
    p_value: float,
    significant: bool,
    n_groups: int,
    traits: List[str],
    univariate: List[UnivariateResult],
    alpha: float,
) -> str:
    """Generate a plain-English thesis-quality MANOVA interpretation."""
    sig_str = f"p {'< ' if significant else '= '}{p_value:.4f}"

    if significant:
        sig_traits = [u.trait for u in univariate if u.significant]
        nonsig_traits = [u.trait for u in univariate if not u.significant]

        parts = [
            f"MANOVA revealed a significant overall effect of the grouping factor "
            f"across the {len(traits)}-trait multivariate response "
            f"({test_stat_name} = {stat_value:.4f}, F = {f_stat:.3f}, {sig_str}). "
            f"The {n_groups} groups differ significantly in their combined trait profile."
        ]

        if sig_traits:
            parts.append(
                f" Follow-up univariate ANOVAs at alpha = {alpha} indicated significant "
                f"group differences for: {', '.join(sig_traits)}."
            )
        if nonsig_traits:
            parts.append(
                f" Non-significant univariate effects were observed for: "
                f"{', '.join(nonsig_traits)}, suggesting these traits contribute "
                f"less to the multivariate discrimination."
            )

        parts.append(
            " MANOVA controls the family-wise error rate and accounts for trait "
            "correlations, providing a more powerful overall test than separate ANOVAs."
        )
        return "".join(parts)
    else:
        return (
            f"MANOVA found no significant overall multivariate effect of the grouping "
            f"factor ({test_stat_name} = {stat_value:.4f}, F = {f_stat:.3f}, {sig_str}). "
            f"The groups do not differ significantly across the combined {len(traits)}-trait "
            f"profile at alpha = {alpha}. Consider whether the sample size is adequate "
            f"(recommended: >= 20 observations per group for MANOVA robustness) or whether "
            f"the chosen traits are discriminating for the factor of interest."
        )


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/analysis/manova",
    response_model=MANOVAResponse,
    summary="Run MANOVA — multivariate analysis of variance across multiple traits",
)
async def analysis_manova(request: MANOVARequest):
    """
    Test whether a grouping factor (e.g. genotype) significantly affects
    multiple traits simultaneously using Multivariate Analysis of Variance.

    Provides:
      • Overall MANOVA test result (Wilks, Pillai, Hotelling-Lawley, or Roy)
      • Follow-up univariate ANOVAs per trait
      • Partial eta-squared effect sizes

    Requires >= 2 trait columns and a dataset_token from POST /upload/dataset.
    """
    if len(request.trait_columns) < 2:
        raise HTTPException(
            status_code=400,
            detail="MANOVA requires at least 2 trait columns.",
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

    # Validate columns
    all_cols = request.trait_columns + [request.factor_column] + request.covariates
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Columns not found in dataset: {missing}",
        )

    for trait in request.trait_columns:
        if not pd.api.types.is_numeric_dtype(df[trait]):
            raise HTTPException(
                status_code=400,
                detail=f"Trait column '{trait}' must be numeric.",
            )

    test_stat_key = request.test_statistic.lower()
    if test_stat_key not in _STAT_MAP:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported test_statistic '{request.test_statistic}'. "
                f"Choose: Wilks, Pillai, Hotelling-Lawley, Roy."
            ),
        )

    try:
        result = _compute_manova(
            df,
            request.trait_columns,
            request.factor_column,
            test_stat_key,
            request.alpha,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("MANOVA computation error: %s", exc)
        raise HTTPException(status_code=500, detail=f"MANOVA failed: {exc}") from exc

    interpretation = _build_interpretation(
        test_stat_name=result["test_statistic_name"],
        stat_value=result["test_statistic_value"],
        f_stat=result["f_statistic"],
        p_value=result["p_value"],
        significant=result["significant"],
        n_groups=result["n_groups"],
        traits=request.trait_columns,
        univariate=result["univariate_results"],
        alpha=request.alpha,
    )

    assumptions_note = (
        "MANOVA assumes multivariate normality and homogeneity of covariance matrices "
        "(Box's M test). With n > 20 observations per group the test is robust to "
        "moderate violations of multivariate normality (Tabachnick & Fidell, 2013). "
        "Pillai's Trace is the most robust statistic when assumptions are uncertain."
    )

    return MANOVAResponse(
        status="success",
        n_traits=len(request.trait_columns),
        n_groups=result["n_groups"],
        n_observations=result["n_observations"],
        traits=request.trait_columns,
        factor=request.factor_column,
        test_statistic_name=result["test_statistic_name"],
        test_statistic_value=result["test_statistic_value"],
        f_statistic=result["f_statistic"],
        df_hypothesis=result["df_hypothesis"],
        df_error=result["df_error"],
        p_value=result["p_value"],
        significant=result["significant"],
        univariate_results=result["univariate_results"],
        effect_sizes=result["effect_sizes"],
        interpretation=interpretation,
        assumptions_note=assumptions_note,
    )
