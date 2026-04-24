"""
VivaSense – Non-Parametric Tests Module

POST /analysis/nonparametric

Provides robust alternatives to parametric ANOVA when normality assumptions
are violated. Suitable for ordinal data, disease scores, insect counts, and
skewed agronomic measurements.

Supported tests:
  • Kruskal-Wallis H-test — non-parametric one-way ANOVA
  • Friedman test       — non-parametric repeated-measures ANOVA
  • Dunn's post-hoc    — pairwise comparisons after Kruskal-Wallis

References:
  Kruskal, W.H. & Wallis, W.A. (1952). Use of ranks in one-criterion
  variance analysis. JASA, 47(260), 583-621.

  Friedman, M. (1937). The use of ranks to avoid the assumption of
  normality implicit in the analysis of variance. JASA, 32(200), 675-701.

  Dunn, O.J. (1964). Multiple comparisons using rank sums.
  Technometrics, 6(3), 241-252.
"""

import base64
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from scipy import stats

import dataset_cache
from module_schemas import (
    GroupMedian,
    NonparametricRequest,
    NonparametricResponse,
    PairwiseComparison,
)
from multitrait_upload_routes import read_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


# ============================================================================
# COMPUTATION HELPERS
# ============================================================================

def _kruskal_wallis(
    data: pd.DataFrame, trait: str, group: str
) -> Dict[str, Any]:
    """Kruskal-Wallis H-test (non-parametric one-way ANOVA)."""
    group_names = data[group].dropna().unique().tolist()
    groups = [data[data[group] == g][trait].dropna() for g in group_names]

    # Remove empty groups
    non_empty = [(g_name, g) for g_name, g in zip(group_names, groups) if len(g) > 0]
    if len(non_empty) < 2:
        raise ValueError("Need at least 2 non-empty groups for Kruskal-Wallis test.")

    group_names_clean = [x[0] for x in non_empty]
    groups_clean = [x[1] for x in non_empty]

    h_stat, p_value = stats.kruskal(*groups_clean)
    df = len(groups_clean) - 1

    # Group summaries with global ranks
    all_values = pd.concat([g.reset_index(drop=True) for g in groups_clean])
    all_ranks = all_values.rank()
    offset = 0
    group_medians: List[GroupMedian] = []
    for g_name, g in zip(group_names_clean, groups_clean):
        g_ranks = all_ranks.iloc[offset : offset + len(g)]
        group_medians.append(
            GroupMedian(
                group_name=str(g_name),
                median=float(g.median()),
                n=len(g),
                rank_sum=float(g_ranks.sum()),
            )
        )
        offset += len(g)

    return {
        "statistic": float(h_stat),
        "statistic_name": "H",
        "p_value": float(p_value),
        "df": int(df),
        "group_medians": group_medians,
        "n_groups": len(groups_clean),
        "n_observations": sum(len(g) for g in groups_clean),
    }


def _friedman_test(
    data: pd.DataFrame, trait: str, group: str, block: str
) -> Dict[str, Any]:
    """Friedman test (non-parametric repeated-measures ANOVA)."""
    # Pivot to wide format: blocks × groups
    try:
        pivot = data.pivot_table(values=trait, index=block, columns=group, aggfunc="mean")
    except Exception as exc:
        raise ValueError(f"Could not reshape data for Friedman test: {exc}") from exc

    pivot_clean = pivot.dropna()

    if len(pivot_clean) < 2:
        raise ValueError(
            "Friedman test requires at least 2 complete blocks after removing missing data."
        )
    if pivot_clean.shape[1] < 2:
        raise ValueError("Friedman test requires at least 2 treatment groups.")

    chi2_stat, p_value = stats.friedmanchisquare(
        *[pivot_clean[col].values for col in pivot_clean.columns]
    )
    df = pivot_clean.shape[1] - 1

    # Mean ranks across blocks
    rank_matrix = pivot_clean.rank(axis=1)
    group_medians: List[GroupMedian] = []
    for col in pivot_clean.columns:
        group_medians.append(
            GroupMedian(
                group_name=str(col),
                median=float(pivot_clean[col].median()),
                n=int(len(pivot_clean)),
                mean_rank=float(rank_matrix[col].mean()),
            )
        )

    return {
        "statistic": float(chi2_stat),
        "statistic_name": "Chi-square",
        "p_value": float(p_value),
        "df": int(df),
        "group_medians": group_medians,
        "n_groups": int(pivot_clean.shape[1]),
        "n_observations": int(pivot_clean.shape[0] * pivot_clean.shape[1]),
    }


def _dunn_posthoc(
    data: pd.DataFrame, trait: str, group: str, alpha: float = 0.05
) -> List[PairwiseComparison]:
    """Dunn's post-hoc test for pairwise comparisons (after Kruskal-Wallis).

    Uses Bonferroni correction. Falls back to a manual implementation when
    scikit-posthocs is not installed.
    """
    try:
        import scikit_posthocs as sp  # optional dependency

        posthoc = sp.posthoc_dunn(
            data, val_col=trait, group_col=group, p_adjust="bonferroni"
        )
        groups = posthoc.index.tolist()
        comparisons: List[PairwiseComparison] = []
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i < j:
                    p_val = float(posthoc.loc[g1, g2])
                    comparisons.append(
                        PairwiseComparison(
                            group1=str(g1),
                            group2=str(g2),
                            p_value=p_val,
                            significant=p_val < alpha,
                            adjustment="bonferroni",
                        )
                    )
        return comparisons

    except ImportError:
        # Manual Dunn's test fallback
        return _dunn_manual(data, trait, group, alpha)


def _dunn_manual(
    data: pd.DataFrame, trait: str, group: str, alpha: float
) -> List[PairwiseComparison]:
    """Manual Dunn's test with Bonferroni correction (no external dependency)."""
    clean = data[[trait, group]].dropna()
    n_total = len(clean)

    # Assign global ranks
    clean = clean.copy()
    clean["_rank"] = clean[trait].rank(method="average")

    group_names = clean[group].unique().tolist()
    k = len(group_names)
    n_comparisons = k * (k - 1) // 2

    # Tie correction factor
    tie_groups = clean["_rank"].value_counts()
    tie_correction = sum(t**3 - t for t in tie_groups.values)

    comparisons: List[PairwiseComparison] = []
    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            if i >= j:
                continue
            n1 = len(clean[clean[group] == g1])
            n2 = len(clean[clean[group] == g2])
            r1 = clean[clean[group] == g1]["_rank"].mean()
            r2 = clean[clean[group] == g2]["_rank"].mean()

            denom = math.sqrt(
                (n_total * (n_total + 1) / 12.0 - tie_correction / (12.0 * (n_total - 1)))
                * (1.0 / n1 + 1.0 / n2)
            )
            if denom == 0:
                z = 0.0
            else:
                z = (r1 - r2) / denom

            raw_p = 2 * (1 - stats.norm.cdf(abs(z)))
            bonferroni_p = min(1.0, raw_p * n_comparisons)

            comparisons.append(
                PairwiseComparison(
                    group1=str(g1),
                    group2=str(g2),
                    p_value=float(bonferroni_p),
                    significant=bonferroni_p < alpha,
                    adjustment="bonferroni",
                )
            )
    return comparisons


def _build_interpretation(
    test_type: str,
    statistic_name: str,
    statistic: float,
    p_value: float,
    significant: bool,
    n_groups: int,
    alpha: float,
    posthoc: Optional[List[PairwiseComparison]],
) -> str:
    """Generate a plain-English thesis-quality interpretation."""
    sig_str = f"p {'< ' if significant else '= '}{p_value:.4f}"

    if test_type in ("kruskal-wallis", "dunn"):
        if significant:
            parts = [
                f"The Kruskal-Wallis H-test detected significant differences among "
                f"the {n_groups} groups ({statistic_name} = {statistic:.3f}, df = {n_groups - 1}, "
                f"{sig_str}). Groups differ significantly in median trait values, indicating "
                f"that at least one group has a different distribution."
            ]
            if posthoc:
                n_sig = sum(1 for c in posthoc if c.significant)
                parts.append(
                    f" Dunn's post-hoc test (Bonferroni-corrected) identified "
                    f"{n_sig} of {len(posthoc)} pairwise comparisons as significant at α = {alpha}."
                )
            parts.append(
                " This test is appropriate when ANOVA normality assumptions are violated, "
                "as commonly occurs with disease scores, insect counts, and ordinal ratings."
            )
            return "".join(parts)
        else:
            return (
                f"The Kruskal-Wallis H-test found no significant differences among the "
                f"{n_groups} groups ({statistic_name} = {statistic:.3f}, df = {n_groups - 1}, "
                f"{sig_str}). There is insufficient evidence to conclude that groups differ "
                f"in median trait values at α = {alpha}. Use this non-parametric test when "
                "ANOVA normality assumptions cannot be met."
            )

    else:  # friedman
        if significant:
            return (
                f"The Friedman test detected a significant treatment effect across blocks "
                f"({statistic_name} = {statistic:.3f}, df = {n_groups - 1}, {sig_str}). "
                f"Treatments differ significantly in their ranked performance. "
                f"This non-parametric alternative to repeated-measures ANOVA is robust "
                f"to non-normality and heterogeneity of variance."
            )
        else:
            return (
                f"The Friedman test found no significant treatment effect across blocks "
                f"({statistic_name} = {statistic:.3f}, df = {n_groups - 1}, {sig_str}). "
                f"There is insufficient evidence that treatments differ in ranked performance "
                f"at α = {alpha}."
            )


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/analysis/nonparametric",
    response_model=NonparametricResponse,
    summary="Run non-parametric tests (Kruskal-Wallis, Friedman, Dunn's post-hoc)",
)
async def analysis_nonparametric(request: NonparametricRequest):
    """
    Run a non-parametric statistical test on a trait column grouped by a
    factor column.

    Supported tests:
      • **kruskal-wallis** — Non-parametric one-way ANOVA (≥2 independent groups)
      • **friedman**       — Non-parametric repeated-measures ANOVA (requires block_column)
      • **dunn**           — Kruskal-Wallis + Dunn's pairwise post-hoc

    Requires a dataset_token from POST /upload/dataset.
    """
    # Retrieve dataset
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
    for col in [request.trait_column, request.group_column]:
        if col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{col}' not found in dataset.",
            )

    if not pd.api.types.is_numeric_dtype(df[request.trait_column]):
        raise HTTPException(
            status_code=400,
            detail=f"Trait column '{request.trait_column}' must be numeric.",
        )

    test_type = request.test_type.lower()
    if test_type not in ("kruskal-wallis", "friedman", "dunn"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported test_type '{request.test_type}'. Choose: kruskal-wallis, friedman, dunn.",
        )

    # Friedman requires a block column
    if test_type == "friedman" and not request.block_column:
        raise HTTPException(
            status_code=400,
            detail="Friedman test requires block_column to be specified.",
        )

    if test_type == "friedman" and request.block_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Block column '{request.block_column}' not found in dataset.",
        )

    # Run test
    try:
        if test_type == "friedman":
            result = _friedman_test(
                df, request.trait_column, request.group_column, request.block_column
            )
            posthoc_results = None
        else:
            result = _kruskal_wallis(df, request.trait_column, request.group_column)
            if test_type == "dunn" and result.get("significant", True):
                posthoc_results = _dunn_posthoc(
                    df, request.trait_column, request.group_column, request.alpha
                )
            else:
                posthoc_results = None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Non-parametric test error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Non-parametric test failed: {exc}") from exc

    significant = result["p_value"] < request.alpha

    # Run Dunn's post-hoc automatically when Kruskal-Wallis is significant
    if test_type == "kruskal-wallis" and significant:
        try:
            posthoc_results = _dunn_posthoc(
                df, request.trait_column, request.group_column, request.alpha
            )
        except Exception as exc:
            logger.warning("Dunn's post-hoc failed (non-fatal): %s", exc)
            posthoc_results = None

    # Assumptions metadata
    n_obs = result["n_observations"]
    n_groups = result["n_groups"]
    assumptions = {
        "minimum_n_per_group": 5,
        "sample_size_adequate": all(gm.n >= 5 for gm in result["group_medians"]),
        "ties_present": bool(df[request.trait_column].duplicated().any()),
        "note": (
            "Kruskal-Wallis assumes independent random samples and at least "
            "ordinal measurement scale. Ties are handled by average-rank method."
            if test_type != "friedman"
            else "Friedman test assumes blocks are independent and measurements are at least ordinal."
        ),
    }

    interpretation = _build_interpretation(
        test_type=test_type,
        statistic_name=result["statistic_name"],
        statistic=result["statistic"],
        p_value=result["p_value"],
        significant=significant,
        n_groups=n_groups,
        alpha=request.alpha,
        posthoc=posthoc_results,
    )

    return NonparametricResponse(
        status="success",
        test_type=test_type,
        trait=request.trait_column,
        group_column=request.group_column,
        n_groups=n_groups,
        n_observations=n_obs,
        statistic=result["statistic"],
        statistic_name=result["statistic_name"],
        p_value=result["p_value"],
        df=result["df"],
        significant=significant,
        group_medians=result["group_medians"],
        posthoc_results=posthoc_results,
        interpretation=interpretation,
        assumptions_met=assumptions,
    )
