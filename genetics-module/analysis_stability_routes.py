"""
VivaSense – Stability Analysis Module

POST /analysis/stability

Computes Eberhart-Russell regression-based stability metrics for genotypes
tested across multiple environments.

Metrics returned per genotype:
  • bi   — regression coefficient (1 = average stability)
  • S²di — deviation from regression (0 = perfectly predictable)
  • mean — genotype mean across environments
  • rank — rank by mean performance (1 = highest)
  • stability_class — "stable" | "responsive_favorable" | "responsive_poor" | "unpredictable"

Also computed:
  • Environmental means per site
  • Shukla stability variance (σ²i)
  • Best-stable genotypes (bi ≈ 1.0, low S²di, above-average mean)
  • Plain-English interpretation

Reference:
  Eberhart, S.A. and Russell, W.A. (1966). Stability parameters for comparing
  varieties. Crop Science, 6(1), 36-40.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from scipy import stats

import dataset_cache
from module_schemas import (
    GenotypeStability,
    StabilityRequest,
    StabilityResponse,
)
from multitrait_upload_routes import read_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


# ============================================================================
# COMPUTATION HELPERS
# ============================================================================

def _compute_eberhart_russell(
    df: pd.DataFrame,
    trait_col: str,
    genotype_col: str,
    env_col: str,
) -> Dict[str, Any]:
    """
    Compute Eberhart-Russell (1966) stability statistics.

    df must have columns: genotype_col, env_col, trait_col.
    Multiple reps per (genotype, env) cell are averaged before regression.
    """
    # Average across reps within each (genotype, environment) cell
    cell_means = (
        df.groupby([genotype_col, env_col])[trait_col]
        .mean()
        .reset_index()
        .rename(columns={trait_col: "cell_mean"})
    )

    # Require at least 2 environments
    environments = cell_means[env_col].unique().tolist()
    n_envs = len(environments)
    if n_envs < 2:
        raise ValueError(
            "Stability analysis requires data from at least 2 environments. "
            f"Only {n_envs} found."
        )

    # Environmental means and indices
    env_means = cell_means.groupby(env_col)["cell_mean"].mean()
    grand_mean = env_means.mean()
    env_index = env_means - grand_mean   # I_j = mean_j - grand_mean

    # Genotype grand means
    geno_means = cell_means.groupby(genotype_col)["cell_mean"].mean()

    # Eberhart-Russell per genotype
    genotypes = sorted(geno_means.index.tolist())
    n_genos = len(genotypes)

    stability_rows: List[Dict[str, Any]] = []
    shukla_vars: Dict[str, float] = {}

    for geno in genotypes:
        geno_env = cell_means[cell_means[genotype_col] == geno].copy()
        geno_env = geno_env.set_index(env_col)["cell_mean"]

        # Align with env_index (some envs may be missing for this genotype)
        aligned_env_idx = env_index.reindex(geno_env.index).dropna()
        aligned_geno = geno_env.reindex(aligned_env_idx.index)

        n_e = len(aligned_env_idx)
        if n_e < 2:
            # Insufficient data for this genotype
            bi = 1.0
            s2di = float("nan")
        else:
            x = aligned_env_idx.values
            y = aligned_geno.values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            bi = float(slope)

            # S²di = mean squared deviation from regression (residual variance)
            y_pred = intercept + bi * x
            ss_res = float(np.sum((y - y_pred) ** 2))
            # Eberhart-Russell: S²di = (SS_dev / (n_e - 2)) - MSe_pooled
            # Simplified: use residual variance directly (degrees of freedom = n_e - 2)
            s2di = ss_res / max(n_e - 2, 1) if n_e > 2 else 0.0

        geno_mean = float(geno_means[geno])

        stability_rows.append({
            "genotype": geno,
            "mean": geno_mean,
            "bi": bi,
            "s2di": s2di if not (isinstance(s2di, float) and math.isnan(s2di)) else 0.0,
        })

        # Shukla (1972) stability variance — sum of squared deviations of
        # (Y_ij - bi * I_j) from the genotype mean, summed over environments.
        if n_e >= 2:
            residuals = aligned_geno.values - (grand_mean + geno_mean - grand_mean + bi * aligned_env_idx.values)
            shukla_vars[geno] = float(np.var(residuals, ddof=1))
        else:
            shukla_vars[geno] = 0.0

    # Sort by mean (descending) and assign ranks
    stability_rows.sort(key=lambda r: r["mean"], reverse=True)
    for rank, row in enumerate(stability_rows, start=1):
        row["rank"] = rank

    # Classify genotypes
    grand_mean_float = float(grand_mean)
    # S²di threshold: 75th percentile of non-NaN values
    s2di_values = [r["s2di"] for r in stability_rows if not math.isnan(r["s2di"])]
    s2di_threshold = float(np.percentile(s2di_values, 75)) if s2di_values else 0.0

    for row in stability_rows:
        bi = row["bi"]
        s2di = row["s2di"]
        mean = row["mean"]
        high_s2di = s2di > s2di_threshold

        if 0.9 <= bi <= 1.1 and not high_s2di:
            row["stability_class"] = "stable"
        elif bi > 1.1 and not high_s2di:
            row["stability_class"] = "responsive_favorable"
        elif bi < 0.9 and not high_s2di:
            row["stability_class"] = "responsive_poor"
        else:
            row["stability_class"] = "unpredictable"

    # Best stable genotypes: stable class + above-average mean
    best_stable = [
        r["genotype"]
        for r in stability_rows
        if r["stability_class"] == "stable" and r["mean"] >= grand_mean_float
    ]
    if not best_stable:
        # Fallback: top-3 by mean among all genotypes
        best_stable = [r["genotype"] for r in stability_rows[:3]]

    return {
        "stability_rows": stability_rows,
        "env_means": {str(k): float(v) for k, v in env_means.items()},
        "grand_mean": grand_mean_float,
        "n_genotypes": n_genos,
        "n_environments": n_envs,
        "best_stable": best_stable,
        "environments": [str(e) for e in environments],
    }


def _generate_stability_interpretation(
    trait: str,
    grand_mean: float,
    n_genotypes: int,
    n_environments: int,
    stability_rows: List[Dict[str, Any]],
    best_stable: List[str],
) -> str:
    """Generate plain-English thesis-quality interpretation."""
    sections: List[tuple] = []

    # 1. Overview
    n_stable = sum(1 for r in stability_rows if r["stability_class"] == "stable")
    n_resp_fav = sum(1 for r in stability_rows if r["stability_class"] == "responsive_favorable")
    n_resp_poor = sum(1 for r in stability_rows if r["stability_class"] == "responsive_poor")
    n_unpred = sum(1 for r in stability_rows if r["stability_class"] == "unpredictable")

    overview = (
        f"Eberhart-Russell (1966) stability analysis was performed for {trait} "
        f"across {n_environments} environments using {n_genotypes} genotypes. "
        f"The grand mean was {grand_mean:.3f}. "
        f"Stability was assessed via the regression coefficient (bi) and deviation "
        f"from regression (S\u00b2di): genotypes with bi \u2248 1.0 and low S\u00b2di are "
        f"considered broadly adapted and stable."
    )
    sections.append(("Overview", overview))

    # 2. Stability Classification Summary
    class_summary = (
        f"Of the {n_genotypes} genotypes evaluated, {n_stable} were classified as "
        f"stable (bi \u2248 1.0, low S\u00b2di), {n_resp_fav} as responsive to favourable "
        f"environments (bi > 1.1, low S\u00b2di), {n_resp_poor} as suited to poor "
        f"environments (bi < 0.9, low S\u00b2di), and {n_unpred} as unpredictable "
        f"(high S\u00b2di, regardless of bi)."
    )
    sections.append(("Stability Classification", class_summary))

    # 3. Top Performers
    top3 = stability_rows[:3]
    top_text = "The highest-yielding genotypes were: " + ", ".join(
        f"{r['genotype']} (mean = {r['mean']:.3f}, bi = {r['bi']:.3f}, S\u00b2di = {r['s2di']:.4f})"
        for r in top3
    ) + "."
    sections.append(("Top Performers", top_text))

    # 4. Recommended Genotypes
    if best_stable:
        rec_text = (
            f"Genotypes recommended for broad adaptation (stable, above-average mean): "
            + ", ".join(best_stable) + ". "
            "These genotypes combine high mean performance with predictable behaviour "
            "across environments, making them suitable candidates for commercial release "
            "across a wide range of growing conditions."
        )
    else:
        rec_text = (
            "No genotype simultaneously showed above-average mean performance and "
            "full stability (bi \u2248 1.0, low S\u00b2di). Breeders should weigh "
            "stability against mean yield according to the target market environment."
        )
    sections.append(("Recommended Genotypes", rec_text))

    # 5. Interpretation of bi and S²di
    interp = (
        "Interpretation guide: bi < 0.9 indicates below-average response to "
        "environmental improvement (suited to low-input or marginal environments); "
        "bi \u2248 1.0 (0.9\u20131.1) reflects average response (broadly adapted); "
        "bi > 1.1 indicates above-average response (suited to high-input, favourable "
        "environments). S\u00b2di close to zero means performance is highly predictable; "
        "large S\u00b2di indicates genotype \u00d7 environment interaction beyond what "
        "the linear regression captures."
    )
    sections.append(("Parameter Interpretation", interp))

    return "\n\n".join(f"{h}\n{c}" for h, c in sections)


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/analysis/stability",
    response_model=StabilityResponse,
    summary="Eberhart-Russell stability analysis across environments",
)
async def analysis_stability(request: StabilityRequest) -> StabilityResponse:
    """
    Compute Eberhart-Russell regression stability metrics for each genotype
    tested across multiple environments.

    Requires:
      - A dataset_token from POST /upload/dataset with environment_column set
      - At least 2 distinct environments
      - At least 3 genotypes for meaningful stability assessment
    """
    ctx = dataset_cache.get_dataset(request.dataset_token)
    if ctx is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Dataset token not found. Please upload your file via "
                "POST /upload/dataset first."
            ),
        )

    # Decode and read the dataset
    try:
        raw_bytes = __import__("base64").b64decode(ctx["base64_content"])
        df = read_file(raw_bytes, ctx["file_type"])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read dataset: {exc}") from exc

    # Resolve column names
    genotype_col: Optional[str] = ctx.get("genotype_column")
    env_col: Optional[str] = ctx.get("environment_column")
    trait_col: str = request.trait_column

    if not genotype_col or genotype_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "Genotype column not found. Ensure the dataset was uploaded "
                "with genotype_column set."
            ),
        )
    if not env_col or env_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                "Environment column not found. Stability analysis requires "
                "multi-environment data. Upload with environment_column set."
            ),
        )
    if trait_col not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Trait column '{trait_col}' not found in dataset.",
        )

    # Coerce trait to numeric
    df[trait_col] = pd.to_numeric(df[trait_col], errors="coerce")
    df = df.dropna(subset=[trait_col, genotype_col, env_col])

    if df.empty:
        raise HTTPException(
            status_code=422,
            detail="No valid numeric observations found after filtering.",
        )

    n_environments = df[env_col].nunique()
    if n_environments < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Stability analysis requires at least 2 environments. "
                f"Only {n_environments} found in column '{env_col}'."
            ),
        )

    n_genotypes = df[genotype_col].nunique()
    if n_genotypes < 3:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Stability analysis requires at least 3 genotypes. "
                f"Only {n_genotypes} found."
            ),
        )

    try:
        result = _compute_eberhart_russell(df, trait_col, genotype_col, env_col)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Stability computation error")
        raise HTTPException(
            status_code=503,
            detail=f"Stability analysis failed: {exc}",
        ) from exc

    interpretation = _generate_stability_interpretation(
        trait=trait_col,
        grand_mean=result["grand_mean"],
        n_genotypes=result["n_genotypes"],
        n_environments=result["n_environments"],
        stability_rows=result["stability_rows"],
        best_stable=result["best_stable"],
    )

    genotype_stability = [
        GenotypeStability(
            genotype=r["genotype"],
            mean=round(r["mean"], 4),
            bi=round(r["bi"], 4),
            s2di=round(r["s2di"], 6),
            rank=r["rank"],
            stability_class=r["stability_class"],
        )
        for r in result["stability_rows"]
    ]

    # Plot data: mean vs bi scatter (frontend-ready)
    plot_data = {
        "x_axis": "bi (regression coefficient)",
        "y_axis": f"mean {trait_col}",
        "points": [
            {
                "genotype": r["genotype"],
                "x": round(r["bi"], 4),
                "y": round(r["mean"], 4),
                "stability_class": r["stability_class"],
            }
            for r in result["stability_rows"]
        ],
        "reference_lines": {
            "bi_stable_low": 0.9,
            "bi_stable_high": 1.1,
            "grand_mean": round(result["grand_mean"], 4),
        },
    }

    return StabilityResponse(
        status="success",
        trait=trait_col,
        n_genotypes=result["n_genotypes"],
        n_environments=result["n_environments"],
        genotype_stability=genotype_stability,
        environment_means={k: round(v, 4) for k, v in result["env_means"].items()},
        grand_mean=round(result["grand_mean"], 4),
        best_stable_genotypes=result["best_stable"],
        interpretation=interpretation,
        plot_data=plot_data,
    )
