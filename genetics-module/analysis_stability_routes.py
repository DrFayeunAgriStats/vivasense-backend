"""
VivaSense – Stability Analysis Module

POST /analysis/stability

Computes stability metrics for genotypes tested across multiple environments.

Supported methods:
  • eberhart-russell — regression coefficient (bi) and deviation from regression (S²di)
  • shukla           — Shukla stability variance (σ²i)
  • ammi             — AMMI analysis with IPCA decomposition and ASV ranking
  • gge-biplot       — GGE biplot: which-won-where / mean-stability views

References:
  Eberhart, S.A. and Russell, W.A. (1966). Stability parameters for comparing
  varieties. Crop Science, 6(1), 36-40.
  Purchase, J.L. et al. (2000). Genotype × environment interaction of winter
  wheat in South Africa. Euphytica, 111, 35-42.
  Yan, W. and Kang, M.S. (2003). GGE Biplot Analysis. CRC Press.
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
    AMMIBiplotData,
    AMMIResults,
    EnvironmentIPCA,
    EnvironmentPC,
    GGEBiplotData,
    GGEResults,
    GenotypeASV,
    GenotypeDistance,
    GenotypeIPCA,
    GenotypePC,
    GenotypeStability,
    MeanVsStability,
    MegaEnvironment,
    StabilityRequest,
    StabilityResponse,
    WhichWonWhere,
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
# AMMI ANALYSIS
# ============================================================================

def _compute_ammi(
    df: pd.DataFrame,
    trait_col: str,
    genotype_col: str,
    env_col: str,
    n_components: int = 2,
) -> Dict[str, Any]:
    """
    Compute AMMI (Additive Main effects and Multiplicative Interaction) analysis.

    Uses SVD on the double-centred GE interaction matrix to extract IPCA axes.
    Computes AMMI Stability Value (ASV) per genotype (Purchase et al. 2000).

    Returns a dict ready to be converted into AMMIResults.
    """
    # ── Cell means ────────────────────────────────────────────────────────────
    cell_means = (
        df.groupby([genotype_col, env_col])[trait_col]
        .mean()
        .reset_index()
        .rename(columns={trait_col: "cell_mean"})
    )

    genotypes = sorted(cell_means[genotype_col].unique().tolist())
    environments = sorted(cell_means[env_col].unique().tolist())
    n_g = len(genotypes)
    n_e = len(environments)

    if n_e < 2:
        raise ValueError("AMMI requires at least 2 environments.")
    if n_g < 2:
        raise ValueError("AMMI requires at least 2 genotypes.")

    # ── GE matrix (genotypes × environments) ─────────────────────────────────
    ge_pivot = cell_means.pivot(index=genotype_col, columns=env_col, values="cell_mean")
    ge_pivot = ge_pivot.reindex(index=genotypes, columns=environments)

    # Fill missing cells with column mean (imputation for unbalanced data)
    for env in environments:
        col_mean = ge_pivot[env].mean()
        ge_pivot[env] = ge_pivot[env].fillna(col_mean)

    ge_matrix = ge_pivot.values.astype(float)  # shape: (n_g, n_e)

    # ── Grand mean, genotype means, environment means ─────────────────────────
    grand_mean = float(ge_matrix.mean())
    geno_means = ge_matrix.mean(axis=1)     # shape: (n_g,)
    env_means_arr = ge_matrix.mean(axis=0)  # shape: (n_e,)

    # ── Interaction matrix (double-centred) ───────────────────────────────────
    interaction = (
        ge_matrix
        - geno_means[:, np.newaxis]
        - env_means_arr[np.newaxis, :]
        + grand_mean
    )

    # ── SVD ───────────────────────────────────────────────────────────────────
    max_components = min(n_g - 1, n_e - 1, n_components)
    U, S, Vt = np.linalg.svd(interaction, full_matrices=False)

    # Variance explained by each IPCA
    ss_total = float(np.sum(S ** 2))
    ss_each = S ** 2
    pct_explained = (ss_each / ss_total * 100).tolist() if ss_total > 0 else [0.0] * len(S)
    pct_explained_trunc = pct_explained[:max_components]
    cumvar = list(np.cumsum(pct_explained_trunc))

    # ── IPCA scores ───────────────────────────────────────────────────────────
    # Symmetric partitioning: multiply by sqrt of singular value
    geno_scores_mat = U[:, :max_components] * S[:max_components] ** 0.5
    env_scores_mat = Vt[:max_components, :].T * S[:max_components] ** 0.5

    genotype_scores: List[GenotypeIPCA] = []
    for i, g in enumerate(genotypes):
        scores = geno_scores_mat[i]
        genotype_scores.append(
            GenotypeIPCA(
                genotype=g,
                mean=round(float(geno_means[i]), 6),
                ipca1=round(float(scores[0]), 6),
                ipca2=round(float(scores[1]), 6) if max_components > 1 else None,
                ipca3=round(float(scores[2]), 6) if max_components > 2 else None,
            )
        )

    environment_scores: List[EnvironmentIPCA] = []
    for j, e in enumerate(environments):
        scores = env_scores_mat[j]
        environment_scores.append(
            EnvironmentIPCA(
                environment=e,
                mean=round(float(env_means_arr[j]), 6),
                ipca1=round(float(scores[0]), 6),
                ipca2=round(float(scores[1]), 6) if max_components > 1 else None,
            )
        )

    # ── ASV (AMMI Stability Value) ────────────────────────────────────────────
    # ASV = sqrt((SS_IPCA1 / SS_IPCA2 * IPCA1)^2 + IPCA2^2)
    if max_components >= 2 and ss_each[1] > 0:
        ss_ratio = float(np.sqrt(ss_each[0] / ss_each[1]))
        asv_values = [
            float(np.sqrt((ss_ratio * geno_scores_mat[i, 0]) ** 2 + geno_scores_mat[i, 1] ** 2))
            for i in range(n_g)
        ]
    else:
        asv_values = [float(abs(geno_scores_mat[i, 0])) for i in range(n_g)]

    asv_sorted_idx = np.argsort(asv_values)
    asv_ranks = [0] * n_g
    for rank_pos, orig_idx in enumerate(asv_sorted_idx, start=1):
        asv_ranks[orig_idx] = rank_pos

    asv_q = np.quantile(asv_values, [0.25, 0.50, 0.75])
    stability_measure: List[GenotypeASV] = []
    for i, g in enumerate(genotypes):
        asv = asv_values[i]
        if asv <= asv_q[0]:
            s_class = "highly stable"
        elif asv <= asv_q[1]:
            s_class = "stable"
        elif asv <= asv_q[2]:
            s_class = "moderately stable"
        else:
            s_class = "unstable"
        stability_measure.append(
            GenotypeASV(
                genotype=g,
                asv=round(asv, 6),
                rank=asv_ranks[i],
                stability_class=s_class,
            )
        )

    # ── Biplot data ───────────────────────────────────────────────────────────
    biplot_data = AMMIBiplotData(
        genotypes=[
            {
                "genotype": gs.genotype,
                "mean": gs.mean,
                "ipca1": gs.ipca1,
                "ipca2": gs.ipca2,
            }
            for gs in genotype_scores
        ],
        environments=[
            {
                "environment": es.environment,
                "mean": es.mean,
                "ipca1": es.ipca1,
                "ipca2": es.ipca2,
            }
            for es in environment_scores
        ],
    )

    interpretation = _generate_ammi_interpretation(
        trait_col, genotype_scores, stability_measure, pct_explained_trunc, grand_mean
    )

    return AMMIResults(
        variance_explained=[round(v, 4) for v in pct_explained_trunc],
        cumulative_variance=[round(v, 4) for v in cumvar],
        genotype_scores=genotype_scores,
        environment_scores=environment_scores,
        stability_measure=stability_measure,
        biplot_data=biplot_data,
        interpretation=interpretation,
    )


def _generate_ammi_interpretation(
    trait: str,
    genotype_scores: List[GenotypeIPCA],
    stability_measure: List[GenotypeASV],
    pct_explained: List[float],
    grand_mean: float,
) -> str:
    """Generate plain-English AMMI interpretation."""
    sections: List[tuple] = []

    # Variance explained
    if len(pct_explained) >= 2:
        var_text = (
            f"AMMI analysis of {trait} partitioned the genotype × environment (GxE) "
            f"interaction into IPCA axes. IPCA1 explained {pct_explained[0]:.1f}% and "
            f"IPCA2 explained {pct_explained[1]:.1f}% of the interaction variation "
            f"(cumulative: {sum(pct_explained[:2]):.1f}%)."
        )
    elif pct_explained:
        var_text = (
            f"AMMI analysis of {trait}: IPCA1 explained {pct_explained[0]:.1f}% "
            f"of the GxE interaction variation."
        )
    else:
        var_text = f"AMMI analysis of {trait} completed."
    sections.append(("Variance Explained", var_text))

    # Stability classification
    n_highly = sum(1 for s in stability_measure if s.stability_class == "highly stable")
    n_stable = sum(1 for s in stability_measure if s.stability_class == "stable")
    n_mod = sum(1 for s in stability_measure if s.stability_class == "moderately stable")
    n_unstable = sum(1 for s in stability_measure if s.stability_class == "unstable")
    stab_text = (
        f"Of {len(stability_measure)} genotypes evaluated by AMMI Stability Value (ASV): "
        f"{n_highly} were highly stable, {n_stable} stable, {n_mod} moderately stable, "
        f"and {n_unstable} unstable. Lower ASV indicates greater stability across environments."
    )
    sections.append(("Stability Classification (ASV)", stab_text))

    # Top stable genotypes with above-average yield
    geno_mean_map = {gs.genotype: gs.mean for gs in genotype_scores}
    top_stable = sorted(
        [s for s in stability_measure if s.stability_class in ("highly stable", "stable")],
        key=lambda s: s.asv,
    )
    above_avg_stable = [s for s in top_stable if geno_mean_map.get(s.genotype, 0) >= grand_mean]
    if above_avg_stable:
        names = ", ".join(
            f"{s.genotype} (ASV={s.asv:.3f}, mean={geno_mean_map[s.genotype]:.3f})"
            for s in above_avg_stable[:3]
        )
        rec_text = (
            f"Genotypes combining stability (low ASV) with above-average {trait}: {names}. "
            "These genotypes are recommended for broad adaptation across environments."
        )
    else:
        top_3 = sorted(stability_measure, key=lambda s: s.asv)[:3]
        names = ", ".join(f"{s.genotype} (ASV={s.asv:.3f})" for s in top_3)
        rec_text = (
            f"Most stable genotypes by ASV: {names}. "
            "No genotype simultaneously showed above-average mean and high stability; "
            "breeders should weigh stability against yield for the target environment."
        )
    sections.append(("Recommended Genotypes", rec_text))

    interp_guide = (
        "Interpretation guide: Genotypes close to the biplot origin (low |IPCA1| and |IPCA2|) "
        "are highly stable across environments. Genotypes far from the origin show strong "
        "genotype × environment interaction, indicating specific adaptation. "
        "ASV combines IPCA1 and IPCA2 into a single stability index weighted by their "
        "proportion of GxE variance explained."
    )
    sections.append(("Parameter Interpretation", interp_guide))

    return "\n\n".join(f"{h}\n{c}" for h, c in sections)


# ============================================================================
# GGE BIPLOT ANALYSIS
# ============================================================================

def _compute_gge_biplot(
    df: pd.DataFrame,
    trait_col: str,
    genotype_col: str,
    env_col: str,
    biplot_type: str = "which-won-where",
) -> Dict[str, Any]:
    """
    Compute GGE (Genotype + Genotype×Environment) biplot.

    Uses SVD on the environment-centred (but not genotype-centred) GE matrix,
    so that both G and GxE variation are captured in the biplot axes.

    Returns a dict ready to be converted into GGEResults.
    """
    # ── Cell means ────────────────────────────────────────────────────────────
    cell_means = (
        df.groupby([genotype_col, env_col])[trait_col]
        .mean()
        .reset_index()
        .rename(columns={trait_col: "cell_mean"})
    )

    genotypes = sorted(cell_means[genotype_col].unique().tolist())
    environments = sorted(cell_means[env_col].unique().tolist())
    n_g = len(genotypes)
    n_e = len(environments)

    if n_e < 2:
        raise ValueError("GGE Biplot requires at least 2 environments.")
    if n_g < 2:
        raise ValueError("GGE Biplot requires at least 2 genotypes.")

    # ── GE matrix (genotypes × environments) ─────────────────────────────────
    ge_pivot = cell_means.pivot(index=genotype_col, columns=env_col, values="cell_mean")
    ge_pivot = ge_pivot.reindex(index=genotypes, columns=environments)

    for env in environments:
        col_mean = ge_pivot[env].mean()
        ge_pivot[env] = ge_pivot[env].fillna(col_mean)

    ge_matrix = ge_pivot.values.astype(float)

    geno_means = ge_matrix.mean(axis=1)
    env_means_arr = ge_matrix.mean(axis=0)
    grand_mean = float(ge_matrix.mean())

    # ── Environment-centred matrix (GGE = G + GxE) ───────────────────────────
    gge_matrix = ge_matrix - env_means_arr[np.newaxis, :]

    # ── SVD ───────────────────────────────────────────────────────────────────
    U, S, Vt = np.linalg.svd(gge_matrix, full_matrices=False)

    ss_total = float(np.sum(S ** 2))
    ss_each = S ** 2
    pct_explained = (ss_each / ss_total * 100).tolist() if ss_total > 0 else [0.0] * len(S)

    # Take first 2 PCs
    n_pc = min(2, len(S))
    geno_scores_mat = U[:, :n_pc] * S[:n_pc] ** 0.5
    env_scores_mat = Vt[:n_pc, :].T * S[:n_pc] ** 0.5

    genotype_scores: List[GenotypePC] = []
    for i, g in enumerate(genotypes):
        genotype_scores.append(
            GenotypePC(
                genotype=g,
                mean=round(float(geno_means[i]), 6),
                pc1=round(float(geno_scores_mat[i, 0]), 6),
                pc2=round(float(geno_scores_mat[i, 1]) if n_pc > 1 else 0.0, 6),
            )
        )

    environment_scores: List[EnvironmentPC] = []
    for j, e in enumerate(environments):
        environment_scores.append(
            EnvironmentPC(
                environment=e,
                mean=round(float(env_means_arr[j]), 6),
                pc1=round(float(env_scores_mat[j, 0]), 6),
                pc2=round(float(env_scores_mat[j, 1]) if n_pc > 1 else 0.0, 6),
            )
        )

    pct_pc = [round(pct_explained[i], 4) for i in range(min(2, len(pct_explained)))]
    cumvar = round(sum(pct_pc), 4)

    # ── Which-Won-Where ───────────────────────────────────────────────────────
    which_won_where: Optional[WhichWonWhere] = None
    if biplot_type == "which-won-where":
        winning_genos: Dict[str, str] = {}
        env_to_sector: Dict[str, int] = {}

        # For each environment find the highest-yielding genotype
        for env in environments:
            env_data = cell_means[cell_means[env_col] == env]
            if env_data.empty:
                continue
            winner_idx = env_data["cell_mean"].idxmax()
            winning_genos[env] = str(env_data.loc[winner_idx, genotype_col])

        # Group environments by winning genotype to form mega-environments
        from collections import defaultdict
        mega_map: Dict[str, List[str]] = defaultdict(list)
        for env, winner in winning_genos.items():
            mega_map[winner].append(env)

        mega_environments: List[MegaEnvironment] = []
        for mega_id, (winner, envs) in enumerate(mega_map.items(), start=1):
            env_yields = [
                cell_means.loc[
                    (cell_means[genotype_col] == winner) & (cell_means[env_col] == e),
                    "cell_mean",
                ].mean()
                for e in envs
            ]
            mean_yield = float(np.nanmean(env_yields)) if env_yields else float("nan")
            mega_environments.append(
                MegaEnvironment(
                    id=mega_id,
                    environments=sorted(envs),
                    best_genotype=winner,
                    mean_yield=round(mean_yield, 4),
                )
            )

        wwi = _generate_gge_wwi_interpretation(trait_col, mega_environments)
        which_won_where = WhichWonWhere(
            mega_environments=mega_environments,
            winning_genotypes=winning_genos,
            interpretation=wwi,
        )

    # ── Mean vs Stability ─────────────────────────────────────────────────────
    mean_vs_stability: Optional[MeanVsStability] = None
    if biplot_type == "mean-stability":
        # Distance from origin in GGE biplot = instability
        distances = [
            float(np.sqrt(geno_scores_mat[i, 0] ** 2 + (geno_scores_mat[i, 1] if n_pc > 1 else 0.0) ** 2))
            for i in range(n_g)
        ]
        # Normalise mean performance (0–1)
        means = [float(geno_means[i]) for i in range(n_g)]
        mean_range = max(means) - min(means) if max(means) != min(means) else 1.0
        norm_means = [(m - min(means)) / mean_range for m in means]
        # Ideal: high normalised mean, low distance
        # Score = norm_mean / (distance + epsilon)
        eps = 1e-9
        scores = [nm / (d + eps) for nm, d in zip(norm_means, distances)]
        ideal_idx = int(np.argmax(scores))
        ideal_geno = genotypes[ideal_idx]

        dist_ranks = list(np.argsort(distances) + 1)  # lower distance = better rank
        geno_dist_sorted = sorted(
            zip(range(n_g), distances, dist_ranks),
            key=lambda x: x[1],
        )
        genotype_distances: List[GenotypeDistance] = [
            GenotypeDistance(
                genotype=genotypes[i],
                distance_from_ideal=round(d, 6),
                rank=r,
            )
            for i, d, r in geno_dist_sorted
        ]
        mvs_interp = _generate_gge_mvs_interpretation(trait_col, ideal_geno, genotype_distances)
        mean_vs_stability = MeanVsStability(
            ideal_genotype=ideal_geno,
            ideal_coordinates={
                "pc1": round(float(geno_scores_mat[ideal_idx, 0]), 6),
                "pc2": round(float(geno_scores_mat[ideal_idx, 1]) if n_pc > 1 else 0.0, 6),
            },
            genotype_distances=genotype_distances,
            interpretation=mvs_interp,
        )

    # ── Biplot data ───────────────────────────────────────────────────────────
    biplot_data = GGEBiplotData(
        genotypes=[
            {"genotype": gs.genotype, "mean": gs.mean, "pc1": gs.pc1, "pc2": gs.pc2}
            for gs in genotype_scores
        ],
        environments=[
            {"environment": es.environment, "mean": es.mean, "pc1": es.pc1, "pc2": es.pc2}
            for es in environment_scores
        ],
    )

    interpretation = _generate_gge_interpretation(
        trait_col, genotype_scores, pct_pc, biplot_type, which_won_where, mean_vs_stability
    )

    return GGEResults(
        variance_explained=pct_pc,
        cumulative_variance=cumvar,
        genotype_scores=genotype_scores,
        environment_scores=environment_scores,
        which_won_where=which_won_where,
        mean_vs_stability=mean_vs_stability,
        biplot_data=biplot_data,
        interpretation=interpretation,
    )


def _generate_gge_wwi_interpretation(
    trait: str,
    mega_environments: List[MegaEnvironment],
) -> str:
    """Interpretation for which-won-where analysis."""
    n_mega = len(mega_environments)
    lines = [
        f"Which-Won-Where analysis of {trait} identified {n_mega} mega-environment(s):"
    ]
    for me in mega_environments:
        envs_str = ", ".join(me.environments)
        lines.append(
            f"  • Mega-environment {me.id} ({envs_str}): "
            f"Winner = {me.best_genotype} (mean = {me.mean_yield:.3f})"
        )
    if n_mega > 1:
        lines.append(
            "Different genotypes are adapted to different mega-environments, "
            "indicating crossover genotype × environment interaction. "
            "Location-specific genotype recommendation is advised."
        )
    else:
        lines.append(
            "A single mega-environment was identified, suggesting consistent "
            "genotype performance across locations. The winning genotype is "
            "broadly adapted."
        )
    return " ".join(lines)


def _generate_gge_mvs_interpretation(
    trait: str,
    ideal_genotype: str,
    genotype_distances: List[GenotypeDistance],
) -> str:
    """Interpretation for mean vs stability view."""
    top3 = genotype_distances[:3]
    top_names = ", ".join(f"{gd.genotype} (rank {gd.rank})" for gd in top3)
    return (
        f"Mean-vs-Stability view for {trait}: the ideal genotype combining high mean "
        f"performance and stability across environments is {ideal_genotype}. "
        f"Genotypes closest to the ideal (ranked by distance): {top_names}. "
        "Genotypes near the Average Environment Coordination (AEC) abscissa are "
        "considered more stable; those far from the AEC ordinate have higher mean performance."
    )


def _generate_gge_interpretation(
    trait: str,
    genotype_scores: List[GenotypePC],
    pct_pc: List[float],
    biplot_type: str,
    which_won_where: Optional[WhichWonWhere],
    mean_vs_stability: Optional[MeanVsStability],
) -> str:
    """Generate a comprehensive GGE Biplot interpretation."""
    sections: List[tuple] = []

    # Variance explained
    if len(pct_pc) >= 2:
        var_text = (
            f"GGE Biplot analysis of {trait}: PC1 explained {pct_pc[0]:.1f}% and "
            f"PC2 explained {pct_pc[1]:.1f}% of the genotype + GxE variation "
            f"(cumulative: {sum(pct_pc):.1f}%). "
            "Higher cumulative variance (>60%) indicates the biplot provides a reliable "
            "visual approximation of the data."
        )
    else:
        var_text = f"GGE Biplot analysis of {trait} completed."
    sections.append(("Variance Explained (GGE)", var_text))

    if which_won_where is not None:
        sections.append(("Which-Won-Where", which_won_where.interpretation))

    if mean_vs_stability is not None:
        sections.append(("Mean vs Stability", mean_vs_stability.interpretation))

    interp_guide = (
        "Interpretation guide: In the GGE Biplot, the polygon vertex genotypes "
        "(outermost points) won in at least one environment. Environments within "
        "the same sector share the same winning genotype (mega-environment). "
        "Genotypes near the biplot origin are stable but not necessarily high-yielding."
    )
    sections.append(("Interpretation Guide", interp_guide))

    return "\n\n".join(f"{h}\n{c}" for h, c in sections)


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post(
    "/analysis/stability",
    response_model=StabilityResponse,
    summary="Comprehensive stability analysis across environments (Eberhart-Russell, Shukla, AMMI, GGE Biplot)",
)
async def analysis_stability(request: StabilityRequest) -> StabilityResponse:
    """
    Compute stability metrics for genotypes tested across multiple environments.

    Supported methods (specify via the ``methods`` field):
      - **eberhart-russell** — regression coefficient (bi) and deviation from regression (S²di)
      - **shukla**           — Shukla stability variance (included with eberhart-russell)
      - **ammi**             — AMMI analysis with IPCA decomposition and ASV ranking
      - **gge-biplot**       — GGE biplot for mega-environment identification

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

    # ── Resolve requested methods ─────────────────────────────────────────────
    methods = [m.lower().strip() for m in (request.methods or ["eberhart-russell", "shukla"])]
    # Shukla is always computed alongside Eberhart-Russell (shared computation)
    if "shukla" in methods and "eberhart-russell" not in methods:
        methods = ["eberhart-russell"] + methods

    methods_computed: List[str] = []

    # ── Eberhart-Russell / Shukla ─────────────────────────────────────────────
    classic_result: Dict[str, Any] = {}
    genotype_stability: List[GenotypeStability] = []
    environment_means: Dict[str, float] = {}
    grand_mean: float = 0.0
    best_stable_genotypes: List[str] = []
    er_interpretation: str = ""
    plot_data: Dict[str, Any] = {}

    run_classic = any(m in methods for m in ("eberhart-russell", "shukla"))
    if run_classic:
        try:
            classic_result = _compute_eberhart_russell(df, trait_col, genotype_col, env_col)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Eberhart-Russell computation error")
            raise HTTPException(
                status_code=503,
                detail=f"Stability analysis failed: {exc}",
            ) from exc

        er_interpretation = _generate_stability_interpretation(
            trait=trait_col,
            grand_mean=classic_result["grand_mean"],
            n_genotypes=classic_result["n_genotypes"],
            n_environments=classic_result["n_environments"],
            stability_rows=classic_result["stability_rows"],
            best_stable=classic_result["best_stable"],
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
            for r in classic_result["stability_rows"]
        ]

        environment_means = {k: round(v, 4) for k, v in classic_result["env_means"].items()}
        grand_mean = round(classic_result["grand_mean"], 4)
        best_stable_genotypes = classic_result["best_stable"]

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
                for r in classic_result["stability_rows"]
            ],
            "reference_lines": {
                "bi_stable_low": 0.9,
                "bi_stable_high": 1.1,
                "grand_mean": grand_mean,
            },
        }

        if "eberhart-russell" in methods:
            methods_computed.append("eberhart-russell")
        if "shukla" in methods:
            methods_computed.append("shukla")

    # ── AMMI ─────────────────────────────────────────────────────────────────
    ammi_results: Optional[AMMIResults] = None
    if "ammi" in methods:
        try:
            ammi_results = _compute_ammi(
                df,
                trait_col,
                genotype_col,
                env_col,
                n_components=request.ammi_components or 2,
            )
            methods_computed.append("ammi")
        except Exception as exc:
            logger.warning("AMMI computation failed: %s", exc)
            # Non-fatal: continue without AMMI results

    # ── GGE Biplot ────────────────────────────────────────────────────────────
    gge_results: Optional[GGEResults] = None
    if "gge-biplot" in methods:
        try:
            gge_results = _compute_gge_biplot(
                df,
                trait_col,
                genotype_col,
                env_col,
                biplot_type=request.biplot_type or "which-won-where",
            )
            methods_computed.append("gge-biplot")
        except Exception as exc:
            logger.warning("GGE Biplot computation failed: %s", exc)
            # Non-fatal: continue without GGE results

    # ── Compose interpretation ────────────────────────────────────────────────
    interp_parts: List[str] = []
    if er_interpretation:
        interp_parts.append(er_interpretation)
    if ammi_results is not None:
        interp_parts.append(ammi_results.interpretation)
    if gge_results is not None:
        interp_parts.append(gge_results.interpretation)
    interpretation = "\n\n---\n\n".join(interp_parts) if interp_parts else (
        "No methods were computed. Specify at least one method in the 'methods' field."
    )

    # Use AMMI grand mean as fallback if classic was not run
    if not run_classic and classic_result == {}:
        n_environments = df[env_col].nunique()
        n_genotypes = df[genotype_col].nunique()

    return StabilityResponse(
        status="success",
        trait=trait_col,
        methods_computed=methods_computed,
        n_genotypes=n_genotypes,
        n_environments=n_environments,
        genotype_stability=genotype_stability,
        environment_means=environment_means,
        grand_mean=grand_mean,
        best_stable_genotypes=best_stable_genotypes,
        ammi_results=ammi_results,
        gge_results=gge_results,
        interpretation=interpretation,
        plot_data=plot_data if plot_data else None,
    )
