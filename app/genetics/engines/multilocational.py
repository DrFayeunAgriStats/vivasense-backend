"""
Combined multi-location ANOVA, AMMI model, and GGE biplot.

Statistical foundations
-----------------------
Combined ANOVA (balanced RCBD across l locations, g genotypes, r reps):
  Source         df             EMS (random-effects model)
  Genotype       g-1            σ²e + r·σ²gl + r·l·σ²g
  Location       l-1            σ²e + r·σ²gl + r·g·σ²l
  G×L            (g-1)(l-1)     σ²e + r·σ²gl
  Rep/Location   l(r-1)         σ²e
  Error          gl(r-1)        σ²e

F-tests use MS_GL as denominator for G and L (mixed model, locations random).

AMMI: Additive Main effects + Multiplicative Interaction model.
GGE:  Genotype + Genotype×Environment biplot (environment-centered).
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _sig(p: float, alpha: float = 0.05) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < alpha:
        return "*"
    return "ns"


class MultilocationEngine:
    """Combined ANOVA → AMMI → GGE for multi-environment breeding trials."""

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------
    # COMBINED ANOVA
    # ------------------------------------------------------------------

    def run_combined_anova(
        self,
        df: pd.DataFrame,
        trait: str,
        geno_col: str,
        loc_col: str,
        rep_col: str,
    ) -> Dict[str, Any]:
        """
        Compute partitioned SS for G, L, G×L, Rep/L, and Error.
        Returns a rich dict consumed by VarianceComponentEngine, StabilityEngine, etc.
        """
        alpha = self.config.alpha

        genotypes = sorted(df[geno_col].unique())
        locations = sorted(df[loc_col].unique())
        g = len(genotypes)
        l = len(locations)

        # Mode number of reps (robust to mild imbalance)
        rep_counts = df.groupby([geno_col, loc_col])[rep_col].nunique()
        r = int(rep_counts.mode().iloc[0])

        grand_mean = float(df[trait].mean())
        geno_means = df.groupby(geno_col)[trait].mean()
        loc_means = df.groupby(loc_col)[trait].mean()

        # Cell means matrix (g × l)
        cell_means = df.groupby([geno_col, loc_col])[trait].mean().unstack(loc_col)
        cell_means = cell_means.reindex(index=genotypes, columns=locations)

        # --- SS calculations ---
        SS_G = l * r * float(((geno_means - grand_mean) ** 2).sum())
        df_G = g - 1

        SS_L = g * r * float(((loc_means - grand_mean) ** 2).sum())
        df_L = l - 1

        # Interaction matrix: cell_mean - geno_mean - loc_mean + grand_mean
        gm = geno_means.reindex(genotypes).values[:, np.newaxis]
        lm = loc_means.reindex(locations).values[np.newaxis, :]
        interaction_matrix = cell_means.values - gm - lm + grand_mean  # shape g×l

        SS_GL = r * float((interaction_matrix ** 2).sum())
        df_GL = (g - 1) * (l - 1)

        SS_total = float(((df[trait] - grand_mean) ** 2).sum())
        df_total = len(df) - 1

        # Rep within Location SS
        rep_loc_means = df.groupby([rep_col, loc_col])[trait].mean()
        SS_rep_loc = 0.0
        for loc in locations:
            loc_rep_means = df[df[loc_col] == loc].groupby(rep_col)[trait].mean()
            SS_rep_loc += g * float(((loc_rep_means - loc_means[loc]) ** 2).sum())
        df_rep_loc = l * (r - 1)

        # Error (residual within cells)
        SS_error = max(0.0, SS_total - SS_G - SS_L - SS_GL - SS_rep_loc)
        df_error = max(1, g * l * (r - 1))

        MS_G = SS_G / df_G if df_G > 0 else 0.0
        MS_L = SS_L / df_L if df_L > 0 else 0.0
        MS_GL = SS_GL / df_GL if df_GL > 0 else 0.0
        MS_error = SS_error / df_error if df_error > 0 else 0.0

        # F tests: G and L tested over MS_GL (mixed model)
        F_G = MS_G / MS_GL if MS_GL > 0 else 0.0
        F_L = MS_L / MS_GL if MS_GL > 0 else 0.0
        F_GL = MS_GL / MS_error if MS_error > 0 else 0.0

        p_G = float(1 - stats.f.cdf(F_G, df_G, df_GL)) if F_G > 0 else 1.0
        p_L = float(1 - stats.f.cdf(F_L, df_L, df_GL)) if F_L > 0 else 1.0
        p_GL = float(1 - stats.f.cdf(F_GL, df_GL, df_error)) if F_GL > 0 else 1.0

        anova_table = {
            "columns": ["Source", "df", "SS", "MS", "F", "p_value", "significance"],
            "rows": [
                ["Genotype",           df_G,     round(SS_G,    4), round(MS_G,    4), round(F_G,  3), round(p_G,  4), _sig(p_G,  alpha)],
                ["Location",           df_L,     round(SS_L,    4), round(MS_L,    4), round(F_L,  3), round(p_L,  4), _sig(p_L,  alpha)],
                ["Genotype×Location",  df_GL,    round(SS_GL,   4), round(MS_GL,   4), round(F_GL, 3), round(p_GL, 4), _sig(p_GL, alpha)],
                ["Rep/Location",       df_rep_loc, round(SS_rep_loc, 4), round(SS_rep_loc / max(df_rep_loc, 1), 4), None, None, None],
                ["Error",              df_error, round(SS_error, 4), round(MS_error, 4), None, None, None],
                ["Total",              df_total, round(SS_total, 4), None, None, None, None],
            ],
        }

        return {
            "anova_table": anova_table,
            "grand_mean": grand_mean,
            "n_genotypes": g,
            "n_locations": l,
            "n_reps": r,
            "genotype_names": genotypes,
            "location_names": locations,
            "geno_means": geno_means,
            "loc_means": loc_means,
            "cell_matrix": cell_means,          # pd.DataFrame g×l
            "interaction_matrix": interaction_matrix,  # np.ndarray g×l
            "MS": {"G": MS_G, "L": MS_L, "GL": MS_GL, "error": MS_error},
            "SS": {"G": SS_G, "L": SS_L, "GL": SS_GL, "error": SS_error, "total": SS_total},
            "df": {"G": df_G, "L": df_L, "GL": df_GL, "error": df_error, "total": df_total},
            "F": {"G": round(F_G, 3), "L": round(F_L, 3), "GL": round(F_GL, 3)},
            "p": {"G": round(p_G, 4), "L": round(p_L, 4), "GL": round(p_GL, 4)},
        }

    # ------------------------------------------------------------------
    # AMMI MODEL
    # ------------------------------------------------------------------

    def fit_ammi(self, anova_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        AMMI via SVD of the G×L interaction matrix.

        Interaction matrix I_ij = ȳ_ij - ȳ_i. - ȳ_.j + ȳ..
        SVD: I = U Σ Vᵀ
        IPCA_k genotype score  = U_ik × √λ_k
        IPCA_k environment score = V_jk × √λ_k  (symmetric biplot scaling)
        """
        W = anova_result["interaction_matrix"]      # np.ndarray shape g×l
        geno_names = anova_result["genotype_names"]
        loc_names = anova_result["location_names"]
        geno_means = anova_result["geno_means"]
        loc_means = anova_result["loc_means"]
        SS_GL = anova_result["SS"]["GL"]

        n_axes = min(self.config.n_ammi_axes, min(W.shape) - 1, min(W.shape))
        n_axes = max(1, n_axes)

        U, s, Vt = np.linalg.svd(W, full_matrices=False)

        total_ipca_ss = float((s ** 2).sum())

        # Gollob (1968) df for AMMI axes
        g, l = len(geno_names), len(loc_names)

        # IPCA scores (symmetric SVD partition)
        ipca_geno = {}  # genotype → {IPCA1: v, IPCA2: v, ...}
        ipca_env = {}   # location → {IPCA1: v, ...}

        for k in range(n_axes):
            sq_s = float(np.sqrt(s[k]))
            for i, geno in enumerate(geno_names):
                ipca_geno.setdefault(geno, {})[f"IPCA{k+1}"] = round(float(U[i, k]) * sq_s, 6)
            for j, loc in enumerate(loc_names):
                ipca_env.setdefault(loc, {})[f"IPCA{k+1}"] = round(float(Vt[k, j]) * sq_s, 6)

        # Explained variance per axis
        explained_variance = []
        cumulative = 0.0
        for k in range(len(s)):
            pct = 100.0 * s[k] ** 2 / total_ipca_ss if total_ipca_ss > 0 else 0.0
            cumulative += pct
            df_axis = g + l - 1 - 2 * (k + 1)  # Gollob df approximation
            explained_variance.append({
                "axis": f"IPCA{k+1}",
                "singular_value": round(float(s[k]), 6),
                "ss": round(float(s[k] ** 2), 4),
                "percent": round(pct, 2),
                "cumulative_percent": round(cumulative, 2),
                "df_gollob": max(1, df_axis),
            })

        ipca1_pct = explained_variance[0]["percent"] if explained_variance else 0
        ipca2_pct = explained_variance[1]["percent"] if len(explained_variance) > 1 else 0

        genotype_points = [
            {
                "id": geno,
                "x": ipca_geno.get(geno, {}).get("IPCA1", 0.0),
                "y": ipca_geno.get(geno, {}).get("IPCA2", 0.0),
                "mean": round(float(geno_means[geno]), 4),
                "type": "genotype",
            }
            for geno in geno_names
        ]
        environment_points = [
            {
                "id": loc,
                "x": ipca_env.get(loc, {}).get("IPCA1", 0.0),
                "y": ipca_env.get(loc, {}).get("IPCA2", 0.0),
                "mean": round(float(loc_means[loc]), 4),
                "type": "environment",
            }
            for loc in loc_names
        ]

        # Interaction matrix as dict for JSON
        interaction_dict = {
            geno_names[i]: {
                loc_names[j]: round(float(W[i, j]), 6)
                for j in range(l)
            }
            for i in range(g)
        }

        return {
            "n_axes_retained": n_axes,
            "grand_mean": round(anova_result["grand_mean"], 4),
            "total_ss_gxe": round(SS_GL, 4),
            "anova_table": anova_result["anova_table"],
            "interaction_matrix": {
                "description": "Cell means minus G main effect minus L main effect (G×E interaction deviations)",
                "formula": "I_ij = ȳ_ij - ȳ_i. - ȳ_.j + ȳ..",
                "rows_are": "genotypes",
                "cols_are": "environments",
                "data": interaction_dict,
            },
            "ipca_scores": {"genotypes": ipca_geno, "environments": ipca_env},
            "explained_variance": explained_variance,
            "biplot_data": {
                "description": "Genotypes as circles, environments as triangles on IPCA1 × IPCA2",
                "x_axis_label": f"IPCA1 ({ipca1_pct:.1f}% of G×E SS)",
                "y_axis_label": f"IPCA2 ({ipca2_pct:.1f}% of G×E SS)",
                "genotype_points": genotype_points,
                "environment_points": environment_points,
            },
        }

    # ------------------------------------------------------------------
    # GGE BIPLOT
    # ------------------------------------------------------------------

    def fit_gge(self, anova_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        GGE biplot via SVD of environment-centered cell means.

        Z_ij = ȳ_ij - ȳ_.j   (only location mean removed — G and G×E retained)
        SVD: Z = U Σ Vᵀ
        PC scores (symmetric scaling):
          genotype i on PC k  = U_ik × √λ_k
          environment j on PCk = V_jk × √λ_k
        """
        M = anova_result["cell_matrix"].values      # g×l
        geno_names = anova_result["genotype_names"]
        loc_names = anova_result["location_names"]
        geno_means = anova_result["geno_means"]
        loc_means = anova_result["loc_means"]

        # Environment-center
        env_col_means = M.mean(axis=0)
        Z = M - env_col_means[np.newaxis, :]

        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        total_ss = float((s ** 2).sum())

        gen_scores: Dict[str, Dict] = {}
        env_scores: Dict[str, Dict] = {}

        n_pc = min(2, len(s))
        for k in range(n_pc):
            sq_s = float(np.sqrt(s[k]))
            for i, geno in enumerate(geno_names):
                gen_scores.setdefault(geno, {})[f"PC{k+1}"] = round(float(U[i, k]) * sq_s, 6)
            for j, loc in enumerate(loc_names):
                env_scores.setdefault(loc, {})[f"PC{k+1}"] = round(float(Vt[k, j]) * sq_s, 6)

        # Add means to genotype scores
        for geno in geno_names:
            gen_scores[geno]["mean"] = round(float(geno_means[geno]), 4)

        explained_variance = []
        cumulative = 0.0
        for k in range(len(s)):
            pct = 100.0 * s[k] ** 2 / total_ss if total_ss > 0 else 0.0
            cumulative += pct
            explained_variance.append({
                "axis": f"PC{k+1}",
                "percent": round(pct, 2),
                "cumulative_percent": round(cumulative, 2),
            })

        # Which-won-where: highest yielding genotype per location
        cell_df = anova_result["cell_matrix"]
        which_won = {}
        for loc in loc_names:
            col_sorted = cell_df[loc].sort_values(ascending=False)
            which_won[loc] = {
                "winner": col_sorted.index[0],
                "runner_up": col_sorted.index[1] if len(col_sorted) > 1 else None,
                "winner_mean": round(float(col_sorted.iloc[0]), 4),
            }

        # Ideal genotype: closest to mean-environment vector tip
        mean_PC1 = float(np.mean([env_scores[l].get("PC1", 0) for l in loc_names]))
        mean_PC2 = float(np.mean([env_scores[l].get("PC2", 0) for l in loc_names]))
        distances = {
            g: float(np.sqrt((gen_scores[g].get("PC1", 0) - mean_PC1) ** 2
                             + (gen_scores[g].get("PC2", 0) - mean_PC2) ** 2))
            for g in geno_names
        }
        ideal = min(distances, key=distances.get)

        # Discriminating environments (vector length from origin)
        env_lengths = {
            loc: float(np.sqrt(env_scores[loc].get("PC1", 0) ** 2
                               + env_scores[loc].get("PC2", 0) ** 2))
            for loc in loc_names
        }
        sorted_envs = sorted(env_lengths, key=env_lengths.get, reverse=True)

        # Convex hull (polygon vertices = mega-environment winners)
        try:
            from scipy.spatial import ConvexHull
            pts = np.column_stack([
                [gen_scores[g].get("PC1", 0) for g in geno_names],
                [gen_scores[g].get("PC2", 0) for g in geno_names],
            ])
            hull = ConvexHull(pts)
            polygon_vertices = [
                {"id": geno_names[i], "x": gen_scores[geno_names[i]].get("PC1", 0),
                 "y": gen_scores[geno_names[i]].get("PC2", 0)}
                for i in hull.vertices
            ]
        except Exception:
            polygon_vertices = []

        pc1_pct = explained_variance[0]["percent"] if explained_variance else 0
        pc2_pct = explained_variance[1]["percent"] if len(explained_variance) > 1 else 0

        return {
            "n_environments": len(loc_names),
            "explained_variance": explained_variance,
            "genotype_scores": gen_scores,
            "environment_scores": env_scores,
            "which_won_where": which_won,
            "ideal_genotype": {
                "identified": ideal,
                "mean_yield": gen_scores[ideal]["mean"],
                "basis": "Closest to ideal marker (mean-environment vector tip)",
            },
            "discriminating_environments": {
                "most_discriminating": sorted_envs[0] if sorted_envs else None,
                "least_discriminating": sorted_envs[-1] if sorted_envs else None,
                "ranking": sorted_envs,
                "basis": "PC vector length from biplot origin",
            },
            "biplot_data": {
                "x_axis_label": f"PC1 ({pc1_pct:.1f}%)",
                "y_axis_label": f"PC2 ({pc2_pct:.1f}%)",
                "genotype_points": [
                    {"id": g, "x": gen_scores[g].get("PC1", 0),
                     "y": gen_scores[g].get("PC2", 0),
                     "mean": gen_scores[g]["mean"], "type": "genotype"}
                    for g in geno_names
                ],
                "environment_vectors": [
                    {"id": loc, "x": env_scores[loc].get("PC1", 0),
                     "y": env_scores[loc].get("PC2", 0),
                     "vector_length": round(env_lengths[loc], 4), "type": "environment"}
                    for loc in loc_names
                ],
                "polygon_vertices": polygon_vertices,
                "ideal_marker": {"x": round(mean_PC1, 4), "y": round(mean_PC2, 4)},
            },
        }
