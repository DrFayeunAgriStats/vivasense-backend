"""
Trait relationship analysis:
  - Phenotypic correlations (Pearson r + significance)
  - Genotypic correlations (from genotype means — approximation)
  - Path coefficient analysis (direct + indirect effects via standardised regression)
  - Smith-Hazel selection index

Genotypic correlation approximation
------------------------------------
True genotypic correlation requires bivariate ANOVA of cross-products.
The standard software approximation (used by CROPSTAT, GENSTAT, R/metan) is:

    r_g(xy) = COV_g(xy) / √(σ²g_x × σ²g_y)

Where COV_g(xy) ≈ [(MSG_sum − MSG_diff) / 4] / (r × l) using the
"sum-and-difference" trick (Searle 1961).  When σ²g estimates are
unavailable for a trait pair, we fall back to the Pearson correlation
of per-genotype means (a common practical approximation).

Path coefficient analysis
--------------------------
Let P = phenotypic correlation matrix (traits as both predictor and target).
For target trait Y and predictors X₁…Xₙ:
  Direct path coefficients p = P_xx⁻¹ · r_xy
  where P_xx = correlation matrix of predictors, r_xy = correlations with target.
  Indirect effect of X_i via X_j = p_j × r_ij.
  Residual = √(1 − R²), where R² = Σ(p_i × r_iy).

Selection index (Smith 1936 / Hazel 1943)
------------------------------------------
  b = P⁻¹ · G · a
  I_i = b' · x_i   (index score for genotype i)
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CorrelationEngine:
    """Phenotypic/genotypic correlations, path analysis, selection index."""

    def __init__(self, config):
        self.config = config

    # ------------------------------------------------------------------
    # PHENOTYPIC CORRELATIONS
    # ------------------------------------------------------------------

    def phenotypic_correlations(
        self,
        df: pd.DataFrame,
        trait_cols: List[str],
        loc_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Pearson r between all trait pairs, with p-values.
        Also computes per-location matrices if loc_col is provided.
        """
        n = len(df)
        matrix = {}
        pmatrix = {}

        for t1 in trait_cols:
            matrix[t1] = {}
            pmatrix[t1] = {}
            for t2 in trait_cols:
                if t1 == t2:
                    matrix[t1][t2] = {"r": 1.0, "p_value": None, "significant": None}
                    continue
                valid = df[[t1, t2]].dropna()
                if len(valid) < 4:
                    matrix[t1][t2] = {"r": None, "p_value": None, "significant": None, "note": "insufficient_data"}
                    continue
                r, p = stats.pearsonr(valid[t1], valid[t2])
                matrix[t1][t2] = {
                    "r": round(float(r), 4),
                    "p_value": round(float(p), 4),
                    "significant": float(p) < self.config.alpha,
                }

        result: Dict[str, Any] = {
            "n_observations": n,
            "formula": "r_p(xy) = Cov_p(x,y) / √(Var_p(x) × Var_p(y))",
            "formula_description": "Pearson product-moment phenotypic correlation",
            "matrix": matrix,
        }

        # Per-location
        if loc_col and loc_col in df.columns:
            per_loc = {}
            for loc, gdf in df.groupby(loc_col):
                loc_mat = {}
                for t1 in trait_cols:
                    loc_mat[t1] = {}
                    for t2 in trait_cols:
                        if t1 == t2:
                            loc_mat[t1][t2] = {"r": 1.0, "p_value": None}
                            continue
                        valid = gdf[[t1, t2]].dropna()
                        if len(valid) < 4:
                            loc_mat[t1][t2] = {"r": None, "p_value": None}
                            continue
                        r, p = stats.pearsonr(valid[t1], valid[t2])
                        loc_mat[t1][t2] = {
                            "r": round(float(r), 4),
                            "p_value": round(float(p), 4),
                            "significant": float(p) < self.config.alpha,
                        }
                per_loc[str(loc)] = {"n_observations": len(gdf), "matrix": loc_mat}
            result["per_location"] = per_loc

        return result

    # ------------------------------------------------------------------
    # GENOTYPIC CORRELATIONS
    # ------------------------------------------------------------------

    def genotypic_correlations(
        self,
        df: pd.DataFrame,
        trait_cols: List[str],
        geno_col: str,
        variance_components: Dict[str, Any],  # {trait: vc_result dict}
    ) -> Dict[str, Any]:
        """
        Approximate genotypic correlations using per-genotype means and
        genotypic variances from the variance component estimates.

        r_g(xy) ≈ r_p(ȳ_x, ȳ_y) × √(PCV_x × PCV_y) / √(GCV_x × GCV_y)
        when σ²g_x and σ²g_y are available (adjusts phenotypic correlation of
        means for the heritabilities of each trait).

        Fallback: Pearson r of per-genotype means (if VC not available).
        """
        geno_means = df.groupby(geno_col)[trait_cols].mean()
        matrix: Dict[str, Dict] = {}

        for t1 in trait_cols:
            matrix[t1] = {}
            s2g_x = None
            s2p_x = None
            if t1 in variance_components:
                comps = variance_components[t1].get("components", {})
                s2g_x = comps.get("sigma2_g", {}).get("value")
                s2p_x = comps.get("sigma2_p", {}).get("value")

            for t2 in trait_cols:
                if t1 == t2:
                    matrix[t1][t2] = {"r_g": 1.0, "formula": "r_g = 1.0 (same trait)"}
                    continue

                s2g_y = None
                s2p_y = None
                if t2 in variance_components:
                    comps2 = variance_components[t2].get("components", {})
                    s2g_y = comps2.get("sigma2_g", {}).get("value")
                    s2p_y = comps2.get("sigma2_p", {}).get("value")

                valid = geno_means[[t1, t2]].dropna()
                if len(valid) < 4:
                    matrix[t1][t2] = {"r_g": None, "note": "insufficient_genotypes"}
                    continue

                r_means, _ = stats.pearsonr(valid[t1], valid[t2])

                # Adjust for heritabilities if σ²g and σ²p are available
                if (s2g_x and s2p_x and s2g_y and s2p_y
                        and s2g_x > 0 and s2g_y > 0 and s2p_x > 0 and s2p_y > 0):
                    # r_g = r_means × √(σ²p_x × σ²p_y) / √(σ²g_x × σ²g_y)
                    adj = (np.sqrt(s2p_x * s2p_y) / np.sqrt(s2g_x * s2g_y))
                    r_g = float(np.clip(r_means * adj, -1.0, 1.0))
                    method = "Adjusted for heritability: r_g = r_ȳ × √(σ²p_x·σ²p_y) / √(σ²g_x·σ²g_y)"
                else:
                    r_g = float(r_means)
                    method = "Pearson r of per-genotype means (σ²g unavailable for adjustment)"

                matrix[t1][t2] = {
                    "r_g": round(r_g, 4),
                    "r_phenotypic_means": round(float(r_means), 4),
                    "formula": "r_g(xy) = COV_g(xy) / √(σ²g_x × σ²g_y)",
                    "method": method,
                }

        return {
            "n_genotypes": len(geno_means),
            "formula": "r_g(xy) = COV_g(xy) / √(σ²g_x × σ²g_y)",
            "formula_description": "Genotypic correlation — proportion of genetic covariance between traits",
            "matrix": matrix,
            "note": "Genotypic correlations estimated from genotype means and variance components.",
        }

    # ------------------------------------------------------------------
    # PATH COEFFICIENT ANALYSIS
    # ------------------------------------------------------------------

    def path_analysis(
        self,
        df: pd.DataFrame,
        trait_cols: List[str],
        geno_col: str,
        target_trait: str,
        loc_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Direct and indirect path coefficients via standardised OLS regression.

        Method:
          1. Compute correlation matrix P of all traits.
          2. Partition into P_xx (predictors) and r_xy (target correlations).
          3. Direct paths: p = P_xx⁻¹ · r_xy.
          4. Indirect effect of X_i via X_j = p_j × r_ij.
          5. R² = Σ(p_i × r_iy).  Residual = √(1 - R²).
        """
        predictors = [t for t in trait_cols if t != target_trait]
        if not predictors:
            return {"status": "error", "error": "Need ≥2 traits for path analysis"}

        def _run_path(data: pd.DataFrame, label: str) -> Dict[str, Any]:
            geno_means = data.groupby(geno_col)[trait_cols].mean().dropna()
            if len(geno_means) < 4:
                return {"error": f"Too few genotypes in {label}"}

            # Full correlation matrix (predictors + target)
            all_cols = predictors + [target_trait]
            corr_mat = geno_means[all_cols].corr().values
            n = len(predictors)

            P_xx = corr_mat[:n, :n]   # predictor-predictor correlations
            r_xy = corr_mat[:n, n]    # predictor-target correlations

            try:
                p_direct = np.linalg.solve(P_xx, r_xy)
            except np.linalg.LinAlgError:
                p_direct = np.linalg.lstsq(P_xx, r_xy, rcond=None)[0]

            R2 = float(np.dot(p_direct, r_xy))
            residual = float(np.sqrt(max(0, 1 - R2)))

            # Direct effects
            direct: Dict[str, Any] = {}
            for i, pred in enumerate(predictors):
                direct[pred] = {
                    "value": round(float(p_direct[i]), 4),
                    "formula": "p_ij = standardised regression coefficient",
                    "correlation_with_target": round(float(r_xy[i]), 4),
                }

            # Indirect effects
            indirect: Dict[str, Dict] = {}
            for i, pred_i in enumerate(predictors):
                indirect[pred_i] = {}
                for j, pred_j in enumerate(predictors):
                    if i == j:
                        continue
                    via = float(p_direct[j]) * float(corr_mat[i, j])
                    indirect[pred_i][f"via_{pred_j}"] = {
                        "value": round(via, 4),
                        "formula": f"p_{pred_j} × r({pred_i},{pred_j})",
                    }

            return {
                "R_squared": round(R2, 4),
                "residual_effect": {
                    "value": round(residual, 4),
                    "formula": "residual = √(1 − R²)",
                },
                "direct_effects": direct,
                "indirect_effects": indirect,
            }

        overall = _run_path(df, "overall")

        per_loc: Dict[str, Any] = {}
        if loc_col and loc_col in df.columns:
            for loc, gdf in df.groupby(loc_col):
                per_loc[str(loc)] = _run_path(gdf, str(loc))

        return {
            "target_trait": target_trait,
            "predictor_traits": predictors,
            "formula_system": "Solve: r_iy = Σ(p_ij × r_jy) for direct path coefficients p_ij",
            "formula_description": "Path analysis partitions correlation into direct and indirect effects",
            **overall,
            "per_location": per_loc if per_loc else None,
        }

    # ------------------------------------------------------------------
    # SELECTION INDEX (Smith 1936 / Hazel 1943)
    # ------------------------------------------------------------------

    def selection_index(
        self,
        df: pd.DataFrame,
        trait_cols: List[str],
        geno_col: str,
        variance_components: Dict[str, Any],
        economic_weights: Optional[Dict[str, float]] = None,
        loc_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Smith-Hazel selection index.

        b = P⁻¹ · G · a
        I_i = b' · x_i   (per-genotype index score)

        where P = phenotypic variance-covariance matrix,
              G = genotypic variance-covariance matrix,
              a = vector of economic weights.
        """
        if len(trait_cols) < 2:
            return {"status": "error", "error": "Selection index requires ≥2 traits"}

        # Default economic weights: equal
        if economic_weights is None:
            economic_weights = {t: 1.0 for t in trait_cols}

        # Align weights to trait_cols order
        a = np.array([economic_weights.get(t, 0.0) for t in trait_cols])

        def _run_index(data: pd.DataFrame, vc_map: Dict[str, Any]) -> Dict[str, Any]:
            geno_means = data.groupby(geno_col)[trait_cols].mean().dropna()
            if len(geno_means) < 4:
                return {"error": "Too few genotypes for selection index"}

            # Phenotypic covariance matrix P (from raw data)
            P = geno_means.cov().values

            # Genotypic covariance matrix G (diagonal = σ²g; off-diag from phenotypic correlations scaled)
            G_diag = np.array([
                vc_map.get(t, {}).get("components", {}).get("sigma2_g", {}).get("value", 0.0) or 0.0
                for t in trait_cols
            ])
            G = np.diag(G_diag)
            # Off-diagonal: approximate via phenotypic correlations × geometric mean of σ²g
            corr_mat = geno_means.corr().values
            for i in range(len(trait_cols)):
                for j in range(len(trait_cols)):
                    if i != j:
                        G[i, j] = corr_mat[i, j] * np.sqrt(G_diag[i] * G_diag[j])

            try:
                P_inv = np.linalg.pinv(P)
                b = P_inv @ G @ a
            except Exception as e:
                return {"error": f"Matrix inversion failed: {e}"}

            # Index scores per genotype
            X = geno_means.values  # shape: n_geno × n_traits
            scores_raw = X @ b
            scores = {geno_means.index[i]: round(float(scores_raw[i]), 4)
                      for i in range(len(geno_means))}

            # Top selections (top ~5%)
            n_sel = max(1, len(scores) // 20)
            top = sorted(scores, key=scores.get, reverse=True)[:max(3, n_sel)]

            # Expected genetic gain ΔG = b' σ²g_i / σ_I
            sigma_I = float(np.sqrt(max(1e-10, b @ P @ b)))
            delta_G = {}
            for i, t in enumerate(trait_cols):
                s2g = G_diag[i]
                delta_g = (b[i] * s2g / sigma_I) * self.config.selection_intensity if sigma_I > 0 else 0.0
                grand = float(data[t].mean())
                delta_G[t] = {
                    "absolute": round(float(delta_g), 4),
                    "percent": round(float(delta_g / grand * 100) if grand != 0 else 0.0, 2),
                }

            index_weights = {trait_cols[i]: round(float(b[i]), 4) for i in range(len(trait_cols))}
            fn_terms = " + ".join(
                f"{round(b[i], 3)}×{trait_cols[i]}"
                for i in range(len(trait_cols))
                if abs(b[i]) > 1e-6
            )

            return {
                "index_weights": index_weights,
                "discriminant_function": f"I = {fn_terms}",
                "genotype_index_scores": dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)),
                "top_selections": top,
                "expected_genetic_gain": delta_G,
                "sigma_index": round(sigma_I, 4),
            }

        overall = _run_index(df, variance_components)

        per_loc: Dict[str, Any] = {}
        if loc_col and loc_col in df.columns:
            for loc, gdf in df.groupby(loc_col):
                per_loc[str(loc)] = _run_index(gdf, variance_components)

        return {
            "method": "Smith-Hazel Selection Index",
            "formula": "b = P⁻¹·G·a",
            "formula_description": (
                "Optimal index weights b from phenotypic covariance matrix P, "
                "genotypic covariance matrix G, and economic weights a"
            ),
            "economic_weights": economic_weights,
            **overall,
            "per_location": per_loc if per_loc else None,
        }
