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

    @staticmethod
    def _sigma2g_oneway(data: pd.DataFrame, geno_col: str, trait: str) -> float:
        """
        One-way ANOVA estimate of σ²g from plot-level data.
        Returns σ²g ≥ 0 (negative estimates clamped to 0).
        """
        valid = data[[geno_col, trait]].dropna()
        if len(valid) < 4:
            return 0.0
        groups = valid.groupby(geno_col)[trait]
        counts = groups.count()
        n_geno = int(len(counts))
        n_total = int(counts.sum())
        if n_geno < 2 or n_total <= n_geno:
            return 0.0
        geno_means_v = groups.mean()
        grand_mean = float(valid[trait].mean())
        # Harmonic mean of group sizes (handles unbalanced design)
        n0 = float(n_geno / (1.0 / counts).sum())
        SS_geno = float(((geno_means_v - grand_mean) ** 2 * counts).sum())
        SS_error = float(groups.apply(lambda x: float(((x - x.mean()) ** 2).sum())).sum())
        df_geno = n_geno - 1
        df_error = n_total - n_geno
        if df_geno <= 0 or df_error <= 0:
            return 0.0
        MS_geno = SS_geno / df_geno
        MS_error = SS_error / df_error
        return max(0.0, (MS_geno - MS_error) / n0)

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
        variance_components: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Experimental genotypic correlations via ANCOVA covariance components.

        Uses the sum-and-difference trick (Searle 1961) on raw plot-level data:
          Let u = x + y,  v = x − y
          σ²g(u) = one-way ANOVA estimate on u
          σ²g(v) = one-way ANOVA estimate on v
          COV_g(x,y) = [σ²g(u) − σ²g(v)] / 4
          r_g = COV_g(x,y) / √(σ²g_x × σ²g_y)

        σ²g estimates are derived from raw plot observations, NOT from the
        passed variance_components dict (which is retained for API compatibility).
        """
        valid_df = df[[geno_col] + trait_cols].dropna()
        n_geno = int(valid_df[geno_col].nunique())
        matrix: Dict[str, Dict] = {}

        # Pre-compute per-trait σ²g from one-way ANOVA on full dataset
        s2g_cache: Dict[str, float] = {}
        for t in trait_cols:
            s2g_cache[t] = self._sigma2g_oneway(valid_df, geno_col, t)

        for t1 in trait_cols:
            matrix[t1] = {}
            for t2 in trait_cols:
                if t1 == t2:
                    matrix[t1][t2] = {"r_g": 1.0, "formula": "r_g = 1.0 (same trait)"}
                    continue

                pair_df = valid_df[[geno_col, t1, t2]].dropna().copy()
                n_pair_geno = int(pair_df[geno_col].nunique())
                if n_pair_geno < 4:
                    matrix[t1][t2] = {"r_g": None, "note": "insufficient_genotypes"}
                    continue

                s2g_x = s2g_cache[t1]
                s2g_y = s2g_cache[t2]

                # Sum-and-difference trick
                pair_df["_u"] = pair_df[t1] + pair_df[t2]
                pair_df["_v"] = pair_df[t1] - pair_df[t2]
                s2g_u = self._sigma2g_oneway(pair_df, geno_col, "_u")
                s2g_v = self._sigma2g_oneway(pair_df, geno_col, "_v")
                cov_g = (s2g_u - s2g_v) / 4.0

                denom = float(np.sqrt(max(0.0, s2g_x) * max(0.0, s2g_y)))
                if denom < 1e-10:
                    # Both σ²g ≈ 0 — fall back to Pearson r of genotype means
                    gm = pair_df.groupby(geno_col)[[t1, t2]].mean()
                    r_fb, _ = stats.pearsonr(gm[t1], gm[t2])
                    matrix[t1][t2] = {
                        "r_g": round(float(np.clip(r_fb, -1.0, 1.0)), 4),
                        "method": "Pearson r of genotype means (σ²g ≈ 0 for one or both traits)",
                        "formula": "r_g ≈ r(ȳ_x, ȳ_y)",
                        "warning": "Near-zero genotypic variance; estimate unreliable",
                    }
                    continue

                r_g = float(np.clip(cov_g / denom, -1.0, 1.0))
                entry: Dict[str, Any] = {
                    "r_g": round(r_g, 4),
                    "cov_g": round(float(cov_g), 6),
                    "sigma2g_x": round(float(s2g_x), 6),
                    "sigma2g_y": round(float(s2g_y), 6),
                    "method": "Experimental genotypic correlation (ANCOVA sum-and-difference, Searle 1961)",
                    "formula": "r_g = COV_g(xy) / √(σ²g_x × σ²g_y)",
                }
                if abs(r_g) >= 0.95:
                    entry["warning"] = "High correlation detected; verify covariance matrix"
                matrix[t1][t2] = entry

        return {
            "n_genotypes": n_geno,
            "method": "ANCOVA covariance components (sum-and-difference trick, Searle 1961)",
            "formula": "r_g(xy) = COV_g(xy) / √(σ²g_x × σ²g_y)",
            "formula_description": (
                "Experimental genotypic correlation from bivariate ANOVA covariance components. "
                "COV_g(xy) estimated via sum-and-difference trick on raw plot-level data."
            ),
            "matrix": matrix,
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

            all_cols = predictors + [target_trait]
            corr_mat = geno_means[all_cols].corr().values
            n = len(predictors)

            P_xx = corr_mat[:n, :n]
            r_xy = corr_mat[:n, n]

            # ── Multicollinearity diagnostics ──────────────────────────────────
            eigvals = np.abs(np.linalg.eigvalsh(P_xx))
            cond = float(eigvals.max() / max(float(eigvals.min()), 1e-10))

            vifs: Dict[str, float] = {}
            for j, pred in enumerate(predictors):
                if n < 2:
                    vifs[pred] = 1.0
                else:
                    r_j = np.delete(P_xx[j, :], j)
                    P_oth = np.delete(np.delete(P_xx, j, axis=0), j, axis=1)
                    try:
                        R2_j = float(r_j @ np.linalg.pinv(P_oth) @ r_j)
                        R2_j = min(R2_j, 0.9999)
                    except Exception:
                        R2_j = 0.0
                    vifs[pred] = round(1.0 / max(1e-10, 1.0 - R2_j), 2)

            max_vif = max(vifs.values()) if vifs else 0.0
            mc_flag = ("High" if max_vif > 10
                       else ("Moderate" if max_vif > 5 else "Low"))

            diag_warnings: List[str] = []
            if max_vif > 10:
                diag_warnings.append(
                    "High multicollinearity detected. Path coefficients may be unstable.")
            if cond > 1000:
                diag_warnings.append(
                    "Correlation matrix is ill-conditioned. "
                    "Consider removing redundant predictors.")

            diagnostics: Dict[str, Any] = {
                "vif": vifs,
                "condition_number": round(cond, 1),
                "multicollinearity_flag": mc_flag,
            }

            # ── Solve for direct path coefficients (ridge if collinear) ─────────
            if max_vif > 10 or cond > 1000:
                ridge_k = 0.1
                P_reg = P_xx + np.eye(n) * ridge_k
                try:
                    p_direct = np.linalg.solve(P_reg, r_xy)
                except np.linalg.LinAlgError:
                    p_direct = np.linalg.lstsq(P_reg, r_xy, rcond=None)[0]
                diag_warnings.append(
                    f"Ridge regression applied (k={ridge_k}) to stabilise path coefficients.")
            else:
                try:
                    p_direct = np.linalg.solve(P_xx, r_xy)
                except np.linalg.LinAlgError:
                    p_direct = np.linalg.lstsq(P_xx, r_xy, rcond=None)[0]

            R2 = float(np.dot(p_direct, r_xy))
            residual = float(np.sqrt(max(0, 1 - R2)))

            direct: Dict[str, Any] = {}
            for i, pred in enumerate(predictors):
                direct[pred] = {
                    "value": round(float(p_direct[i]), 4),
                    "formula": "p_ij = standardised regression coefficient",
                    "correlation_with_target": round(float(r_xy[i]), 4),
                }

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
                "diagnostics": diagnostics,
                "warnings": diag_warnings,
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

            # ── FIX: P from raw plot-level observations, NOT genotype means ────
            # Using geno_means.cov() gives variance of means ≈ σ²p/n_reps,
            # which makes P^{-1} n_reps times too large → b explodes → gains > 1000%.
            raw_valid = data[trait_cols].dropna()
            P = raw_valid.cov().values

            # ── Condition number check ─────────────────────────────────────────
            try:
                eigvals_P = np.abs(np.linalg.eigvalsh(P))
                P_cond = float(eigvals_P.max() / max(float(eigvals_P.min()), 1e-10))
            except Exception:
                P_cond = 0.0

            cond_warnings: List[str] = []
            if P_cond > 1000:
                cond_warnings.append(
                    "Matrix ill-conditioned (condition number "
                    f"{P_cond:.0f}); weights may be unstable.")
                P_inv = np.linalg.pinv(P + np.eye(len(trait_cols)) * 1e-4)
            else:
                P_inv = np.linalg.pinv(P)

            # ── Genotypic covariance matrix G ──────────────────────────────────
            G_diag = np.array([
                vc_map.get(t, {}).get("components", {}).get("sigma2_g", {}).get("value", 0.0) or 0.0
                for t in trait_cols
            ])
            G = np.diag(G_diag)
            corr_mat = geno_means.corr().values
            for i in range(len(trait_cols)):
                for j in range(len(trait_cols)):
                    if i != j:
                        G[i, j] = corr_mat[i, j] * np.sqrt(max(0, G_diag[i]) * max(0, G_diag[j]))

            try:
                b = P_inv @ G @ a
            except Exception as exc:
                return {"error": f"Matrix inversion failed: {exc}"}

            # ── Index scores per genotype ──────────────────────────────────────
            X = geno_means.values
            scores_raw = X @ b
            scores = {geno_means.index[i]: round(float(scores_raw[i]), 4)
                      for i in range(len(geno_means))}

            n_sel = max(1, len(scores) // 20)
            top = sorted(scores, key=scores.get, reverse=True)[:max(3, n_sel)]

            # ── Expected genetic gain ΔG_i = i × σ²g_i × b_i / σ_I ───────────
            sigma_I = float(np.sqrt(max(1e-10, b @ P @ b)))
            sel_i = float(self.config.selection_intensity)

            delta_G: Dict[str, Any] = {}
            for idx_t, t in enumerate(trait_cols):
                s2g = float(G_diag[idx_t])
                gain_abs = sel_i * s2g * float(b[idx_t]) / sigma_I if sigma_I > 0 else 0.0
                grand = float(data[t].mean())
                pct = float(gain_abs / grand * 100) if grand != 0 else 0.0
                flag: Optional[str] = None
                if abs(pct) > 500:
                    flag = "❌ Gain >500% is implausible. Check matrix conditioning and trait scaling."
                elif abs(pct) > 100:
                    flag = "⚠️ Unusually large gain (>100%). Verify economic weights and covariance."
                delta_G[t] = {
                    "absolute": round(float(gain_abs), 4),
                    "percent": round(float(pct), 2),
                    **({"flag": flag} if flag else {}),
                }

            # ── Index accuracy r_IH ───────────────────────────────────────────
            # r_IH = Cov(I,H) / √(Var(I)·Var(H)) = b'Ga / √(b'Pb · a'Ga)
            b_Ga = float(b @ G @ a)
            var_I = max(1e-10, float(b @ P @ b))
            var_H = max(1e-10, float(a @ G @ a))
            r_IH = float(np.clip(b_Ga / np.sqrt(var_I * var_H), 0.0, 1.0))

            index_weights = {trait_cols[i]: round(float(b[i]), 4) for i in range(len(trait_cols))}
            fn_terms = " + ".join(
                f"{round(b[i], 3)}×{trait_cols[i]}"
                for i in range(len(trait_cols))
                if abs(b[i]) > 1e-6
            )

            result: Dict[str, Any] = {
                "index_weights": index_weights,
                "discriminant_function": f"I = {fn_terms}",
                "genotype_index_scores": dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)),
                "top_selections": top,
                "expected_genetic_gain": delta_G,
                "sigma_index": round(sigma_I, 4),
                "index_accuracy_r_IH": round(r_IH, 4),
                "matrix_diagnostics": {
                    "P_condition_number": round(P_cond, 1),
                    "P_matrix_source": "raw plot-level observations",
                },
                **({"warnings": cond_warnings} if cond_warnings else {}),
            }
            return result

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
