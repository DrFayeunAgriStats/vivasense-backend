"""
Trait relationship analysis:
  - Phenotypic correlations (Pearson r + significance)
  - Genotypic correlations (RCBD two-way ANOVA on cross-products)
  - Path coefficient analysis (direct + indirect effects via standardised regression)
  - Smith-Hazel selection index

Genotypic correlation method
------------------------------
Uses RCBD two-way ANOVA on the cross-product (Falconer & Mackay 1996;
Lynch & Walsh 1998).  For each trait pair (x, y):

  1.  For each trait, run two-way ANOVA (genotype + block/rep) to obtain
      MS_G(x), MS_e(x), MS_G(y), MS_e(y) and n_reps (harmonic mean).
  2.  Compute cross-product: cp_i = (x_i − x̄)(y_i − ȳ)
  3.  Run two-way ANOVA on cp to obtain MS_G(cp) and MS_e(cp).
  4.  Genotypic covariance:  COV_g(xy) = (MS_G(cp) − MS_e(cp)) / n_reps
  5.  Genotypic variances:   σ²g(x)    = (MS_G(x)  − MS_e(x))  / n_reps
                              σ²g(y)    = (MS_G(y)  − MS_e(y))  / n_reps
  6.  r_g = COV_g(xy) / √(σ²g(x) × σ²g(y)), clamped to [−1, 1].

When rep_col is absent (no block structure) the sum-and-difference trick
(Searle 1961) is used as a fallback:
  COV_g(xy) ≈ [σ²g(x+y) − σ²g(x−y)] / 4  via one-way ANOVA.

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
        Used as fallback when rep_col is unavailable.
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

    @staticmethod
    def _rcbd_anova_ms(
        data: pd.DataFrame,
        geno_col: str,
        rep_col: str,
        value_col: str,
    ) -> Tuple[float, float, float]:
        """
        Two-way RCBD ANOVA (genotype + block) for value_col.

        Returns (MS_G, MS_error, n_reps_harmonic).

        Partitions:
          SS_total = SS_G + SS_block + SS_error
          df_G     = n_genotypes  − 1
          df_block = n_reps       − 1
          df_error = n − n_genotypes − n_reps + 1   [unbalanced approx]
        """
        valid = data[[geno_col, rep_col, value_col]].dropna()
        n = len(valid)
        if n < 4:
            return 0.0, 0.0, 1.0

        grand_mean = float(valid[value_col].mean())

        geno_groups  = valid.groupby(geno_col)[value_col]
        rep_groups   = valid.groupby(rep_col)[value_col]
        geno_means   = geno_groups.mean()
        rep_means    = rep_groups.mean()
        geno_counts  = geno_groups.count()
        rep_counts   = rep_groups.count()

        n_geno = int(len(geno_means))
        n_reps = int(len(rep_means))
        if n_geno < 2 or n_reps < 2:
            # Not enough structure — fall back to one-way (no rep correction)
            n0 = float(n_geno / (1.0 / geno_counts).sum()) if n_geno > 1 else 1.0
            SS_G = float(((geno_means - grand_mean) ** 2 * geno_counts).sum())
            SS_total = float(((valid[value_col] - grand_mean) ** 2).sum())
            SS_e = max(0.0, SS_total - SS_G)
            df_G = max(1, n_geno - 1)
            df_e = max(1, n - n_geno)
            return SS_G / df_G, SS_e / df_e, n0

        # Harmonic mean of genotype group sizes = effective n_reps
        n0 = float(n_geno / (1.0 / geno_counts).sum())

        SS_total = float(((valid[value_col] - grand_mean) ** 2).sum())
        SS_G     = float((geno_counts * (geno_means - grand_mean) ** 2).sum())
        SS_block = float((rep_counts  * (rep_means  - grand_mean) ** 2).sum())
        SS_e     = max(0.0, SS_total - SS_G - SS_block)

        df_G     = n_geno - 1
        df_block = n_reps - 1
        df_e     = max(1, n - n_geno - n_reps + 1)

        MS_G = SS_G / df_G if df_G > 0 else 0.0
        MS_e = SS_e / df_e if df_e > 0 else 0.0

        return MS_G, MS_e, n0

    @staticmethod
    def _rcbd_cross_product_mcp(
        data: pd.DataFrame,
        geno_col: str,
        rep_col: str,
        t1: str,
        t2: str,
    ) -> Tuple[float, float, float]:
        """
        RCBD two-way ANOVA mean cross-products for the (t1, t2) trait pair.

        Returns (MCP_G, MCP_error, n_reps_harmonic).

        Uses the standard MANOVA cross-product decomposition:
          CP_total = Σ (x_ij − x̄)(y_ij − ȳ)
          CP_G     = Σ_i n_i · (x̄_i − x̄)(ȳ_i − ȳ)   [genotype CP]
          CP_block = Σ_j g_j · (x̄_j − x̄)(ȳ_j − ȳ)   [block CP]
          CP_e     = CP_total − CP_G − CP_block          [error CP]

          MCP_G = CP_G / (n_geno − 1)
          MCP_e = CP_e / (n_total − n_geno − n_reps + 1)

        Reference: Falconer & Mackay (1996) §19, Lynch & Walsh (1998) §A1.4
        """
        valid = data[[geno_col, rep_col, t1, t2]].dropna()
        n = len(valid)
        if n < 4:
            return 0.0, 0.0, 1.0

        grand_x = float(valid[t1].mean())
        grand_y = float(valid[t2].mean())

        geno_grp   = valid.groupby(geno_col)
        rep_grp    = valid.groupby(rep_col)
        geno_mx    = geno_grp[t1].mean()
        geno_my    = geno_grp[t2].mean()
        rep_mx     = rep_grp[t1].mean()
        rep_my     = rep_grp[t2].mean()
        geno_cnt   = geno_grp[t1].count()
        rep_cnt    = rep_grp[t1].count()

        n_geno = int(len(geno_mx))
        n_reps = int(len(rep_mx))
        if n_geno < 2 or n_reps < 2:
            return 0.0, 0.0, 1.0

        # Harmonic mean group size
        n0 = float(n_geno / (1.0 / geno_cnt).sum())

        # Cross-product components
        CP_total = float(((valid[t1] - grand_x) * (valid[t2] - grand_y)).sum())
        CP_G     = float((geno_cnt * (geno_mx - grand_x) * (geno_my - grand_y)).sum())
        CP_block = float((rep_cnt  * (rep_mx  - grand_x) * (rep_my  - grand_y)).sum())
        CP_e     = CP_total - CP_G - CP_block

        df_G = n_geno - 1
        df_e = max(1, n - n_geno - n_reps + 1)

        MCP_G = CP_G / df_G if df_G > 0 else 0.0
        MCP_e = CP_e / df_e if df_e > 0 else 0.0

        return MCP_G, MCP_e, n0

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
        rep_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Genotypic correlations via ANCOVA covariance components.

        Primary method (when rep_col is provided — RCBD):
          For each trait pair (x, y):
            1. Two-way ANOVA (genotype + block) on x, y, and cross-product cp.
            2. σ²g(x)   = (MS_G(x)  - MS_e(x))  / n_reps
               σ²g(y)   = (MS_G(y)  - MS_e(y))  / n_reps
               COV_g(xy)= (MS_G(cp) - MS_e(cp)) / n_reps
            3. r_g = COV_g(xy) / √(σ²g(x) · σ²g(y))

          Reference: Falconer & Mackay (1996), Lynch & Walsh (1998).

        Fallback (when rep_col is absent):
          Sum-and-difference trick (Searle 1961) via one-way ANOVA.

        σ²g estimates are computed fresh from plot-level observations.
        The variance_components dict is accepted for API compatibility only.
        """
        # Build working dataframe — include rep_col if available
        base_cols = [geno_col] + trait_cols
        if rep_col and rep_col in df.columns:
            base_cols = [geno_col, rep_col] + trait_cols
        valid_df = df[base_cols].dropna(subset=[geno_col] + trait_cols)
        n_geno = int(valid_df[geno_col].nunique())

        use_rcbd = bool(rep_col and rep_col in valid_df.columns
                        and valid_df[rep_col].nunique() >= 2)

        matrix: Dict[str, Dict] = {}

        # Pre-compute per-trait σ²g using the best available method
        s2g_cache: Dict[str, float] = {}
        for t in trait_cols:
            if use_rcbd:
                ms_g, ms_e, n0 = self._rcbd_anova_ms(valid_df, geno_col, rep_col, t)
                s2g_cache[t] = max(0.0, (ms_g - ms_e) / n0) if n0 > 0 else 0.0
            else:
                s2g_cache[t] = self._sigma2g_oneway(valid_df, geno_col, t)

        for t1 in trait_cols:
            matrix[t1] = {}
            for t2 in trait_cols:
                if t1 == t2:
                    matrix[t1][t2] = {"r_g": 1.0, "formula": "r_g = 1.0 (same trait)"}
                    continue

                keep = [geno_col, t1, t2]
                if use_rcbd:
                    keep = [geno_col, rep_col, t1, t2]
                pair_df = valid_df[keep].dropna().copy()
                n_pair_geno = int(pair_df[geno_col].nunique())
                if n_pair_geno < 4:
                    matrix[t1][t2] = {"r_g": None, "note": "insufficient_genotypes"}
                    continue

                s2g_x = s2g_cache[t1]
                s2g_y = s2g_cache[t2]

                if use_rcbd:
                    # ── RCBD MANOVA cross-product decomposition ────────────────
                    mcp_g, mcp_e, n0 = self._rcbd_cross_product_mcp(
                        pair_df, geno_col, rep_col, t1, t2
                    )
                    cov_g = (mcp_g - mcp_e) / n0 if n0 > 0 else 0.0

                    method_str = (
                        "RCBD MANOVA cross-product decomposition "
                        "(Falconer & Mackay 1996; Lynch & Walsh 1998)"
                    )
                    formula_str = (
                        "r_g = COV_g(xy) / √(σ²g_x × σ²g_y); "
                        "COV_g = (MCP_G − MCP_e) / n_reps"
                    )
                else:
                    # ── Fallback: sum-and-difference trick ────────────────────
                    pair_df["_u"] = pair_df[t1] + pair_df[t2]
                    pair_df["_v"] = pair_df[t1] - pair_df[t2]
                    s2g_u = self._sigma2g_oneway(pair_df, geno_col, "_u")
                    s2g_v = self._sigma2g_oneway(pair_df, geno_col, "_v")
                    cov_g = (s2g_u - s2g_v) / 4.0
                    method_str = "One-way ANOVA sum-and-difference trick (Searle 1961)"
                    formula_str = "r_g = COV_g(xy) / √(σ²g_x × σ²g_y); COV_g ≈ [σ²g(x+y) − σ²g(x−y)] / 4"

                denom = float(np.sqrt(max(0.0, s2g_x) * max(0.0, s2g_y)))

                if denom < 1e-10:
                    # Near-zero genetic variance — fall back to correlation of genotype means
                    gm = pair_df.groupby(geno_col)[[t1, t2]].mean()
                    if len(gm) >= 4:
                        r_fb, _ = stats.pearsonr(gm[t1], gm[t2])
                        r_g_val = round(float(np.clip(r_fb, -1.0, 1.0)), 4)
                    else:
                        r_g_val = None
                    matrix[t1][t2] = {
                        "r_g": r_g_val,
                        "method": "Pearson r of genotype means (σ²g ≈ 0 for one or both traits)",
                        "formula": "r_g ≈ r(ȳ_x, ȳ_y)",
                        "sigma2g_x": round(float(s2g_x), 6),
                        "sigma2g_y": round(float(s2g_y), 6),
                        "warning": "Near-zero genotypic variance — estimate unreliable",
                    }
                    continue

                r_g_raw = cov_g / denom
                r_g = float(np.clip(r_g_raw, -1.0, 1.0))

                entry: Dict[str, Any] = {
                    "r_g":       round(r_g, 4),
                    "r_g_raw":   round(float(r_g_raw), 6),   # pre-clamp, for diagnostics
                    "cov_g":     round(float(cov_g), 6),
                    "sigma2g_x": round(float(s2g_x), 6),
                    "sigma2g_y": round(float(s2g_y), 6),
                    "method":    method_str,
                    "formula":   formula_str,
                }
                if abs(r_g) >= 0.95:
                    entry["warning"] = (
                        f"Very high genotypic correlation ({r_g:.3f}) — "
                        "verify data quality and check for outliers."
                    )
                elif abs(r_g_raw) > 1.0:
                    entry["warning"] = (
                        f"Raw r_g ({r_g_raw:.4f}) exceeded ±1 and was clamped. "
                        "This may indicate negative genetic variance or insufficient replication."
                    )
                matrix[t1][t2] = entry

        method_used = (
            "RCBD ANCOVA cross-product ANOVA (Falconer & Mackay 1996)"
            if use_rcbd
            else "One-way ANOVA sum-and-difference trick (Searle 1961)"
        )
        return {
            "n_genotypes": n_geno,
            "method": method_used,
            "formula": "r_g(xy) = COV_g(xy) / √(σ²g_x × σ²g_y)",
            "formula_description": (
                "Genotypic correlation from ANOVA covariance components on plot-level data. "
                + ("Two-way RCBD ANOVA used for unbiased block-corrected estimates."
                   if use_rcbd
                   else "One-way ANOVA fallback (no rep_col provided).")
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

            # ── Path matrix (publication format) ─────────────────────────────
            # For each predictor i:
            #   Total_r[i] = direct[i] + Σ_{j≠i} r_ij × p_j  =  r_xy[i]
            # (guaranteed by path analysis identity; used as validation)
            path_matrix: Dict[str, Any] = {}
            for i, pred_i in enumerate(predictors):
                p_i    = float(p_direct[i])
                row: Dict[str, Any] = {"Direct": round(p_i, 4)}

                total_indirect = 0.0
                for j, pred_j in enumerate(predictors):
                    if j == i:
                        continue
                    via_val = float(P_xx[i, j]) * float(p_direct[j])
                    row[f"Indirect_via_{pred_j}"] = round(via_val, 4)
                    total_indirect += via_val

                # Total_r should equal r_xy[i]; use it directly for accuracy
                row["Total_r"] = round(float(r_xy[i]), 4)
                path_matrix[pred_i] = row

            # Residual row — no indirect paths
            residual_row: Dict[str, Any] = {"Direct": round(residual, 4)}
            for pred_j in predictors:
                residual_row[f"Indirect_via_{pred_j}"] = 0.0
            residual_row["Total_r"] = round(residual, 4)
            path_matrix["Residual (Unmeasured)"] = residual_row

            # ── Effect decomposition ──────────────────────────────────────────
            effect_decomposition: Dict[str, Any] = {}
            for i, pred_i in enumerate(predictors):
                p_i   = float(p_direct[i])
                total = float(r_xy[i])        # = direct + sum(indirects)
                indir = total - p_i

                # Avoid division by zero / near-zero total
                if abs(total) > 1e-6:
                    pct_d = round(p_i   / total * 100, 2)
                    pct_i = round(indir / total * 100, 2)
                else:
                    pct_d, pct_i = 0.0, 0.0

                effect_decomposition[pred_i] = {
                    "direct":       round(p_i,   4),
                    "indirect":     round(indir, 4),
                    "total":        round(total, 4),
                    "pct_direct":   pct_d,
                    "pct_indirect": pct_i,
                    "vif":          vifs.get(pred_i),
                }

            # ── Model fit block ───────────────────────────────────────────────
            r2_pct  = round(R2 * 100, 2)
            res_pct = round((1.0 - R2) * 100, 2)
            model_fit: Dict[str, Any] = {
                "r_squared":  round(R2, 4),
                "residual_path": round(residual, 4),
                "residual_path_formula": "p_res = √(1 − R²)",
                "residual_interpretation": (
                    f"Model explains {r2_pct:.1f}% of variation in {target_trait}. "
                    f"{res_pct:.1f}% is attributable to unmeasured variables."
                ),
            }

            # ── Build path_matrix_html using plain HTML (no pandas dependency) ─
            if path_matrix:
                col_keys = list(next(iter(path_matrix.values())).keys())
                header = "<tr><th>Predictor</th>" + "".join(
                    f"<th>{c}</th>" for c in col_keys
                ) + "</tr>"
                rows_html = []
                for row_name, row_vals in path_matrix.items():
                    cells = "".join(
                        f"<td>{row_vals.get(c, '')}</td>" for c in col_keys
                    )
                    rows_html.append(f"<tr><th>{row_name}</th>{cells}</tr>")
                path_matrix_html = (
                    f"<table border='1' style='border-collapse:collapse'>"
                    f"<thead>{header}</thead>"
                    f"<tbody>{''.join(rows_html)}</tbody>"
                    f"</table>"
                )
            else:
                path_matrix_html = ""

            # ── Path diagram (PNG + Plotly JSON) ──────────────────────────
            path_diagram: Dict[str, Any] = {}
            try:
                from ..path_visualization import build_path_diagram as _build_pd
                path_diagram = _build_pd(
                    target_trait=target_trait,
                    predictor_traits=predictors,
                    p_direct={pred: float(p_direct[i])
                               for i, pred in enumerate(predictors)},
                    residual=residual,
                    r_squared=R2,
                )
            except Exception as _pd_exc:
                logger.warning("Path diagram generation failed: %s", _pd_exc)

            return {
                "R_squared": round(R2, 4),
                "residual_effect": {
                    "value": round(residual, 4),
                    "formula": "residual = sqrt(1 - R^2)",
                },
                "direct_effects":       direct,
                "indirect_effects":     indirect,
                "path_matrix":          path_matrix,
                "path_matrix_html":     path_matrix_html,
                "effect_decomposition": effect_decomposition,
                "model_fit":            model_fit,
                "path_diagram":         path_diagram,
                "diagnostics":          diagnostics,
                "warnings":             diag_warnings,
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
