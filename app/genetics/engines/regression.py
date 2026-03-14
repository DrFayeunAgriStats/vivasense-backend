"""
Multiple Linear Regression engine for VivaSense genetics module.

Provides:
  - OLS regression via statsmodels
  - Coefficient table (β, SE, t-statistic, p-value, 95% CI, significance)
  - Model fit statistics (R², adjusted R², F-statistic, F p-value, RMSE, AIC, BIC)
  - Variance Inflation Factor (VIF) — multicollinearity detection
  - Assumption tests:
      * Shapiro-Wilk (n ≤ 5000) or Kolmogorov-Smirnov (n > 5000) on residuals
      * Breusch-Pagan test for homoscedasticity
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _safe_float(v: Any) -> Optional[float]:
    """Cast to float, returning None for nan/inf/non-numeric."""
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _sig_stars(p: Optional[float]) -> str:
    if p is None:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "N/A"
    return f"{p:.4f}" if p >= 0.001 else "<0.001"


class MultipleRegressionEngine:
    """OLS multiple regression with full diagnostics."""

    def __init__(self, config=None):
        self.config = config

    def analyze(
        self,
        df: pd.DataFrame,
        response: str,
        predictors: List[str],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Run OLS multiple regression and compute all diagnostics.

        Parameters
        ----------
        df         : input DataFrame (must contain response + predictor columns)
        response   : name of the dependent variable column
        predictors : list of predictor (independent variable) column names
        alpha      : significance level for CI and assumption tests (default 0.05)

        Returns
        -------
        dict with keys:
          status, regression { model, coefficients, model_fit, vif, assumptions, warnings }
        """
        warnings: List[str] = []

        # ── Validate inputs ───────────────────────────────────────────────────
        missing_cols = [c for c in [response] + predictors if c not in df.columns]
        if missing_cols:
            return {
                "status": "failed",
                "error": f"Column(s) not found in data: {missing_cols}",
                "warnings": warnings,
            }

        cols = [response] + predictors
        data = df[cols].dropna()
        n = len(data)
        k = len(predictors)

        if n < k + 2:
            return {
                "status": "failed",
                "error": (
                    f"Insufficient observations ({n}) for {k} predictor(s). "
                    f"Need at least {k + 2} complete rows."
                ),
                "warnings": warnings,
            }

        # Check for constant predictors
        for pred in predictors:
            if data[pred].nunique() <= 1:
                warnings.append(
                    f"Predictor '{pred}' has zero or near-zero variance — "
                    "it may cause perfect multicollinearity."
                )

        # ── Import statsmodels ────────────────────────────────────────────────
        try:
            import statsmodels.api as sm
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            from statsmodels.stats.diagnostic import het_breuschpagan
        except ImportError as exc:
            return {
                "status": "failed",
                "error": f"Missing dependency: {exc}",
                "warnings": warnings,
            }

        y = data[response].values.astype(float)
        X = data[predictors].values.astype(float)
        X_const = sm.add_constant(X, has_constant="add")

        # ── Fit OLS ───────────────────────────────────────────────────────────
        try:
            result = sm.OLS(y, X_const).fit()
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"OLS fit failed: {exc}",
                "warnings": warnings,
            }

        # ── Coefficients table ────────────────────────────────────────────────
        term_names = ["Intercept"] + list(predictors)
        params     = result.params
        bse        = result.bse
        tvalues    = result.tvalues
        pvalues    = result.pvalues
        conf       = np.asarray(result.conf_int(alpha=alpha))

        coefficients: List[Dict[str, Any]] = []
        for i, name in enumerate(term_names):
            p = _safe_float(pvalues[i])
            coefficients.append({
                "term":        name,
                "estimate":    _safe_float(params[i]),
                "std_error":   _safe_float(bse[i]),
                "t_statistic": _safe_float(tvalues[i]),
                "p_value":     p,
                "p_display":   _fmt_p(p),
                "significance": _sig_stars(p),
                "ci_lower":    _safe_float(float(conf[i, 0])),
                "ci_upper":    _safe_float(float(conf[i, 1])),
            })

        # ── Model fit ─────────────────────────────────────────────────────────
        residuals = result.resid
        rmse      = _safe_float(float(np.sqrt(np.mean(residuals ** 2))))
        f_p       = _safe_float(result.f_pvalue)

        model_fit: Dict[str, Any] = {
            "r_squared":     _safe_float(result.rsquared),
            "adj_r_squared": _safe_float(result.rsquared_adj),
            "f_statistic":   _safe_float(result.fvalue),
            "f_p_value":     f_p,
            "f_p_display":   _fmt_p(f_p),
            "rmse":          rmse,
            "n_obs":         int(n),
            "n_predictors":  int(k),
            "aic":           _safe_float(result.aic),
            "bic":           _safe_float(result.bic),
        }

        # ── VIF ───────────────────────────────────────────────────────────────
        vif: Dict[str, Any] = {}
        if k == 1:
            # VIF is always 1.0 for a single predictor
            vif[predictors[0]] = {"vif": 1.0, "flag": "ok",
                                  "interpretation": "Single predictor — VIF not applicable."}
        else:
            try:
                for j, pred in enumerate(predictors):
                    vif_val = _safe_float(variance_inflation_factor(X_const, j + 1))
                    if vif_val is None:
                        flag = "unknown"
                        msg = "Could not compute VIF."
                    elif vif_val > 10:
                        flag = "severe"
                        msg = f"VIF = {vif_val:.2f} — severe multicollinearity (>10). Consider removing or combining this predictor."
                        warnings.append(f"'{pred}': VIF = {vif_val:.2f} — severe multicollinearity.")
                    elif vif_val > 5:
                        flag = "moderate"
                        msg = f"VIF = {vif_val:.2f} — moderate multicollinearity (>5). Review predictor relationships."
                        warnings.append(f"'{pred}': VIF = {vif_val:.2f} — moderate multicollinearity.")
                    else:
                        flag = "ok"
                        msg = f"VIF = {vif_val:.2f} — acceptable (<5)."
                    vif[pred] = {"vif": vif_val, "flag": flag, "interpretation": msg}
            except Exception as exc:
                logger.warning("VIF computation failed: %s", exc)
                warnings.append(f"VIF computation skipped: {exc}")

        # ── Assumption tests ──────────────────────────────────────────────────
        assumptions: Dict[str, Any] = {}

        # Normality of residuals
        resid_arr = np.asarray(residuals, dtype=float)
        if len(resid_arr) >= 3:
            if len(resid_arr) <= 5000:
                try:
                    sw_stat, sw_p = stats.shapiro(resid_arr)
                    sw_p_f = _safe_float(sw_p)
                    passed = bool(sw_p >= alpha) if sw_p_f is not None else None
                    assumptions["normality"] = {
                        "test":           "Shapiro-Wilk",
                        "statistic":      _safe_float(sw_stat),
                        "p_value":        sw_p_f,
                        "p_display":      _fmt_p(sw_p_f),
                        "passed":         passed,
                        "interpretation": (
                            "Residuals are normally distributed (p ≥ α)."
                            if passed
                            else "Residuals deviate from normality (p < α). "
                                 "Consider a Box-Cox or log transformation of the response."
                        ),
                    }
                except Exception as exc:
                    assumptions["normality"] = {"test": "Shapiro-Wilk", "error": str(exc)}
            else:
                try:
                    mean_r = float(np.mean(resid_arr))
                    std_r  = float(np.std(resid_arr, ddof=1))
                    ks_stat, ks_p = stats.kstest(
                        resid_arr, "norm", args=(mean_r, std_r)
                    )
                    ks_p_f = _safe_float(ks_p)
                    passed = bool(ks_p >= alpha) if ks_p_f is not None else None
                    assumptions["normality"] = {
                        "test":           "Kolmogorov-Smirnov",
                        "statistic":      _safe_float(ks_stat),
                        "p_value":        ks_p_f,
                        "p_display":      _fmt_p(ks_p_f),
                        "passed":         passed,
                        "interpretation": (
                            "Residuals are normally distributed (p ≥ α)."
                            if passed
                            else "Residuals deviate from normality (p < α)."
                        ),
                    }
                except Exception as exc:
                    assumptions["normality"] = {"test": "KS", "error": str(exc)}
        else:
            assumptions["normality"] = {
                "test": "Shapiro-Wilk",
                "passed": None,
                "interpretation": "Too few residuals for normality test.",
            }

        # Homoscedasticity — Breusch-Pagan
        try:
            bp_lm, bp_p, _bp_fval, _bp_fp = het_breuschpagan(resid_arr, X_const)
            bp_p_f = _safe_float(bp_p)
            passed = bool(bp_p >= alpha) if bp_p_f is not None else None
            assumptions["homoscedasticity"] = {
                "test":           "Breusch-Pagan",
                "statistic":      _safe_float(bp_lm),
                "p_value":        bp_p_f,
                "p_display":      _fmt_p(bp_p_f),
                "passed":         passed,
                "interpretation": (
                    "Homoscedasticity assumption is met — constant residual variance (p ≥ α)."
                    if passed
                    else "Heteroscedasticity detected (p < α). "
                         "Consider weighted least squares or robust standard errors (HC3)."
                ),
            }
        except Exception as exc:
            logger.warning("Breusch-Pagan test failed: %s", exc)
            assumptions["homoscedasticity"] = {
                "test": "Breusch-Pagan",
                "error": str(exc),
                "passed": None,
            }

        # ── Assemble result ───────────────────────────────────────────────────
        formula = f"{response} ~ {' + '.join(predictors)}"
        return {
            "status": "success",
            "regression": {
                "model":        formula,
                "response":     response,
                "predictors":   predictors,
                "coefficients": coefficients,
                "model_fit":    model_fit,
                "vif":          vif,
                "assumptions":  assumptions,
                "warnings":     warnings,
            },
        }
