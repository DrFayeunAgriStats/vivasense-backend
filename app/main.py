"""
VivaSense Statistical Backend (V1)
FastAPI + pandas + statsmodels + scipy + matplotlib

Features:
- Robust ANOVA (Type II default; Type III optional)
- Tukey HSD + compact letter display
- Mean plots with error bars, box plots
- Correlation, regression, heatmaps
- Assumption diagnostics + diagnostic plots
- Downloadable tables (CSV/Excel) + plots (PNG) via ZIP
- Pay-per-analysis plumbing via Paystack (initialize + webhook)
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import traceback
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # server-safe
import matplotlib.pyplot as plt

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from scipy import stats

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import linear_rainbow

# If httpx is not installed in your environment, add it to requirements.txt
try:
    import httpx
except Exception:
    httpx = None  # paystack endpoints will error clearly if not available


# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("vivasense")


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# =========================
# CONFIG
# =========================
@dataclass
class AnalysisConfig:
    alpha: float = 0.05
    anova_type: Literal[2, 3] = 2  # 2 (Type II) default; 3 for Type III
    figure_dpi: int = 250
    figure_format: str = "png"
    include_interactions: bool = True
    max_interaction_level: int = 2  # 2-way interactions by default
    # effect-size thresholds (eta-squared)
    effect_small: float = 0.01
    effect_medium: float = 0.06
    effect_large: float = 0.14

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssumptionTest:
    test_name: str
    statistic: float
    p_value: Optional[float]
    passed: bool
    message: str = ""


@dataclass
class EffectSize:
    eta_squared: float
    omega_squared: float
    cohens_f: float
    interpretation: str


# =========================
# IN-MEM STORES (V1)
# =========================
# In production: swap this with Redis / DB.
ANALYSIS_STORE: Dict[str, Dict[str, Any]] = {}     # analysis_id -> results/metadata/status
PAYMENT_STORE: Dict[str, Dict[str, Any]] = {}      # reference -> payment status + analysis_id


# =========================
# UTIL
# =========================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (np.floating, float, int, np.integer)):
            return float(x)
        return float(x)
    except Exception:
        return None


def df_hash(df: pd.DataFrame) -> str:
    # stable enough for cache (V1)
    s = df.to_json(date_format="iso", orient="split")
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def ensure_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce")


def to_rowwise_anova(anova_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Convert statsmodels anova_lm DataFrame into row-wise dict:
    {
      "FactorA": {"df":..., "sum_sq":..., "mean_sq":..., "F":..., "PR(>F)":...},
      ...
    }
    """
    out: Dict[str, Dict[str, Any]] = {}
    for idx, row in anova_df.iterrows():
        r = {}
        for col in anova_df.columns:
            r[col] = safe_float(row[col]) if col != "df" else safe_float(row[col])
        out[str(idx)] = r
    # add mean_sq if not present
    if "mean_sq" not in anova_df.columns and "sum_sq" in anova_df.columns and "df" in anova_df.columns:
        for k in out:
            dfv = out[k].get("df") or 0.0
            ssv = out[k].get("sum_sq") or 0.0
            out[k]["mean_sq"] = (ssv / dfv) if dfv > 0 else None
    return out


# =========================
# CORE ANALYZER
# =========================
class StatisticalAnalyzer:
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()

    # ---------- data detection ----------
    def detect_variable_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        categorical: List[str] = []
        continuous: List[str] = []
        block_keywords = ["block", "rep", "replicate", "batch", "plot", "field"]

        for col in df.columns:
            if col is None:
                continue
            c = str(col).strip()
            low = c.lower()

            # categorical heuristics
            if (
                df[c].dtype == "object"
                or str(df[c].dtype).startswith("category")
                or low in block_keywords
                or any(k in low for k in block_keywords)
                or (pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) < 10)
            ):
                categorical.append(c)
            else:
                continuous.append(c)

        return categorical, continuous

    def validate_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        errors: List[str] = []
        warns: List[str] = []

        if len(df) < 10:
            errors.append(f"Sample size too small ({len(df)} rows). Minimum 10 rows recommended.")

        missing = df.isna().sum()
        if (missing > 0).any():
            cols = missing[missing > 0].index.tolist()
            for c in cols:
                pct = float(missing[c] / len(df) * 100)
                warns.append(f"Column '{c}' has {pct:.1f}% missing values.")

        for c in df.columns:
            try:
                if df[c].nunique(dropna=True) <= 1:
                    warns.append(f"Column '{c}' is constant (may be uninformative).")
            except Exception:
                pass

        # extreme outliers check (numeric only)
        for c in df.select_dtypes(include=[np.number]).columns:
            s = df[c].dropna()
            if len(s) >= 10 and float(s.std()) > 0:
                z = np.abs((s - float(s.mean())) / float(s.std()))
                n_out = int((z > 6).sum())
                if n_out > 0:
                    warns.append(f"Column '{c}' has {n_out} extreme outliers (>6 SD).")

        return {"errors": errors, "warnings": warns}

    # ---------- model ----------
    def build_formula(self, response: str, predictors: List[str], blocks: Optional[List[str]] = None) -> str:
        all_terms: List[str] = []
        all_terms.extend(predictors)
        if blocks:
            for b in blocks:
                if b not in all_terms:
                    all_terms.append(b)

        if not all_terms:
            raise ValueError("No predictors provided for model formula.")

        formula = f"Q('{response}') ~ " + " + ".join([f"Q('{t}')" for t in all_terms])

        # add interactions among predictors only (not blocks)
        if self.config.include_interactions and len(predictors) >= 2:
            inter_terms: List[str] = []
            preds = predictors[:]
            # 2-way
            if self.config.max_interaction_level >= 2:
                for i in range(len(preds)):
                    for j in range(i + 1, len(preds)):
                        inter_terms.append(f"Q('{preds[i]}'):Q('{preds[j]}')")

            # 3-way (optional)
            if self.config.max_interaction_level >= 3 and len(preds) >= 3:
                for i in range(len(preds)):
                    for j in range(i + 1, len(preds)):
                        for k in range(j + 1, len(preds)):
                            inter_terms.append(f"Q('{preds[i]}'):Q('{preds[j]}'):Q('{preds[k]}')")

            if inter_terms:
                formula += " + " + " + ".join(inter_terms)

        return formula

    # ---------- effect sizes ----------
    def calculate_effect_sizes(self, anova_df: pd.DataFrame) -> Dict[str, EffectSize]:
        effect_sizes: Dict[str, EffectSize] = {}
        if "sum_sq" not in anova_df.columns or "df" not in anova_df.columns:
            return effect_sizes

        ss_total = float(anova_df["sum_sq"].sum())
        ss_error = float(anova_df.loc["Residual", "sum_sq"]) if "Residual" in anova_df.index else 0.0
        df_error = float(anova_df.loc["Residual", "df"]) if "Residual" in anova_df.index else 0.0
        ms_error = (ss_error / df_error) if df_error > 0 else 0.0

        for eff in anova_df.index:
            eff_name = str(eff)
            if eff_name == "Residual":
                continue
            ss_eff = float(anova_df.loc[eff, "sum_sq"])
            df_eff = float(anova_df.loc[eff, "df"])
            ms_eff = (ss_eff / df_eff) if df_eff > 0 else 0.0

            eta_sq = (ss_eff / ss_total) if ss_total > 0 else 0.0
            if ss_total > 0 and ms_error > 0:
                omega_sq = (ss_eff - (df_eff * ms_error)) / (ss_total + ms_error)
            else:
                omega_sq = eta_sq
            omega_sq = max(0.0, float(omega_sq))

            if ms_error > 0:
                cohens_f = float(np.sqrt(ms_eff / ms_error)) if ms_error > 0 else 0.0
            else:
                cohens_f = float(np.sqrt(eta_sq / (1 - eta_sq))) if 0 <= eta_sq < 1 else float("inf")

            if eta_sq < self.config.effect_small:
                interp = "negligible"
            elif eta_sq < self.config.effect_medium:
                interp = "small"
            elif eta_sq < self.config.effect_large:
                interp = "medium"
            else:
                interp = "large"

            effect_sizes[eff_name] = EffectSize(
                eta_squared=float(eta_sq),
                omega_squared=float(omega_sq),
                cohens_f=float(cohens_f),
                interpretation=interp,
            )

        return effect_sizes

    # ---------- assumptions ----------
    def check_assumptions(self, model) -> Dict[str, AssumptionTest]:
        out: Dict[str, AssumptionTest] = {}
        resid = model.resid
        fitted = model.fittedvalues

        # Normality: Shapiro if <=5000 else KS
        try:
            if len(resid) <= 5000:
                st, p = stats.shapiro(resid)
                out["normality"] = AssumptionTest(
                    test_name="Shapiro-Wilk",
                    statistic=float(st),
                    p_value=float(p),
                    passed=bool(p > self.config.alpha),
                    message=("Residuals appear normal" if p > self.config.alpha else "Residuals deviate from normality"),
                )
            else:
                st, p = stats.kstest((resid - np.mean(resid)) / (np.std(resid) + 1e-12), "norm")
                out["normality"] = AssumptionTest(
                    test_name="Kolmogorov-Smirnov (std.)",
                    statistic=float(st),
                    p_value=float(p),
                    passed=bool(p > self.config.alpha),
                    message=("Residuals appear normal" if p > self.config.alpha else "Residuals deviate from normality"),
                )
        except Exception as e:
            out["normality"] = AssumptionTest("Normality", 0.0, None, False, f"Normality test failed: {e}")

        # Independence: Durbin–Watson
        try:
            dw = float(durbin_watson(resid))
            out["independence"] = AssumptionTest(
                test_name="Durbin-Watson",
                statistic=dw,
                p_value=None,
                passed=bool(1.5 < dw < 2.5),
                message=f"DW={dw:.3f} (≈2 suggests independence)",
            )
        except Exception as e:
            out["independence"] = AssumptionTest("Durbin-Watson", 0.0, None, False, f"DW failed: {e}")

        # Linearity (rainbow)
        try:
            st, p = linear_rainbow(model)
            out["linearity"] = AssumptionTest(
                test_name="Rainbow Test",
                statistic=float(st),
                p_value=float(p),
                passed=bool(p > self.config.alpha),
                message=("No strong non-linearity signal" if p > self.config.alpha else "Possible non-linearity"),
            )
        except Exception as e:
            out["linearity"] = AssumptionTest("Rainbow", 0.0, None, False, f"Rainbow failed: {e}")

        # Residual-vs-fitted pattern check (simple)
        try:
            corr = float(np.corrcoef(np.asarray(fitted), np.asarray(np.abs(resid)))[0, 1])
            out["heteroscedasticity_hint"] = AssumptionTest(
                test_name="Abs(resid) vs fitted corr",
                statistic=corr,
                p_value=None,
                passed=bool(abs(corr) < 0.4),
                message=("No strong funnel pattern" if abs(corr) < 0.4 else "Possible heteroscedasticity"),
            )
        except Exception as e:
            out["heteroscedasticity_hint"] = AssumptionTest("Heteroscedasticity hint", 0.0, None, False, f"Check failed: {e}")

        return out

    def levene_test(self, df: pd.DataFrame, response: str, group: str) -> Optional[AssumptionTest]:
        try:
            gvals: List[np.ndarray] = []
            for level in df[group].dropna().unique():
                vals = df.loc[df[group] == level, response].dropna().to_numpy()
                if len(vals) > 0:
                    gvals.append(vals)
            if len(gvals) < 2:
                return None
            st, p = stats.levene(*gvals)
            return AssumptionTest(
                test_name="Levene",
                statistic=float(st),
                p_value=float(p),
                passed=bool(p > self.config.alpha),
                message=("Variances appear equal" if p > self.config.alpha else "Variances may differ"),
            )
        except Exception:
            return None

    # ---------- tukey + letters ----------
    def tukey_letters(self, df: pd.DataFrame, response: str, group: str) -> Dict[str, str]:
        """
        Compute compact letter display (simple greedy algorithm) using Tukey decisions.
        Groups that are NOT significantly different can share a letter.
        """
        # Prepare data
        y = ensure_numeric_series(df[response])
        g = df[group].astype(str)

        # Drop missing
        mask = (~y.isna()) & (~g.isna())
        y = y[mask]
        g = g[mask]

        if g.nunique() < 2:
            return {}

        tk = pairwise_tukeyhsd(endog=y, groups=g, alpha=self.config.alpha)

        groups = list(map(str, tk.groupsunique))
        n = len(groups)

        # Build significance matrix: sig[i,j]=True if different
        sig = np.zeros((n, n), dtype=bool)
        # tk.summary() rows contain group1, group2, meandiff, p-adj, lower, upper, reject
        summ = tk.summary()
        # Data starts at row 1 (row 0 is header)
        for row in summ.data[1:]:
            g1, g2, _, p_adj, _, _, reject = row
            i = groups.index(str(g1))
            j = groups.index(str(g2))
            sig[i, j] = bool(reject)
            sig[j, i] = bool(reject)

        # Order by mean descending
        means = df.assign(_y=y, _g=g).groupby("_g")["_y"].mean().sort_values(ascending=False)
        ordered = list(means.index.astype(str))

        letters: Dict[str, str] = {gr: "" for gr in groups}
        current_letter_ord = ord("A")

        # Greedy: create letter groups
        for gr in ordered:
            if letters[gr]:
                continue
            letter = chr(current_letter_ord)
            current_letter_ord += 1
            letters[gr] = letter

            # Try to add other groups to same letter if they are NOT significantly different from all in this letter group
            letter_group = [gr]
            for other in ordered:
                if letters[other]:
                    continue
                ok = True
                for member in letter_group:
                    i = groups.index(str(member))
                    j = groups.index(str(other))
                    if sig[i, j]:  # significantly different -> cannot share
                        ok = False
                        break
                if ok:
                    letters[other] = letter
                    letter_group.append(other)

        return letters

    # ---------- descriptive stats ----------
    def descriptive(self, df: pd.DataFrame, response: str, predictors: List[str]) -> Dict[str, Any]:
        y = ensure_numeric_series(df[response])
        out: Dict[str, Any] = {
            "overall": {
                "n": int(y.notna().sum()),
                "mean": safe_float(y.mean()),
                "std": safe_float(y.std()),
                "sem": safe_float(stats.sem(y.dropna())) if y.notna().sum() > 1 else None,
                "cv": safe_float((y.std() / y.mean()) * 100) if safe_float(y.mean()) not in (None, 0.0) else None,
                "min": safe_float(y.min()),
                "max": safe_float(y.max()),
                "median": safe_float(y.median()),
                "q1": safe_float(y.quantile(0.25)),
                "q3": safe_float(y.quantile(0.75)),
            }
        }

        for p in predictors:
            grp = df.groupby(p)[response].agg(["count", "mean", "std", "min", "max"])
            grp["sem"] = df.groupby(p)[response].apply(lambda s: float(stats.sem(pd.to_numeric(s, errors="coerce").dropna())) if s.dropna().shape[0] > 1 else np.nan)
            grp["cv"] = (grp["std"] / grp["mean"] * 100.0)
            grp = grp.replace([np.inf, -np.inf], np.nan).round(6)
            out[p] = grp.fillna(np.nan).to_dict(orient="index")

        return out

    # ---------- plots ----------
    def _fig_to_b64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format=self.config.figure_format, dpi=self.config.figure_dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def mean_plot(self, df: pd.DataFrame, response: str, group: str) -> str:
        y = ensure_numeric_series(df[response])
        d = df.copy()
        d[response] = y
        d = d.dropna(subset=[response, group])

        stats_g = d.groupby(group)[response].agg(["mean", "count", "std"])
        stats_g["sem"] = stats_g["std"] / np.sqrt(stats_g["count"])
        stats_g = stats_g.sort_values("mean", ascending=False)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        x = np.arange(len(stats_g))
        ax.bar(x, stats_g["mean"].values, yerr=stats_g["sem"].values, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(stats_g.index.astype(str), rotation=45, ha="right")
        ax.set_xlabel(group)
        ax.set_ylabel(response)
        ax.set_title(f"Mean {response} by {group} (±SEM)")
        ax.grid(True, axis="y", alpha=0.25)

        return self._fig_to_b64(fig)

    def box_plot(self, df: pd.DataFrame, response: str, group: str) -> str:
        y = ensure_numeric_series(df[response])
        d = df.copy()
        d[response] = y
        d = d.dropna(subset=[response, group])

        order = d.groupby(group)[response].mean().sort_values(ascending=False).index.astype(str).tolist()

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        data = [d.loc[d[group].astype(str) == g, response].values for g in order]
        ax.boxplot(data, labels=order, vert=True, patch_artist=True)
        ax.set_xlabel(group)
        ax.set_ylabel(response)
        ax.set_title(f"{response} distribution by {group}")
        ax.grid(True, axis="y", alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        return self._fig_to_b64(fig)

    def residual_plots(self, model) -> Dict[str, str]:
        resid = model.resid
        fitted = model.fittedvalues

        # Residuals vs fitted
        fig1 = plt.figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        ax1.scatter(fitted, resid, alpha=0.7, edgecolors="black", linewidths=0.3)
        ax1.axhline(0, linestyle="--")
        ax1.set_xlabel("Fitted")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Fitted")
        ax1.grid(True, alpha=0.25)

        # QQ plot
        fig2 = plt.figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        sm.qqplot(resid, line="45", ax=ax2)
        ax2.set_title("Q-Q Plot (Residuals)")
        ax2.grid(True, alpha=0.25)

        return {"residuals_vs_fitted": self._fig_to_b64(fig1), "qq_plot": self._fig_to_b64(fig2)}

    def correlation_heatmap(self, df: pd.DataFrame, cols: Optional[List[str]] = None) -> Tuple[Dict[str, Any], str]:
        num = df.select_dtypes(include=[np.number]).copy()
        if cols:
            num = num[[c for c in cols if c in num.columns]].copy()
        corr = num.corr(numeric_only=True)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        im = ax.imshow(corr.values, aspect="auto")
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        ax.set_title("Correlation Heatmap")
        fig.colorbar(im, ax=ax, shrink=0.8)

        return corr.round(6).to_dict(), self._fig_to_b64(fig)

    def regression(self, df: pd.DataFrame, y_col: str, x_col: str) -> Dict[str, Any]:
        y = ensure_numeric_series(df[y_col])
        x = ensure_numeric_series(df[x_col])
        d = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(d) < 5:
            raise ValueError("Not enough data for regression (need at least 5 complete rows).")

        X = sm.add_constant(d["x"].values)
        model = sm.OLS(d["y"].values, X).fit()

        # plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.scatter(d["x"], d["y"], alpha=0.7, edgecolors="black", linewidths=0.3)
        xs = np.linspace(d["x"].min(), d["x"].max(), 100)
        ys = model.params[0] + model.params[1] * xs
        ax.plot(xs, ys)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Regression: {y_col} ~ {x_col}")
        ax.grid(True, alpha=0.25)

        return {
            "n": int(len(d)),
            "coef_intercept": safe_float(model.params[0]),
            "coef_slope": safe_float(model.params[1]),
            "r_squared": safe_float(model.rsquared),
            "p_value_slope": safe_float(model.pvalues[1]) if len(model.pvalues) > 1 else None,
            "stderr_slope": safe_float(model.bse[1]) if len(model.bse) > 1 else None,
            "plot_b64": self._fig_to_b64(fig),
        }

    # ---------- ANOVA run ----------
    def run_anova(
        self,
        df: pd.DataFrame,
        response: str,
        predictors: List[str],
        blocks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if response not in df.columns:
            raise ValueError(f"Response '{response}' not found.")

        # clean numeric response
        d = df.copy()
        d[response] = ensure_numeric_series(d[response])

        # drop missing in used columns
        used = [response] + predictors + (blocks or [])
        d = d.dropna(subset=list(dict.fromkeys(used)))

        if len(d) < 5:
            raise ValueError("Not enough complete rows after dropping missing values.")

        formula = self.build_formula(response, predictors, blocks)
        model = ols(formula, data=d).fit()

        # ANOVA
        typ = int(self.config.anova_type)
        anova_df = sm.stats.anova_lm(model, typ=typ)
        anova_rowwise = to_rowwise_anova(anova_df)

        # effect sizes
        es = self.calculate_effect_sizes(anova_df)
        es_dict = {k: asdict(v) for k, v in es.items()}

        # assumptions
        assumptions = self.check_assumptions(model)
        # add Levene on first predictor (best effort)
        if predictors:
            lev = self.levene_test(d, response, predictors[0])
            if lev:
                assumptions["homogeneity"] = lev

        assumptions_dict = {k: asdict(v) for k, v in assumptions.items()}

        # means + tukey letters per predictor (only for main effects present)
        means: Dict[str, Dict[str, float]] = {}
        letters: Dict[str, Dict[str, str]] = {}
        for p in predictors:
            grp_means = d.groupby(p)[response].mean().sort_values(ascending=False)
            means[p] = {str(k): float(v) for k, v in grp_means.to_dict().items()}

            # if main effect exists and is significant, do tukey letters
            # In statsmodels, effect name may be "Q('Factor')" when using Q(). We'll match loosely.
            # We'll check p-value by scanning anova index strings.
            pval = None
            for idx_name, row in anova_rowwise.items():
                if p in idx_name:
                    pval = row.get("PR(>F)")
                    break
            if pval is not None and float(pval) < self.config.alpha:
                try:
                    letters[p] = self.tukey_letters(d, response, p)
                except Exception as e:
                    logger.warning(f"Tukey letters failed for {p}: {e}")
                    letters[p] = {}
            else:
                letters[p] = {}

        desc = self.descriptive(d, response, predictors)

        return {
            "formula": formula,
            "n_used": int(len(d)),
            "r_squared": safe_float(model.rsquared),
            "adj_r_squared": safe_float(model.rsquared_adj),
            "f_value": safe_float(getattr(model, "fvalue", None)),
            "f_pvalue": safe_float(getattr(model, "f_pvalue", None)),
            "anova": anova_rowwise,
            "means": means,
            "letters": letters,
            "effect_sizes": es_dict,
            "assumptions": assumptions_dict,
            "descriptive_stats": desc,
        }


# =========================
# BACKEND ORCHESTRATOR
# =========================
class VivaSenseBackend:
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.analyzer = StatisticalAnalyzer(self.config)

    def cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        f = CACHE_DIR / f"{key}.json"
        if f.exists():
            try:
                return json.loads(f.read_text())
            except Exception:
                return None
        return None

    def cache_set(self, key: str, data: Dict[str, Any]) -> None:
        f = CACHE_DIR / f"{key}.json"
        try:
            f.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    def process_dataframe(self, df: pd.DataFrame, filename: str, config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # optional per-request config override
        if config_override:
            cfg = AnalysisConfig(**{**self.config.to_dict(), **config_override})
            analyzer = StatisticalAnalyzer(cfg)
        else:
            cfg = self.config
            analyzer = self.analyzer

        validation = analyzer.validate_data(df)
        if validation["errors"]:
            return {"status": "error", "errors": validation["errors"], "warnings": validation["warnings"]}

        # variable detection
        categorical, continuous = analyzer.detect_variable_types(df)
        blocks = [c for c in categorical if any(k in c.lower() for k in ["block", "rep", "replicate", "batch"])]

        if not categorical:
            return {"status": "error", "errors": ["No categorical predictors found."], "warnings": validation["warnings"]}

        if not continuous:
            return {"status": "error", "errors": ["No continuous response variables found."], "warnings": validation["warnings"]}

        # cache
        h = df_hash(df)
        cache_key = hashlib.md5(json.dumps({"h": h, "cfg": cfg.to_dict()}, sort_keys=True).encode()).hexdigest()
        cached = self.cache_get(cache_key)
        if cached:
            cached["metadata"]["cache_hit"] = True
            return cached

        analysis_id = str(uuid.uuid4())
        result: Dict[str, Any] = {
            "status": "success",
            "metadata": {
                "analysis_id": analysis_id,
                "filename": filename,
                "timestamp": now_iso(),
                "n_rows": int(len(df)),
                "n_cols": int(len(df.columns)),
                "categorical_vars": categorical,
                "continuous_vars": continuous,
                "blocks": blocks,
                "config": cfg.to_dict(),
                "cache_hit": False,
            },
            "warnings": validation["warnings"],
            "traits": {},
            "extras": {},
        }

        # per-trait ANOVA + plots
        for trait in continuous:
            try:
                anova_res = analyzer.run_anova(df, trait, categorical, blocks)

                # plots based on first categorical
                group0 = categorical[0]
                plots = {
                    "mean_plot": analyzer.mean_plot(df, trait, group0),
                    "box_plot": analyzer.box_plot(df, trait, group0),
                }
                # diagnostics plots from fitted model: we need to refit quickly
                # (reusing formula built in run_anova would require returning model; keep V1 simple)
                # We'll re-fit a minimal model trait ~ group0
                try:
                    dd = df.copy()
                    dd[trait] = ensure_numeric_series(dd[trait])
                    dd = dd.dropna(subset=[trait, group0])
                    m = ols(f"Q('{trait}') ~ Q('{group0}')", data=dd).fit()
                    plots.update(analyzer.residual_plots(m))
                except Exception:
                    pass

                result["traits"][trait] = {
                    "status": "success",
                    "statistical_results": anova_res,
                    "plots": plots,
                }
            except Exception as e:
                logger.error(f"Trait analysis failed for {trait}: {e}")
                result["traits"][trait] = {"status": "failed", "error": str(e)}

        # correlation/heatmap on numeric
        try:
            corr_tbl, heat_b64 = analyzer.correlation_heatmap(df)
            result["extras"]["correlation"] = corr_tbl
            result["extras"]["correlation_heatmap_b64"] = heat_b64
        except Exception as e:
            result["extras"]["correlation_error"] = str(e)

        self.cache_set(cache_key, result)
        return result

    def export_zip(self, analysis: Dict[str, Any]) -> Path:
        """
        Creates a ZIP containing:
        - anova tables (json + csv per trait)
        - descriptive stats (json)
        - correlation table (csv)
        - plots (png)
        """
        analysis_id = analysis.get("metadata", {}).get("analysis_id", str(uuid.uuid4()))
        out_zip = RESULTS_DIR / f"vivasense_{analysis_id}.zip"

        def b64_to_bytes(b64: str) -> bytes:
            return base64.b64decode(b64.encode("utf-8"))

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("metadata.json", json.dumps(analysis.get("metadata", {}), indent=2))
            z.writestr("warnings.json", json.dumps(analysis.get("warnings", []), indent=2))

            # extras
            if "correlation" in analysis.get("extras", {}):
                corr = pd.DataFrame(analysis["extras"]["correlation"])
                with io.StringIO() as s:
                    corr.to_csv(s)
                    z.writestr("extras/correlation.csv", s.getvalue())
                if analysis["extras"].get("correlation_heatmap_b64"):
                    z.writestr("extras/correlation_heatmap.png", b64_to_bytes(analysis["extras"]["correlation_heatmap_b64"]))

            # traits
            for trait, tdata in analysis.get("traits", {}).items():
                if tdata.get("status") != "success":
                    z.writestr(f"traits/{trait}/error.txt", str(tdata.get("error", "failed")))
                    continue

                stats_res = tdata.get("statistical_results", {})
                z.writestr(f"traits/{trait}/anova.json", json.dumps(stats_res.get("anova", {}), indent=2))
                z.writestr(f"traits/{trait}/descriptive_stats.json", json.dumps(stats_res.get("descriptive_stats", {}), indent=2))
                z.writestr(f"traits/{trait}/means.json", json.dumps(stats_res.get("means", {}), indent=2))
                z.writestr(f"traits/{trait}/letters.json", json.dumps(stats_res.get("letters", {}), indent=2))
                z.writestr(f"traits/{trait}/effect_sizes.json", json.dumps(stats_res.get("effect_sizes", {}), indent=2))
                z.writestr(f"traits/{trait}/assumptions.json", json.dumps(stats_res.get("assumptions", {}), indent=2))

                # anova csv
                anova_tbl = pd.DataFrame(stats_res.get("anova", {})).T
                with io.StringIO() as s:
                    anova_tbl.to_csv(s)
                    z.writestr(f"traits/{trait}/anova.csv", s.getvalue())

                # plots
                for pname, pb64 in (tdata.get("plots") or {}).items():
                    try:
                        z.writestr(f"traits/{trait}/plots/{pname}.png", b64_to_bytes(pb64))
                    except Exception:
                        pass

        return out_zip


backend = VivaSenseBackend()


# =========================
# PAYSTACK (V1 plumbing)
# =========================
PAYSTACK_SECRET_KEY = os.environ.get("PAYSTACK_SECRET_KEY", "").strip()
PAYSTACK_PUBLIC_KEY = os.environ.get("PAYSTACK_PUBLIC_KEY", "").strip()
PAYSTACK_BASE_URL = "https://api.paystack.co"


async def paystack_initialize(email: str, amount_kobo: int, reference: str, callback_url: Optional[str] = None) -> Dict[str, Any]:
    if httpx is None:
        raise RuntimeError("httpx not installed. Add 'httpx' to requirements.txt.")
    if not PAYSTACK_SECRET_KEY:
        raise RuntimeError("PAYSTACK_SECRET_KEY not set in environment.")

    payload = {"email": email, "amount": int(amount_kobo), "reference": reference}
    if callback_url:
        payload["callback_url"] = callback_url

    headers = {"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(f"{PAYSTACK_BASE_URL}/transaction/initialize", json=payload, headers=headers)
        data = r.json()
        if r.status_code >= 400 or not data.get("status"):
            raise RuntimeError(f"Paystack initialize failed: {data}")
        return data


async def paystack_verify(reference: str) -> Dict[str, Any]:
    if httpx is None:
        raise RuntimeError("httpx not installed. Add 'httpx' to requirements.txt.")
    if not PAYSTACK_SECRET_KEY:
        raise RuntimeError("PAYSTACK_SECRET_KEY not set in environment.")

    headers = {"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(f"{PAYSTACK_BASE_URL}/transaction/verify/{reference}", headers=headers)
        data = r.json()
        if r.status_code >= 400 or not data.get("status"):
            raise RuntimeError(f"Paystack verify failed: {data}")
        return data


# =========================
# FASTAPI
# =========================
app = FastAPI(
    title="VivaSense Statistical Engine",
    version="1.0.0",
    description="V1 statistical analysis backend (ANOVA, Tukey, plots, diagnostics, downloads, Paystack).",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# API MODELS (lightweight dict validation)
# =========================
def require_keys(obj: Dict[str, Any], keys: List[str]) -> None:
    for k in keys:
        if k not in obj:
            raise HTTPException(status_code=400, detail=f"Missing required field: '{k}'")


# =========================
# ROUTES
# =========================
@app.get("/")
async def root():
    return {"service": "VivaSense", "version": "1.0.0", "status": "ok", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": now_iso()}


# ---------- analysis (free/direct) ----------
@app.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...), config: Optional[str] = None):
    """
    Direct analysis endpoint (no paywall).
    Upload CSV/XLSX and get results JSON.
    Optional: config JSON string (override AnalysisConfig fields).
    """
    if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only .csv, .xlsx, .xls supported.")

    try:
        content = await file.read()
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))

        cfg_override = json.loads(config) if config else None
        res = backend.process_dataframe(df, filename=file.filename, config_override=cfg_override)

        # store in memory
        if res.get("status") == "success":
            ANALYSIS_STORE[res["metadata"]["analysis_id"]] = res

        return res

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analyze failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/json")
async def analyze_json(payload: Dict[str, Any]):
    """
    { "data": [ {col: val, ...}, ... ], "filename": "optional", "config": {...optional...} }
    """
    require_keys(payload, ["data"])
    df = pd.DataFrame(payload["data"])
    if df.empty:
        raise HTTPException(status_code=400, detail="No rows in data.")

    filename = payload.get("filename", "json_upload")
    cfg_override = payload.get("config")
    try:
        res = backend.process_dataframe(df, filename=filename, config_override=cfg_override)
        if res.get("status") == "success":
            ANALYSIS_STORE[res["metadata"]["analysis_id"]] = res
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    if analysis_id in ANALYSIS_STORE:
        return ANALYSIS_STORE[analysis_id]
    raise HTTPException(status_code=404, detail="analysis_id not found")


@app.get("/analysis/{analysis_id}/download.zip")
async def download_zip(analysis_id: str):
    if analysis_id not in ANALYSIS_STORE:
        raise HTTPException(status_code=404, detail="analysis_id not found")
    z = backend.export_zip(ANALYSIS_STORE[analysis_id])
    return FileResponse(path=str(z), filename=z.name, media_type="application/zip")


# ---------- correlation & regression (explicit endpoints) ----------
@app.post("/stats/correlation")
async def correlation(payload: Dict[str, Any]):
    """
    { "data": [...], "columns": ["optional", ...] }
    Returns correlation table + heatmap b64.
    """
    require_keys(payload, ["data"])
    df = pd.DataFrame(payload["data"])
    cols = payload.get("columns")
    try:
        tbl, heat = backend.analyzer.correlation_heatmap(df, cols=cols)
        return {"status": "success", "correlation": tbl, "heatmap_b64": heat}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stats/regression")
async def regression(payload: Dict[str, Any]):
    """
    { "data": [...], "y": "col", "x": "col" }
    """
    require_keys(payload, ["data", "y", "x"])
    df = pd.DataFrame(payload["data"])
    try:
        res = backend.analyzer.regression(df, y_col=payload["y"], x_col=payload["x"])
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Paystack pay-per-analysis flow (V1) ----------
@app.post("/pay/initialize")
async def pay_initialize(payload: Dict[str, Any]):
    """
    Initialize Paystack payment for an analysis request.
    Expected:
      {
        "email": "...",
        "amount_kobo": 50000,
        "analysis_id": "...",      # must already exist in ANALYSIS_STORE OR you can create a "pending" analysis id
        "callback_url": "optional"
      }
    """
    require_keys(payload, ["email", "amount_kobo", "analysis_id"])
    analysis_id = payload["analysis_id"]
    if analysis_id not in ANALYSIS_STORE:
        raise HTTPException(status_code=404, detail="analysis_id not found (create analysis first or use paywalled request flow).")

    reference = f"VS-{uuid.uuid4().hex[:16]}"
    PAYMENT_STORE[reference] = {
        "status": "pending",
        "analysis_id": analysis_id,
        "email": payload["email"],
        "amount_kobo": int(payload["amount_kobo"]),
        "created_at": now_iso(),
    }

    try:
        data = await paystack_initialize(
            email=payload["email"],
            amount_kobo=int(payload["amount_kobo"]),
            reference=reference,
            callback_url=payload.get("callback_url"),
        )
        return {"status": "success", "reference": reference, "paystack": data.get("data", {})}
    except Exception as e:
        PAYMENT_STORE[reference]["status"] = "failed"
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pay/verify/{reference}")
async def pay_verify(reference: str):
    """
    Verify payment status via Paystack.
    """
    if reference not in PAYMENT_STORE:
        raise HTTPException(status_code=404, detail="reference not found")
    try:
        data = await paystack_verify(reference)
        pay_status = data.get("data", {}).get("status")  # typically "success" when paid
        if pay_status == "success":
            PAYMENT_STORE[reference]["status"] = "paid"
            PAYMENT_STORE[reference]["verified_at"] = now_iso()
        return {"status": "success", "reference": reference, "payment": PAYMENT_STORE[reference], "paystack": data.get("data", {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pay/webhook")
async def pay_webhook(request: Request):
    """
    Paystack webhook receiver (V1).
    NOTE: In production you MUST verify signature using 'x-paystack-signature' HMAC SHA512 of raw body.
    Here we do "verify(reference)" as a safer fallback.
    """
    body = await request.body()
    try:
        event = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    ref = (event.get("data") or {}).get("reference")
    if not ref:
        return {"status": "ignored", "reason": "no reference"}

    if ref not in PAYMENT_STORE:
        # still verify to be sure
        PAYMENT_STORE[ref] = {"status": "unknown", "created_at": now_iso()}

    # verify with Paystack
    try:
        data = await paystack_verify(ref)
        pay_status = data.get("data", {}).get("status")
        if pay_status == "success":
            PAYMENT_STORE[ref]["status"] = "paid"
            PAYMENT_STORE[ref]["verified_at"] = now_iso()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Webhook verify failed: {e}")
        return {"status": "error", "detail": str(e)}


@app.get("/config")
async def get_config():
    return backend.config.to_dict()


@app.post("/config")
async def set_config(payload: Dict[str, Any]):
    """
    Update global config (server-wide). Prefer per-request override for safety.
    """
    try:
        backend.config = AnalysisConfig(**{**backend.config.to_dict(), **payload})
        backend.analyzer = StatisticalAnalyzer(backend.config)
        return {"status": "success", "config": backend.config.to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
