"""
genetics/compat.py
==================
Stand-alone clones of VivaSense V2.2 helper functions, scoped for the
genetics module.  Import from here rather than reaching into main.py so
that the genetics package remains a self-contained sub-application.

All implementations mirror V2.2 behaviour exactly.
"""
from __future__ import annotations

import base64
import io
import math
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import HTTPException
from scipy import stats
from scipy.stats import levene as scipy_levene, shapiro as scipy_shapiro


# ── Numeric helpers ───────────────────────────────────────────────────────────

def round_val(x: Any, digits: int = 4) -> Any:
    """Round a numeric value; pass non-numeric through unchanged."""
    try:
        return round(float(x), digits)
    except (TypeError, ValueError):
        return x


def fmt_p(p: Any) -> float:
    """Floor p-values at 0.0001 (matches V2.2 df_to_records convention)."""
    try:
        v = float(p)
        return max(v, 0.0001)
    except (TypeError, ValueError):
        return p


def fmt_p_display(p: Any) -> str:
    """Return a publication-style p-value string (e.g. '< 0.001', '0.0432')."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return str(p)
    if v < 0.001:
        return "< 0.001"
    if v < 0.05:
        return f"{v:.4f}"
    return f"{v:.4f}"


def sig_stars(p: Any) -> str:
    """Return significance stars: *** <0.001, ** <0.01, * <0.05, ns otherwise."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return ""
    if v < 0.001:
        return "***"
    if v < 0.01:
        return "**"
    if v < 0.05:
        return "*"
    return "ns"


def cv_percent(series: pd.Series) -> float:
    """Coefficient of variation as a percentage (σ/μ × 100)."""
    m = float(series.mean())
    if m == 0 or math.isnan(m):
        return float("nan")
    return float(series.std() / m * 100)


def coerce_numeric(series: pd.Series) -> pd.Series:
    """pd.to_numeric wrapper with coerce — V2.2 convention."""
    return pd.to_numeric(series, errors="coerce")


# ── Data validation helpers ───────────────────────────────────────────────────

def require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """Raise HTTP 400 if any column in *cols* is missing from *df*."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Missing required column(s): {missing}. "
                f"Available columns: {list(df.columns)}"
            ),
        )


def clean_for_genetics(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Drop NaN rows in *group_col* or *value_col*, coerce *value_col* to float.
    Raises HTTP 400 if fewer than 4 rows remain.
    """
    df = df.copy()
    df[value_col] = coerce_numeric(df[value_col])
    df = df.dropna(subset=[group_col, value_col])
    if len(df) < 4:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Fewer than 4 complete observations for '{value_col}'. "
                "Cannot fit model."
            ),
        )
    return df


# ── DataFrame serialisation ───────────────────────────────────────────────────

_P_COLS: set = {"PR(>F)", "p-value", "pvalue", "p_value", "Pr(>F)", "p"}


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame to a list of dicts.
    P-value columns are floored at 0.0001 (V2.2 convention).
    NumPy scalars are cast to Python natives.
    """
    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean: Dict[str, Any] = {}
        for k, v in row.items():
            if k in _P_COLS:
                try:
                    clean[k] = fmt_p(float(v))
                except (TypeError, ValueError):
                    clean[k] = v
            elif isinstance(v, np.integer):
                clean[k] = int(v)
            elif isinstance(v, np.floating):
                clean[k] = None if (math.isnan(v) or math.isinf(v)) else float(v)
            elif isinstance(v, np.bool_):
                clean[k] = bool(v)
            elif isinstance(v, float):
                clean[k] = None if (math.isnan(v) or math.isinf(v)) else v
            else:
                clean[k] = v
        records.append(clean)
    return records


def add_p_display_to_anova(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Post-process ANOVA records to add a *PR(>F)_display* field
    (e.g. '< 0.001') for every row that carries a PR(>F) value.
    """
    for rec in records:
        p_raw = rec.get("PR(>F)")
        if p_raw is not None:
            try:
                rec["PR(>F)_display"] = fmt_p_display(float(p_raw))
            except (TypeError, ValueError):
                rec["PR(>F)_display"] = str(p_raw)
    return records


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _b64_png(fig: plt.Figure) -> str:
    """Serialise a matplotlib Figure to a base64 PNG at 170 DPI (V2.2 standard)."""
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Statistical assumption tests ──────────────────────────────────────────────

def shapiro_test(residuals: np.ndarray) -> Dict[str, Any]:
    """
    Shapiro-Wilk (or Kolmogorov-Smirnov for n>5000) normality test on
    model residuals.  Returns a dict matching the V2.2 assumption-record shape.
    """
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    n = len(residuals)

    if n < 3:
        return {
            "test": "Shapiro-Wilk",
            "statistic": None,
            "p_value": None,
            "p_display": "N/A",
            "passed": None,
            "interpretation": "Insufficient data for normality test.",
        }

    if n > 5000:
        stat, p = stats.kstest(
            residuals, "norm",
            args=(float(residuals.mean()), float(residuals.std(ddof=1))),
        )
        test_name = "Kolmogorov-Smirnov"
    else:
        stat, p = scipy_shapiro(residuals)
        test_name = "Shapiro-Wilk"

    passed = bool(p > 0.05)
    return {
        "test": test_name,
        "statistic": round_val(float(stat)),
        "p_value": fmt_p(float(p)),
        "p_display": fmt_p_display(float(p)),
        "passed": passed,
        "interpretation": (
            "Residuals appear normally distributed (p > 0.05)."
            if passed
            else (
                "Residuals deviate from normality (p ≤ 0.05). "
                "Consider data transformation."
            )
        ),
    }


def levene_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> Dict[str, Any]:
    """
    Levene's test for homogeneity of variance across groups.
    Returns a dict matching the V2.2 assumption-record shape.
    """
    groups = [
        grp[value_col].dropna().values
        for _, grp in df.groupby(group_col)
        if len(grp[value_col].dropna()) >= 2
    ]
    if len(groups) < 2:
        return {
            "test": "Levene",
            "statistic": None,
            "p_value": None,
            "p_display": "N/A",
            "passed": None,
            "interpretation": "Insufficient groups for Levene test.",
        }

    stat, p = scipy_levene(*groups)
    passed = bool(p > 0.05)
    return {
        "test": "Levene",
        "statistic": round_val(float(stat)),
        "p_value": fmt_p(float(p)),
        "p_display": fmt_p_display(float(p)),
        "passed": passed,
        "interpretation": (
            "Variances are homogeneous across groups (p > 0.05)."
            if passed
            else (
                "Heterogeneous variances detected (p ≤ 0.05). "
                "Results may be affected."
            )
        ),
    }


def assumption_guidance(
    shapiro_rec: Dict[str, Any],
    levene_rec: Dict[str, Any],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Locked verdict aggregating both assumption tests (V2.2 pattern).
    Produces verdict_code for downstream filtering.
    """
    norm_ok = shapiro_rec.get("passed")
    hom_ok  = levene_rec.get("passed")

    if norm_ok is True and hom_ok is True:
        verdict_code = "ALL_PASS"
        overall = "All parametric assumptions met. ANOVA results are reliable."
        robustness = "high"
    elif norm_ok is False and hom_ok is False:
        verdict_code = "BOTH_FAIL"
        overall = (
            "Both normality and homogeneity assumptions violated. "
            "Consider non-parametric alternatives (Kruskal-Wallis)."
        )
        robustness = "low"
    elif norm_ok is False:
        verdict_code = "NORM_FAIL"
        overall = (
            "Normality assumption violated; homogeneity met. "
            "ANOVA is moderately robust to non-normality with balanced designs."
        )
        robustness = "moderate"
    elif hom_ok is False:
        verdict_code = "HOM_FAIL"
        overall = (
            "Homogeneity of variance violated; normality met. "
            "Use Welch-corrected F or apply variance-stabilising transformation."
        )
        robustness = "moderate"
    else:
        verdict_code = "UNKNOWN"
        overall = "One or more assumption tests could not be performed."
        robustness = "unknown"

    return {
        "verdict_code": verdict_code,
        "overall": overall,
        "robustness": robustness,
        "normality": {
            "status": "pass" if norm_ok else ("fail" if norm_ok is False else "na"),
            "detail": shapiro_rec.get("interpretation", ""),
        },
        "homogeneity": {
            "status": "pass" if hom_ok else ("fail" if hom_ok is False else "na"),
            "detail": levene_rec.get("interpretation", ""),
        },
    }


# ── Genetics interpretation ───────────────────────────────────────────────────

def interpret_genetics(
    analysis_type: str,
    p_map: Dict[str, Optional[float]],
    trait: str,
    alpha: float = 0.05,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Plain-English interpretation of genetics results.
    Mirrors V2.2 interpret_anova() with genetics-specific language.
    """
    lines: List[str] = [f"**{analysis_type} — {trait}**"]

    sig = {src: p for src, p in p_map.items() if p is not None and p <= alpha}
    ns  = {src: p for src, p in p_map.items() if p is not None and p >  alpha}

    if sig:
        sig_str = ", ".join(
            f"{src} (p = {fmt_p_display(p)})" for src, p in sig.items()
        )
        lines.append(f"Significant effects detected for: {sig_str}.")
    else:
        lines.append("No statistically significant effects were detected.")

    if ns:
        ns_str = ", ".join(ns.keys())
        lines.append(f"Non-significant source(s): {ns_str}.")

    if extra:
        h2 = extra.get("heritability")
        ga_pct = extra.get("ga_percent")
        if h2 is not None:
            cat = "high" if h2 >= 0.6 else ("moderate" if h2 >= 0.3 else "low")
            lines.append(
                f"Broad-sense heritability H\u00b2 = {h2:.2%} ({cat}), "
                "indicating the proportion of phenotypic variance attributable to genotype."
            )
        if ga_pct is not None:
            lines.append(
                f"Expected genetic advance under 5\u202f% selection intensity: "
                f"{ga_pct:.1f}\u202f% of the mean."
            )

    return "  ".join(lines)
