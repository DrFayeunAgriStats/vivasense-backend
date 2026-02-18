"""
VivaSense V1 — app/main.py (single-file FastAPI backend)

Features (V1 plan)
- Descriptive statistics (optionally grouped)
- One-way ANOVA (CRD) + Tukey HSD + compact letter display
- Two-way ANOVA (CRD factorial) + Tukey on A:B combinations
- Factorial in RCBD (block + A*B) + Tukey on A:B combinations
- Split-plot ANOVA (correct main-plot error term using Block:A) + Tukey on A:B combinations
- Multi-trait runner for any of the above designs

All endpoints accept multipart/form-data (Form + File), NOT JSON body models.
Returns JSON with:
  meta, tables, plots (base64 PNG), interpretation

Dependencies:
  fastapi, uvicorn, pandas, numpy, scipy, statsmodels, matplotlib
"""

from __future__ import annotations

import io
import base64
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="VivaSense V1", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    # If GET is defined, FastAPI will also respond to HEAD properly (no 405 surprises).
    return {"name": "VivaSense V1", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ----------------------------
# Helpers
# ----------------------------
def _b64_png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


async def load_csv(upload: UploadFile) -> pd.DataFrame:
    if upload is None:
        raise HTTPException(status_code=400, detail="Missing file.")
    if not (upload.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    content = await upload.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required column(s): {missing}. Available: {list(df.columns)}",
        )


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_for_model(df: pd.DataFrame, y: str, x_cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    require_cols(d, [y] + x_cols)

    d[y] = coerce_numeric(d[y])
    for c in x_cols:
        d[c] = d[c].astype(str)

    d = d.dropna(subset=[y] + x_cols)

    if d.shape[0] < 3:
        raise HTTPException(status_code=400, detail=f"Not enough valid rows for analysis after cleaning ({y}).")

    # Ensure at least 2 groups where relevant (helps Tukey/Levene)
    if len(x_cols) == 1:
        if d[x_cols[0]].nunique(dropna=True) < 2:
            raise HTTPException(status_code=400, detail=f"Need at least 2 levels in '{x_cols[0]}' for ANOVA.")
    return d


def cv_percent(y: pd.Series) -> Optional[float]:
    y = pd.to_numeric(y, errors="coerce")
    m = float(np.nanmean(y))
    if np.isnan(m) or m == 0:
        return None
    s = float(np.nanstd(y, ddof=1)) if y.dropna().shape[0] > 1 else np.nan
    if np.isnan(s):
        return None
    return float((s / m) * 100.0)


def shapiro_test(resid: np.ndarray) -> Dict[str, Any]:
    resid = np.asarray(resid)
    if resid.size < 3:
        return {"test": "Shapiro-Wilk", "stat": None, "p_value": None, "note": "Not enough residuals."}
    stat, p = stats.shapiro(resid[:5000])
    return {"test": "Shapiro-Wilk", "stat": float(stat), "p_value": float(p)}


def levene_test(df: pd.DataFrame, y: str, group: str) -> Dict[str, Any]:
    groups = []
    for _, g in df.groupby(group):
        vals = g[y].dropna().values
        if len(vals) > 0:
            groups.append(vals)
    if len(groups) < 2:
        return {"test": "Levene", "stat": None, "p_value": None, "note": "Need at least 2 groups."}
    stat, p = stats.levene(*groups, center="median")
    return {"test": "Levene", "stat": float(stat), "p_value": float(p)}


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df.replace({np.nan: None}).to_dict(orient="records")


def mean_table(df: pd.DataFrame, y: str, group: str) -> pd.DataFrame:
    g = df.groupby(group)[y]
    out = pd.DataFrame({
        group: g.mean().index.astype(str),
        "n": g.count().values,
        "mean": g.mean().values,
        "sd": g.std(ddof=1).values,
        "se": (g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))).values,
    })
    # sort by mean descending for nicer reporting
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


def mean_plot(df: pd.DataFrame, y: str, group: str, title: str) -> str:
    mt = mean_table(df, y, group)
    fig = plt.figure(figsize=(7.8, 4.3))
    ax = fig.add_subplot(111)
    x = np.arange(len(mt))
    ax.errorbar(x, mt["mean"].values, yerr=mt["se"].values, fmt="o", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(mt[group].astype(str).tolist(), rotation=35, ha="right")
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    return _b64_png(fig)


def box_plot(df: pd.DataFrame, y: str, group: str, title: str) -> str:
    fig = plt.figure(figsize=(7.8, 4.3))
    ax = fig.add_subplot(111)

    levels = df[group].astype(str).unique().tolist()
    data = [df.loc[df[group].astype(str) == lvl, y].dropna().values for lvl in levels]

    ax.boxplot(data, labels=levels, showmeans=True)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    return _b64_png(fig)


def compact_letter_display(groups: List[str], means: Dict[str, float], sig: Dict[Tuple[str, str], bool]) -> Dict[str, str]:
    """
    Simple compact letter display:
    sig[(a,b)] == True means significantly different.
    """
    ordered = sorted(groups, key=lambda g: means.get(g, -np.inf), reverse=True)
    letters_for = {g: "" for g in ordered}
    letter_sets: List[List[str]] = []

    def conflicts(g: str, members: List[str]) -> bool:
        for m in members:
            if g == m:
                continue
            key = (g, m) if (g, m) in sig else (m, g)
            if sig.get(key, False):
                return True
        return False

    for g in ordered:
        placed = False
        for members in letter_sets:
            if not conflicts(g, members):
                members.append(g)
                placed = True
                break
        if not placed:
            letter_sets.append([g])

    for i, members in enumerate(letter_sets):
        letter = chr(ord("a") + i)
        for g in members:
            letters_for[g] += letter

    return letters_for


def tukey_letters(df: pd.DataFrame, y: str, group: str, alpha: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - Tukey table
      - Means table with CLD letters
    """
    mt = mean_table(df, y, group)

    # Safety: Tukey requires >=2 groups and >=2 observations total
    if df[group].nunique() < 2:
        raise HTTPException(status_code=400, detail=f"Need at least 2 levels in '{group}' for Tukey.")
    if df[y].dropna().shape[0] < 3:
        raise HTTPException(status_code=400, detail=f"Not enough numeric observations in '{y}' for Tukey.")

    res = pairwise_tukeyhsd(endog=df[y].values, groups=df[group].astype(str).values, alpha=alpha)
    tukey_df = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])

    uniq = mt[group].astype(str).tolist()
    means = {row[group]: float(row["mean"]) for _, row in mt.iterrows()}
    sig: Dict[Tuple[str, str], bool] = {}

    for _, r in tukey_df.iterrows():
        a = str(r["group1"])
        b = str(r["group2"])
        sig[(a, b)] = bool(r["reject"])

    letters = compact_letter_display(uniq, means, sig)
    mt["letters"] = mt[group].astype(str).map(letters)

    return tukey_df, mt


def interpret_anova(title: str, p_map: Dict[str, Optional[float]], cv: Optional[float], alpha: float) -> str:
    lines = [f"{title} interpretation:"]
    if cv is not None:
        lines.append(f"- The coefficient of variation (CV) is {cv:.2f}%. Lower CV generally indicates more precise measurements.")
    for term, p in p_map.items():
        if p is None:
            continue
        if p < alpha:
            lines.append(f"- **{term}** is significant (p = {p:.4f}), meaning it has a real effect on the trait.")
        else:
            lines.append(f"- **{term}** is not significant (p = {p:.4f}); differences may be due to random variation.")
    lines.append("- Assumptions: check residual normality (Shapiro) and equal variances (Levene).")
    return "\n".join(lines)


# ----------------------------
# Analysis engines
# ----------------------------
def oneway_engine(df: pd.DataFrame, factor: str, trait: str, alpha: float) -> Dict[str, Any]:
    d = clean_for_model(df, trait, [factor])
    d[factor] = d[factor].astype(str)

    model = ols(f"{trait} ~ C({factor})", data=d).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})
    anova["ms"] = anova["sum_sq"] / anova["df"]

    sh = shapiro_test(model.resid)
    lv = levene_test(d, trait, factor)

    tukey_df, means_letters = tukey_letters(d, trait, factor, alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, factor, f"Means ± SE by {factor}"),
        "box_plot": box_plot(d, trait, factor, f"Boxplot of {trait} by {factor}"),
    }

    # p-value for factor term
    p_factor = None
    m = anova["source"].astype(str).str.contains(f"C({factor})")
    if m.any():
        p_factor = float(anova.loc[m, "PR(>F)"].iloc[0])

    return {
        "meta": {
            "design": "CRD one-way",
            "factor": factor,
            "trait": trait,
            "alpha": alpha,
            "n_rows_used": int(d.shape[0]),
            "levels": sorted(d[factor].unique().tolist()),
            "cv_percent": cv_percent(d[trait]),
        },
        "tables": {
            "anova": df_to_records(anova[["source", "df", "sum_sq", "ms", "F", "PR(>F)"]]),
            "means": df_to_records(means_letters),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
        },
        "plots": plots,
        "interpretation": interpret_anova("One-way ANOVA", {"Treatment": p_factor}, cv_percent(d[trait]), alpha),
    }


def twoway_engine(df: pd.DataFrame, a: str, b: str, trait: str, alpha: float) -> Dict[str, Any]:
    d = clean_for_model(df, trait, [a, b])
    d[a] = d[a].astype(str)
    d[b] = d[b].astype(str)
    d["_AB_"] = d[a].astype(str) + ":" + d[b].astype(str)

    model = ols(f"{trait} ~ C({a}) * C({b})", data=d).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})
    anova["ms"] = anova["sum_sq"] / anova["df"]

    sh = shapiro_test(model.resid)
    lv = levene_test(d, trait, "_AB_")

    tukey_df, means_letters = tukey_letters(d, trait, "_AB_", alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, "_AB_", f"Means ± SE by {a}:{b}"),
        "box_plot": box_plot(d, trait, "_AB_", f"Boxplot of {trait} by {a}:{b}"),
    }

    def p_of(term: str) -> Optional[float]:
        m = anova["source"].astype(str).str.contains(term)
        if m.any():
            return float(anova.loc[m, "PR(>F)"].iloc[0])
        return None

    p_map = {
        a: p_of(f"C({a})"),
        b: p_of(f"C({b})"),
        f"{a}×{b}": p_of(":"),
    }

    return {
        "meta": {
            "design": "CRD two-way (factorial)",
            "factor_a": a,
            "factor_b": b,
            "trait": trait,
            "alpha": alpha,
            "n_rows_used": int(d.shape[0]),
            "cv_percent": cv_percent(d[trait]),
        },
        "tables": {
            "anova": df_to_records(anova[["source", "df", "sum_sq", "ms", "F", "PR(>F)"]]),
            "means": df_to_records(means_letters.rename(columns={"_AB_": "A:B"})),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
        },
        "plots": plots,
        "interpretation": interpret_anova("Two-way ANOVA", p_map, cv_percent(d[trait]), alpha),
    }


def rcbd_factorial_engine(df: pd.DataFrame, block: str, a: str, b: str, trait: str, alpha: float) -> Dict[str, Any]:
    d = clean_for_model(df, trait, [block, a, b])
    for c in [block, a, b]:
        d[c] = d[c].astype(str)
    d["_AB_"] = d[a].astype(str) + ":" + d[b].astype(str)

    model = ols(f"{trait} ~ C({block}) + C({a}) * C({b})", data=d).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})
    anova["ms"] = anova["sum_sq"] / anova["df"]

    sh = shapiro_test(model.resid)
    lv = levene_test(d, trait, "_AB_")

    tukey_df, means_letters = tukey_letters(d, trait, "_AB_", alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, "_AB_", f"Means ± SE by {a}:{b} (RCBD)"),
        "box_plot": box_plot(d, trait, "_AB_", f"Boxplot of {trait} by {a}:{b} (RCBD)"),
    }

    def p_of(term: str) -> Optional[float]:
        m = anova["source"].astype(str).str.contains(term)
        if m.any():
            return float(anova.loc[m, "PR(>F)"].iloc[0])
        return None

    p_map = {
        "Block": p_of(f"C({block})"),
        a: p_of(f"C({a})"),
        b: p_of(f"C({b})"),
        f"{a}×{b}": p_of(":"),
    }

    return {
        "meta": {
            "design": "Factorial in RCBD",
            "block": block,
            "factor_a": a,
            "factor_b": b,
            "trait": trait,
            "alpha": alpha,
            "n_rows_used": int(d.shape[0]),
            "cv_percent": cv_percent(d[trait]),
        },
        "tables": {
            "anova": df_to_records(anova[["source", "df", "sum_sq", "ms", "F", "PR(>F)"]]),
            "means": df_to_records(means_letters.rename(columns={"_AB_": "A:B"})),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
        },
        "plots": plots,
        "interpretation": interpret_anova("Factorial RCBD ANOVA", p_map, cv_percent(d[trait]), alpha),
    }


def splitplot_engine(df: pd.DataFrame, block: str, main: str, sub: str, trait: str, alpha: float) -> Dict[str, Any]:
    """
    Split-plot correct testing:
      Fit: y ~ block + main*sub + block:main
      Test main using MS(block:main) as denominator
      Test sub and main:sub using residual MS
    """
    d = clean_for_model(df, trait, [block, main, sub])
    for c in [block, main, sub]:
        d[c] = d[c].astype(str)
    d["_AB_"] = d[main].astype(str) + ":" + d[sub].astype(str)

    formula = f"{trait} ~ C({block}) + C({main}) * C({sub}) + C({block}):C({main})"
    model = ols(formula, data=d).fit()
    an0 = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})
    an0["ms"] = an0["sum_sq"] / an0["df"]

    src = an0["source"].astype(str)

    term_A = f"C({main})"
    term_B = f"C({sub})"
    term_AB = f"C({main}):C({sub})"
    term_blockA = f"C({block}):C({main})"
    term_resid = "Residual"

    def get_row(term: str) -> Optional[pd.Series]:
        m = src == term
        if m.any():
            return an0.loc[m].iloc[0]
        return None

    row_A = get_row(term_A)
    row_B = get_row(term_B)
    row_AB = get_row(term_AB)
    row_blockA = get_row(term_blockA)
    row_resid = get_row(term_resid)

    if row_A is None or row_blockA is None or row_resid is None:
        raise HTTPException(status_code=500, detail="Split-plot ANOVA term parsing failed (check column names).")

    ms_A = float(row_A["ms"])
    df_A = float(row_A["df"])
    ms_blockA = float(row_blockA["ms"])
    df_blockA = float(row_blockA["df"])

    ms_resid = float(row_resid["ms"])
    df_resid = float(row_resid["df"])

    F_A = ms_A / ms_blockA if ms_blockA > 0 else np.nan
    p_A = float(stats.f.sf(F_A, df_A, df_blockA)) if np.isfinite(F_A) else None

    def f_p(row: Optional[pd.Series]) -> Tuple[Optional[float], Optional[float]]:
        if row is None:
            return None, None
        ms = float(row["ms"])
        df1 = float(row["df"])
        Fv = ms / ms_resid if ms_resid > 0 else np.nan
        pv = float(stats.f.sf(Fv, df1, df_resid)) if np.isfinite(Fv) else None
        return float(Fv), pv

    F_B, p_B = f_p(row_B)
    F_AB, p_AB = f_p(row_AB)

    # Corrected table
    an_corr = an0.copy()
    an_corr["F_corrected"] = None
    an_corr["p_corrected"] = None

    def set_corr(term: str, Fv: Optional[float], pv: Optional[float]) -> None:
        m = an_corr["source"].astype(str) == term
        if m.any():
            idx = an_corr.index[m][0]
            an_corr.loc[idx, "F_corrected"] = Fv
            an_corr.loc[idx, "p_corrected"] = pv

    set_corr(term_A, float(F_A), p_A)
    set_corr(term_B, F_B, p_B)
    set_corr(term_AB, F_AB, p_AB)

    sh = shapiro_test(model.resid)
    lv = levene_test(d, trait, "_AB_")

    tukey_df, means_letters = tukey_letters(d, trait, "_AB_", alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, "_AB_", f"Means ± SE by {main}:{sub} (Split-plot)"),
        "box_plot": box_plot(d, trait, "_AB_", f"Boxplot of {trait} by {main}:{sub} (Split-plot)"),
    }

    p_map = {
        f"Main plot ({main})": p_A,
        f"Sub plot ({sub})": p_B,
        f"{main}×{sub}": p_AB,
    }

    return {
        "meta": {
            "design": "Split-plot",
            "block": block,
            "main_plot_factor": main,
            "sub_plot_factor": sub,
            "trait": trait,
            "alpha": alpha,
            "n_rows_used": int(d.shape[0]),
            "cv_percent": cv_percent(d[trait]),
            "note": "Main plot factor is tested using Block:Main as the error term (correct split-plot test).",
        },
        "tables": {
            "anova_raw": df_to_records(an0.replace({np.nan: None})),
            "anova_corrected": df_to_records(an_corr[["source", "df", "sum_sq", "ms", "F_corrected", "p_corrected"]].replace({np.nan: None})),
            "means": df_to_records(means_letters.rename(columns={"_AB_": "Main:Sub"})),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
        },
        "plots": plots,
        "interpretation": interpret_anova("Split-plot ANOVA", p_map, cv_percent(d[trait]), alpha),
    }


# ----------------------------
# Endpoints — Descriptive
# ----------------------------
@app.post("/analyze/descriptive")
async def descriptive(
    file: UploadFile = File(...),
    columns: List[str] = Form(...),
    by: Optional[str] = Form(None),
):
    df = await load_csv(file)
    require_cols(df, columns)
    if by is not None:
        require_cols(df, [by])
        df[by] = df[by].astype(str)

    if by is None:
        rows = []
        for c in columns:
            s = coerce_numeric(df[c])
            rows.append({
                "column": c,
                "n": int(s.count()),
                "missing": int(s.isna().sum()),
                "mean": float(np.nanmean(s)) if s.count() else None,
                "sd": float(np.nanstd(s, ddof=1)) if s.count() > 1 else None,
                "se": float(np.nanstd(s, ddof=1) / np.sqrt(s.count())) if s.count() > 1 else None,
                "min": float(np.nanmin(s)) if s.count() else None,
                "max": float(np.nanmax(s)) if s.count() else None,
                "cv_percent": cv_percent(s),
            })
        return {
            "meta": {"by": None, "columns": columns, "n_rows": int(df.shape[0])},
            "tables": {"descriptive": rows},
            "plots": {},
            "interpretation": "Descriptive statistics computed for the selected columns.",
        }

    # Grouped
    rows = []
    for c in columns:
        d2 = df[[by, c]].copy()
        d2[c] = coerce_numeric(d2[c])
        d2 = d2.dropna(subset=[by, c])
        if d2.empty:
            continue
        g = d2.groupby(by)[c]
        tmp = pd.DataFrame({
            by: g.mean().index.astype(str),
            "n": g.count().values,
            "mean": g.mean().values,
            "sd": g.std(ddof=1).values,
            "se": (g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))).values,
        })
        tmp["column"] = c
        rows.extend(df_to_records(tmp))

    return {
        "meta": {"by": by, "columns": columns, "n_rows": int(df.shape[0])},
        "tables": {"descriptive_by": rows},
        "plots": {},
        "interpretation": f"Descriptive statistics computed by group ({by}).",
    }


# ----------------------------
# Endpoints — ANOVA single trait
# ----------------------------
@app.post("/analyze/anova/oneway")
async def analyze_anova_oneway(
    file: UploadFile = File(...),
    factor: str = Form(...),
    trait: str = Form(...),
    alpha: float = Form(0.05),
):
    df = await load_csv(file)
    require_cols(df, [factor, trait])
    return oneway_engine(df, factor, trait, float(alpha))


@app.post("/analyze/anova/twoway")
async def analyze_anova_twoway(
    file: UploadFile = File(...),
    factor_a: str = Form(...),
    factor_b: str = Form(...),
    trait: str = Form(...),
    alpha: float = Form(0.05),
):
    df = await load_csv(file)
    require_cols(df, [factor_a, factor_b, trait])
    return twoway_engine(df, factor_a, factor_b, trait, float(alpha))


@app.post("/analyze/anova/rcbd_factorial")
async def analyze_anova_rcbd_factorial(
    file: UploadFile = File(...),
    block: str = Form(...),
    factor_a: str = Form(...),
    factor_b: str = Form(...),
    trait: str = Form(...),
    alpha: float = Form(0.05),
):
    df = await load_csv(file)
    require_cols(df, [block, factor_a, factor_b, trait])
    return rcbd_factorial_engine(df, block, factor_a, factor_b, trait, float(alpha))


@app.post("/analyze/anova/splitplot")
async def analyze_anova_splitplot(
    file: UploadFile = File(...),
    block: str = Form(...),
    main_plot: str = Form(...),
    sub_plot: str = Form(...),
    trait: str = Form(...),
    alpha: float = Form(0.05),
):
    df = await load_csv(file)
    require_cols(df, [block, main_plot, sub_plot, trait])
    return splitplot_engine(df, block, main_plot, sub_plot, trait, float(alpha))


# ----------------------------
# Endpoint — Multi-trait ANOVA runner (robust, never 500 due to one bad trait)
# ----------------------------
@app.post("/analyze/anova/multitrait")
async def analyze_anova_multitrait(
    file: UploadFile = File(...),
    design: str = Form(...),                     # oneway | twoway | rcbd_factorial | splitplot
    trait_columns: List[str] = Form(...),        # repeated fields OR comma list (Swagger)
    factor: Optional[str] = Form(None),
    factor_a: Optional[str] = Form(None),
    factor_b: Optional[str] = Form(None),
    block: Optional[str] = Form(None),
    main_plot: Optional[str] = Form(None),
    sub_plot: Optional[str] = Form(None),
    alpha: float = Form(0.05),
):
    df = await load_csv(file)

    design = (design or "").strip().lower()
    alpha = float(alpha)

    # Normalize trait_columns (Swagger may send one item like "Yield,Brix,Height")
    normalized: List[str] = []
    for item in trait_columns or []:
        if item is None:
            continue
        parts = [p.strip() for p in str(item).split(",") if p.strip()]
        normalized.extend(parts)
    trait_columns = normalized

    if not trait_columns:
        raise HTTPException(status_code=400, detail="trait_columns is required (repeat the field or supply comma-separated list).")

    # Validate traits exist
    for t in trait_columns:
        if t not in df.columns:
            raise HTTPException(status_code=400, detail=f"Trait column '{t}' not found in CSV.")

    results_by_trait: Dict[str, Any] = {}
    summary: List[Dict[str, Any]] = []

    for t in trait_columns:
        try:
            if design == "oneway":
                if not factor:
                    raise HTTPException(status_code=400, detail="factor is required for design=oneway.")
                r = oneway_engine(df, factor, t, alpha)

            elif design == "twoway":
                if not factor_a or not factor_b:
                    raise HTTPException(status_code=400, detail="factor_a and factor_b are required for design=twoway.")
                r = twoway_engine(df, factor_a, factor_b, t, alpha)

            elif design == "rcbd_factorial":
                if not block or not factor_a or not factor_b:
                    raise HTTPException(status_code=400, detail="block, factor_a and factor_b are required for design=rcbd_factorial.")
                r = rcbd_factorial_engine(df, block, factor_a, factor_b, t, alpha)

            elif design == "splitplot":
                if not block or not main_plot or not sub_plot:
                    raise HTTPException(status_code=400, detail="block, main_plot and sub_plot are required for design=splitplot.")
                r = splitplot_engine(df, block, main_plot, sub_plot, t, alpha)

            else:
                raise HTTPException(status_code=400, detail="design must be one of: oneway, twoway, rcbd_factorial, splitplot.")

            results_by_trait[t] = r
            meta = (r or {}).get("meta", {})
            summary.append({
                "trait": t,
                "design": meta.get("design"),
                "n_rows_used": meta.get("n_rows_used"),
                "cv_percent": meta.get("cv_percent"),
            })

        except HTTPException as e:
            results_by_trait[t] = {"error": True, "detail": e.detail}
            summary.append({"trait": t, "error": True, "detail": e.detail})

        except Exception as e:
            # Never crash the whole request because one trait failed
            results_by_trait[t] = {"error": True, "detail": str(e)}
            summary.append({"trait": t, "error": True, "detail": str(e)})

    return {
        "meta": {
            "design": design,
            "alpha": alpha,
            "traits": trait_columns,
            "n_rows": int(df.shape[0]),
        },
        "tables": {
            "summary": summary,
        },
        "results_by_trait": results_by_trait,
        "plots": {},
        "interpretation": "Multi-trait analysis completed. Any problematic trait is returned as an error object instead of crashing the whole run.",
    }
