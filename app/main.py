"""
VivaSense V1 (FastAPI) — single-file backend: app/main.py

Scope (per your new V1 plan)
- Descriptive statistics (single + multi-trait)
- One-way ANOVA (CRD)
- Two-way ANOVA (factorial CRD)
- Factorial in RCBD (ANOVA with Block as factor)
- Split-plot ANOVA (correct A test using Block:A as error term)
- Multi-trait runner for any of the above (runs trait list)

All endpoints accept multipart/form-data (Form + File), NOT JSON models.
Returns JSON:
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
    return {"name": "VivaSense V1", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ----------------------------
# Utilities
# ----------------------------
def _b64_png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


async def load_csv(upload: UploadFile) -> pd.DataFrame:
    if not upload:
        raise HTTPException(status_code=400, detail="Missing file.")
    if not upload.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    content = await upload.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
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
        raise HTTPException(status_code=400, detail="Not enough valid rows after cleaning.")
    return d


def shapiro_test(resid: np.ndarray) -> Dict[str, Any]:
    resid = np.asarray(resid)
    if resid.size < 3:
        return {"test": "Shapiro-Wilk", "p_value": None, "note": "Not enough residuals."}
    # Shapiro has an upper practical limit; still ok for typical agri datasets
    stat, p = stats.shapiro(resid[:5000])
    return {"test": "Shapiro-Wilk", "stat": float(stat), "p_value": float(p)}


def levene_test(df: pd.DataFrame, y: str, group: str) -> Dict[str, Any]:
    groups = []
    for _, g in df.groupby(group):
        vals = g[y].dropna().values
        if len(vals) > 0:
            groups.append(vals)
    if len(groups) < 2:
        return {"test": "Levene", "p_value": None, "note": "Need at least 2 groups."}
    stat, p = stats.levene(*groups, center="median")
    return {"test": "Levene", "stat": float(stat), "p_value": float(p)}


def cv_percent(y: pd.Series) -> Optional[float]:
    m = float(np.nanmean(y))
    s = float(np.nanstd(y, ddof=1)) if y.dropna().shape[0] > 1 else np.nan
    if m == 0 or np.isnan(m) or np.isnan(s):
        return None
    return float((s / m) * 100.0)


def mean_table(df: pd.DataFrame, y: str, group: str) -> pd.DataFrame:
    g = df.groupby(group)[y]
    out = pd.DataFrame({
        "n": g.count(),
        "mean": g.mean(),
        "sd": g.std(ddof=1),
        "se": g.std(ddof=1) / np.sqrt(g.count().clip(lower=1)),
    }).reset_index()
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


def mean_plot(df: pd.DataFrame, y: str, group: str, title: str) -> str:
    mt = mean_table(df, y, group)
    fig = plt.figure(figsize=(7.5, 4.2))
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
    fig = plt.figure(figsize=(7.5, 4.2))
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
    Simple CLD (letters) from pairwise significance matrix.
    sig[(a,b)] == True means significantly different.
    """
    # Order groups by mean descending
    ordered = sorted(groups, key=lambda g: means.get(g, -np.inf), reverse=True)

    letters_for = {g: "" for g in ordered}
    letter_sets: List[List[str]] = []

    def conflict(g: str, members: List[str]) -> bool:
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
            if not conflict(g, members):
                members.append(g)
                placed = True
                break
        if not placed:
            letter_sets.append([g])

    # assign letters a, b, c...
    for i, members in enumerate(letter_sets):
        letter = chr(ord("a") + i)
        for g in members:
            letters_for[g] += letter

    return letters_for


def tukey_letters(df: pd.DataFrame, y: str, group: str, alpha: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - tukey table dataframe
      - means table with letters
    """
    mt = mean_table(df, y, group)
    # Tukey needs arrays:
    endog = df[y].values
    groups = df[group].astype(str).values
    res = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=alpha)

    tukey_df = pd.DataFrame(
        data=res._results_table.data[1:],
        columns=res._results_table.data[0]
    )

    # Build significance map
    uniq = mt[group].astype(str).tolist()
    means = {row[group]: float(row["mean"]) for _, row in mt.iterrows()}
    sig = {}
    for _, r in tukey_df.iterrows():
        a = str(r["group1"])
        b = str(r["group2"])
        sig[(a, b)] = bool(r["reject"])

    letters = compact_letter_display(uniq, means, sig)
    mt["group"] = mt[group].astype(str)
    mt["letters"] = mt["group"].map(letters)
    mt = mt.drop(columns=["group"])
    return tukey_df, mt


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return df.replace({np.nan: None}).to_dict(orient="records")


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
    lines.append("- Check assumptions: normality of residuals and homogeneity of variances (Levene).")
    return "\n".join(lines)


# ----------------------------
# Core ANOVA engines
# ----------------------------
def oneway_engine(df: pd.DataFrame, factor: str, trait: str, alpha: float) -> Dict[str, Any]:
    d = clean_for_model(df, trait, [factor])
    d[factor] = d[factor].astype(str)

    model = ols(f"{trait} ~ C({factor})", data=d).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})
    anova["ms"] = anova["sum_sq"] / anova["df"]

    # Assumptions
    sh = shapiro_test(model.resid)
    lv = levene_test(d, trait, factor)

    # Means + Tukey
    tukey_df, means_letters = tukey_letters(d, trait, factor, alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, factor, f"Means ± SE by {factor}"),
        "box_plot": box_plot(d, trait, factor, f"Boxplot of {trait} by {factor}"),
    }

    p_map = {f"{factor}": float(anova.loc[anova["source"].str.contains(factor), "PR(>F)"].iloc[0]) if (anova["source"].str.contains(factor).any()) else None}

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
        "interpretation": interpret_anova("One-way ANOVA", p_map, cv_percent(d[trait]), alpha),
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
    # Levene across interaction groups is a practical choice
    lv = levene_test(d, trait, "_AB_")

    tukey_df, means_letters = tukey_letters(d, trait, "_AB_", alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, "_AB_", f"Means ± SE by {a}:{b}"),
        "box_plot": box_plot(d, trait, "_AB_", f"Boxplot of {trait} by {a}:{b}"),
    }

    # p-values
    def p_of(term_contains: str) -> Optional[float]:
        m = anova["source"].astype(str).str.contains(term_contains)
        if not m.any():
            return None
        return float(anova.loc[m, "PR(>F)"].iloc[0])

    p_map = {
        a: p_of(a),
        b: p_of(b),
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
    """
    Factorial in RCBD.
    Practical V1 approach: OLS with block included as a factor.
      y ~ block + A*B
    """
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

    def p_contains(term: str) -> Optional[float]:
        m = anova["source"].astype(str).str.contains(term)
        if not m.any():
            return None
        return float(anova.loc[m, "PR(>F)"].iloc[0])

    p_map = {
        "Block": p_contains(block),
        a: p_contains(a),
        b: p_contains(b),
        f"{a}×{b}": p_contains(":"),
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
    Split-plot:
      - Main plot factor: A (main)
      - Subplot factor: B (sub)
      - Blocks: block

    V1 correct-testing approach:
      Fit: y ~ block + A*B + block:A
      Use MS(block:A) as denominator for A test (main plot error).
      Use residual MS as denominator for B and A:B.
    """
    d = clean_for_model(df, trait, [block, main, sub])
    for c in [block, main, sub]:
        d[c] = d[c].astype(str)

    d["_AB_"] = d[main].astype(str) + ":" + d[sub].astype(str)

    formula = f"{trait} ~ C({block}) + C({main}) * C({sub}) + C({block}):C({main})"
    model = ols(formula, data=d).fit()
    an0 = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})

    # Add MS
    an0["ms"] = an0["sum_sq"] / an0["df"]

    # Identify rows
    src = an0["source"].astype(str)
    # Depending on statsmodels formatting, term strings look like "C(A)" etc.
    term_A = f"C({main})"
    term_B = f"C({sub})"
    term_AB = f"C({main}):C({sub})"
    term_blockA = f"C({block}):C({main})"
    term_block = f"C({block})"

    def get_row(term: str) -> Optional[pd.Series]:
        m = src == term
        if m.any():
            return an0.loc[m].iloc[0]
        # fallback contains
        m = src.str.contains(term.replace("C(", "").replace(")", ""))
        if m.any():
            return an0.loc[m].iloc[0]
        return None

    row_A = get_row(term_A)
    row_B = get_row(term_B)
    row_AB = get_row(term_AB)
    row_blockA = get_row(term_blockA)
    row_resid = get_row("Residual")

    if row_A is None or row_blockA is None or row_resid is None:
        raise HTTPException(status_code=500, detail="Split-plot ANOVA term parsing failed.")

    # Compute correct F-tests
    ms_A = float(row_A["ms"])
    df_A = float(row_A["df"])
    ms_blockA = float(row_blockA["ms"])
    df_blockA = float(row_blockA["df"])
    ms_resid = float(row_resid["ms"])
    df_resid = float(row_resid["df"])

    # A tested against block:A
    F_A = ms_A / ms_blockA if ms_blockA > 0 else np.nan
    p_A = float(stats.f.sf(F_A, df_A, df_blockA)) if np.isfinite(F_A) else None

    # B and A:B tested against residual
    def f_p(row: Optional[pd.Series]) -> Tuple[Optional[float], Optional[float]]:
        if row is None:
            return None, None
        ms = float(row["ms"])
        df1 = float(row["df"])
        F = ms / ms_resid if ms_resid > 0 else np.nan
        p = float(stats.f.sf(F, df1, df_resid)) if np.isfinite(F) else None
        return float(F), p

    F_B, p_B = f_p(row_B)
    F_AB, p_AB = f_p(row_AB)

    # Build a “final” ANOVA table with corrected F/p for A, B, A:B
    an = an0.copy()
    an["F_corrected"] = None
    an["p_corrected"] = None

    def set_term(term: str, Fv: Optional[float], pv: Optional[float]):
        m = an["source"].astype(str) == term
        if not m.any():
            m = an["source"].astype(str).str.contains(term.replace("C(", "").replace(")", ""))
        if m.any():
            idx = an.index[m][0]
            an.loc[idx, "F_corrected"] = Fv
            an.loc[idx, "p_corrected"] = pv

    set_term(term_A, float(F_A), p_A)
    set_term(term_B, F_B, p_B)
    set_term(term_AB, F_AB, p_AB)

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
            "anova_corrected": df_to_records(an[["source", "df", "sum_sq", "ms", "F_corrected", "p_corrected"]].replace({np.nan: None})),
            "means": df_to_records(means_letters.rename(columns={"_AB_": "Main:Sub"})),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
        },
        "plots": plots,
        "interpretation": interpret_anova("Split-plot ANOVA", p_map, cv_percent(d[trait]), alpha),
    }


# ----------------------------
# Endpoints: Descriptive
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

    out_tables = {}

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
        out_tables["descriptive"] = rows
        interp = "Descriptive statistics computed for the selected columns."
    else:
        df[by] = df[by].astype(str)
        rows = []
        for c in columns:
            s = coerce_numeric(df[c])
            d2 = df.copy()
            d2[c] = s
            d2 = d2.dropna(subset=[c, by])
            g = d2.groupby(by)[c]
            tmp = pd.DataFrame({
                by: g.mean().index.astype(str),
                "n": g.count().values,
                "mean": g.mean().values,
                "sd": g.std(ddof=1).values,
                "se": (g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))).values,
                "cv_percent": [cv_percent(d2.loc[d2[by] == lvl, c]) for lvl in g.mean().index.astype(str)],
            })
            tmp["column"] = c
            rows.extend(df_to_records(tmp))
        out_tables["descriptive_by"] = rows
        interp = f"Descriptive statistics computed by group ({by})."

    return {
        "meta": {"columns": columns, "by": by, "n_rows": int(df.shape[0])},
        "tables": out_tables,
        "plots": {},
        "interpretation": interp,
    }


# ----------------------------
# Endpoints: ANOVA (single-trait)
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
# Endpoint: Multi-trait runner
# ----------------------------
@app.post("/analyze/anova/multitrait")
async def analyze_anova_multitrait(
    file: UploadFile = File(...),
    design: str = Form(...),  # "oneway" | "twoway" | "rcbd_factorial" | "splitplot"
    trait_columns: List[str] = Form(...),  # repeated fields: trait_columns=yield&trait_columns=brix...
    # Design columns (some optional depending on design)
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

    # Validate trait columns
    if not trait_columns:
        raise HTTPException(status_code=400, detail="trait_columns is required (repeat the field for each trait).")
    for t in trait_columns:
        if t not in df.columns:
            raise HTTPException(status_code=400, detail=f"Trait column '{t}' not found in CSV.")

    results: Dict[str, Any] = {}
    summary_rows = []

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

            results[t] = r

            # build a compact summary
            meta = r.get("meta", {})
            cvv = meta.get("cv_percent", None)
            summary_rows.append({
                "trait": t,
                "design": meta.get("design"),
                "n_rows_used": meta.get("n_rows_used"),
                "cv_percent": cvv,
            })

        except HTTPException as e:
            results[t] = {"error": True, "detail": e.detail}
        except Exception as e:
            results[t] = {"error": True, "detail": str(e)}

    return {
        "meta": {
            "design": design,
            "alpha": alpha,
            "traits": trait_columns,
            "n_rows": int(df.shape[0]),
        },
        "tables": {
            "summary": summary_rows,
        },
        "results_by_trait": results,
        "plots": {},
        "interpretation": "Multi-trait analysis completed. Open each trait result to view ANOVA, mean separation, plots, and interpretation.",
    }
