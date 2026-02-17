"""
VivaSense V1 - FastAPI Statistical Backend (single file: main.py)

✅ Fixes included:
- Root route "/" to stop Render 404
- Endpoints accept multipart/form-data using Form(...) + File(...)
- One-way ANOVA supports optional Block (RCBD) via model: trait ~ factor + block
- Multi-trait one-way ANOVA supports optional Block
- Correlation + heatmap via multipart form
- Simple regression via multipart form
- Corrected Tukey parsing + Compact Letter Display (CLD)

Deploy note (Render):
Your start command should point to this file, e.g.
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import io
import base64
import warnings
from typing import Any, Dict, List, Optional

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
# App setup
# ----------------------------
app = FastAPI(title="VivaSense V1", version="1.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Helpers
# ----------------------------
def _read_csv(upload: UploadFile) -> pd.DataFrame:
    try:
        raw = upload.file.read()
        df = pd.read_csv(io.BytesIO(raw))
        if df.empty:
            raise ValueError("CSV is empty.")
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")


def _b64_png_from_fig(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")


def _ensure_numeric(series: pd.Series, name: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().shape[0] < 3:
        raise HTTPException(status_code=400, detail=f"Trait '{name}' has too few numeric values.")
    return s


def _clean_factor(series: pd.Series, name: str) -> pd.Series:
    s = series.astype(str).fillna("NA")
    if s.nunique(dropna=False) < 2:
        raise HTTPException(status_code=400, detail=f"Factor '{name}' must have at least 2 groups.")
    return s


def _cv_percent(y: pd.Series) -> Optional[float]:
    y = y.dropna()
    if y.shape[0] < 3:
        return None
    m = float(np.mean(y))
    s = float(np.std(y, ddof=1))
    if m == 0:
        return None
    return (s / m) * 100.0


def _assumption_notes(shapiro_p: Optional[float], levene_p: Optional[float], alpha: float) -> List[str]:
    notes: List[str] = []
    if shapiro_p is not None:
        if shapiro_p < alpha:
            notes.append(
                f"Normality check (Shapiro on residuals): p={shapiro_p:.4g} < {alpha} → residuals deviate from normality. "
                f"Consider transformation (log/sqrt) or robust/nonparametric alternatives depending on design."
            )
        else:
            notes.append(f"Normality check (Shapiro on residuals): p={shapiro_p:.4g} ≥ {alpha} → normality looks acceptable.")
    if levene_p is not None:
        if levene_p < alpha:
            notes.append(
                f"Homogeneity check (Levene): p={levene_p:.4g} < {alpha} → variances differ among groups. "
                f"Consider transformation or Welch-type approach if appropriate."
            )
        else:
            notes.append(f"Homogeneity check (Levene): p={levene_p:.4g} ≥ {alpha} → variance homogeneity looks acceptable.")
    return notes


def _agri_interpretation_oneway(alpha: float, p: float, trait: str, factor: str, block: Optional[str]) -> str:
    design = "RCBD (with Block)" if block else "CRD (no Block)"
    if p < alpha:
        return (
            f"Design: {design}. ANOVA indicates a statistically significant effect of '{factor}' on '{trait}' "
            f"(p={p:.4g} < {alpha}). This suggests real differences among groups. "
            f"Use the Tukey mean separation and letters to identify which groups differ. "
            f"Groups that do NOT share the same letter are significantly different."
        )
    return (
        f"Design: {design}. ANOVA indicates no statistically significant effect of '{factor}' on '{trait}' "
        f"(p={p:.4g} ≥ {alpha}). Observed differences in group means are likely due to random variation. "
        f"You may still report descriptive means and variability (SE/CV), but avoid claiming treatment/genotype superiority."
    )


def _summary_stats(tmp: pd.DataFrame, factor: str, trait: str) -> pd.DataFrame:
    g = tmp.groupby(factor)[trait]
    out = pd.DataFrame({
        "n": g.count(),
        "mean": g.mean(),
        "sd": g.std(ddof=1),
    })
    out["se"] = out["sd"] / np.sqrt(out["n"].replace(0, np.nan))
    out = out.reset_index()
    return out


def _mean_plot(summary_df: pd.DataFrame, factor: str, trait: str) -> str:
    fig = plt.figure()
    x = np.arange(summary_df.shape[0])
    means = summary_df["mean"].values
    ses = summary_df["se"].values
    plt.errorbar(x, means, yerr=ses, fmt="o", capsize=4)
    plt.xticks(x, summary_df[factor].astype(str).values, rotation=30, ha="right")
    plt.title(f"Mean ± SE: {trait} by {factor}")
    plt.xlabel(factor)
    plt.ylabel(trait)
    return _b64_png_from_fig(fig)


def _boxplot(tmp: pd.DataFrame, factor: str, trait: str) -> str:
    fig = plt.figure()
    groups = [g[trait].dropna().values for _, g in tmp.groupby(factor)]
    labels = [str(k) for k, _ in tmp.groupby(factor)]
    plt.boxplot(groups, labels=labels, showmeans=True)
    plt.xticks(rotation=30, ha="right")
    plt.title(f"Boxplot: {trait} by {factor}")
    plt.xlabel(factor)
    plt.ylabel(trait)
    return _b64_png_from_fig(fig)


# ---- Compact Letter Display (CLD)
def _build_cld_from_tukey(tukey: Any, groups_order: List[str]) -> Dict[str, str]:
    """
    Returns mapping {group: letters}. Groups that share a letter are NOT significantly different.
    """
    idx = {g: i for i, g in enumerate(groups_order)}
    k = len(groups_order)
    sig = np.zeros((k, k), dtype=bool)

    # Row format: group1, group2, meandiff, p-adj, lower, upper, reject
    table = tukey.summary().data[1:]
    for row in table:
        g1 = str(row[0])
        g2 = str(row[1])
        reject = bool(row[6])
        if g1 in idx and g2 in idx:
            i, j = idx[g1], idx[g2]
            sig[i, j] = reject
            sig[j, i] = reject

    letters: List[List[str]] = []
    group_letters: Dict[str, List[str]] = {g: [] for g in groups_order}

    def can_share(g: str, letter_groups: List[str]) -> bool:
        gi = idx[g]
        for h in letter_groups:
            hi = idx[h]
            if sig[gi, hi]:
                return False
        return True

    for g in groups_order:
        placed_any = False
        for li, letter_groups in enumerate(letters):
            if can_share(g, letter_groups):
                letter_groups.append(g)
                group_letters[g].append(chr(ord("a") + li))
                placed_any = True
        if not placed_any:
            letters.append([g])
            group_letters[g].append(chr(ord("a") + (len(letters) - 1)))

    return {g: "".join(group_letters[g]) for g in groups_order}


def _anova_oneway(
    df: pd.DataFrame,
    factor: str,
    trait: str,
    alpha: float,
    block: Optional[str] = None,
) -> Dict[str, Any]:
    # Clean + keep needed cols
    cols = [factor, trait] + ([block] if block else [])
    tmp = df[cols].copy()

    tmp[factor] = _clean_factor(tmp[factor], factor)
    tmp[trait] = _ensure_numeric(tmp[trait], trait)

    if block:
        tmp[block] = _clean_factor(tmp[block], block)

    tmp = tmp.dropna(subset=[trait, factor] + ([block] if block else []))

    # Model
    if block:
        formula = f"`{trait}` ~ C(`{factor}`) + C(`{block}`)"
    else:
        formula = f"`{trait}` ~ C(`{factor}`)"

    model = ols(formula, data=tmp).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    term = f"C(`{factor}`)"
    if term not in anova_table.index:
        raise HTTPException(status_code=400, detail=f"Could not compute ANOVA for factor '{factor}'.")

    p = float(anova_table.loc[term, "PR(>F)"])

    # Diagnostics
    resid = pd.Series(model.resid).dropna()
    shapiro_p = None
    if 3 <= resid.shape[0] <= 5000:
        try:
            shapiro_p = float(stats.shapiro(resid)[1])
        except Exception:
            shapiro_p = None

    levene_p = None
    try:
        arrays = [g[trait].dropna().values for _, g in tmp.groupby(factor)]
        if len(arrays) >= 2:
            levene_p = float(stats.levene(*arrays, center="median")[1])
    except Exception:
        levene_p = None

    # Summary stats (raw means by factor)
    summ = _summary_stats(tmp, factor, trait)
    cv = _cv_percent(tmp[trait])

    # Tukey HSD (on raw groups; for RCBD this is a practical approximation)
    tuk = None
    tukey_rows: List[Dict[str, Any]] = []
    try:
        tuk = pairwise_tukeyhsd(endog=tmp[trait].values, groups=tmp[factor].values, alpha=alpha)
        for r in tuk.summary().data[1:]:
            tukey_rows.append({
                "group1": str(r[0]),
                "group2": str(r[1]),
                "meandiff": float(r[2]),
                "p_adj": float(r[3]),
                "lower": float(r[4]),
                "upper": float(r[5]),
                "reject": bool(r[6]),
            })
    except Exception:
        tuk = None
        tukey_rows = []

    # Order groups by mean (descending) for nicer letters
    summ_sorted = summ.sort_values("mean", ascending=False).reset_index(drop=True)
    groups_order = [str(x) for x in summ_sorted[factor].tolist()]

    cld = {}
    if tuk is not None and len(groups_order) >= 2:
        try:
            cld = _build_cld_from_tukey(tuk, groups_order)
        except Exception:
            cld = {}

    summ_sorted["letters"] = summ_sorted[factor].astype(str).map(cld).fillna("")

    # Plots
    mean_plot_b64 = _mean_plot(summ_sorted, factor, trait)
    boxplot_b64 = _boxplot(tmp, factor, trait)

    return {
        "meta": {
            "analysis": "one_way_anova",
            "design": "rcbd" if block else "crd",
            "factor": factor,
            "block": block,
            "trait": trait,
            "alpha": alpha,
            "n_rows_used": int(tmp.shape[0]),
            "cv_percent": None if cv is None else float(cv),
        },
        "anova_table": anova_table.reset_index().rename(columns={"index": "term"}).to_dict(orient="records"),
        "assumptions": {
            "shapiro_p": shapiro_p,
            "levene_p": levene_p,
            "notes": _assumption_notes(shapiro_p, levene_p, alpha),
        },
        "group_summary": summ_sorted.to_dict(orient="records"),
        "tukey_hsd": tukey_rows,
        "plots": {
            "mean_plot_png_b64": mean_plot_b64,
            "boxplot_png_b64": boxplot_b64,
        },
        "interpretation": _agri_interpretation_oneway(alpha, p, trait, factor, block),
        "notes": (
            "If 'block' is provided (RCBD), ANOVA adjusts for block in the model. "
            "Tukey here is computed on raw treatment groups (pragmatic). "
            "Next upgrade: block-adjusted mean separation (LSMeans)."
            if block else ""
        ),
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "VivaSense V1", "status": "ok", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "VivaSense V1"}


@app.post("/analyze/anova/oneway")
async def analyze_oneway(
    file: UploadFile = File(...),
    factor: str = Form(...),
    trait: str = Form(...),
    alpha: float = Form(0.05),
    block: Optional[str] = Form(None),
) -> Dict[str, Any]:
    df = _read_csv(file)
    needed = [factor, trait] + ([block] if block else [])
    _require_columns(df, needed)
    return _anova_oneway(df, factor, trait, alpha, block=block)


@app.post("/analyze/anova/multitrait")
async def analyze_multitrait(
    file: UploadFile = File(...),
    factor: str = Form(...),
    traits: List[str] = Form(...),
    alpha: float = Form(0.05),
    block: Optional[str] = Form(None),
) -> Dict[str, Any]:
    df = _read_csv(file)
    needed = [factor] + traits + ([block] if block else [])
    _require_columns(df, needed)

    results: List[Dict[str, Any]] = []
    for t in traits:
        try:
            results.append(_anova_oneway(df, factor, t, alpha, block=block))
        except HTTPException as e:
            results.append({
                "meta": {"analysis": "one_way
