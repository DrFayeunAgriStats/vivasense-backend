from __future__ import annotations

import io
import base64
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
app = FastAPI(title="VivaSense V1", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Request models
# ----------------------------
class OneWayAnovaRequest(BaseModel):
    factor: str = Field(..., description="Grouping/factor column (e.g., Treatment, Genotype)")
    trait: str = Field(..., description="Numeric trait column (e.g., Yield)")
    alpha: float = Field(0.05, ge=0.0001, le=0.2)


class MultiTraitAnovaRequest(BaseModel):
    factor: str = Field(..., description="Grouping/factor column (e.g., Treatment, Genotype)")
    traits: List[str] = Field(..., description="List of numeric trait columns")
    alpha: float = Field(0.05, ge=0.0001, le=0.2)


class CorrelationRequest(BaseModel):
    columns: Optional[List[str]] = Field(None, description="Columns to include (numeric only). If null, use all numeric columns.")
    method: str = Field("pearson", description="pearson or spearman")


class RegressionRequest(BaseModel):
    x: str = Field(..., description="Predictor column (numeric)")
    y: str = Field(..., description="Response column (numeric)")


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


def _agri_interpretation_oneway(alpha: float, p: float, trait: str, factor: str) -> str:
    if p < alpha:
        return (
            f"ANOVA indicates a statistically significant effect of '{factor}' on '{trait}' (p={p:.4g} < {alpha}). "
            f"This suggests real differences among the groups. Use the Tukey mean separation and letters to identify "
            f"which groups are different. Groups that do NOT share the same letter are significantly different."
        )
    return (
        f"ANOVA indicates no statistically significant effect of '{factor}' on '{trait}' (p={p:.4g} ≥ {alpha}). "
        f"Observed differences in group means are likely due to random variation under this design. "
        f"You may still report descriptive means and variability (SE/CV), but avoid claiming treatment/genotype superiority."
    )


def _assumption_notes(shapiro_p: Optional[float], levene_p: Optional[float], alpha: float) -> List[str]:
    notes = []
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


def _summary_stats(df: pd.DataFrame, factor: str, trait: str) -> pd.DataFrame:
    g = df.groupby(factor)[trait]
    out = pd.DataFrame({
        "n": g.count(),
        "mean": g.mean(),
        "sd": g.std(ddof=1),
    })
    out["se"] = out["sd"] / np.sqrt(out["n"].replace(0, np.nan))
    out = out.reset_index()
    return out


def _cv_percent(y: pd.Series) -> Optional[float]:
    y = y.dropna()
    if y.shape[0] < 3:
        return None
    m = float(np.mean(y))
    s = float(np.std(y, ddof=1))
    if m == 0:
        return None
    return (s / m) * 100.0


# ---- Compact Letter Display (simple greedy algorithm)
def _build_cld_from_tukey(tukey: Any, groups_order: List[str], alpha: float) -> Dict[str, str]:
    """
    Returns mapping {group: letters}. Groups with shared letter are NOT significantly different.
    This is a pragmatic CLD (not perfect for every edge-case), but works well for typical agricultural layouts.
    """
    # Build significance matrix
    idx = {g: i for i, g in enumerate(groups_order)}
    k = len(groups_order)
    sig = np.zeros((k, k), dtype=bool)

    # tukey.summary() gives rows with group1 group2 meandiff p-adj lower upper reject
    table = tukey.summary().data[1:]
    for row in table:
        g1, g2, p_adj, reject = row[0], row[1], float(row[3]), bool(row[5])
        i, j = idx[str(g1)], idx[str(g2)]
        sig[i, j] = reject
        sig[j, i] = reject

    # Greedy letter assignment
    letters = []
    group_letters: Dict[str, List[str]] = {g: [] for g in groups_order}

    def can_share_letter(g: str, letter_groups: List[str]) -> bool:
        gi = idx[g]
        for h in letter_groups:
            hi = idx[h]
            if sig[gi, hi]:
                return False
        return True

    for g in groups_order:
        placed = False
        for li, letter_groups in enumerate(letters):
            if can_share_letter(g, letter_groups):
                letter_groups.append(g)
                group_letters[g].append(chr(ord("a") + li))
                placed = True
        if not placed:
            letters.append([g])
            group_letters[g].append(chr(ord("a") + (len(letters) - 1)))

    return {g: "".join(group_letters[g]) for g in groups_order}


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


def _boxplot(df: pd.DataFrame, factor: str, trait: str) -> str:
    fig = plt.figure()
    groups = [g[trait].dropna().values for _, g in df.groupby(factor)]
    labels = [str(k) for k, _ in df.groupby(factor)]
    plt.boxplot(groups, labels=labels, showmeans=True)
    plt.xticks(rotation=30, ha="right")
    plt.title(f"Boxplot: {trait} by {factor}")
    plt.xlabel(factor)
    plt.ylabel(trait)
    return _b64_png_from_fig(fig)


def _anova_oneway(df: pd.DataFrame, factor: str, trait: str, alpha: float) -> Dict[str, Any]:
    # Clean
    tmp = df[[factor, trait]].copy()
    tmp[factor] = _clean_factor(tmp[factor], factor)
    tmp[trait] = _ensure_numeric(tmp[trait], trait)
    tmp = tmp.dropna(subset=[trait, factor])

    # Model
    model = ols(f"`{trait}` ~ C(`{factor}`)", data=tmp).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # p-value
    p = float(anova_table.loc[f"C(`{factor}`)", "PR(>F)"])

    # Diagnostics
    resid = pd.Series(model.resid).dropna()
    shapiro_p = None
    if resid.shape[0] >= 3 and resid.shape[0] <= 5000:
        try:
            shapiro_p = float(stats.shapiro(resid)[1])
        except Exception:
            shapiro_p = None

    # Levene across groups
    levene_p = None
    try:
        arrays = [g[trait].dropna().values for _, g in tmp.groupby(factor)]
        if len(arrays) >= 2:
            levene_p = float(stats.levene(*arrays, center="median")[1])
    except Exception:
        levene_p = None

    # Summary stats
    summ = _summary_stats(tmp, factor, trait)
    cv = _cv_percent(tmp[trait])

    # Tukey (always compute; useful even if ANOVA ns)
    try:
        tuk = pairwise_tukeyhsd(endog=tmp[trait].values, groups=tmp[factor].values, alpha=alpha)
        tukey_rows = []
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
    except Exception as e:
        tuk = None
        tukey_rows = []
    
    # Order groups by mean (descending) for nice letters
    summ_sorted = summ.sort_values("mean", ascending=False).reset_index(drop=True)
    groups_order = [str(x) for x in summ_sorted[factor].tolist()]

    cld = {}
    if tuk is not None and len(groups_order) >= 2:
        try:
            cld = _build_cld_from_tukey(tuk, groups_order, alpha)
        except Exception:
            cld = {}

    # Attach letters to summary table
    summ_sorted["letters"] = summ_sorted[factor].astype(str).map(cld).fillna("")

    # Plots
    mean_plot_b64 = _mean_plot(summ_sorted, factor, trait)
    boxplot_b64 = _boxplot(tmp, factor, trait)

    # Output
    result = {
        "meta": {
            "analysis": "one_way_anova",
            "factor": factor,
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
        "interpretation": _agri_interpretation_oneway(alpha, p, trait, factor),
    }
    return result


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "VivaSense V1"}


@app.post("/analyze/anova/oneway")
async def analyze_oneway(req: OneWayAnovaRequest, file: UploadFile = File(...)) -> Dict[str, Any]:
    df = _read_csv(file)
    _require_columns(df, [req.factor, req.trait])
    return _anova_oneway(df, req.factor, req.trait, req.alpha)


@app.post("/analyze/anova/multitrait")
async def analyze_multitrait(req: MultiTraitAnovaRequest, file: UploadFile = File(...)) -> Dict[str, Any]:
    df = _read_csv(file)
    _require_columns(df, [req.factor] + req.traits)

    results = []
    for t in req.traits:
        try:
            results.append(_anova_oneway(df, req.factor, t, req.alpha))
        except HTTPException as e:
            results.append({
                "meta": {"analysis": "one_way_anova", "factor": req.factor, "trait": t, "alpha": req.alpha},
                "error": e.detail
            })
        except Exception as e:
            results.append({
                "meta": {"analysis": "one_way_anova", "factor": req.factor, "trait": t, "alpha": req.alpha},
                "error": str(e)
            })

    return {
        "meta": {"analysis": "multi_trait_oneway_anova", "factor": req.factor, "alpha": req.alpha, "n_traits": len(req.traits)},
        "results": results
    }


@app.post("/analyze/correlation")
async def analyze_correlation(req: CorrelationRequest, file: UploadFile = File(...)) -> Dict[str, Any]:
    df = _read_csv(file)

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise HTTPException(status_code=400, detail="No numeric columns found for correlation.")

    cols = req.columns if req.columns else numeric_df.columns.tolist()
    _require_columns(df, cols)

    data = df[cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(axis=0, how="any")
    if data.shape[0] < 3:
        raise HTTPException(status_code=400, detail="Too few complete rows for correlation (need at least 3).")

    method = req.method.lower().strip()
    if method not in ["pearson", "spearman"]:
        raise HTTPException(status_code=400, detail="method must be 'pearson' or 'spearman'.")

    corr = data.corr(method=method)

    # Heatmap plot (matplotlib only)
    fig = plt.figure()
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title(f"{method.title()} Correlation Heatmap")
    plt.colorbar()
    heatmap_b64 = _b64_png_from_fig(fig)

    return {
        "meta": {"analysis": "correlation", "method": method, "n_rows_used": int(data.shape[0])},
        "correlation_matrix": corr.reset_index().rename(columns={"index": "variable"}).to_dict(orient="records"),
        "plots": {"heatmap_png_b64": heatmap_b64},
        "interpretation": (
            "Correlation measures association (not causation). Values near +1/-1 indicate strong positive/negative association, "
            "while values near 0 indicate weak linear/monotonic association."
        )
    }


@app.post("/analyze/regression/simple")
async def analyze_regression(req: RegressionRequest, file: UploadFile = File(...)) -> Dict[str, Any]:
    df = _read_csv(file)
    _require_columns(df, [req.x, req.y])

    tmp = df[[req.x, req.y]].copy()
    tmp[req.x] = pd.to_numeric(tmp[req.x], errors="coerce")
    tmp[req.y] = pd.to_numeric(tmp[req.y], errors="coerce")
    tmp = tmp.dropna()

    if tmp.shape[0] < 5:
        raise HTTPException(status_code=400, detail="Too few rows for regression (need at least 5 complete observations).")

    X = sm.add_constant(tmp[req.x].values)
    y = tmp[req.y].values
    model = sm.OLS(y, X).fit()

    # Scatter + fitted line
    fig = plt.figure()
    plt.scatter(tmp[req.x].values, tmp[req.y].values)
    x_line = np.linspace(tmp[req.x].min(), tmp[req.x].max(), 100)
    y_line = model.params[0] + model.params[1] * x_line
    plt.plot(x_line, y_line)
    plt.xlabel(req.x)
    plt.ylabel(req.y)
    plt.title("Simple Linear Regression")
    plot_b64 = _b64_png_from_fig(fig)

    return {
        "meta": {"analysis": "simple_regression", "x": req.x, "y": req.y, "n_rows_used": int(tmp.shape[0])},
        "model": {
            "intercept": float(model.params[0]),
            "slope": float(model.params[1]),
            "r_squared": float(model.rsquared),
            "p_value_slope": float(model.pvalues[1]),
            "stderr_slope": float(model.bse[1]),
        },
        "plots": {"scatter_fit_png_b64": plot_b64},
        "interpretation": (
            f"The slope indicates the expected change in '{req.y}' per unit increase in '{req.x}'. "
            f"R² shows how much variation in '{req.y}' is explained by '{req.x}' in this linear model."
        )
    }
