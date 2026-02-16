"""
VivaSense V1 - One-way ANOVA backend (FastAPI)
- Upload CSV
- One-way ANOVA
- Tukey HSD + compact letter display
- CV, assumption checks (Shapiro, Levene)
- Mean plot (SE bars) + boxplot
- Agricultural expert interpretation embedded in JSON
"""

from __future__ import annotations

import io
import base64
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

app = FastAPI(title="VivaSense V1", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request/Response Schemas
# ----------------------------

class AnovaRequest(BaseModel):
    response_col: str
    factor_col: str
    trait_name: Optional[str] = None
    unit: Optional[str] = ""
    higher_is_better: bool = True
    control_level: Optional[str] = None
    alpha: float = 0.05


# ----------------------------
# Utilities
# ----------------------------

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def _fmt(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.{nd}f}"

def compute_cv(mse: float, overall_mean: float) -> Optional[float]:
    if overall_mean is None or np.isnan(overall_mean) or overall_mean == 0 or mse is None or np.isnan(mse):
        return None
    return (np.sqrt(mse) / overall_mean) * 100.0

def n_per_group(df: pd.DataFrame, factor_col: str, response_col: str) -> Dict[str, int]:
    g = df.dropna(subset=[response_col]).groupby(factor_col)[response_col].size()
    return {str(k): int(v) for k, v in g.items()}

# ----------------------------
# Tukey letters (compact letter display)
# ----------------------------

def tukey_letters(df: pd.DataFrame, factor_col: str, response_col: str, alpha: float) -> pd.DataFrame:
    """
    Returns means table with Tukey group letters using a simple, deterministic algorithm.
    Columns: factor, n, mean, sd, se, group
    """
    d = df[[factor_col, response_col]].dropna()
    # Summary stats
    summ = d.groupby(factor_col)[response_col].agg(["count", "mean", "std"]).reset_index()
    summ.rename(columns={"count": "n", "std": "sd"}, inplace=True)
    summ["se"] = summ["sd"] / np.sqrt(summ["n"].clip(lower=1))

    # Tukey test
    tuk = pairwise_tukeyhsd(endog=d[response_col].values, groups=d[factor_col].astype(str).values, alpha=alpha)
    tdf = pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0])
    # tdf columns: group1, group2, meandiff, p-adj, lower, upper, reject

    levels = summ[factor_col].astype(str).tolist()
    means = {str(r[factor_col]): float(r["mean"]) for _, r in summ.iterrows()}

    # Sort by mean descending (higher is "better" by default in many agronomy traits)
    levels_sorted = sorted(levels, key=lambda x: means[x], reverse=True)

    # Build significance matrix: sig[a][b] = True if significantly different
    sig = {a: {b: False for b in levels_sorted} for a in levels_sorted}
    for _, r in tdf.iterrows():
        a, b = str(r["group1"]), str(r["group2"])
        rej = bool(r["reject"])
        sig[a][b] = rej
        sig[b][a] = rej

    # Assign letters
    letters = {}
    letter_list = list("abcdefghijklmnopqrstuvwxyz")
    used = 0
    groups_for_letter: List[List[str]] = []

    for lvl in levels_sorted:
        placed = False
        for gi, members in enumerate(groups_for_letter):
            # Can we put lvl in this letter group? Only if NOT significantly different from all members
            if all(sig[lvl][m] is False for m in members):
                members.append(lvl)
                letters[lvl] = letter_list[gi]
                placed = True
                break
        if not placed:
            if used >= len(letter_list):
                # fallback if too many groups
                letters[lvl] = f"g{used+1}"
                groups_for_letter.append([lvl])
            else:
                letters[lvl] = letter_list[used]
                groups_for_letter.append([lvl])
            used += 1

    out = summ.copy()
    out["group"] = out[factor_col].astype(str).map(letters)
    return out.sort_values("mean", ascending=False).reset_index(drop=True)

# ----------------------------
# Interpretation Engine (agricultural expert tone, rule-based)
# ----------------------------

@dataclass
class InterpretationConfig:
    trait_name: str = "Trait"
    unit: str = ""
    higher_is_better: bool = True
    factor_label: str = "Treatment"
    control_level: Optional[str] = None
    alpha: float = 0.05

def _pct_change(best: float, ref: float) -> Optional[float]:
    if ref is None or np.isnan(ref) or ref == 0:
        return None
    return (best - ref) / ref * 100.0

def _cv_comment(cv: Optional[float]) -> Tuple[str, str]:
    if cv is None or np.isnan(cv):
        return ("info", "CV could not be computed.")
    if cv < 10:
        return ("info", f"Low CV ({_fmt(cv)}%) indicates excellent experimental precision.")
    if cv < 20:
        return ("info", f"Moderate CV ({_fmt(cv)}%) suggests good experimental precision.")
    if cv < 30:
        return ("warn", f"CV is somewhat high ({_fmt(cv)}%). Interpret small differences cautiously.")
    return ("critical", f"High CV ({_fmt(cv)}%) indicates low precision; results may be unreliable without improved control/replication.")

def _assumption_comment(shapiro_p: Optional[float], levene_p: Optional[float], alpha: float) -> List[Dict[str, Any]]:
    notes = []
    if shapiro_p is not None and not np.isnan(shapiro_p):
        if shapiro_p < alpha:
            notes.append({"severity": "warn", "message": f"Residual normality may be questionable (Shapiro p={_fmt(shapiro_p,3)}). Consider transformation or robust methods."})
        else:
            notes.append({"severity": "info", "message": f"Residual normality appears acceptable (Shapiro p={_fmt(shapiro_p,3)})."})
    if levene_p is not None and not np.isnan(levene_p):
        if levene_p < alpha:
            notes.append({"severity": "warn", "message": f"Variance homogeneity may be violated (Levene p={_fmt(levene_p,3)}). Consider transformation/GLM or report robust SEs."})
        else:
            notes.append({"severity": "info", "message": f"Variance homogeneity appears acceptable (Levene p={_fmt(levene_p,3)})."})
    return notes

def _design_health_comment(npg: Dict[str, int], missing_count: int) -> List[Dict[str, Any]]:
    notes = []
    if missing_count > 0:
        notes.append({"severity": "warn", "message": f"Dataset contains {missing_count} missing value(s). Verify data entry and handling method."})
    if npg:
        ns = np.array(list(npg.values()))
        if len(ns) > 1 and (ns.max() != ns.min()):
            notes.append({"severity": "warn", "message": f"Data are unbalanced (n ranges from {int(ns.min())} to {int(ns.max())} across groups). Mean separation should be interpreted cautiously."})
        else:
            notes.append({"severity": "info", "message": f"Group sample sizes are balanced (n={int(ns[0])} per group)." if len(ns) else "Group sizes look fine."})
    return notes

def generate_interpretation(
    anova_p: Optional[float],
    means_table: pd.DataFrame,
    group_col: str,
    mean_col: str,
    tukey_group_col: Optional[str],
    cv: Optional[float],
    shapiro_p: Optional[float],
    levene_p: Optional[float],
    npg: Dict[str, int],
    missing_count: int,
    cfg: InterpretationConfig
) -> Dict[str, Any]:
    warnings_list: List[Dict[str, Any]] = []
    bullets: List[str] = []
    recs: List[str] = []

    sev, cv_msg = _cv_comment(cv)
    warnings_list.append({"severity": sev, "message": cv_msg})
    warnings_list.extend(_assumption_comment(shapiro_p, levene_p, cfg.alpha))
    warnings_list.extend(_design_health_comment(npg, missing_count))

    mt = means_table.dropna(subset=[mean_col]).copy()
    if mt.empty:
        return {
            "summary_bullets": ["No interpretable means were produced."],
            "supervisor_ready_paragraph": "The analysis could not generate a valid means summary. Please check input data and variable selection.",
            "warnings": warnings_list,
            "recommendations": ["Check that the response variable is numeric and groups have data."]
        }

    mt_sorted = mt.sort_values(mean_col, ascending=not cfg.higher_is_better)
    best = mt_sorted.iloc[0]
    worst = mt_sorted.iloc[-1]
    best_name = str(best[group_col])
    worst_name = str(worst[group_col])
    best_mean = float(best[mean_col])
    worst_mean = float(worst[mean_col])

    # Main effect statement
    if anova_p is None or np.isnan(anova_p):
        bullets.append(f"Overall effect of {cfg.factor_label} could not be determined (p-value unavailable).")
    else:
        if anova_p < cfg.alpha:
            bullets.append(f"{cfg.factor_label} had a statistically significant effect on {cfg.trait_name} (p={_fmt(anova_p,3)}).")
        else:
            bullets.append(f"No statistically significant differences were detected among {cfg.factor_label} levels for {cfg.trait_name} (p={_fmt(anova_p,3)}).")

    # Best performer + % change vs control (if provided)
    if cfg.control_level and cfg.control_level in set(mt[group_col].astype(str)):
        ctrl_mean = float(mt[mt[group_col].astype(str) == cfg.control_level].iloc[0][mean_col])
        pct = _pct_change(best_mean, ctrl_mean)
        if pct is not None:
            direction = "increased" if pct > 0 else "reduced"
            bullets.append(f"{best_name} recorded the best mean ({_fmt(best_mean)}{cfg.unit}) and {direction} {cfg.trait_name} by {_fmt(abs(pct),1)}% relative to {cfg.control_level}.")
        else:
            bullets.append(f"{best_name} recorded the best mean ({_fmt(best_mean)}{cfg.unit}).")
    else:
        bullets.append(f"{best_name} recorded the best mean ({_fmt(best_mean)}{cfg.unit}), while {worst_name} had the lowest mean ({_fmt(worst_mean)}{cfg.unit}).")

    # Tukey lettering narrative
    if tukey_group_col and tukey_group_col in mt.columns:
        best_letter = str(best.get(tukey_group_col, "")).strip()
        if best_letter:
            same = mt[mt[tukey_group_col].astype(str) == best_letter]
            if len(same) > 1:
                bullets.append(f"Mean separation indicates {best_name} is statistically similar to {', '.join(map(str, same[group_col].tolist()[:5]))}{'...' if len(same)>5 else ''} at α={cfg.alpha}.")
            else:
                bullets.append(f"Mean separation indicates {best_name} is significantly superior to most other levels at α={cfg.alpha}.")
        else:
            bullets.append("Mean separation lettering was not available for the top-performing level.")

    # Recommendations
    if anova_p is not None and not np.isnan(anova_p) and anova_p < cfg.alpha:
        recs.append(f"Report the ANOVA as significant and present Tukey groupings; highlight {best_name} as the top performer for {cfg.trait_name}.")
    else:
        recs.append("Differences were not significant. Consider increasing replication, improving blocking/control of variability, or testing across additional environments.")
    if sev in ("warn", "critical"):
        recs.append("High variability suggests improving experimental control (blocking, uniform management) or considering transformation/robust analysis.")
    if any(w["severity"] in ("warn", "critical") for w in warnings_list):
        recs.append("Include assumption diagnostics in your report and justify any remedial steps taken (e.g., transformation, robust methods).")

    sig_text = "was significant" if (anova_p is not None and not np.isnan(anova_p) and anova_p < cfg.alpha) else "was not significant"
    paragraph = (
        f"The effect of {cfg.factor_label} on {cfg.trait_name} {sig_text} (α={cfg.alpha}). "
        f"The highest mean {cfg.trait_name} was observed under {best_name} ({_fmt(best_mean)}{cfg.unit}), "
        f"while {worst_name} recorded the lowest mean ({_fmt(worst_mean)}{cfg.unit}). "
        f"{cv_msg}"
    )
    key_warns = [w for w in warnings_list if w["severity"] in ("warn", "critical")]
    if key_warns:
        paragraph += " Key diagnostic note(s): " + " ".join([w["message"] for w in key_warns[:2]])

    return {
        "summary_bullets": bullets,
        "supervisor_ready_paragraph": paragraph,
        "warnings": warnings_list,
        "recommendations": recs
    }

# ----------------------------
# Endpoints
# ----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/anova/oneway")
async def anova_oneway(req: AnovaRequest, file: UploadFile = File(...)) -> Dict[str, Any]:
    # Read CSV
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Validate columns
    if req.response_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"response_col '{req.response_col}' not found in CSV.")
    if req.factor_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"factor_col '{req.factor_col}' not found in CSV.")

    # Coerce response to numeric
    df = df.copy()
    df[req.response_col] = pd.to_numeric(df[req.response_col], errors="coerce")
    df[req.factor_col] = df[req.factor_col].astype(str)

    missing_count = int(df[req.response_col].isna().sum())

    # Remove rows with missing response for modeling
    dfit = df.dropna(subset=[req.response_col, req.factor_col]).copy()
    if dfit.empty:
        raise HTTPException(status_code=400, detail="No valid rows after removing missing response/factor values.")

    # Fit one-way ANOVA
    formula = f"`{req.response_col}` ~ C(`{req.factor_col}`)"
    model = ols(formula, data=dfit).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)

    # Extract p-value for factor
    factor_term = f"C(`{req.factor_col}`)"
    anova_p = float(anova_tbl.loc[factor_term, "PR(>F)"]) if factor_term in anova_tbl.index else None

    # MSE for CV
    mse = float(anova_tbl.loc["Residual", "sum_sq"] / anova_tbl.loc["Residual", "df"]) if "Residual" in anova_tbl.index else np.nan
    overall_mean = float(dfit[req.response_col].mean())
    cv_value = compute_cv(mse, overall_mean)

    # Assumptions
    resid = model.resid.values
    shapiro_p = None
    if len(resid) >= 3 and len(resid) <= 5000:  # scipy Shapiro recommended bounds
        shapiro_p = float(stats.shapiro(resid).pvalue)

    # Levene on groups
    levene_p = None
    groups = [g[req.response_col].values for _, g in dfit.groupby(req.factor_col)]
    if len(groups) >= 2:
        levene_p = float(stats.levene(*groups, center="median").pvalue)

    # Means + Tukey letters
    means_df = tukey_letters(dfit, req.factor_col, req.response_col, req.alpha)
    means_df.rename(columns={req.factor_col: "level"}, inplace=True)

    # Plots
    # Mean plot with SE bars
    fig1 = plt.figure()
    x = np.arange(len(means_df))
    plt.errorbar(x, means_df["mean"].values, yerr=means_df["se"].values, fmt="o", capsize=4)
    plt.xticks(x, means_df["level"].tolist(), rotation=45, ha="right")
    plt.xlabel(req.factor_col)
    plt.ylabel(req.trait_name or req.response_col)
    plt.title("Means (±SE)")
    mean_plot_b64 = fig_to_base64(fig1)

    # Box plot
    fig2 = plt.figure()
    dfit.boxplot(column=req.response_col, by=req.factor_col, grid=False)
    plt.title("Boxplot by group")
    plt.suptitle("")
    plt.xlabel(req.factor_col)
    plt.ylabel(req.trait_name or req.response_col)
    box_plot_b64 = fig_to_base64(fig2)

    # Build interpretation
    interp = generate_interpretation(
        anova_p=anova_p,
        means_table=means_df.rename(columns={"level": req.factor_col}),
        group_col=req.factor_col,
        mean_col="mean",
        tukey_group_col="group" if "group" in means_df.columns else None,
        cv=cv_value,
        shapiro_p=shapiro_p,
        levene_p=levene_p,
        npg=n_per_group(dfit, req.factor_col, req.response_col),
        missing_count=missing_count,
        cfg=InterpretationConfig(
            trait_name=req.trait_name or req.response_col,
            unit=req.unit or "",
            higher_is_better=req.higher_is_better,
            factor_label=req.factor_col,
            control_level=req.control_level,
            alpha=req.alpha
        )
    )

    # Return
    return {
        "meta": {
            "analysis": "one-way anova",
            "response_col": req.response_col,
            "factor_col": req.factor_col,
            "alpha": req.alpha,
            "n_rows_used": int(dfit.shape[0]),
            "missing_response_count": missing_count,
        },
        "anova_table": anova_tbl.reset_index().rename(columns={"index": "term"}).to_dict(orient="records"),
        "cv_percent": cv_value,
        "assumptions": {
            "shapiro_p": shapiro_p,
            "levene_p": levene_p,
        },
        "means": means_df.to_dict(orient="records"),
        "plots": {
            "means_plot_base64_png": mean_plot_b64,
            "boxplot_base64_png": box_plot_b64,
        },
        "interpretation": interp,
    }
