"""
VivaSense V1 - One-way ANOVA backend (FastAPI)

V1 Features included in this single main.py:
- Upload CSV
- One-way ANOVA for CRD and RCBD (block/rep)
- Tukey HSD + compact letter display (a, b, c...)
- Descriptive statistics (overall, by group, by block if RCBD)
- CV, assumption checks (Shapiro, Levene)
- Mean plot (SE bars) + boxplot
- Agricultural expert interpretation embedded in JSON
- Multi-trait analysis endpoint (run many traits at once)

Endpoints:
- GET  /           (root)
- GET  /health
- POST /anova/oneway                 (single trait)
- POST /anova/oneway/multitrait      (many traits)
"""

from __future__ import annotations

import io
import base64
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple

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

# =========================
# APP SETUP
# =========================

app = FastAPI(title="VivaSense V1", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# BASIC ROUTES
# =========================

@app.get("/")
def root():
    return {
        "name": "VivaSense Backend",
        "status": "ok",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# UTILITIES
# =========================

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

def descriptive_stats(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    n = int(s.notna().sum())
    miss = int(s.isna().sum())
    if n == 0:
        return {"n": 0, "missing": miss}

    mean = float(s.mean())
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    se = float(sd / np.sqrt(n)) if n > 0 else None
    cv = float((sd / mean) * 100.0) if mean not in (0, None) and not np.isnan(mean) else None

    return {
        "n": n,
        "missing": miss,
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": mean,
        "sd": sd,
        "se": se,
        "cv_percent": cv,
        "median": float(s.median()),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
    }

def descriptive_by_group(df: pd.DataFrame, group_col: str, value_col: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for level, g in df.groupby(group_col):
        d = descriptive_stats(g[value_col])
        d[group_col] = str(level)
        out.append(d)
    out.sort(key=lambda r: (r.get("mean") is None, -(r.get("mean") or -1e18)))
    return out


# =========================
# TUKEY LETTERS (COMPACT LETTER DISPLAY)
# =========================

def tukey_letters(df: pd.DataFrame, factor_col: str, response_col: str, alpha: float) -> pd.DataFrame:
    """
    Returns means table with Tukey group letters using a simple, deterministic algorithm.
    Columns: factor_col, n, mean, sd, se, group
    """
    d = df[[factor_col, response_col]].dropna()

    summ = d.groupby(factor_col)[response_col].agg(["count", "mean", "std"]).reset_index()
    summ.rename(columns={"count": "n", "std": "sd"}, inplace=True)
    summ["se"] = summ["sd"] / np.sqrt(summ["n"].clip(lower=1))

    # Tukey test
    tuk = pairwise_tukeyhsd(endog=d[response_col].values, groups=d[factor_col].astype(str).values, alpha=alpha)
    tdf = pd.DataFrame(data=tuk._results_table.data[1:], columns=tuk._results_table.data[0])

    levels = summ[factor_col].astype(str).tolist()
    means = {str(r[factor_col]): float(r["mean"]) for _, r in summ.iterrows()}
    levels_sorted = sorted(levels, key=lambda x: means[x], reverse=True)

    sig = {a: {b: False for b in levels_sorted} for a in levels_sorted}
    for _, r in tdf.iterrows():
        a, b = str(r["group1"]), str(r["group2"])
        rej = bool(r["reject"])
        sig[a][b] = rej
        sig[b][a] = rej

    letters: Dict[str, str] = {}
    letter_list = list("abcdefghijklmnopqrstuvwxyz")
    used = 0
    groups_for_letter: List[List[str]] = []

    for lvl in levels_sorted:
        placed = False
        for gi, members in enumerate(groups_for_letter):
            if all(sig[lvl][m] is False for m in members):
                members.append(lvl)
                letters[lvl] = letter_list[gi] if gi < len(letter_list) else f"g{gi+1}"
                placed = True
                break
        if not placed:
            letters[lvl] = letter_list[used] if used < len(letter_list) else f"g{used+1}"
            groups_for_letter.append([lvl])
            used += 1

    out = summ.copy()
    out["group"] = out[factor_col].astype(str).map(letters)
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


# =========================
# INTERPRETATION ENGINE
# =========================

@dataclass
class InterpretationConfig:
    trait_name: str = "Trait"
    unit: str = ""
    higher_is_better: bool = True
    factor_label: str = "Treatment"
    control_level: Optional[str] = None
    alpha: float = 0.05
    design: str = "CRD"
    block_label: Optional[str] = None

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
    cfg: InterpretationConfig,
    block_p: Optional[float] = None,
) -> Dict[str, Any]:
    warnings_list: List[Dict[str, Any]] = []
    bullets: List[str] = []
    recs: List[str] = []

    sev, cv_msg = _cv_comment(cv)
    warnings_list.append({"severity": sev, "message": cv_msg})
    warnings_list.extend(_assumption_comment(shapiro_p, levene_p, cfg.alpha))
    warnings_list.extend(_design_health_comment(npg, missing_count))

    # Design bullet
    if cfg.design == "RCBD":
        bullets.append(f"Design: RCBD (blocking by {cfg.block_label or 'block'}).")
        if block_p is not None and not np.isnan(block_p):
            if block_p < cfg.alpha:
                bullets.append(f"Block effect is significant (p={_fmt(block_p,3)}), suggesting field variability existed and blocking improved precision.")
            else:
                bullets.append(f"Block effect is not significant (p={_fmt(block_p,3)}); field variability across blocks may be low, but RCBD remains acceptable.")
    else:
        bullets.append("Design: CRD (no blocking).")

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

    # Main effect
    if anova_p is None or np.isnan(anova_p):
        bullets.append(f"Overall effect of {cfg.factor_label} could not be determined (p-value unavailable).")
    else:
        if anova_p < cfg.alpha:
            bullets.append(f"{cfg.factor_label} had a statistically significant effect on {cfg.trait_name} (p={_fmt(anova_p,3)}).")
        else:
            bullets.append(f"No statistically significant differences were detected among {cfg.factor_label} levels for {cfg.trait_name} (p={_fmt(anova_p,3)}).")

    # Best performer
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

    # Tukey narrative
    if tukey_group_col and tukey_group_col in mt.columns:
        best_letter = str(best.get(tukey_group_col, "")).strip()
        if best_letter:
            same = mt[mt[tukey_group_col].astype(str) == best_letter]
            if len(same) > 1:
                bullets.append(
                    f"Mean separation indicates {best_name} is statistically similar to "
                    f"{', '.join(map(str, same[group_col].tolist()[:5]))}{'...' if len(same)>5 else ''} at α={cfg.alpha}."
                )
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


# =========================
# CORE ENGINE: RUN ONE TRAIT (REUSABLE)
# =========================

def run_oneway_single_trait(
    df: pd.DataFrame,
    response_col: str,
    factor_col: str,
    design: str,
    block_col: Optional[str],
    trait_name: str,
    unit: str,
    higher_is_better: bool,
    control_level: Optional[str],
    alpha: float,
) -> Dict[str, Any]:
    d = df.copy()
    d[response_col] = pd.to_numeric(d[response_col], errors="coerce")
    d[factor_col] = d[factor_col].astype(str)
    if design == "RCBD" and block_col:
        d[block_col] = d[block_col].astype(str)

    missing_count = int(d[response_col].isna().sum())
    needed = [response_col, factor_col] + ([block_col] if design == "RCBD" and block_col else [])
    dfit = d.dropna(subset=needed).copy()
    if dfit.empty:
        raise ValueError("No valid rows after removing missing values for this trait.")

    # Model
    if design == "CRD":
        formula = f"`{response_col}` ~ C(`{factor_col}`)"
    else:
        formula = f"`{response_col}` ~ C(`{factor_col}`) + C(`{block_col}`)"

    model = ols(formula, data=dfit).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)

    factor_term = f"C(`{factor_col}`)"
    anova_p = float(anova_tbl.loc[factor_term, "PR(>F)"]) if factor_term in anova_tbl.index else None

    block_p = None
    if design == "RCBD" and block_col:
        block_term = f"C(`{block_col}`)"
        if block_term in anova_tbl.index:
            block_p = float(anova_tbl.loc[block_term, "PR(>F)"])

    mse = float(anova_tbl.loc["Residual", "sum_sq"] / anova_tbl.loc["Residual", "df"]) if "Residual" in anova_tbl.index else np.nan
    overall_mean = float(dfit[response_col].mean())
    cv_value = compute_cv(mse, overall_mean)

    # Assumptions
    resid = model.resid.values
    shapiro_p = None
    if 3 <= len(resid) <= 5000:
        shapiro_p = float(stats.shapiro(resid).pvalue)

    levene_p = None
    groups = [g[response_col].values for _, g in dfit.groupby(factor_col)]
    if len(groups) >= 2:
        levene_p = float(stats.levene(*groups, center="median").pvalue)

    # Means + Tukey
    means_df = tukey_letters(dfit, factor_col, response_col, alpha)
    means_df.rename(columns={factor_col: "level"}, inplace=True)

    # Descriptives
    overall_desc = descriptive_stats(dfit[response_col])
    by_group_desc = descriptive_by_group(dfit, factor_col, response_col)
    by_block_desc = descriptive_by_group(dfit, block_col, response_col) if design == "RCBD" and block_col else None

    # Plots
    fig1 = plt.figure()
    x = np.arange(len(means_df))
    plt.errorbar(x, means_df["mean"].values, yerr=means_df["se"].values, fmt="o", capsize=4)
    plt.xticks(x, means_df["level"].tolist(), rotation=45, ha="right")
    plt.xlabel(factor_col)
    plt.ylabel(trait_name or response_col)
    plt.title(f"{trait_name or response_col}: Means (±SE)")
    mean_plot_b64 = fig_to_base64(fig1)

    fig2 = plt.figure()
    dfit.boxplot(column=response_col, by=factor_col, grid=False)
    plt.title(f"{trait_name or response_col}: Boxplot by group")
    plt.suptitle("")
    plt.xlabel(factor_col)
    plt.ylabel(trait_name or response_col)
    box_plot_b64 = fig_to_base64(fig2)

    # Interpretation
    interp = generate_interpretation(
        anova_p=anova_p,
        means_table=means_df.rename(columns={"level": factor_col}),
        group_col=factor_col,
        mean_col="mean",
        tukey_group_col="group" if "group" in means_df.columns else None,
        cv=cv_value,
        shapiro_p=shapiro_p,
        levene_p=levene_p,
        npg=n_per_group(dfit, factor_col, response_col),
        missing_count=missing_count,
        cfg=InterpretationConfig(
            trait_name=trait_name or response_col,
            unit=unit or "",
            higher_is_better=higher_is_better,
            factor_label=factor_col,
            control_level=control_level,
            alpha=alpha,
            design=design,
            block_label=block_col if design == "RCBD" else None,
        ),
        block_p=block_p
    )

    return {
        "meta": {
            "trait": response_col,
            "trait_name": trait_name or response_col,
            "n_rows_used": int(dfit.shape[0]),
            "missing_response_count": missing_count,
        },
        "descriptives": {
            "overall": overall_desc,
            "by_group": by_group_desc,
            "by_block": by_block_desc
        },
        "anova_table": anova_tbl.reset_index().rename(columns={"index": "term"}).to_dict(orient="records"),
        "p_values": {"factor_p": anova_p, "block_p": block_p},
        "cv_percent": cv_value,
        "assumptions": {"shapiro_p": shapiro_p, "levene_p": levene_p},
        "means": means_df.to_dict(orient="records"),
        "plots": {
            "means_plot_base64_png": mean_plot_b64,
            "boxplot_base64_png": box_plot_b64
        },
        "interpretation": interp
    }


# =========================
# ENDPOINTS
# =========================

@app.post("/anova/oneway")
async def anova_oneway(
    file: UploadFile = File(...),
    response_col: str = Form(...),
    factor_col: str = Form(...),

    design: str = Form("CRD"),   # "CRD" or "RCBD"
    block_col: str = Form(""),   # required if RCBD

    trait_name: str = Form(""),
    unit: str = Form(""),
    higher_is_better: bool = Form(True),
    control_level: str = Form(""),
    alpha: float = Form(0.05),
) -> Dict[str, Any]:

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    design_norm = (design or "CRD").strip().upper()
    if design_norm not in ("CRD", "RCBD"):
        raise HTTPException(status_code=400, detail="design must be 'CRD' or 'RCBD'.")

    # Validate columns
    if response_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"response_col '{response_col}' not found in CSV.")
    if factor_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"factor_col '{factor_col}' not found in CSV.")

    block_col_norm = block_col.strip()
    if design_norm == "RCBD":
        if not block_col_norm:
            raise HTTPException(status_code=400, detail="block_col is required for RCBD.")
        if block_col_norm not in df.columns:
            raise HTTPException(status_code=400, detail=f"block_col '{block_col_norm}' not found in CSV.")

    out = run_oneway_single_trait(
        df=df,
        response_col=response_col,
        factor_col=factor_col,
        design=design_norm,
        block_col=block_col_norm if design_norm == "RCBD" else None,
        trait_name=trait_name or response_col,
        unit=unit or "",
        higher_is_better=higher_is_better,
        control_level=control_level.strip() or None,
        alpha=alpha,
    )

    return {
        "meta": {
            "analysis": "one-way anova",
            "design": design_norm,
            "response_col": response_col,
            "factor_col": factor_col,
            "block_col": block_col_norm if design_norm == "RCBD" else None,
            "alpha": alpha,
        },
        "result": out
    }


@app.post("/anova/oneway/multitrait")
async def anova_oneway_multitrait(
    file: UploadFile = File(...),

    factor_col: str = Form(...),
    traits: str = Form(...),  # comma-separated: "yield,height,days_to_flower"

    design: str = Form("CRD"),
    block_col: str = Form(""),

    alpha: float = Form(0.05),
    higher_is_better: bool = Form(True),
) -> Dict[str, Any]:

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    design_norm = (design or "CRD").strip().upper()
    if design_norm not in ("CRD", "RCBD"):
        raise HTTPException(status_code=400, detail="design must be 'CRD' or 'RCBD'.")

    if factor_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"factor_col '{factor_col}' not found in CSV.")

    block_col_norm = block_col.strip()
    if design_norm == "RCBD":
        if not block_col_norm:
            raise HTTPException(status_code=400, detail="block_col is required for RCBD.")
        if block_col_norm not in df.columns:
            raise HTTPException(status_code=400, detail=f"block_col '{block_col_norm}' not found in CSV.")

    trait_list = [t.strip() for t in traits.split(",") if t.strip()]
    if not trait_list:
        raise HTTPException(status_code=400, detail="traits must be a comma-separated list of column names.")

    missing_traits = [t for t in trait_list if t not in df.columns]
    if missing_traits:
        raise HTTPException(status_code=400, detail=f"Trait column(s) not found: {missing_traits}")

    results: Dict[str, Any] = {}
    summary: List[Dict[str, Any]] = []

    for t in trait_list:
        try:
            out = run_oneway_single_trait(
                df=df,
                response_col=t,
                factor_col=factor_col,
                design=design_norm,
                block_col=block_col_norm if design_norm == "RCBD" else None,
                trait_name=t,
                unit="",
                higher_is_better=higher_is_better,
                control_level=None,
                alpha=alpha,
            )
            results[t] = out

            fp = out.get("p_values", {}).get("factor_p", None)
            cv = out.get("cv_percent", None)
            top = None
            means = out.get("means", [])
            if means:
                top = means[0].get("level")

            summary.append({
                "trait": t,
                "factor_p": fp,
                "cv_percent": cv,
                "top_level": top,
                "status": "ok"
            })
        except Exception as e:
            results[t] = {"error": str(e)}
            summary.append({
                "trait": t,
                "status": "error",
                "error": str(e)
            })

    return {
        "meta": {
            "analysis": "one-way anova multitrait",
            "design": design_norm,
            "factor_col": factor_col,
            "block_col": block_col_norm if design_norm == "RCBD" else None,
            "alpha": alpha,
            "n_traits_requested": len(trait_list)
        },
        "summary": summary,
        "results": results
    }
