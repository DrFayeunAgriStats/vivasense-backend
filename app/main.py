"""
VivaSense V2 + FIA AI Proxy - app/main.py
==========================================
V2 improvements over V1:
  - Number rounding (all floats to 4dp, CV to 2dp)
  - RCBD one-way ANOVA added (most common Nigerian field trial design)
  - Assumption guidance: plain-English verdict + specific alternative test recommendation
  - Better error messages with column names shown
  - Improved plot styling (larger, cleaner)
  - User context fields in /api/interpret (crop, treatment, objective, location)
  - HEAD method support on root and health endpoints (fixes UptimeRobot)
  - Significance stars (*, **, ***) in interpretation text

Environment variables required (set in Render dashboard):
  ANTHROPIC_API_KEY = sk-ant-api03-...

Analysis endpoints:
  GET  /                               - root
  GET  /health                         - health check
  POST /analyze/descriptive            - descriptive statistics
  POST /analyze/anova/oneway           - one-way ANOVA (CRD)
  POST /analyze/anova/oneway_rcbd      - one-way ANOVA (RCBD) [NEW in V2]
  POST /analyze/anova/twoway           - two-way ANOVA (CRD factorial)
  POST /analyze/anova/rcbd_factorial   - factorial in RCBD
  POST /analyze/anova/splitplot        - split-plot ANOVA

AI proxy endpoints:
  GET  /api/health                     - AI proxy health check
  POST /api/chat                       - CSP 811 Dr. Fayeun tutor chat
  POST /api/interpret                  - VivaSense AI interpretation (with user context)
  POST /api/followup                   - VivaSense post-analysis follow-up chat
"""

from __future__ import annotations

import io
import os
import json
import base64
import warnings
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import httpx

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")


# ============================================================
#  APP SETUP
# ============================================================

app = FastAPI(title="VivaSense V2 + FIA AI Proxy", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fieldtoinsightacademy.com.ng",
        "https://www.fieldtoinsightacademy.com.ng",
        "http://localhost:3000",
        "http://localhost:5173",
        "*",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)


# ============================================================
#  ANTHROPIC CONFIG
# ============================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-20250514"

ANTHROPIC_HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
}

request_counts: dict = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 20


def is_rate_limited(ip: str) -> bool:
    now = datetime.now()
    cutoff = now - timedelta(minutes=1)
    request_counts[ip] = [t for t in request_counts[ip] if t > cutoff]
    if len(request_counts[ip]) >= MAX_REQUESTS_PER_MINUTE:
        return True
    request_counts[ip].append(now)
    return False


# ============================================================
#  AI SYSTEM PROMPTS
# ============================================================

CSP811_SYSTEM = """You are Dr. Fayeun, AI tutor for CSP 811 Biometrical Genetics at FUTA Nigeria, created by Prof. Lawrence Stephen Fayeun of the Field-to-Insight Academy (www.fieldtoinsightacademy.com.ng).

STYLE: Warm, precise, step-by-step. Use Nigerian crops (cassava, cowpea, maize, sorghum) as examples. Show formula, explain symbols, then worked example. End longer answers with "- Dr. Fayeun". Invite follow-up.

EXAM TIPS (share when relevant): State assumptions. Show working. Include units. Check h2 is less than or equal to 1. Interpret results for breeding decisions.

CORE CONTENT COVERS: variance components, heritability estimation, generation mean analysis, diallel analysis, ANOVA experimental designs, regression and path analysis, GxE interaction, stability analysis (AMMI/WAASB), selection indices (Smith-Hazel), machine learning in plant breeding (rrBLUP, GBLUP, LASSO, Random Forest, Bayesian methods).

For R code questions: provide complete runnable code with comments.
For ANOVA questions: always present the full table.
Link statistical results to breeding decisions."""


VIVASENSE_INTERPRET_SYSTEM = """You are Dr. Fayeun, Professor of Quantitative Genetics at FUTA and founder of Field-to-Insight Academy. You are interpreting statistical analysis results for a researcher who used VivaSense.

Write exactly as you would in a peer-reviewed journal article or when supervising a postgraduate student's thesis.

STRUCTURE YOUR RESPONSE AS:
1. Overall Finding - one clear sentence stating the key result with the most important statistic
2. Statistical Evidence - reference specific F-value, p-value, means, and CV from the results
3. Treatment/Genotype Comparisons - interpret the Tukey grouping letters specifically, name the best and worst performers
4. Assumptions Check - state clearly whether normality and homogeneity were met; if violated, name the specific alternative test and why
5. Research/Breeding Recommendation - actionable conclusion for this specific researcher or breeder
6. Thesis/Paper Sentence - one publication-ready sentence they can copy directly into their write-up

PERSONALISATION RULES (critical):
- If crop is provided: use it specifically throughout ("among the cowpea genotypes", "for sugarcane Brix content")
- If treatment description is provided: reference it meaningfully ("nitrogen fertilizer rates", "irrigation regimes")
- If study objective is provided: align the recommendation directly to that objective
- If location is provided: contextualise findings to that environment
- Always reference ACTUAL numbers from the results provided - never speak generally
- Keep language accessible to final-year undergraduates and postgraduate researchers
- End with: "- Dr. Fayeun, VivaSense"

You are what separates a confused student from a confident researcher."""


VIVASENSE_FOLLOWUP_SYSTEM = """You are Dr. Fayeun, Professor of Quantitative Genetics at FUTA. A researcher has run a statistical analysis on VivaSense and you have already provided an interpretation. They are now asking follow-up questions.

You have full knowledge of their analysis results and study context provided below. Answer their specific question directly and concisely.

- If they ask about thesis writing: give exact, publication-ready sentences
- If they ask about assumption violations: give specific, actionable advice naming the correct test
- If they ask why a result is unexpected: reason through it using their actual numbers
- If they ask about the best treatment: reference the specific means and Tukey letters
- If they ask for R code: provide complete runnable code for their specific analysis

Always ground your answer in THEIR specific numbers and context.
End with "- Dr. Fayeun" on detailed answers."""


# ============================================================
#  AI REQUEST MODELS
# ============================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    topic: Optional[str] = None
    stream: Optional[bool] = True


class InterpretRequest(BaseModel):
    analysis_type: str
    results: dict
    crop: Optional[str] = None
    treatment_description: Optional[str] = None
    study_objective: Optional[str] = None
    location: Optional[str] = None
    additional_context: Optional[str] = None
    stream: Optional[bool] = True


class FollowupRequest(BaseModel):
    messages: List[Message]
    analysis_results: dict
    user_context: Optional[dict] = None
    stream: Optional[bool] = True


# ============================================================
#  AI HELPERS
# ============================================================

async def stream_anthropic(
    system: str,
    messages: list,
    model: str = HAIKU_MODEL,
    max_tokens: int = 1024
):
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "stream": True,
    }

    async def generate():
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", ANTHROPIC_URL,
                headers=ANTHROPIC_HEADERS,
                json=payload
            ) as response:
                if response.status_code != 200:
                    error = await response.aread()
                    yield f"data: {json.dumps({'error': error.decode()})}\n\n"
                    return
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


async def call_anthropic(
    system: str,
    messages: list,
    model: str = HAIKU_MODEL,
    max_tokens: int = 1024
) -> str:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            ANTHROPIC_URL, headers=ANTHROPIC_HEADERS, json=payload
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.json().get("error", {}).get("message", "API error")
            )
        data = response.json()
        return data["content"][0]["text"]


# ============================================================
#  AI ENDPOINTS
# ============================================================

@app.api_route("/api/health", methods=["GET", "HEAD"])
async def api_health():
    return {
        "status": "ok",
        "service": "FIA AI Proxy",
        "version": "2.0.0",
        "api_key_configured": bool(ANTHROPIC_API_KEY),
        "endpoints": ["/api/chat", "/api/interpret", "/api/followup"],
    }


@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    client_ip = request.client.host
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not configured.")

    system = CSP811_SYSTEM
    if body.topic:
        system += f"\n\n[Student is currently studying: {body.topic}]"

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream:
        return await stream_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=1024)
    else:
        text = await call_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=1024)
        return {"content": text}


@app.post("/api/interpret")
async def interpret(request: Request, body: InterpretRequest):
    client_ip = request.client.host
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not configured.")

    context_lines = []
    if body.crop:
        context_lines.append(f"Crop: {body.crop}")
    if body.treatment_description:
        context_lines.append(f"Treatment description: {body.treatment_description}")
    if body.study_objective:
        context_lines.append(f"Study objective: {body.study_objective}")
    if body.location:
        context_lines.append(f"Study location: {body.location}")
    if body.additional_context:
        context_lines.append(f"Additional context: {body.additional_context}")

    context_block = ""
    if context_lines:
        context_block = "RESEARCHER'S STUDY CONTEXT:\n" + "\n".join(context_lines) + "\n\n"

    results_text = json.dumps(body.results, indent=2)

    user_message = f"""Please interpret these {body.analysis_type} results for the researcher.

{context_block}Statistical Results:
{results_text}

Provide a complete, personalised interpretation using the study context provided above."""

    messages = [{"role": "user", "content": user_message}]

    if body.stream:
        return await stream_anthropic(
            VIVASENSE_INTERPRET_SYSTEM, messages,
            model=SONNET_MODEL, max_tokens=1500
        )
    else:
        text = await call_anthropic(
            VIVASENSE_INTERPRET_SYSTEM, messages,
            model=SONNET_MODEL, max_tokens=1500
        )
        return {"interpretation": text}


@app.post("/api/followup")
async def followup(request: Request, body: FollowupRequest):
    client_ip = request.client.host
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not configured.")

    results_context = json.dumps(body.analysis_results, indent=2)
    context_block = ""
    if body.user_context:
        context_block = "\n\nSTUDY CONTEXT:\n" + json.dumps(body.user_context, indent=2)

    system = (
        VIVASENSE_FOLLOWUP_SYSTEM
        + f"\n\nANALYSIS RESULTS:\n{results_context}"
        + context_block
    )

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream:
        return await stream_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=1000)
    else:
        text = await call_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=1000)
        return {"content": text}


# ============================================================
#  ROOT + HEALTH (HEAD method added for UptimeRobot)
# ============================================================

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"name": "VivaSense V2", "status": "ok", "docs": "/docs", "version": "2.0.0"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "healthy", "version": "2.0.0"}


# ============================================================
#  NUMBER FORMATTING HELPERS
# ============================================================

def round_val(v, decimals: int = 4):
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, decimals)
    except (TypeError, ValueError):
        return v


def fmt_p(p) -> Optional[float]:
    if p is None:
        return None
    f = float(p)
    if np.isnan(f):
        return None
    return round(f, 4)


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].apply(lambda x: round_val(x, 4))
    return df.replace({np.nan: None}).to_dict(orient="records")


# ============================================================
#  VIVASENSE STATISTICAL HELPERS
# ============================================================

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
    df.columns = [str(c).strip() for c in df.columns]
    return df


def require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing column(s): {missing}. Your file contains: {list(df.columns)}",
        )


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_for_model(df: pd.DataFrame, y: str, x_cols: List[str]) -> pd.DataFrame:
    d = df.copy()
    require_cols(d, [y] + x_cols)
    d[y] = coerce_numeric(d[y])
    for c in x_cols:
        d[c] = d[c].astype(str).str.strip()
    d = d.dropna(subset=[y] + x_cols)
    if d.shape[0] < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Only {d.shape[0]} valid rows found for '{y}'. Minimum 3 required."
        )
    if len(x_cols) == 1 and d[x_cols[0]].nunique(dropna=True) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{x_cols[0]}' has only 1 unique value. Need at least 2 levels for ANOVA."
        )
    return d


def cv_percent(y: pd.Series) -> Optional[float]:
    y = pd.to_numeric(y, errors="coerce")
    m = float(np.nanmean(y))
    if np.isnan(m) or m == 0:
        return None
    if y.dropna().shape[0] < 2:
        return None
    s = float(np.nanstd(y, ddof=1))
    if np.isnan(s):
        return None
    return round(float((s / m) * 100.0), 2)


def shapiro_test(resid: np.ndarray) -> Dict[str, Any]:
    resid = np.asarray(resid)
    if resid.size < 3:
        return {"test": "Shapiro-Wilk", "stat": None, "p_value": None,
                "passed": None, "verdict": "Not enough residuals to test."}
    stat, p = stats.shapiro(resid[:5000])
    passed = bool(float(p) >= 0.05)
    return {
        "test": "Shapiro-Wilk",
        "stat": round_val(stat, 4),
        "p_value": round_val(p, 4),
        "passed": passed,
        "verdict": "Normality met." if passed
                   else "Normality violated (p<0.05).",
    }


def levene_test(df: pd.DataFrame, y: str, group: str) -> Dict[str, Any]:
    groups = []
    for _, g in df.groupby(group):
        vals = g[y].dropna().values
        if len(vals) > 0:
            groups.append(vals)
    if len(groups) < 2:
        return {"test": "Levene", "stat": None, "p_value": None,
                "passed": None, "verdict": "Need at least 2 groups."}
    stat, p = stats.levene(*groups, center="median")
    passed = bool(float(p) >= 0.05)
    return {
        "test": "Levene",
        "stat": round_val(stat, 4),
        "p_value": round_val(p, 4),
        "passed": passed,
        "verdict": "Equal variances met." if passed
                   else "Equal variances violated (p<0.05).",
    }


def assumption_guidance(
    shapiro: Dict, levene: Dict, design: str
) -> Dict[str, Any]:
    """Plain-English assumption verdict with specific alternative test recommendation."""
    s_pass = shapiro.get("passed")
    l_pass = levene.get("passed")

    alternatives = {
        "CRD one-way": "Kruskal-Wallis test",
        "RCBD one-way": "Friedman test",
        "CRD two-way (factorial)": "Log or square-root data transformation, then re-run ANOVA",
        "Factorial in RCBD": "Log or square-root data transformation, then re-run ANOVA",
        "Split-plot": "Log or square-root data transformation, then re-run ANOVA",
    }
    alt = alternatives.get(design, "Data transformation or non-parametric method")

    if s_pass is None or l_pass is None:
        return {"overall": "Assumptions could not be evaluated.", "alternative": None, "action": None}

    if s_pass and l_pass:
        return {
            "overall": "Both assumptions met. ANOVA results are valid and reliable.",
            "normality_ok": True,
            "homogeneity_ok": True,
            "alternative": None,
            "action": None,
        }
    elif not s_pass and l_pass:
        return {
            "overall": "Normality violated but equal variances met. ANOVA is moderately robust; results are likely valid. Verify with the recommended alternative.",
            "normality_ok": False,
            "homogeneity_ok": True,
            "alternative": alt,
            "action": f"Recommended: run {alt} to confirm your findings.",
        }
    elif s_pass and not l_pass:
        return {
            "overall": "Normality met but equal variances violated. This affects ANOVA validity more seriously.",
            "normality_ok": True,
            "homogeneity_ok": False,
            "alternative": alt,
            "action": f"Recommended: use {alt} as your primary analysis.",
        }
    else:
        return {
            "overall": "Both assumptions violated. ANOVA results should be treated with caution.",
            "normality_ok": False,
            "homogeneity_ok": False,
            "alternative": alt,
            "action": f"Recommended: use {alt} as your primary analysis.",
        }


def mean_table(df: pd.DataFrame, y: str, group: str) -> pd.DataFrame:
    g = df.groupby(group)[y]
    out = pd.DataFrame({
        group: g.mean().index.astype(str),
        "n": g.count().values,
        "mean": g.mean().values.round(4),
        "sd": g.std(ddof=1).values.round(4),
        "se": (g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))).values.round(4),
    })
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


def mean_plot(df: pd.DataFrame, y: str, group: str, title: str) -> str:
    mt = mean_table(df, y, group)
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    x = np.arange(len(mt))
    ax.errorbar(x, mt["mean"].values, yerr=mt["se"].values,
                fmt="o", capsize=5, color="#2E7D32", markersize=8,
                linewidth=1.8, elinewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(mt[group].astype(str).tolist(), rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(y, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _b64_png(fig)


def box_plot(df: pd.DataFrame, y: str, group: str, title: str) -> str:
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    levels = df[group].astype(str).unique().tolist()
    data = [df.loc[df[group].astype(str) == lvl, y].dropna().values for lvl in levels]
    bp = ax.boxplot(data, labels=levels, showmeans=True, patch_artist=True,
                    meanprops={"marker": "D", "markerfacecolor": "#E65100",
                               "markeredgecolor": "#E65100", "markersize": 6})
    for patch in bp["boxes"]:
        patch.set_facecolor("#E8F5E9")
        patch.set_alpha(0.8)
    ax.set_ylabel(y, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    return _b64_png(fig)


def compact_letter_display(
    groups: List[str],
    means: Dict[str, float],
    sig: Dict[Tuple[str, str], bool]
) -> Dict[str, str]:
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


def tukey_letters(
    df: pd.DataFrame, y: str, group: str, alpha: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mt = mean_table(df, y, group)
    if df[group].nunique() < 2:
        raise HTTPException(status_code=400,
                            detail=f"Need at least 2 levels in '{group}' for Tukey HSD.")
    if df[y].dropna().shape[0] < 3:
        raise HTTPException(status_code=400,
                            detail=f"Not enough observations in '{y}' for Tukey HSD.")

    res = pairwise_tukeyhsd(
        endog=df[y].values,
        groups=df[group].astype(str).values,
        alpha=alpha
    )
    tukey_df = pd.DataFrame(
        res._results_table.data[1:],
        columns=res._results_table.data[0]
    )

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


def sig_stars(p: Optional[float]) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def interpret_anova(
    title: str,
    p_map: Dict[str, Optional[float]],
    cv: Optional[float],
    alpha: float
) -> str:
    lines = [f"{title}:"]
    if cv is not None:
        qual = "good" if cv < 15 else ("acceptable" if cv < 25 else "high - check data quality")
        lines.append(f"- CV = {cv:.2f}% ({qual} experimental precision).")
    for term, p in p_map.items():
        if p is None:
            continue
        stars = sig_stars(p)
        if p < alpha:
            lines.append(f"- {term}: significant (p = {p:.4f}) {stars}")
        else:
            lines.append(f"- {term}: not significant (p = {p:.4f}) {stars}")
    lines.append("- See assumption guidance below for normality and equal variance checks.")
    return "\n".join(lines)


# ============================================================
#  ANALYSIS ENGINES
# ============================================================

def oneway_engine(df: pd.DataFrame, factor: str, trait: str, alpha: float) -> Dict[str, Any]:
    d = clean_for_model(df, trait, [factor])
    model = ols(f"{trait} ~ C({factor})", data=d).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})
    anova["ms"] = anova["sum_sq"] / anova["df"]

    sh = shapiro_test(model.resid)
    lv = levene_test(d, trait, factor)
    guidance = assumption_guidance(sh, lv, "CRD one-way")
    tukey_df, means_letters = tukey_letters(d, trait, factor, alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, factor, f"Means +/- SE by {factor}"),
        "box_plot": box_plot(d, trait, factor, f"Distribution of {trait} by {factor}"),
    }

    p_factor = None
    m = anova["source"].astype(str).str.contains(f"C\\({factor}\\)")
    if m.any():
        p_factor = fmt_p(anova.loc[m, "PR(>F)"].iloc[0])

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
            "assumption_guidance": guidance,
        },
        "plots": plots,
        "interpretation": interpret_anova(
            "One-way ANOVA (CRD)", {factor: p_factor}, cv_percent(d[trait]), alpha
        ),
    }


def oneway_rcbd_engine(
    df: pd.DataFrame, block: str, treatment: str, trait: str, alpha: float
) -> Dict[str, Any]:
    """One-way ANOVA in RCBD - most common Nigerian field trial design."""
    d = clean_for_model(df, trait, [block, treatment])

    model = ols(f"{trait} ~ C({block}) + C({treatment})", data=d).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})
    anova["ms"] = anova["sum_sq"] / anova["df"]

    sh = shapiro_test(model.resid)
    lv = levene_test(d, trait, treatment)
    guidance = assumption_guidance(sh, lv, "RCBD one-way")
    tukey_df, means_letters = tukey_letters(d, trait, treatment, alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, treatment, f"Means +/- SE by {treatment} (RCBD)"),
        "box_plot": box_plot(d, trait, treatment, f"Distribution of {trait} by {treatment} (RCBD)"),
    }

    def p_of(term: str) -> Optional[float]:
        m = anova["source"].astype(str).str.contains(term)
        return fmt_p(anova.loc[m, "PR(>F)"].iloc[0]) if m.any() else None

    p_block = p_of(f"C\\({block}\\)")
    p_treatment = p_of(f"C\\({treatment}\\)")

    resid_row = anova[anova["source"] == "Residual"]
    lsd = None
    if not resid_row.empty:
        ms_error = float(resid_row["ms"].iloc[0])
        df_error = float(resid_row["df"].iloc[0])
        r = d[block].nunique()
        t_crit = stats.t.ppf(1 - alpha / 2, df_error)
        lsd = round_val(t_crit * np.sqrt(2 * ms_error / r), 4)

    return {
        "meta": {
            "design": "RCBD one-way",
            "block": block,
            "treatment": treatment,
            "trait": trait,
            "alpha": alpha,
            "n_rows_used": int(d.shape[0]),
            "n_blocks": int(d[block].nunique()),
            "n_treatments": int(d[treatment].nunique()),
            "levels": sorted(d[treatment].unique().tolist()),
            "cv_percent": cv_percent(d[trait]),
            "lsd": lsd,
        },
        "tables": {
            "anova": df_to_records(anova[["source", "df", "sum_sq", "ms", "F", "PR(>F)"]]),
            "means": df_to_records(means_letters),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
            "assumption_guidance": guidance,
        },
        "plots": plots,
        "interpretation": interpret_anova(
            "One-way ANOVA (RCBD)",
            {f"Block ({block})": p_block, treatment: p_treatment},
            cv_percent(d[trait]),
            alpha
        ),
    }


def twoway_engine(
    df: pd.DataFrame, a: str, b: str, trait: str, alpha: float
) -> Dict[str, Any]:
    d = clean_for_model(df, trait, [a, b])
    d["_AB_"] = d[a].astype(str) + ":" + d[b].astype(str)

    model = ols(f"{trait} ~ C({a}) * C({b})", data=d).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})
    anova["ms"] = anova["sum_sq"] / anova["df"]

    sh = shapiro_test(model.resid)
    lv = levene_test(d, trait, "_AB_")
    guidance = assumption_guidance(sh, lv, "CRD two-way (factorial)")
    tukey_df, means_letters = tukey_letters(d, trait, "_AB_", alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, "_AB_", f"Means +/- SE by {a}:{b}"),
        "box_plot": box_plot(d, trait, "_AB_", f"Distribution of {trait} by {a}:{b}"),
    }

    def p_of(term: str) -> Optional[float]:
        m = anova["source"].astype(str).str.contains(term)
        return fmt_p(anova.loc[m, "PR(>F)"].iloc[0]) if m.any() else None

    p_map = {
        a: p_of(f"C\\({a}\\)"),
        b: p_of(f"C\\({b}\\)"),
        f"{a}x{b} interaction": p_of(":"),
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
            "assumption_guidance": guidance,
        },
        "plots": plots,
        "interpretation": interpret_anova(
            "Two-way Factorial ANOVA (CRD)", p_map, cv_percent(d[trait]), alpha
        ),
    }


def rcbd_factorial_engine(
    df: pd.DataFrame, block: str, a: str, b: str, trait: str, alpha: float
) -> Dict[str, Any]:
    d = clean_for_model(df, trait, [block, a, b])
    d["_AB_"] = d[a].astype(str) + ":" + d[b].astype(str)

    model = ols(f"{trait} ~ C({block}) + C({a}) * C({b})", data=d).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})
    anova["ms"] = anova["sum_sq"] / anova["df"]

    sh = shapiro_test(model.resid)
    lv = levene_test(d, trait, "_AB_")
    guidance = assumption_guidance(sh, lv, "Factorial in RCBD")
    tukey_df, means_letters = tukey_letters(d, trait, "_AB_", alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, "_AB_", f"Means +/- SE by {a}:{b} (RCBD)"),
        "box_plot": box_plot(d, trait, "_AB_", f"Distribution of {trait} by {a}:{b} (RCBD)"),
    }

    def p_of(term: str) -> Optional[float]:
        m = anova["source"].astype(str).str.contains(term)
        return fmt_p(anova.loc[m, "PR(>F)"].iloc[0]) if m.any() else None

    p_map = {
        f"Block ({block})": p_of(f"C\\({block}\\)"),
        a: p_of(f"C\\({a}\\)"),
        b: p_of(f"C\\({b}\\)"),
        f"{a}x{b} interaction": p_of(":"),
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
            "assumption_guidance": guidance,
        },
        "plots": plots,
        "interpretation": interpret_anova(
            "Factorial RCBD ANOVA", p_map, cv_percent(d[trait]), alpha
        ),
    }


def splitplot_engine(
    df: pd.DataFrame, block: str, main: str, sub: str, trait: str, alpha: float
) -> Dict[str, Any]:
    d = clean_for_model(df, trait, [block, main, sub])
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
        return an0.loc[m].iloc[0] if m.any() else None

    row_A = get_row(term_A)
    row_B = get_row(term_B)
    row_AB = get_row(term_AB)
    row_blockA = get_row(term_blockA)
    row_resid = get_row(term_resid)

    if row_A is None or row_blockA is None or row_resid is None:
        raise HTTPException(
            status_code=400,
            detail="Split-plot parsing failed. Check your Block, Main plot, and Sub plot column names."
        )

    ms_A = float(row_A["ms"])
    df_A = float(row_A["df"])
    ms_blockA = float(row_blockA["ms"])
    df_blockA = float(row_blockA["df"])
    ms_resid = float(row_resid["ms"])
    df_resid = float(row_resid["df"])

    F_A = ms_A / ms_blockA if ms_blockA > 0 else np.nan
    p_A = fmt_p(stats.f.sf(F_A, df_A, df_blockA)) if np.isfinite(F_A) else None

    def f_p(row: Optional[pd.Series]) -> Tuple[Optional[float], Optional[float]]:
        if row is None:
            return None, None
        ms = float(row["ms"])
        df1 = float(row["df"])
        Fv = ms / ms_resid if ms_resid > 0 else np.nan
        pv = fmt_p(stats.f.sf(Fv, df1, df_resid)) if np.isfinite(Fv) else None
        return round_val(Fv, 4), pv

    F_B, p_B = f_p(row_B)
    F_AB, p_AB = f_p(row_AB)

    an_corr = an0.copy()
    an_corr["F_corrected"] = None
    an_corr["p_corrected"] = None

    def set_corr(term: str, Fv: Optional[float], pv: Optional[float]) -> None:
        m = an_corr["source"].astype(str) == term
        if m.any():
            idx = an_corr.index[m][0]
            an_corr.loc[idx, "F_corrected"] = Fv
            an_corr.loc[idx, "p_corrected"] = pv

    set_corr(term_A, round_val(F_A, 4), p_A)
    set_corr(term_B, F_B, p_B)
    set_corr(term_AB, F_AB, p_AB)

    sh = shapiro_test(model.resid)
    lv = levene_test(d, trait, "_AB_")
    guidance = assumption_guidance(sh, lv, "Split-plot")
    tukey_df, means_letters = tukey_letters(d, trait, "_AB_", alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, "_AB_", f"Means +/- SE: {main} x {sub} (Split-plot)"),
        "box_plot": box_plot(d, trait, "_AB_", f"Distribution: {trait} by {main} x {sub}"),
    }

    p_map = {
        f"Main plot ({main})": p_A,
        f"Sub plot ({sub})": p_B,
        f"{main}x{sub} interaction": p_AB,
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
            "note": "Main plot factor tested against Block:Main error (correct split-plot test).",
        },
        "tables": {
            "anova_raw": df_to_records(an0.replace({np.nan: None})),
            "anova_corrected": df_to_records(
                an_corr[["source", "df", "sum_sq", "ms",
                          "F_corrected", "p_corrected"]].replace({np.nan: None})
            ),
            "means": df_to_records(means_letters.rename(columns={"_AB_": "Main:Sub"})),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
            "assumption_guidance": guidance,
        },
        "plots": plots,
        "interpretation": interpret_anova(
            "Split-plot ANOVA", p_map, cv_percent(d[trait]), alpha
        ),
    }


# ============================================================
#  ANALYSIS ENDPOINTS
# ============================================================

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
        df[by] = df[by].astype(str).str.strip()

    if by is None:
        rows = []
        for c in columns:
            s = coerce_numeric(df[c])
            rows.append({
                "column": c,
                "n": int(s.count()),
                "missing": int(s.isna().sum()),
                "mean": round_val(np.nanmean(s), 4) if s.count() else None,
                "sd": round_val(np.nanstd(s, ddof=1), 4) if s.count() > 1 else None,
                "se": round_val(np.nanstd(s, ddof=1) / np.sqrt(s.count()), 4) if s.count() > 1 else None,
                "min": round_val(np.nanmin(s), 4) if s.count() else None,
                "max": round_val(np.nanmax(s), 4) if s.count() else None,
                "cv_percent": cv_percent(s),
            })
        return {
            "meta": {"by": None, "columns": columns, "n_rows": int(df.shape[0])},
            "tables": {"descriptive": rows},
            "plots": {},
            "interpretation": "Descriptive statistics computed for the selected columns.",
        }

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
            "mean": g.mean().values.round(4),
            "sd": g.std(ddof=1).values.round(4),
            "se": (g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))).values.round(4),
        })
        tmp["column"] = c
        rows.extend(df_to_records(tmp))

    return {
        "meta": {"by": by, "columns": columns, "n_rows": int(df.shape[0])},
        "tables": {"descriptive_by": rows},
        "plots": {},
        "interpretation": f"Descriptive statistics computed by group ({by}).",
    }


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


@app.post("/analyze/anova/oneway_rcbd")
async def analyze_anova_oneway_rcbd(
    file: UploadFile = File(...),
    block: str = Form(...),
    treatment: str = Form(...),
    trait: str = Form(...),
    alpha: float = Form(0.05),
):
    """One-way ANOVA in RCBD - most common Nigerian field trial design."""
    df = await load_csv(file)
    require_cols(df, [block, treatment, trait])
    return oneway_rcbd_engine(df, block, treatment, trait, float(alpha))


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



# ============================================================
#  MULTI-TRAIT ENGINE
# ============================================================

def correlation_heatmap(corr_matrix: pd.DataFrame) -> str:
    """Generate a styled correlation heatmap as base64 PNG."""
    n = len(corr_matrix)
    fig_size = max(6, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    data = corr_matrix.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")

    labels = list(corr_matrix.columns)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    ax.set_title("Pearson Correlation Matrix", fontsize=12, fontweight="bold", pad=12)
    ax.spines[:].set_visible(False)
    return _b64_png(fig)


def pca_biplot(df_numeric: pd.DataFrame, group_col: str, df_full: pd.DataFrame) -> str:
    """Generate PCA biplot showing traits as vectors and groups as scatter."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA as SklearnPCA

    scaler = StandardScaler()
    X = scaler.fit_transform(df_numeric.values)
    pca = SklearnPCA(n_components=2)
    scores = pca.fit_transform(X)
    loadings = pca.components_.T
    var_exp = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 7))

    groups = df_full[group_col].astype(str).values
    unique_groups = sorted(set(groups))
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_groups)))
    color_map = {g: c for g, c in zip(unique_groups, colors)}

    for grp in unique_groups:
        mask = groups == grp
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   color=color_map[grp], label=grp, alpha=0.75, s=60, edgecolors="white", linewidth=0.5)

    scale = np.max(np.abs(scores)) / np.max(np.abs(loadings)) * 0.45
    for i, trait in enumerate(df_numeric.columns):
        ax.arrow(0, 0,
                 loadings[i, 0] * scale,
                 loadings[i, 1] * scale,
                 head_width=0.08, head_length=0.05,
                 fc="#C62828", ec="#C62828", linewidth=1.5, alpha=0.85)
        ax.text(loadings[i, 0] * scale * 1.12,
                loadings[i, 1] * scale * 1.12,
                trait, fontsize=8.5, color="#C62828", fontweight="bold",
                ha="center", va="center")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=10)
    ax.set_title("PCA Biplot: Trait Vectors and Treatment Groups", fontsize=11, fontweight="bold", pad=10)
    ax.legend(title=group_col, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _b64_png(fig)


def significance_heatmap(
    summary_rows: List[Dict], traits: List[str], factors: List[str]
) -> str:
    """Heatmap of p-values across traits and factors — green=significant, red=NS."""
    p_matrix = np.full((len(factors), len(traits)), np.nan)
    for row in summary_rows:
        trait = row.get("trait")
        factor = row.get("factor")
        p = row.get("p_value")
        if trait in traits and factor in factors and p is not None:
            i = factors.index(factor)
            j = traits.index(trait)
            p_matrix[i, j] = float(p)

    fig, ax = plt.subplots(figsize=(max(7, len(traits) * 1.1), max(3.5, len(factors) * 0.9)))
    masked = np.ma.masked_invalid(p_matrix)
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=0.2, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="p-value (green = significant)")

    ax.set_xticks(range(len(traits)))
    ax.set_yticks(range(len(factors)))
    ax.set_xticklabels(traits, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(factors, fontsize=9)

    for i in range(len(factors)):
        for j in range(len(traits)):
            val = p_matrix[i, j]
            if not np.isnan(val):
                stars = sig_stars(val)
                label = f"{val:.3f}\n{stars}" if stars != "ns" else f"{val:.3f}\nns"
                color = "white" if val < 0.05 else "black"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="bold")

    ax.set_title("Significance Summary: p-values by Trait and Factor",
                 fontsize=11, fontweight="bold", pad=10)
    ax.spines[:].set_visible(False)
    return _b64_png(fig)


def multitrait_engine(
    df: pd.DataFrame,
    design: str,
    traits: List[str],
    alpha: float,
    factor: Optional[str] = None,
    block: Optional[str] = None,
    factor_a: Optional[str] = None,
    factor_b: Optional[str] = None,
    main_plot: Optional[str] = None,
    sub_plot: Optional[str] = None,
    treatment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the same ANOVA design across multiple traits simultaneously.
    Returns per-trait results + cross-trait summary table + correlation matrix + PCA biplot.
    """
    if len(traits) < 2:
        raise HTTPException(
            status_code=400,
            detail="Multi-trait analysis requires at least 2 trait columns."
        )
    if len(traits) > 15:
        raise HTTPException(
            status_code=400,
            detail="Maximum 15 traits per multi-trait run. Please reduce the trait list."
        )

    per_trait_results = {}
    summary_rows = []
    failed_traits = []
    factors_used = []

    for trait in traits:
        try:
            if design == "oneway":
                result = oneway_engine(df, factor, trait, alpha)
                p_map = {factor: result["tables"]["anova"][0].get("PR(>F)")}
                if factor not in factors_used:
                    factors_used.append(factor)

            elif design == "oneway_rcbd":
                result = oneway_rcbd_engine(df, block, treatment, trait, alpha)
                anova_rows = result["tables"]["anova"]
                p_map = {}
                for row in anova_rows:
                    src = str(row.get("source", ""))
                    if treatment in src:
                        p_map[treatment] = row.get("PR(>F)")
                    elif block in src:
                        p_map[f"Block ({block})"] = row.get("PR(>F)")
                if treatment not in factors_used:
                    factors_used.append(treatment)

            elif design == "twoway":
                result = twoway_engine(df, factor_a, factor_b, trait, alpha)
                anova_rows = result["tables"]["anova"]
                p_map = {}
                for row in anova_rows:
                    src = str(row.get("source", ""))
                    if ":" in src:
                        key = f"{factor_a}x{factor_b}"
                    elif factor_a in src:
                        key = factor_a
                    elif factor_b in src:
                        key = factor_b
                    else:
                        continue
                    p_map[key] = row.get("PR(>F)")
                    if key not in factors_used:
                        factors_used.append(key)

            elif design == "rcbd_factorial":
                result = rcbd_factorial_engine(df, block, factor_a, factor_b, trait, alpha)
                anova_rows = result["tables"]["anova"]
                p_map = {}
                for row in anova_rows:
                    src = str(row.get("source", ""))
                    if ":" in src and block not in src:
                        key = f"{factor_a}x{factor_b}"
                    elif factor_a in src and ":" not in src:
                        key = factor_a
                    elif factor_b in src and ":" not in src:
                        key = factor_b
                    elif block in src:
                        key = f"Block ({block})"
                    else:
                        continue
                    p_map[key] = row.get("PR(>F)")
                    if key not in factors_used:
                        factors_used.append(key)

            elif design == "splitplot":
                result = splitplot_engine(df, block, main_plot, sub_plot, trait, alpha)
                anova_rows = result["tables"].get("anova_corrected", [])
                p_map = {}
                for row in anova_rows:
                    src = str(row.get("source", ""))
                    p = row.get("p_corrected")
                    if p is None:
                        continue
                    if ":" in src and block not in src:
                        key = f"{main_plot}x{sub_plot}"
                    elif main_plot in src and ":" not in src:
                        key = f"Main ({main_plot})"
                    elif sub_plot in src and ":" not in src:
                        key = f"Sub ({sub_plot})"
                    else:
                        continue
                    p_map[key] = p
                    if key not in factors_used:
                        factors_used.append(key)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown design: {design}")

            per_trait_results[trait] = result

            # Build summary row for this trait
            for factor_key, p_val in p_map.items():
                summary_rows.append({
                    "trait": trait,
                    "factor": factor_key,
                    "p_value": round_val(p_val, 4),
                    "significant": bool(p_val is not None and float(p_val) < alpha),
                    "stars": sig_stars(p_val),
                    "cv_percent": result["meta"].get("cv_percent"),
                })

        except HTTPException as e:
            failed_traits.append({"trait": trait, "error": e.detail})
        except Exception as e:
            failed_traits.append({"trait": trait, "error": str(e)})

    if not per_trait_results:
        raise HTTPException(
            status_code=400,
            detail=f"All traits failed. Errors: {failed_traits}"
        )

    # --- Cross-trait correlation matrix ---
    numeric_cols = []
    for trait in traits:
        if trait in per_trait_results:
            col = pd.to_numeric(df[trait], errors="coerce")
            if col.dropna().shape[0] >= 3:
                numeric_cols.append(trait)

    corr_matrix = None
    corr_plot = None
    corr_table = []
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if corr_df.shape[0] >= 3:
            corr_matrix = corr_df.corr(method="pearson").round(4)
            corr_plot = correlation_heatmap(corr_matrix)
            corr_table = df_to_records(corr_matrix.reset_index().rename(columns={"index": "trait"}))

    # --- PCA biplot ---
    pca_plot = None
    pca_summary = None
    group_col_for_pca = factor or treatment or factor_a or main_plot
    if group_col_for_pca and len(numeric_cols) >= 2:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA as SklearnPCA

            pca_df = df[numeric_cols + [group_col_for_pca]].dropna()
            if pca_df.shape[0] >= 3:
                X = StandardScaler().fit_transform(pca_df[numeric_cols].values)
                n_components = min(len(numeric_cols), pca_df.shape[0], 5)
                pca = SklearnPCA(n_components=n_components)
                pca.fit(X)
                var_exp = (pca.explained_variance_ratio_ * 100).round(2).tolist()
                pca_summary = {
                    "n_components": n_components,
                    "variance_explained_percent": var_exp,
                    "cumulative_variance_pc1_pc2": round(sum(var_exp[:2]), 2),
                    "loadings": {
                        f"PC{i+1}": {
                            trait: round_val(pca.components_[i][j], 4)
                            for j, trait in enumerate(numeric_cols)
                        }
                        for i in range(min(n_components, 3))
                    }
                }
                if n_components >= 2:
                    pca_plot = pca_biplot(pca_df[numeric_cols], group_col_for_pca, pca_df)
        except ImportError:
            pca_summary = {"note": "scikit-learn not installed. PCA unavailable."}
        except Exception as e:
            pca_summary = {"note": f"PCA failed: {str(e)}"}

    # --- Significance summary heatmap ---
    sig_plot = None
    if summary_rows and factors_used and len(traits) >= 2:
        try:
            successful_traits = [t for t in traits if t in per_trait_results]
            sig_plot = significance_heatmap(summary_rows, successful_traits, factors_used)
        except Exception:
            pass

    # --- Overall multi-trait summary ---
    significant_count = sum(1 for r in summary_rows if r.get("significant"))
    total_tests = len(summary_rows)

    return {
        "meta": {
            "design": design,
            "n_traits_requested": len(traits),
            "n_traits_successful": len(per_trait_results),
            "n_traits_failed": len(failed_traits),
            "traits_analysed": list(per_trait_results.keys()),
            "traits_failed": failed_traits,
            "alpha": alpha,
            "total_significance_tests": total_tests,
            "significant_results": significant_count,
        },
        "summary_table": summary_rows,
        "per_trait": per_trait_results,
        "correlation": {
            "table": corr_table,
            "note": "Pearson correlation coefficients between all numeric traits.",
        },
        "pca": pca_summary,
        "plots": {
            "significance_heatmap": sig_plot,
            "correlation_heatmap": corr_plot,
            "pca_biplot": pca_plot,
        },
    }


# ============================================================
#  MULTI-TRAIT ENDPOINTS
# ============================================================

@app.post("/analyze/anova/multitrait/oneway")
async def multitrait_oneway(
    file: UploadFile = File(...),
    factor: str = Form(...),
    traits: str = Form(...),
    alpha: float = Form(0.05),
):
    """Multi-trait one-way ANOVA (CRD). traits = comma-separated column names."""
    df = await load_csv(file)
    trait_list = [t.strip() for t in traits.split(",") if t.strip()]
    require_cols(df, [factor] + trait_list)
    return multitrait_engine(df, "oneway", trait_list, float(alpha), factor=factor)


@app.post("/analyze/anova/multitrait/oneway_rcbd")
async def multitrait_oneway_rcbd(
    file: UploadFile = File(...),
    block: str = Form(...),
    treatment: str = Form(...),
    traits: str = Form(...),
    alpha: float = Form(0.05),
):
    """Multi-trait one-way ANOVA (RCBD). traits = comma-separated column names."""
    df = await load_csv(file)
    trait_list = [t.strip() for t in traits.split(",") if t.strip()]
    require_cols(df, [block, treatment] + trait_list)
    return multitrait_engine(
        df, "oneway_rcbd", trait_list, float(alpha), block=block, treatment=treatment
    )


@app.post("/analyze/anova/multitrait/twoway")
async def multitrait_twoway(
    file: UploadFile = File(...),
    factor_a: str = Form(...),
    factor_b: str = Form(...),
    traits: str = Form(...),
    alpha: float = Form(0.05),
):
    """Multi-trait two-way factorial ANOVA (CRD). traits = comma-separated column names."""
    df = await load_csv(file)
    trait_list = [t.strip() for t in traits.split(",") if t.strip()]
    require_cols(df, [factor_a, factor_b] + trait_list)
    return multitrait_engine(
        df, "twoway", trait_list, float(alpha), factor_a=factor_a, factor_b=factor_b
    )


@app.post("/analyze/anova/multitrait/rcbd_factorial")
async def multitrait_rcbd_factorial(
    file: UploadFile = File(...),
    block: str = Form(...),
    factor_a: str = Form(...),
    factor_b: str = Form(...),
    traits: str = Form(...),
    alpha: float = Form(0.05),
):
    """Multi-trait factorial RCBD. traits = comma-separated column names."""
    df = await load_csv(file)
    trait_list = [t.strip() for t in traits.split(",") if t.strip()]
    require_cols(df, [block, factor_a, factor_b] + trait_list)
    return multitrait_engine(
        df, "rcbd_factorial", trait_list, float(alpha),
        block=block, factor_a=factor_a, factor_b=factor_b
    )


@app.post("/analyze/anova/multitrait/splitplot")
async def multitrait_splitplot(
    file: UploadFile = File(...),
    block: str = Form(...),
    main_plot: str = Form(...),
    sub_plot: str = Form(...),
    traits: str = Form(...),
    alpha: float = Form(0.05),
):
    """Multi-trait split-plot ANOVA. traits = comma-separated column names."""
    df = await load_csv(file)
    trait_list = [t.strip() for t in traits.split(",") if t.strip()]
    require_cols(df, [block, main_plot, sub_plot] + trait_list)
    return multitrait_engine(
        df, "splitplot", trait_list, float(alpha),
        block=block, main_plot=main_plot, sub_plot=sub_plot
    )
