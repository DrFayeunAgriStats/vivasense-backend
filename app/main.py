"""
VivaSense V1 + FIA AI Proxy - app/main.py
==========================================
Single FastAPI app combining:
  - VivaSense statistical analysis endpoints
  - FIA AI proxy endpoints (CSP 811 tutor + VivaSense interpretation)

Environment variables required (set in Render dashboard):
  ANTHROPIC_API_KEY = sk-ant-api03-...

Analysis endpoints:
  GET  /                           - root
  GET  /health                     - health check
  POST /analyze/descriptive        - descriptive statistics
  POST /analyze/anova/oneway       - one-way ANOVA (CRD)
  POST /analyze/anova/twoway       - two-way ANOVA (CRD factorial)
  POST /analyze/anova/rcbd_factorial - factorial in RCBD
  POST /analyze/anova/splitplot    - split-plot ANOVA

AI proxy endpoints:
  GET  /api/health                 - AI proxy health check
  POST /api/chat                   - CSP 811 Dr. Fayeun tutor chat
  POST /api/interpret              - VivaSense AI interpretation
  POST /api/followup               - VivaSense post-analysis follow-up chat
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

app = FastAPI(title="VivaSense V1 + FIA AI Proxy", version="1.0.0")

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
    allow_methods=["*"],
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

# Simple in-memory rate limiting
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

Write as you would supervise a postgraduate student or write in a peer-reviewed paper.

STRUCTURE YOUR RESPONSE AS:
1. Overall Finding - one sentence stating the key result
2. Statistical Evidence - reference specific F-value, p-value, and means from the results
3. Treatment/Genotype Comparisons - interpret the post-hoc groupings specifically
4. Assumptions - comment on whether assumptions were met; flag any violations with advice
5. Research Recommendation - what the researcher should do or conclude
6. Thesis/Paper Sentence - give them one ready-to-use sentence they can copy into their write-up

RULES:
- Always reference the actual numbers provided, never speak generally
- Use Nigerian/West African crop context where the crop is mentioned
- Keep language accessible to final-year undergraduates and postgraduates
- If assumptions are violated, suggest the appropriate non-parametric alternative
- End with: "- Dr. Fayeun, VivaSense"

You are what separates a confused student from a confident researcher."""


VIVASENSE_FOLLOWUP_SYSTEM = """You are Dr. Fayeun, Professor of Quantitative Genetics at FUTA. A researcher has run a statistical analysis on VivaSense and you have already provided an interpretation. They are now asking follow-up questions.

You have full knowledge of their analysis results provided below. Answer their specific question directly and concisely. If they ask about writing for a thesis, give them exact sentences. If they ask about assumption violations, give specific actionable advice. If they ask why a result is unexpected, reason through it with them.

Always ground your answer in THEIR specific numbers, not generic advice.
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
    stream: Optional[bool] = True


class FollowupRequest(BaseModel):
    messages: List[Message]
    analysis_results: dict
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
    """Stream response from Anthropic API as SSE."""
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
    """Non-streaming call - returns full text."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            ANTHROPIC_URL,
            headers=ANTHROPIC_HEADERS,
            json=payload
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

@app.get("/api/health")
async def api_health():
    """Health check for AI proxy."""
    return {
        "status": "ok",
        "service": "FIA AI Proxy",
        "api_key_configured": bool(ANTHROPIC_API_KEY),
        "endpoints": ["/api/chat", "/api/interpret", "/api/followup"],
    }


@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    """CSP 811 Dr. Fayeun AI tutor chat."""
    client_ip = request.client.host
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")

    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not configured. Contact administrator.")

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
    """Generate Dr. Fayeun's interpretation of VivaSense analysis results."""
    client_ip = request.client.host
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")

    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not configured. Contact administrator.")

    results_text = json.dumps(body.results, indent=2)
    crop_context = f"Crop/study context: {body.crop}" if body.crop else ""

    user_message = f"""Please interpret these {body.analysis_type} results for the researcher.

{crop_context}

Results:
{results_text}

Provide a complete interpretation following your structured format."""

    messages = [{"role": "user", "content": user_message}]

    if body.stream:
        return await stream_anthropic(
            VIVASENSE_INTERPRET_SYSTEM, messages,
            model=SONNET_MODEL, max_tokens=1200
        )
    else:
        text = await call_anthropic(
            VIVASENSE_INTERPRET_SYSTEM, messages,
            model=SONNET_MODEL, max_tokens=1200
        )
        return {"interpretation": text}


@app.post("/api/followup")
async def followup(request: Request, body: FollowupRequest):
    """Post-analysis chat - user asks follow-up questions about their results."""
    client_ip = request.client.host
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")

    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not configured. Contact administrator.")

    results_context = json.dumps(body.analysis_results, indent=2)
    system = VIVASENSE_FOLLOWUP_SYSTEM + f"\n\nANALYSIS RESULTS IN CONTEXT:\n{results_context}"

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream:
        return await stream_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=800)
    else:
        text = await call_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=800)
        return {"content": text}


# ============================================================
#  VIVASENSE ROOT + HEALTH
# ============================================================

@app.get("/")
def root():
    return {"name": "VivaSense V1", "status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


# ============================================================
#  VIVASENSE HELPERS
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
            detail=f"Missing required column(s): {missing}. Available columns: {list(df.columns)}",
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
            detail=f"Not enough valid rows after cleaning for trait '{y}'."
        )
    if len(x_cols) == 1 and d[x_cols[0]].nunique(dropna=True) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 levels in '{x_cols[0]}' for ANOVA."
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
    df: pd.DataFrame,
    y: str,
    group: str,
    alpha: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mt = mean_table(df, y, group)

    if df[group].nunique() < 2:
        raise HTTPException(status_code=400, detail=f"Need at least 2 levels in '{group}' for Tukey.")
    if df[y].dropna().shape[0] < 3:
        raise HTTPException(status_code=400, detail=f"Not enough numeric observations in '{y}' for Tukey.")

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


def interpret_anova(
    title: str,
    p_map: Dict[str, Optional[float]],
    cv: Optional[float],
    alpha: float
) -> str:
    lines = [f"{title} interpretation:"]
    if cv is not None:
        lines.append(f"- CV = {cv:.2f}%. Lower CV generally indicates more precise measurements.")
    for term, p in p_map.items():
        if p is None:
            continue
        if p < alpha:
            lines.append(f"- {term} is significant (p = {p:.4f}).")
        else:
            lines.append(f"- {term} is not significant (p = {p:.4f}).")
    lines.append("- Assumptions: check Shapiro (normality) and Levene (equal variance).")
    return "\n".join(lines)


# ============================================================
#  VIVASENSE ANALYSIS ENGINES
# ============================================================

def oneway_engine(df: pd.DataFrame, factor: str, trait: str, alpha: float) -> Dict[str, Any]:
    d = clean_for_model(df, trait, [factor])
    model = ols(f"{trait} ~ C({factor})", data=d).fit()
    anova = sm.stats.anova_lm(model, typ=2).reset_index().rename(columns={"index": "source"})
    anova["ms"] = anova["sum_sq"] / anova["df"]

    sh = shapiro_test(model.resid)
    lv = levene_test(d, trait, factor)
    tukey_df, means_letters = tukey_letters(d, trait, factor, alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, factor, f"Means +/- SE by {factor}"),
        "box_plot": box_plot(d, trait, factor, f"Boxplot of {trait} by {factor}"),
    }

    p_factor = None
    m = anova["source"].astype(str).str.contains(f"C\\({factor}\\)")
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
        "interpretation": interpret_anova(
            "One-way ANOVA", {factor: p_factor}, cv_percent(d[trait]), alpha
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
    tukey_df, means_letters = tukey_letters(d, trait, "_AB_", alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, "_AB_", f"Means +/- SE by {a}:{b}"),
        "box_plot": box_plot(d, trait, "_AB_", f"Boxplot of {trait} by {a}:{b}"),
    }

    def p_of(term: str) -> Optional[float]:
        m = anova["source"].astype(str).str.contains(term)
        return float(anova.loc[m, "PR(>F)"].iloc[0]) if m.any() else None

    p_map = {
        a: p_of(f"C\\({a}\\)"),
        b: p_of(f"C\\({b}\\)"),
        f"{a}x{b}": p_of(":"),
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
        "interpretation": interpret_anova(
            "Two-way ANOVA", p_map, cv_percent(d[trait]), alpha
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
    tukey_df, means_letters = tukey_letters(d, trait, "_AB_", alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, "_AB_", f"Means +/- SE by {a}:{b} (RCBD)"),
        "box_plot": box_plot(d, trait, "_AB_", f"Boxplot of {trait} by {a}:{b} (RCBD)"),
    }

    def p_of(term: str) -> Optional[float]:
        m = anova["source"].astype(str).str.contains(term)
        return float(anova.loc[m, "PR(>F)"].iloc[0]) if m.any() else None

    p_map = {
        "Block": p_of(f"C\\({block}\\)"),
        a: p_of(f"C\\({a}\\)"),
        b: p_of(f"C\\({b}\\)"),
        f"{a}x{b}": p_of(":"),
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
            detail="Split-plot term parsing failed. Check factor column names."
        )

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
        "mean_plot": mean_plot(d, trait, "_AB_", f"Means +/- SE by {main}:{sub} (Split-plot)"),
        "box_plot": box_plot(d, trait, "_AB_", f"Boxplot of {trait} by {main}:{sub} (Split-plot)"),
    }

    p_map = {
        f"Main plot ({main})": p_A,
        f"Sub plot ({sub})": p_B,
        f"{main}x{sub}": p_AB,
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
            "note": "Main plot factor tested using Block:Main error term (correct split-plot test).",
        },
        "tables": {
            "anova_raw": df_to_records(an0.replace({np.nan: None})),
            "anova_corrected": df_to_records(
                an_corr[["source", "df", "sum_sq", "ms", "F_corrected", "p_corrected"]].replace({np.nan: None})
            ),
            "means": df_to_records(means_letters.rename(columns={"_AB_": "Main:Sub"})),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
        },
        "plots": plots,
        "interpretation": interpret_anova(
            "Split-plot ANOVA", p_map, cv_percent(d[trait]), alpha
        ),
    }


# ============================================================
#  VIVASENSE ANALYSIS ENDPOINTS
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


@app.post("/analyze/anova/multitrait")
async def analyze_anova_multitrait_disabled():
    raise HTTPException(
        status_code=410,
        detail=(
            "Multi-trait analysis is disabled in VivaSense V1 for stability. "
            "Run one-way/two-way/RCBD/split-plot per trait using the single-trait endpoints."
        ),
    )
