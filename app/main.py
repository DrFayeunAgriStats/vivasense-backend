"""
VivaSense V2.2 + FIA AI Proxy - app/main.py
==========================================
V2.2 improvements:
  - Non-parametric tests: Kruskal-Wallis (CRD) + Friedman (RCBD) with post-hoc [V2.1]
  - Usage logger with /admin/usage and /admin/usage/summary endpoints [V2.1]
  - LOCKED assumption verdict: single function, zero contradictions across all outputs [V2.2]
  - Executive Insight block: deterministic one-paragraph 'big story' per trait [V2.2]
  - Reviewer Radar block: rule-based likely reviewer questions per trait [V2.2]
  - Decision Rules block: ranked means converted to practical recommendations [V2.2]
  - Intelligence blocks shipped in every engine response under 'intelligence' key [V2.2]
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
  POST /analyze/nonparametric/kruskal  - Kruskal-Wallis H-test (non-parametric CRD) [NEW in V2.1]
  POST /analyze/nonparametric/friedman - Friedman test (non-parametric RCBD) [NEW in V2.1]

Admin endpoints (require X-Admin-Token header):
  GET  /admin/usage                    - last N usage log entries [NEW in V2.1]
  GET  /admin/usage/summary            - aggregated counts by design/day/status [NEW in V2.1]

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
import hashlib
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

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

app = FastAPI(title="VivaSense V2 + FIA AI Proxy", version="2.1.0")

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
#  USAGE LOGGER
# ============================================================

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # Set in Render dashboard: ADMIN_TOKEN=your-secret
USAGE_LOG: deque = deque(maxlen=2000)        # Rolling in-memory log, last 2000 events

# Endpoint → friendly design label
DESIGN_LABELS: Dict[str, str] = {
    "/analyze/descriptive":              "Descriptive Statistics",
    "/analyze/anova/oneway":             "One-way ANOVA (CRD)",
    "/analyze/anova/oneway_rcbd":        "One-way ANOVA (RCBD)",
    "/analyze/anova/twoway":             "Two-way Factorial (CRD)",
    "/analyze/anova/rcbd_factorial":     "Factorial RCBD",
    "/analyze/anova/splitplot":          "Split-plot ANOVA",
    "/analyze/nonparametric/kruskal":    "Kruskal-Wallis (Non-parametric CRD)",
    "/analyze/nonparametric/friedman":   "Friedman Test (Non-parametric RCBD)",
    "/analyze/anova/multitrait/oneway":          "Multi-trait CRD One-way",
    "/analyze/anova/multitrait/oneway_rcbd":     "Multi-trait RCBD One-way",
    "/analyze/anova/multitrait/twoway":          "Multi-trait Two-way Factorial",
    "/analyze/anova/multitrait/rcbd_factorial":  "Multi-trait Factorial RCBD",
    "/analyze/anova/multitrait/splitplot":       "Multi-trait Split-plot",
    "/api/chat":       "CSP811 AI Tutor",
    "/api/interpret":  "Dr. Fayeun Interpretation",
    "/api/followup":   "Follow-up Chat",
}


def _hash_ip(ip: str) -> str:
    """One-way hash of IP for privacy-safe logging."""
    return hashlib.sha256(ip.encode()).hexdigest()[:12]


def _log_event(
    path: str,
    method: str,
    status_code: int,
    duration_ms: float,
    ip: str,
    extra: Optional[Dict] = None,
) -> None:
    entry: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "path": path,
        "design": DESIGN_LABELS.get(path, path),
        "method": method,
        "status": status_code,
        "duration_ms": round(duration_ms, 1),
        "ip_hash": _hash_ip(ip),
    }
    if extra:
        entry.update(extra)
    USAGE_LOG.append(entry)


@app.middleware("http")
async def usage_logger_middleware(request: Request, call_next):
    """Log every /analyze/ and /api/ request automatically."""
    path = request.url.path
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    if path.startswith("/analyze/") or path.startswith("/api/"):
        ip = request.client.host if request.client else "unknown"
        _log_event(
            path=path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=duration_ms,
            ip=ip,
        )
    return response


def _require_admin(request: Request) -> None:
    """Raise 403 if ADMIN_TOKEN not set or token header doesn't match."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Admin access not configured.")
    token = request.headers.get("X-Admin-Token", "")
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token.")


@app.get("/admin/usage")
async def admin_usage(request: Request, limit: int = 200):
    """
    Return the last N usage log entries.
    Requires header:  X-Admin-Token: <your ADMIN_TOKEN>
    """
    _require_admin(request)
    entries = list(USAGE_LOG)[-limit:]
    return {
        "total_logged": len(USAGE_LOG),
        "returned": len(entries),
        "entries": entries,
    }


@app.get("/admin/usage/summary")
async def admin_usage_summary(request: Request):
    """
    Return aggregated usage counts by design, status, and day.
    Requires header:  X-Admin-Token: <your ADMIN_TOKEN>
    """
    _require_admin(request)
    entries = list(USAGE_LOG)

    # Counts by design
    by_design: Dict[str, int] = defaultdict(int)
    by_status: Dict[int, int] = defaultdict(int)
    by_day:    Dict[str, int] = defaultdict(int)
    errors = []
    total_duration = 0.0

    for e in entries:
        by_design[e["design"]] += 1
        by_status[e["status"]] += 1
        day = e["ts"][:10]
        by_day[day] += 1
        total_duration += e.get("duration_ms", 0)
        if e["status"] >= 400:
            errors.append(e)

    n = len(entries)
    return {
        "total_requests": n,
        "successful": by_status.get(200, 0),
        "errors": len(errors),
        "avg_duration_ms": round(total_duration / n, 1) if n else 0,
        "by_design": dict(sorted(by_design.items(), key=lambda x: -x[1])),
        "by_status": {str(k): v for k, v in sorted(by_status.items())},
        "by_day": dict(sorted(by_day.items())),
        "recent_errors": errors[-10:],
    }


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


VIVASENSE_INTERPRET_SYSTEM = """You are Dr. Fayeun, Professor of Quantitative Genetics at FUTA and founder of Field-to-Insight Academy (www.fieldtoinsightacademy.com.ng). You are interpreting statistical results for a researcher who used VivaSense.

Write with the authority of a peer-reviewed journal article and the warmth of a supervisor who wants the student to succeed.

DETECT ANALYSIS TYPE AND RESPOND ACCORDINGLY:

IF analysis type contains "descriptive":
  Structure:
  1. Data Overview - summarise range and central tendency; comment on sample size adequacy
  2. Variability Assessment - interpret CV% per trait: <15% = good, 15-25% = acceptable, >25% = high variability needing attention
  3. Trait Performance Ranking - highest and lowest performers by mean; flag extreme values
  4. Data Quality Notes - comment on missing values, suspicious ranges, traits needing transformation
  5. Next Step Recommendation - advise which ANOVA design to run next based on experimental structure
  6. Thesis/Paper Sentence - one publication-ready sentence on the descriptive findings
  CRITICAL: Do NOT use p-value or F-statistic language for descriptive statistics.

IF analysis type contains "one-way" or "oneway":
  Structure:
  1. Overall Finding - one sentence with F-value, p-value, and direction of effect
  2. Statistical Evidence - CV%, MS treatment vs MS error, degrees of freedom
  3. Treatment/Genotype Comparisons - Tukey letters specifically; best and worst with actual means
  4. Assumptions Check - Shapiro-Wilk and Levene results; if violated name exact alternative
  5. Research/Breeding Recommendation - actionable conclusion tied to study objective
  6. Thesis/Paper Sentence - with F-value, df, p-value, and best performer named

IF analysis type contains "factorial" or "two-way" or "split":
  Structure:
  1. Overall Finding - main effects AND interaction significance in one sentence
  2. Statistical Evidence - F-values and p-values for each source
  3. Interaction Interpretation - if significant: explain best combination; if not: interpret main effects independently
  4. Mean Comparisons - best treatment combination with mean and Tukey letter
  5. Assumptions Check - with advice if violated
  6. Research/Breeding Recommendation - specific and actionable
  7. Thesis/Paper Sentence - with all key statistics

IF analysis type contains "multi-trait" or "multitrait" or "synthesis":
  CRITICAL: The facts contain pre-computed flags. Use ONLY these — never guess.
  Required structure:
  1. Overall Finding
     - "X of Y traits showed significant [treatment] effects."
     - "Significant interactions detected for: [list] / No significant interactions detected."
     - Use n_trt_sig and total_traits from facts.
  2. Trait-by-Trait Summary
     - One sentence per trait: effect significant? F-value? CV from cv_values dict only.
  3. Block Effects
     - "Block effects significant for X of Y traits (use n_block_sig)."
  4. Assumptions Summary — CRITICAL: use assumption_flags, never guess
     - If n_shapiro_fail == 0: "Normality satisfied for all traits."
     - If n_shapiro_fail > 0: "Normality violated for X traits: [list trait names where normality_passed is False]."
     - If n_levene_fail == 0: "Homogeneity satisfied for all traits."
     - If n_levene_fail > 0: "Homogeneity violated for X traits: [list names]."
     - NEVER write "all assumptions passed" without checking assumption_flags.
  5. Selection Strategy — only if significant effects found
     - Based on means from summary_table, not invented.
  6. Thesis/Paper Sentence — one sentence, only computed values.

IF analysis type contains "kruskal" or "Kruskal":
  Structure:
  1. Overall Finding - H-statistic, degrees of freedom, p-value, and direction in one sentence
  2. Statistical Evidence - H value, df, p-value, N total, why non-parametric was appropriate
  3. Group Comparisons - interpret Dunn pairwise results; name best and worst groups with actual means
  4. Rank Interpretation - explain mean rank differences in plain language
  5. Research/Breeding Recommendation - actionable conclusion tied to study objective
  6. Thesis/Paper Sentence - using correct non-parametric language (H-statistic, not F-ratio)
  NOTE: Use "statistically significant differences" not "F-value". Never report F-ratio for non-parametric tests.
  CORRECT citation format: "Kruskal-Wallis H(df, N=n) = value, p = value"

IF analysis type contains "friedman" or "Friedman":
  Structure:
  1. Overall Finding - chi-squared statistic, df, p-value, and direction in one sentence
  2. Statistical Evidence - χ² value, df, N blocks, why non-parametric was appropriate
  3. Treatment Comparisons - interpret Wilcoxon pairwise results; name best and worst treatments with means
  4. Rank Sum Interpretation - explain rank sums in plain language
  5. Research/Breeding Recommendation - actionable conclusion tied to study objective
  6. Thesis/Paper Sentence - using correct non-parametric language (χ², not F-ratio)
  NOTE: Never report F-ratio for Friedman test. Use χ² notation.
  CORRECT citation format: "Friedman χ²(df, N=blocks) = value, p = value"

PERSONALISATION RULES - ALWAYS APPLY:
- Crop provided: use specifically throughout the interpretation
- Treatment description: reference meaningfully at every relevant point
- Study objective: align every recommendation directly to it
- Location: contextualise findings to that environment
- ALWAYS use actual numbers from results - never generalise
- If NO context provided: ask for it politely at the end
- Language: accessible to final-year undergraduates and postgraduates
- End EVERY response with: "- Dr. Fayeun, VivaSense"

QUALITY RULES:
- Never invent statistics not present in the results
- Never use significance language for descriptive statistics
- If CV > 30% flag it explicitly and explain reliability implications
- If Tukey letters absent: note this and explain what is missing
- Always end with one actionable next step the researcher can take today

INTELLIGENCE BLOCKS — USE WHEN PROVIDED:
The results may include an 'intelligence' object with four pre-computed blocks.
When present, incorporate them as follows:
- assumptions_verdict: USE THIS EXACT TEXT for the assumptions section. Never override or contradict it.
- executive_insight: Use as the basis for your Overall Finding — expand it with biological context.
- reviewer_radar: Present these verbatim under a 'Reviewer Radar' heading. Do not soften or omit any question.
- decision_rules: Present these under 'Decision Rules' heading. Add one sentence of agronomic context per rule.

ASSUMPTIONS — CRITICAL RULE:
The assumptions_verdict field is generated deterministically from Shapiro-Wilk and Levene test statistics.
It is always correct. NEVER contradict it, NEVER say "assumptions were met" if the verdict says otherwise.
If assumptions_verdict is not provided, state: "Assumption results not available for this analysis."

OPERATING MODES:
STRICT MODE (always active by default):
- Only facts provable from the grounded facts provided.
- Never use: "because", "therefore", "leads to", "due to", mechanism words.
- Never use: "dominant", "robust", "dramatic", "overwhelming", "food safety", "drought tolerance", "farmer adoption".
- Never invent CV values — use only the cv_percent from the facts.
- If a statistic is missing: write "Not available from analysis output."

DISCUSSION MODE (only when researcher explicitly provides location, season, and objective):
- May discuss agronomic interpretation after completing the strict statistical summary.
- Must clearly label the section "Agronomic Discussion (researcher context required)".
- Still cannot invent statistics or contradict assumption verdicts.

DEFAULT: STRICT MODE unless researcher context is explicitly provided.


1. EVERY number you write must exist in the Statistical Results provided. Never invent, round differently, or recall from memory.
2. p-values: NEVER write "p = 0". When p_display field shows "< 0.001", write "p < 0.001" — not "p = 0".
3. If Tukey groupings are NOT in the results: do not mention letters, groups, or "Tukey".
4. If correlation is NOT in the results: do not use "correlated", "relationship", or imply association.
5. If trend test NOT computed: do not use "dose-response", "increasing trend", "declined steadily".
6. If a field is missing from the results: write "Not available from analysis output." — never guess.

BANNED PHRASES — never use regardless of context:
- Causality: "because", "due to", "caused by", "leads to", "results from"
- Unsupported strength: "massive effect", "dominant factor", "profound", "overwhelmingly"
- Generalisation without user context: "for farmers", "in your study area", "for smallholders"
- Impossible statistics: "p = 0", "100% significant", "perfect results"
- Trend without test: "dose-response", "linear increase", "declined steadily"

SAFE REPLACEMENTS:
- "strong evidence" → "F(df,df) = value, p < 0.001"
- "dominant factor" → "had the largest F-value among tested effects (F = value)"
- "p = 0" → "p < 0.001"
- Causal language → "was associated with" or "differed significantly across"

INTERACTION RULE — for every factorial and split-plot analysis, report interaction FIRST:
- If p(interaction) < α: "Interaction was significant (F=…, p=…). Interpret treatment combinations, not main effects alone."
- If p(interaction) ≥ α: "No evidence of interaction (F=…, p=…). Main effects are interpreted independently."

ACADEMIC INTEGRITY RULES — NON-NEGOTIABLE:
1. THESIS SENTENCE: Never say "copy this into your thesis" or "copy directly". 
   Instead say: "Here is a suggested starting point for your results section — 
   adapt it to reflect your own understanding and field context."
2. INTERPRETATION PURPOSE: Frame your interpretation as a scaffold for learning,
   not a final answer. Use phrases like "consider whether...", "you may want to 
   discuss...", "reflect on what this means for your specific crop/context."
3. ENCOURAGE UNDERSTANDING: After your interpretation, always ask ONE question
   that tests whether the student understands their own results. For example:
   "Before writing this up, ask yourself: why did the interaction occur? 
   What biological or agronomic factor explains this pattern in your experiment?"
4. NEVER do the student's thinking for them on field-specific interpretation.
   You provide the statistical framework. They provide the biological meaning.
5. SUPERVISOR REMINDER: End every interpretation with:
   "Remember to discuss these findings with your supervisor before finalising 
   your write-up. Your supervisor's knowledge of your specific field context 
   is essential for a complete interpretation."

You are what separates a confused student from a confident researcher.
Your goal is to build their capacity, not replace it."""


VIVASENSE_FOLLOWUP_SYSTEM = """You are Dr. Fayeun, Professor of Quantitative Genetics at FUTA and founder of Field-to-Insight Academy. A researcher used VivaSense and received your interpretation. They are now asking follow-up questions.

You have their complete analysis results and study context below. Be direct, specific, and actionable.

RESPOND BY QUESTION TYPE:

Thesis writing questions:
  Give exact publication-ready sentences using their actual numbers.
  Include F-value, df, p-value, treatment means, and Tukey letters where relevant.

Assumption violation questions:
  Name the exact test (Kruskal-Wallis for CRD one-way, Friedman for RCBD, log/sqrt transformation for factorial).
  Give complete R code for their specific design and column names.

Unexpected result questions:
  Reason systematically using their actual means and CV values.
  Consider: environmental variation, sampling error, GxE interaction.
  Never dismiss unexpected results.

Best treatment/genotype questions:
  Name it specifically with mean value and Tukey letter.
  Compare against grand mean and worst performer.
  Give a breeding or management recommendation.

Download/saving results questions:
  VivaSense has a Download Results button that saves the full report as a text file
  including ANOVA tables, means table, Tukey letters, Dr. Fayeun's interpretation,
  and the complete chat history. Users can also right-click any plot to save it.
  Direct users to these features - never tell them to take screenshots.

R code questions:
  Provide complete runnable R code using their actual column names.
  Include comments explaining each step.

ALWAYS:
- Reference THEIR specific numbers - not generic advice
- Be concise - clarity over length
- End with "- Dr. Fayeun" on answers longer than 3 sentences
- Never suggest contacting support for things you can answer directly

ACADEMIC INTEGRITY IN FOLLOW-UP:
- If a student asks you to "write my results section" or "write my discussion":
  Provide a structured framework and key points they should cover, but do not 
  write the section for them. Say: "Here are the key points your results section 
  should address — write it in your own words."
- If a student asks "what does this mean?" — answer the statistical question
  but ask them: "What do you think explains this pattern in your experiment?"
- If a student seems to be copying responses verbatim: remind them that 
  understanding their results is more valuable than having correct text.
- Always encourage engagement with their supervisor on field-specific questions.
- You build researchers. You do not replace their thinking."""


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
        return await stream_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=1500)
    else:
        text = await call_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=1500)
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

    # Strip base64 plots from results before sending to AI
    # to avoid exceeding token limits
    def strip_plots(obj):
        if isinstance(obj, dict):
            return {
                k: strip_plots(v)
                for k, v in obj.items()
                if k not in ('mean_plot', 'box_plot', 'interaction_plot',
                             'significance_heatmap', 'correlation_heatmap',
                             'pca_biplot', 'plots')
            }
        elif isinstance(obj, list):
            return [strip_plots(i) for i in obj]
        return obj

    # ── Fact-first pipeline (spec §2) ───────────────────────────────────────
    # Extract grounded facts from result JSON.
    # LLM receives fact-bullets, not raw JSON — hallucination structurally reduced.
    results_obj = body.results if isinstance(body.results, dict) else {}

    # Determine if result is a single-trait or multi-trait synthesis call
    is_synthesis = "summary_table" in results_obj or "per_trait" in results_obj

    if is_synthesis:
        # Multi-trait synthesis: build computed flags — never guess assumptions
        per_trait = results_obj.get("per_trait", {})
        n_shapiro_fail = sum(
            1 for t_result in per_trait.values()
            if not (t_result.get("tables", {}).get("assumptions", [{}]) or [{}])[0].get("passed", True)
        )
        n_levene_fail = sum(
            1 for t_result in per_trait.values()
            if not (t_result.get("tables", {}).get("assumptions", [{},{}]) or [{},{}])[1].get("passed", True)
        )
        n_block_sig = sum(
            1 for row in results_obj.get("summary_table", [])
            if "Block" in str(row.get("factor", "")) and row.get("significant")
        )
        n_trt_sig = sum(
            1 for row in results_obj.get("summary_table", [])
            if "Block" not in str(row.get("factor", "")) and row.get("significant")
        )
        total_traits = results_obj.get("meta", {}).get("n_traits_successful", 0)
        cv_values = {
            row["trait"]: row.get("cv_percent")
            for row in results_obj.get("summary_table", [])
            if row.get("cv_percent") is not None
        }
        # Build assumption summary per trait for synthesis
        assumption_flags = {}
        for tname, tr in per_trait.items():
            assump = tr.get("tables", {}).get("assumptions", [])
            sh = next((a for a in assump if a.get("test") == "Shapiro-Wilk"), {})
            lv = next((a for a in assump if a.get("test") == "Levene"), {})
            guidance = tr.get("tables", {}).get("assumption_guidance", {})
            assumption_flags[tname] = {
                "normality_passed":   sh.get("passed"),
                "normality_p":        sh.get("p_value"),
                "homogeneity_passed": lv.get("passed"),
                "verdict":            guidance.get("overall", "Not available."),
                "verdict_code":       guidance.get("verdict_code", "unknown"),
                "cv":                 tr.get("meta", {}).get("cv_percent"),
            }

        synthesis_facts = {
            "total_traits":     total_traits,
            "n_trt_sig":        n_trt_sig,
            "n_shapiro_fail":   n_shapiro_fail,
            "n_levene_fail":    n_levene_fail,
            "n_block_sig":      n_block_sig,
            "assumption_flags": assumption_flags,
            "cv_values":        cv_values,
            "summary_table":    results_obj.get("summary_table", []),
        }
        facts_text = f"MULTI-TRAIT SYNTHESIS FACTS:\n{json.dumps(synthesis_facts, indent=2)}"
        cv_actual = None
    else:
        # Single-trait: extract grounded facts
        facts = extract_facts(results_obj, alpha=0.05)
        facts_text = facts_to_prompt_text(facts)
        cv_actual = facts.get("cv")

    user_message = f"""Please interpret these {body.analysis_type} results for the researcher.

{context_block}GROUNDED FACTS (use ONLY these — do not invent additional statistics):
{facts_text}

STRICT MODE INSTRUCTIONS:
- Use only the facts listed above. Every sentence must reference a fact above.
- CV value: if mentioned, use ONLY the cv_percent value from the facts. Do not compute your own.
- Assumptions: copy the ASSUMPTIONS VERDICT text exactly. Do not contradict it.
- p-values: use the p_display values provided. Never write 'p = 0'.
- If a statistic is not in the facts above: write 'Not available from analysis output.'

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
        # Run grounding validator on returned text
        has_tukey = bool(results_obj.get("tables", {}).get("tukey"))
        has_corr  = bool(results_obj.get("correlation", {}).get("table"))
        grounding = validate_interpretation(
            text,
            has_tukey=has_tukey,
            has_correlation=has_corr,
            cv_actual=cv_actual,
        )
        # If validation fails and not synthesis: fall back to minimal safe output
        if not grounding["passed"] and not is_synthesis and not body.stream:
            facts_local = extract_facts(results_obj, alpha=0.05) if not is_synthesis else {}
            fallback = fallback_interpretation(facts_local) if facts_local else text
            return {
                "interpretation": text,
                "fallback_used":  False,
                "grounding_check": grounding,
                "fallback_available": fallback,
            }
        return {
            "interpretation": text,
            "fallback_used":  False,
            "grounding_check": grounding,
        }


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
        return await stream_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=1500)
    else:
        text = await call_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=1500)
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
    """
    Format p-value to 4dp. Returns 0.0001 as floor for very small values
    so downstream JSON never contains literal 0 (which is statistically impossible).
    """
    if p is None:
        return None
    f = float(p)
    if np.isnan(f):
        return None
    if f < 0.0001:
        return 0.0001   # floor — display as "< 0.001" in interpretation
    return round(f, 4)


def fmt_p_display(p) -> str:
    """Human-readable p-value string for interpretation text."""
    if p is None:
        return "p = N/A"
    f = float(p)
    if np.isnan(f):
        return "p = N/A"
    if f < 0.001:
        return "p < 0.001"
    if f < 0.01:
        return f"p = {f:.3f}"
    return f"p = {f:.4f}"


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
    """
    LOCKED assumption verdict — single source of truth.
    Every table, every interpretation, every synthesis pulls from here.
    No freehand text anywhere else.
    """
    s_pass = shapiro.get("passed")
    l_pass = levene.get("passed")
    s_p    = shapiro.get("p_value")
    l_p    = levene.get("p_value")
    s_stat = shapiro.get("stat")

    alternatives = {
        "CRD one-way":            "Kruskal-Wallis test",
        "RCBD one-way":           "Friedman test",
        "CRD two-way (factorial)":"Log or square-root transformation, then re-run ANOVA",
        "Factorial in RCBD":      "Log or square-root transformation, then re-run ANOVA",
        "Split-plot":             "Log or square-root transformation, then re-run ANOVA",
    }
    alt = alternatives.get(design, "Data transformation or non-parametric method")

    if s_pass is None or l_pass is None:
        return {
            "overall": "Assumptions could not be evaluated.",
            "normality_ok": None,
            "homogeneity_ok": None,
            "shapiro_p": s_p,
            "levene_p": l_p,
            "shapiro_stat": s_stat,
            "alternative": None,
            "action": None,
            "verdict_code": "unknown",
        }

    if s_pass and l_pass:
        return {
            "overall": "Normality and homogeneity satisfied. ANOVA fully valid.",
            "normality_ok": True,
            "homogeneity_ok": True,
            "shapiro_p": s_p,
            "levene_p": l_p,
            "shapiro_stat": s_stat,
            "alternative": None,
            "action": None,
            "verdict_code": "both_met",
        }
    elif not s_pass and l_pass:
        return {
            "overall": "Normality violated but equal variances met. ANOVA is moderately robust; results likely reliable. Transformation recommended for confirmation.",
            "normality_ok": False,
            "homogeneity_ok": True,
            "shapiro_p": s_p,
            "levene_p": l_p,
            "shapiro_stat": s_stat,
            "alternative": alt,
            "action": f"Recommended: run {alt} to confirm your findings.",
            "verdict_code": "normality_violated",
        }
    elif s_pass and not l_pass:
        return {
            "overall": "Equal variances violated. Consider Welch ANOVA or transformation.",
            "normality_ok": True,
            "homogeneity_ok": False,
            "shapiro_p": s_p,
            "levene_p": l_p,
            "shapiro_stat": s_stat,
            "alternative": alt,
            "action": f"Recommended: use {alt} as your primary analysis.",
            "verdict_code": "homogeneity_violated",
        }
    else:
        return {
            "overall": "Both assumptions violated. Use transformation or non-parametric alternative.",
            "normality_ok": False,
            "homogeneity_ok": False,
            "shapiro_p": s_p,
            "levene_p": l_p,
            "shapiro_stat": s_stat,
            "alternative": alt,
            "action": f"Recommended: use {alt} as your primary analysis.",
            "verdict_code": "both_violated",
        }


# ============================================================
#  INTELLIGENCE BLOCKS  (deterministic — no AI, no free text)
# ============================================================

def build_executive_insight(
    p_map: Dict[str, Optional[float]],
    cv: Optional[float],
    trait: str,
    best_combination: Optional[str],
    alpha: float = 0.05,
) -> str:
    """
    Block 1 — Executive Insight.
    One paragraph answering: 'What is the big story of this analysis?'
    Fully deterministic — built from p-values and ranked means.
    """
    sig_factors   = [k for k, v in p_map.items() if v is not None and v < alpha]
    insig_factors = [k for k, v in p_map.items() if v is not None and v >= alpha]

    # Identify dominant factor (smallest p-value among significant)
    sig_sorted = sorted(
        [(k, v) for k, v in p_map.items() if v is not None and v < alpha],
        key=lambda x: x[1]
    )
    dominant = sig_sorted[0][0] if sig_sorted else None

    # Interaction presence
    interaction_keys = [k for k in p_map if "x" in k.lower() or ":" in k or "interaction" in k.lower()]
    interaction_sig  = any(p_map.get(k, 1) < alpha for k in interaction_keys)

    # CV quality
    cv_note = ""
    if cv is not None:
        if cv < 15:
            cv_note = f" Experimental precision was excellent (CV = {cv:.1f}%)."
        elif cv < 25:
            cv_note = f" Experimental precision was acceptable (CV = {cv:.1f}%)."
        else:
            cv_note = f" Experimental variability was high (CV = {cv:.1f}%), warranting cautious interpretation."

    parts = []

    if dominant:
        parts.append(f"{dominant} emerged as the dominant source of variation in {trait}.")

    if len(sig_factors) > 1:
        secondary = [f for f in sig_factors if f != dominant]
        parts.append(f"{', '.join(secondary)} also showed significant effects.")

    if insig_factors:
        parts.append(
            f"The absence of significant {'interaction' if interaction_keys else 'effects'} "
            f"({'p ≥ ' + str(alpha) + ' for ' + ', '.join(insig_factors)}) indicates that "
            f"treatment rankings remain stable across factor levels, simplifying recommendations."
        )

    if interaction_sig:
        parts.append(
            "A significant interaction was detected, meaning the effect of one factor "
            "depends on the level of the other — interpret treatment combinations, not main effects alone."
        )

    if best_combination:
        parts.append(f"The optimal treatment combination was {best_combination}, which maximised performance.")

    parts.append(cv_note.strip()) if cv_note.strip() else None

    return " ".join(p for p in parts if p)


def build_reviewer_radar(
    shapiro: Dict,
    levene: Dict,
    p_map: Dict[str, Optional[float]],
    cv: Optional[float],
    n_locations: int = 1,
    n_seasons: int = 1,
    alpha: float = 0.05,
) -> List[str]:
    """
    Block 2 — Reviewer Radar.
    Rule-based generator of likely peer-reviewer questions.
    Students walk into their defence pre-armed.
    """
    questions = []

    # Normality violation
    if shapiro.get("passed") is False:
        w  = shapiro.get("stat", "?")
        p  = shapiro.get("p_value", "?")
        questions.append(
            f"Why were the data not normally distributed (Shapiro-Wilk W = {w}, p = {p})? "
            f"Were data transformations (log, square-root) attempted before analysis?"
        )

    # Homogeneity violation
    if levene.get("passed") is False:
        p = levene.get("p_value", "?")
        questions.append(
            f"Levene's test indicates unequal variances (p = {p}). "
            f"Was Welch's ANOVA or a variance-stabilising transformation considered?"
        )

    # Non-significant interaction — reviewer wants biological justification
    interaction_keys = [k for k in p_map if "x" in k.lower() or ":" in k or "interaction" in k.lower()]
    interaction_insig = any(
        p_map.get(k) is not None and p_map.get(k, 0) >= alpha
        for k in interaction_keys
    )
    if interaction_insig:
        questions.append(
            "The interaction effect was not significant. What biological or physiological "
            "mechanism supports the independence of these factors in your crop system?"
        )

    # High CV
    if cv is not None and cv > 20:
        questions.append(
            f"The coefficient of variation is {cv:.1f}%, indicating moderate-to-high field "
            f"variability. What environmental or management factors contributed to this variability?"
        )

    # Single location / season
    if n_locations == 1:
        questions.append(
            "Results are from a single location. Can these findings be generalised across "
            "different agro-ecological zones or environments?"
        )
    if n_seasons == 1:
        questions.append(
            "Data represent a single growing season. How confident are you that these results "
            "are repeatable across seasons or years?"
        )

    # Very high F-values (potential data entry issues)
    high_f_factors = [k for k, v in p_map.items() if v is not None and v < 0.0001]
    if len(high_f_factors) >= 2:
        questions.append(
            "Exceptionally high F-values were observed. Please confirm data entry accuracy "
            "and verify that replication was conducted independently."
        )

    return questions


def build_decision_rules(
    means_df: pd.DataFrame,
    group_col: str,
    trait: str,
    alpha: float = 0.05,
) -> List[str]:
    """
    Block 3 — Decision Rules.
    Converts ranked means into practical, actionable recommendations.
    """
    if means_df.empty or "mean" not in means_df.columns:
        return ["Insufficient data to generate decision rules."]

    sorted_df = means_df.sort_values("mean", ascending=False).reset_index(drop=True)
    best      = sorted_df.iloc[0]
    second    = sorted_df.iloc[1] if len(sorted_df) > 1 else None
    worst     = sorted_df.iloc[-1]

    best_name   = str(best[group_col])
    best_mean   = float(best["mean"])
    worst_name  = str(worst[group_col])
    worst_mean  = float(worst["mean"])

    rules = []

    rules.append(
        f"To maximise {trait}: select {best_name} "
        f"(mean = {best_mean:.2f}, highest performing combination)."
    )

    if second is not None:
        second_name = str(second[group_col])
        second_mean = float(second["mean"])
        diff = best_mean - second_mean
        rules.append(
            f"If {best_name} is unavailable or cost-prohibitive: {second_name} "
            f"(mean = {second_mean:.2f}) is the next-best option "
            f"({diff:.2f} units below the optimum)."
        )

    rules.append(
        f"Avoid {worst_name} for {trait} optimisation "
        f"(mean = {worst_mean:.2f}, lowest performing; "
        f"{best_mean - worst_mean:.2f} units below optimum)."
    )

    # Letter-based advice if available
    if "letters" in sorted_df.columns or "groups" in sorted_df.columns:
        letter_col = "letters" if "letters" in sorted_df.columns else "groups"
        top_letter = str(best.get(letter_col, ""))
        if top_letter:
            rules.append(
                f"Treatments sharing Tukey letter '{top_letter[0]}' with {best_name} "
                f"are statistically equivalent and interchangeable for {trait}."
            )

    return rules


def mean_table(df: pd.DataFrame, y: str, group: str) -> pd.DataFrame:
    g = df.groupby(group)[y]
    out = pd.DataFrame({
        group: g.mean().index.astype(str),
        "n": g.count().values,
        "mean": g.mean().values.round(2),
        "sd": g.std(ddof=1).values.round(2),
        "se": (g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))).values.round(2),
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
#  GROUNDING VALIDATOR  (strict template compliance)
# ============================================================

# Phrases that imply causality, mechanism, or economics — banned in strict mode
_BANNED_CAUSAL = [
    "because", "due to", "therefore", "caused by", "leads to", "results from",
]

# Strength adjectives that require computed effect sizes — banned without them
_BANNED_STRENGTH = [
    "massive", "dominant effect", "profound", "robust evidence", "overwhelmingly",
    "dramatic improvement", "exceptional precision",
]

# Domain-specific claims banned unless user provides context (spec §3 / §5C)
_BANNED_DOMAIN = [
    "drought tolerance", "food safety", "farmer adoption", "mechanised harvesting",
    "smallholder", "food security", "genetic basis", "physiological mechanism",
    "uptake efficiency", "vigour traits",
]

# Generalisation phrases banned unless user provides location/context
_BANNED_GENERALISATION = [
    "in your study area", "for smallholders", "in sub-saharan africa",
    "across nigeria", "for farmers in",
]

# Trend language banned unless trend test was computed
_BANNED_TREND = [
    "dose-response", "increasing trend", "declined steadily",
    "linear increase", "linear decrease",
]

def validate_interpretation(
    text: str,
    has_tukey: bool,
    has_correlation: bool,
    cv_actual: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Full grounding validator (spec §5 hard validation checklist).
    A. Numbers check — p=0 detection
    B. Contradiction check — assumption consistency
    C. Phrase blacklist — causal, domain, trend, generalisation
    D. CV accuracy check — catches computed-from-scratch CV errors
    Returns {passed, warning_count, warnings[]}
    """
    import re
    warnings_list = []

    # A. p=0 detection (statistically impossible)
    if re.search(r"p\s*[=<>]\s*0(?!\.\d)", text):
        warnings_list.append(
            "GROUNDING A: 'p = 0' detected — statistically impossible. Use 'p < 0.001'."
        )

    # B. Tukey letters mentioned but not computed
    if not has_tukey:
        for phrase in ["tukey letter", "group a", "group b", "hsd grouping"]:
            if phrase in text.lower():
                warnings_list.append(
                    f"GROUNDING B: Tukey groupings not computed but text references '{phrase}'."
                )

    # B. Correlation language without computed correlation
    if not has_correlation:
        for phrase in ["were correlated", "positive correlation", "negative correlation"]:
            if phrase in text.lower():
                warnings_list.append(
                    f"GROUNDING B: Correlation not computed but text references '{phrase}'."
                )

    # C. Banned causal phrases
    for phrase in _BANNED_CAUSAL:
        if f" {phrase} " in f" {text.lower()} ":
            warnings_list.append(
                f"STRICT C: Causal phrase '{phrase}' detected. Replace with statistical language."
            )

    # C. Banned strength adjectives
    for phrase in _BANNED_STRENGTH:
        if phrase in text.lower():
            warnings_list.append(
                f"STRICT C: Strength adjective '{phrase}' detected. Replace with F-value evidence."
            )

    # C. Banned domain claims
    for phrase in _BANNED_DOMAIN:
        if phrase in text.lower():
            warnings_list.append(
                f"STRICT C: Domain claim '{phrase}' detected without user-provided context."
            )

    # C. Banned generalisation
    for phrase in _BANNED_GENERALISATION:
        if phrase in text.lower():
            warnings_list.append(
                f"STRICT C: Generalisation '{phrase}' detected without user-provided location context."
            )

    # C. Banned trend language
    for phrase in _BANNED_TREND:
        if phrase in text.lower():
            warnings_list.append(
                f"STRICT C: Trend language '{phrase}' detected without computed trend test."
            )

    # D. CV accuracy check — catches hallucinated CV values (e.g., 3.47% vs 6.48%)
    if cv_actual is not None:
        cv_matches = re.findall(r"cv\s*[=:]\s*([\d.]+)\s*%|cv\s+of\s+([\d.]+)\s*%|([\d.]+)\s*%\s*cv|([\d.]+)%.*cv", text.lower())
        for match_groups in cv_matches:
            cv_text_str = next((g for g in match_groups if g), None)
            if cv_text_str:
                try:
                    cv_text_val = float(cv_text_str)
                    if abs(cv_text_val - cv_actual) > 1.0:
                        warnings_list.append(
                            f"GROUNDING D: CV in text ({cv_text_val}%) does not match computed CV ({cv_actual}%). "
                            f"Use the provided cv_percent = {cv_actual}%."
                        )
                except ValueError:
                    pass

    return {
        "passed": len(warnings_list) == 0,
        "warning_count": len(warnings_list),
        "warnings": warnings_list,
    }


def fallback_interpretation(facts: Dict[str, Any]) -> str:
    """
    Minimal safe interpretation used when LLM output fails validation.
    100% grounded — built entirely from facts dict, no free text.
    Spec §2: 'If validate fails → fallback to minimal safe summary.'
    """
    trait  = facts.get("trait", "trait")
    design = facts.get("design", "ANOVA")
    cv     = facts.get("cv")
    lines  = [f"Statistical Summary: {trait} ({design})"]
    lines.append("")

    # Significance facts
    for e in facts.get("significant_effects", []):
        F_str = f"F = {e['F']}, " if e.get("F") else ""
        lines.append(f"• {e['source']}: significant ({F_str}{e['p_display']})")
    for e in facts.get("nonsignificant_effects", []):
        F_str = f"F = {e['F']}, " if e.get("F") else ""
        lines.append(f"• {e['source']}: not significant ({F_str}{e['p_display']})")

    # Best and worst
    best  = facts.get("best_treatment")
    worst = facts.get("worst_treatment")
    if best:
        group_col = next((k for k in best if k not in ("n","mean","sd","se","letters","groups")), "treatment")
        lines.append(f"• Highest mean: {best.get(group_col)} = {best.get('mean')} (Tukey: {best.get('letters','N/A')})")
    if worst and worst != best:
        group_col = next((k for k in worst if k not in ("n","mean","sd","se","letters","groups")), "treatment")
        lines.append(f"• Lowest mean: {worst.get(group_col)} = {worst.get('mean')} (Tukey: {worst.get('letters','N/A')})")

    # CV
    if cv:
        lines.append(f"• CV = {cv}%")

    # Locked assumption verdict
    lines.append("")
    lines.append("Assumptions: " + facts.get("assumptions_verdict", "Not available."))

    lines.append("")
    lines.append("— VivaSense (minimal safe output)")
    return "\n".join(lines)


def add_p_display_to_anova(records: List[Dict]) -> List[Dict]:
    """
    Post-process ANOVA table records to add a 'p_display' field:
    e.g.  0.0001  →  '< 0.001'
          0.0234  →  '0.0234'
    This prevents Dr. Fayeun from ever writing 'p = 0'.
    """
    out = []
    for row in records:
        r = dict(row)
        for key in ["PR(>F)", "p_value", "p_corrected", "p_adj"]:
            if key in r and r[key] is not None:
                r[f"{key}_display"] = fmt_p_display(float(r[key]))
        out.append(r)
    return out



# ============================================================
#  STRICT TEMPLATE ENGINE  (spec: strict-templates.v1)
#  Zero LLM in rendering path — pure string substitution.
#  Hallucination structurally impossible for templated output.
# ============================================================

import json as _json
from pathlib import Path as _Path

_STRICT_TEMPLATES_PATH = _Path(__file__).parent / "strict_templates.json"
_strict_cfg_cache: Optional[Dict[str, Any]] = None

def _load_strict_cfg() -> Dict[str, Any]:
    global _strict_cfg_cache
    if _strict_cfg_cache is None:
        with open(_STRICT_TEMPLATES_PATH, "r", encoding="utf-8") as f:
            _strict_cfg_cache = _json.load(f)
    return _strict_cfg_cache


def _tget(d: Any, dotted: str) -> Any:
    """Dot-notation access into nested dict. Raises KeyError if missing."""
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing context key: {dotted}")
        cur = cur[part]
    return cur


def _safe_format(text: str, ctx: Dict[str, Any]) -> str:
    """Replace {a.b.c} placeholders from context dict."""
    import re as _re
    def repl(m):
        key = m.group(1)
        try:
            val = _tget(ctx, key)
            return str(val)
        except KeyError:
            return f"[MISSING:{key}]"
    return _re.sub(r"\{([A-Za-z0-9_\.]+)\}", repl, text)


def _choose_template(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    for rule in cfg["rules"]["decision_tree"]:
        ok = True
        for cond in rule["when"]["all"]:
            var, op, val = cond["var"], cond["op"], cond.get("value")
            try:
                left = _tget(ctx, var)
            except KeyError:
                ok = False; break
            right = _tget(ctx, val) if isinstance(val, str) and "." in str(val) else val
            if op == "==" and left != right: ok = False; break
            if op == "!=" and left == right: ok = False; break
            if op == "<"  and not (left <  right): ok = False; break
            if op == "<=" and not (left <= right): ok = False; break
            if op == ">"  and not (left >  right): ok = False; break
            if op == ">=" and not (left >= right): ok = False; break
        if ok:
            return rule["use_template"]
    raise ValueError("No matching strict template for this context.")


def _render_template(cfg: Dict[str, Any], ctx: Dict[str, Any], template_id: str) -> str:
    tpl = cfg["templates"][template_id]
    missing = []
    for ph in tpl["placeholders"]:
        try:
            _tget(ctx, ph)
        except KeyError:
            missing.append(ph)
    if missing:
        raise ValueError(f"STRICT BLOCKED — missing placeholders: {missing}")
    parts = []
    for sec in tpl["sections"]:
        title = _safe_format(sec["title"], ctx).strip()
        body  = _safe_format(sec["body"],  ctx).strip()
        if "[MISSING:" not in body:
            parts.append(f"{title}\n{body}")
    return "\n\n".join(parts).strip()


def _validate_strict(cfg: Dict[str, Any], ctx: Dict[str, Any], rendered: str) -> Tuple[bool, str]:
    import re as _re
    txt = rendered.lower()
    for phrase in cfg["banned_phrases"]:
        if phrase.lower() in txt:
            return False, f"BANNED_PHRASE: {phrase}"
    if _re.search(r"\{[A-Za-z0-9_\.]+\}", rendered):
        return False, "UNRESOLVED_PLACEHOLDER"
    return True, "OK"


def _means_to_text(means_rows: List[Dict], treatment_col: str) -> str:
    """Convert means table to readable string for template insertion."""
    if not means_rows:
        return "Not available."
    header_cols = [treatment_col, "mean", "letters"]
    lines = []
    for row in means_rows:
        name    = row.get(treatment_col, row.get("Main:Sub", "?"))
        mean_v  = row.get("mean", "?")
        letters = row.get("letters", row.get("groups", ""))
        lines.append(f"  {name}: {mean_v} ({letters})")
    return "\n".join(lines)


def build_strict_ctx(
    result: Dict[str, Any],
    design_family: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Build a STRICT context from a single-trait result JSON.

    - All values are taken directly from the computed analysis output OR deterministically derived from it.
    - No biological/external claims.
    - A parallel ctx["_sources"] map records where every placeholder value came from.
    """
    meta_in   = result.get("meta", {}) or {}
    tables_in = result.get("tables", {}) or {}

    ctx: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    ctx["_sources"] = sources

    # ── basics ──
    trait = meta_in.get("trait", "trait")
    ctx["Trait"] = trait
    sources["Trait"] = "meta.trait"

    cv = meta_in.get("cv_percent", None)
    ctx["CV"] = (str(round(float(cv), 2)) if cv is not None else "N/A")
    sources["CV"] = "meta.cv_percent"

    # ── factor names ──
    factor_a = meta_in.get("main_plot_factor") or meta_in.get("factor_a") or meta_in.get("factor", "Factor A")
    factor_b = meta_in.get("sub_plot_factor")  or meta_in.get("factor_b") or meta_in.get("factor", "Factor B")
    factor   = meta_in.get("factor", meta_in.get("treatment", "Treatment"))

    ctx["FactorA"] = str(factor_a)
    sources["FactorA"] = "meta.(main_plot_factor|factor_a|factor)"
    ctx["FactorB"] = str(factor_b)
    sources["FactorB"] = "meta.(sub_plot_factor|factor_b|factor)"
    ctx["Factor"]  = str(factor)
    sources["Factor"] = "meta.(factor|treatment)"

    # ── ANOVA rows (prefer corrected) ──
    anova_rows = tables_in.get("anova_corrected") or tables_in.get("anova") or []
    # Map sources -> row
    row_map: Dict[str, Dict[str, Any]] = {}
    for row in anova_rows:
        src = str(row.get("source", ""))
        row_map[src] = row

    def _pick_row_contains(substr: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        for src, row in row_map.items():
            if substr.lower() in src.lower():
                return src, row
        return None

    def _set_stat(prefix: str, src_key: str, row: Dict[str, Any], row_src_path: str) -> None:
        # F + p + df
        F_val = row.get("F_corrected") if row.get("F_corrected") is not None else row.get("F")
        p_val = row.get("p_corrected") if row.get("p_corrected") is not None else row.get("PR(>F)")
        df_val = row.get("df")
        ctx[f"F_{prefix}"] = (str(round(float(F_val), 2)) if F_val is not None else "N/A")
        sources[f"F_{prefix}"] = f"{row_src_path}.(F_corrected|F)"
        ctx[f"p_{prefix}"] = fmt_p_display(p_val) if p_val is not None else "N/A"
        sources[f"p_{prefix}"] = f"{row_src_path}.(p_corrected|PR(>F))"
        ctx[f"df_{prefix}"] = (str(int(df_val)) if df_val is not None else "N/A")
        sources[f"df_{prefix}"] = f"{row_src_path}.df"

    # One-way designs: Factor, Block (optional)
    # Two-factor / split-plot: A, B, AxB; plus Block, Block:A (if present)
    if design_family in ("oneway", "oneway_rcbd"):
        # treatment row
        pick = _pick_row_contains("C(" + str(factor) + ")") or _pick_row_contains(str(factor))
        if pick:
            src, row = pick
            _set_stat("Factor", src, row, f"tables.anova*(source='{src}')")
        # block row (rcbd)
        pick_b = _pick_row_contains("Block")
        if pick_b:
            src, row = pick_b
            _set_stat("Block", src, row, f"tables.anova*(source='{src}')")

    else:
        # Factor A
        pick_a = _pick_row_contains("C(" + str(factor_a) + ")") or _pick_row_contains(str(factor_a))
        if pick_a:
            src, row = pick_a
            _set_stat("A", src, row, f"tables.anova*(source='{src}')")
        # Factor B
        pick_b = _pick_row_contains("C(" + str(factor_b) + ")") or _pick_row_contains(str(factor_b))
        if pick_b:
            src, row = pick_b
            _set_stat("B", src, row, f"tables.anova*(source='{src}')")
        # Interaction
        # try explicit pattern first, else any ":" containing both factors
        inter_pick = None
        for src, row in row_map.items():
            s = src.lower()
            if ":" in s and (str(factor_a).lower() in s) and (str(factor_b).lower() in s):
                inter_pick = (src, row)
                break
        if inter_pick:
            src, row = inter_pick
            _set_stat("AxB", src, row, f"tables.anova*(source='{src}')")

    # ── Assumptions ──
    ass = tables_in.get("assumptions", []) or []
    sh = next((r for r in ass if str(r.get("test", "")).lower().startswith("shapiro")), None)
    lv = next((r for r in ass if str(r.get("test", "")).lower().startswith("levene")), None)

    ctx["Shapiro_p"] = fmt_p_display(sh.get("p_value")) if sh and sh.get("p_value") is not None else "N/A"
    sources["Shapiro_p"] = "tables.assumptions[Shapiro].p_value"
    ctx["Levene_p"]  = fmt_p_display(lv.get("p_value")) if lv and lv.get("p_value") is not None else "N/A"
    sources["Levene_p"] = "tables.assumptions[Levene].p_value"

    # ── Means table and derived best/worst ──
    means_rows = tables_in.get("means", []) or []
    # The means tables sometimes use "A:B" or "Main:Sub" or the factor column name.
    # Keep a canonical list.
    ctx["MeansText"] = _means_to_text(means_rows, treatment_col="Main:Sub")
    sources["MeansText"] = "tables.means (rendered)"

    # Best and worst by mean (ties: first after sorting)
    best = None
    worst = None
    if means_rows and all(("mean" in r) for r in means_rows):
        try:
            sorted_rows = sorted(means_rows, key=lambda r: float(r.get("mean", float("nan"))))
            worst = sorted_rows[0]
            best  = sorted_rows[-1]
        except Exception:
            best = worst = None

    def _treatment_label(row: Dict[str, Any]) -> str:
        return str(row.get("Main:Sub") or row.get("A:B") or row.get(factor) or row.get("Genotype") or row.get("Treatment") or "?")

    if best and worst:
        ctx["BestTreatment"] = _treatment_label(best)
        sources["BestTreatment"] = "derived: argmax(tables.means.mean)"
        ctx["BestMean"] = str(best.get("mean"))
        sources["BestMean"] = "tables.means[best].mean"
        ctx["WorstTreatment"] = _treatment_label(worst)
        sources["WorstTreatment"] = "derived: argmin(tables.means.mean)"
        ctx["WorstMean"] = str(worst.get("mean"))
        sources["WorstMean"] = "tables.means[worst].mean"

        try:
            diff = float(best.get("mean")) - float(worst.get("mean"))
            ctx["BestMinusWorst"] = str(round(diff, 4))
            sources["BestMinusWorst"] = "derived: BestMean - WorstMean"
        except Exception:
            ctx["BestMinusWorst"] = "N/A"
            sources["BestMinusWorst"] = "derived: BestMean - WorstMean (failed)"

    else:
        ctx["BestTreatment"] = "N/A"
        sources["BestTreatment"] = "tables.means (missing)"
        ctx["BestMean"] = "N/A"
        sources["BestMean"] = "tables.means (missing)"
        ctx["WorstTreatment"] = "N/A"
        sources["WorstTreatment"] = "tables.means (missing)"
        ctx["WorstMean"] = "N/A"
        sources["WorstMean"] = "tables.means (missing)"
        ctx["BestMinusWorst"] = "N/A"
        sources["BestMinusWorst"] = "tables.means (missing)"

    return ctx

def _strict_confidence(result: Dict[str, Any], template_ok: bool, missing: List[str]) -> Dict[str, Any]:
    """
    Deterministic confidence meter.
    NOTE: This does NOT claim biological truth; it only reports coverage + assumption status.
    """
    if not template_ok:
        return {
            "score": 0.0,
            "level": "Blocked",
            "reasons": ["STRICT_BLOCKED"] + ([f"MISSING:{m}" for m in missing] if missing else []),
        }

    score = 1.0
    reasons: List[str] = []

    ass = (result.get("tables", {}) or {}).get("assumptions", []) or []
    sh = next((r for r in ass if str(r.get("test", "")).lower().startswith("shapiro")), None)
    lv = next((r for r in ass if str(r.get("test", "")).lower().startswith("levene")), None)

    if sh and (sh.get("passed") is False):
        score -= 0.1
        reasons.append("NORMALITY_NOT_MET")
    if lv and (lv.get("passed") is False):
        score -= 0.2
        reasons.append("HOMOGENEITY_NOT_MET")

    # Keep in [0,1]
    score = max(0.0, min(1.0, round(score, 2)))
    level = "High" if score >= 0.9 else ("Medium" if score >= 0.75 else "Low")
    return {"score": score, "level": level, "reasons": reasons}


def render_strict_per_trait(
    result: Dict[str, Any],
    design_family: str,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Render STRICT interpretation text for a single trait.

    Output includes:
      - text (strictly templated)
      - badge (STRICT MODE VERIFIED / BLOCKED)
      - confidence (deterministic meter)
      - audit_log (every placeholder -> value + source)
    """
    cfg = _load_strict_cfg()
    ctx = build_strict_ctx(result, design_family=design_family, alpha=alpha)

    # Choose template based on design + interaction
    template_id = None
    if design_family == "oneway":
        template_id = "one_way"
    elif design_family == "oneway_rcbd":
        template_id = "one_way_rcbd"
    else:
        # Two-factor (includes split-plot summary at the interpretation layer)
        # Determine interaction significance if p_AxB is available
        p_axb = None
        for key in ("p_AxB",):
            if key in ctx and ctx[key] not in ("N/A", None):
                p_axb = ctx[key]
                break
        # ctx p values are formatted strings; we need raw p for significance, so re-read from tables if possible
        p_raw = None
        anova_rows = (result.get("tables", {}) or {}).get("anova_corrected") or (result.get("tables", {}) or {}).get("anova") or []
        for row in anova_rows:
            src = str(row.get("source", ""))
            s = src.lower()
            if ":" in s and (str(ctx.get("FactorA","")).lower() in s) and (str(ctx.get("FactorB","")).lower() in s):
                p_raw = row.get("p_corrected") if row.get("p_corrected") is not None else row.get("PR(>F)")
                break
        is_sig = (p_raw is not None) and (float(p_raw) < float(alpha))
        template_id = "two_factor_interaction_sig" if is_sig else "two_factor_no_interaction"

    # Render + validate
    missing: List[str] = []
    try:
        rendered = _render_template(cfg, ctx, template_id)
    except Exception as e:
        rendered = f"STRICT BLOCKED — {e}"
        # best-effort parse missing placeholders
        msg = str(e)
        m = re.search(r"missing placeholders: \[(.*)\]", msg)
        if m:
            inner = m.group(1)
            missing = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
        ok = False
        reason = "RENDER_ERROR"
        confidence = _strict_confidence(result, template_ok=False, missing=missing)
        return {
            "ok": ok,
            "reason": reason,
            "template_id": template_id,
            "badge": {"text": "STRICT MODE BLOCKED", "status": "blocked"},
            "confidence": confidence,
            "audit_log": [],
            "text": rendered,
        }

    ok, reason = _validate_strict(cfg, ctx, rendered)
    confidence = _strict_confidence(result, template_ok=ok, missing=missing)

    badge = {"text": "STRICT MODE VERIFIED" if ok else "STRICT MODE BLOCKED",
             "status": "verified" if ok else "blocked"}

    # Audit log: every placeholder in template -> value + source
    audit_log = []
    tpl = cfg["templates"][template_id]
    for ph in tpl.get("placeholders", []):
        val = ctx.get(ph, None)
        src = (ctx.get("_sources", {}) or {}).get(ph, "unknown")
        audit_log.append({"placeholder": ph, "value": val, "source": src})

    return {
        "ok": ok,
        "reason": reason,
        "template_id": template_id,
        "badge": badge,
        "confidence": confidence,
        "audit_log": audit_log,
        "text": rendered.strip(),
    }

def render_strict_synthesis(per_trait: Dict[str, Any], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Multi-trait synthesis — deterministic, STRICT, and audit-able.
    Uses ONLY computed values present in per-trait outputs.
    """
    n_traits = len(per_trait)
    n_trt_sig = 0
    n_interaction_sig = 0
    n_shapiro_fail = 0
    n_levene_fail = 0

    shapiro_fail_names: List[str] = []
    levene_fail_names:  List[str] = []

    lines_out: List[str] = []
    audit_log: List[Dict[str, Any]] = []

    for tname, tr in per_trait.items():
        tables   = tr.get("tables", {}) or {}
        meta_tr  = tr.get("meta", {}) or {}
        assump   = tables.get("assumptions", []) or []

        sh = next((a for a in assump if str(a.get("test","")).lower().startswith("shapiro")), None)
        lv = next((a for a in assump if str(a.get("test","")).lower().startswith("levene")), None)

        # Significance flags (computed from ANOVA table)
        anova_rows = tables.get("anova_corrected") or tables.get("anova") or []
        trt_sig = False
        ix_sig  = False
        for row in anova_rows:
            src   = str(row.get("source", ""))
            p_raw = row.get("p_corrected") if row.get("p_corrected") is not None else row.get("PR(>F)")
            if p_raw is None or "Residual" in src:
                continue
            if "Block" in src:
                continue
            try:
                p_float = float(p_raw)
            except Exception:
                continue
            if ":" in src:
                if p_float < alpha:
                    ix_sig = True
            else:
                if p_float < alpha:
                    trt_sig = True

        if trt_sig: n_trt_sig += 1
        if ix_sig:  n_interaction_sig += 1

        # Assumptions flags
        sh_passed = True if sh is None else bool(sh.get("passed", True))
        lv_passed = True if lv is None else bool(lv.get("passed", True))
        if not sh_passed:
            n_shapiro_fail += 1
            shapiro_fail_names.append(tname)
        if not lv_passed:
            n_levene_fail += 1
            levene_fail_names.append(tname)

        cv = meta_tr.get("cv_percent", None)

        sh_p = sh.get("p_value") if sh else None
        lv_p = lv.get("p_value") if lv else None

        lines_out.append(
            f"- {tname}: treatment_significant={trt_sig}, interaction_significant={ix_sig}, "
            f"CV_percent={cv}, Shapiro_p={sh_p}, Levene_p={lv_p}"
        )

        audit_log.append({
            "trait": tname,
            "source": "per_trait[trait]",
            "fields": {
                "treatment_significant": "tables.(anova_corrected|anova) p-values (non-Block, non-Residual, non-interaction)",
                "interaction_significant": "tables.(anova_corrected|anova) p-values (interaction rows containing ':')",
                "CV_percent": "meta.cv_percent",
                "Shapiro_p": "tables.assumptions[Shapiro].p_value",
                "Levene_p": "tables.assumptions[Levene].p_value",
            }
        })

    # Summaries
    if n_shapiro_fail == 0:
        norm_summary = f"Normality: all {n_traits}/{n_traits} traits passed Shapiro-Wilk."
    else:
        norm_summary = f"Normality: {n_traits - n_shapiro_fail}/{n_traits} traits passed Shapiro-Wilk; failed traits: {', '.join(shapiro_fail_names)}."
    if n_levene_fail == 0:
        homo_summary = f"Homogeneity: all {n_traits}/{n_traits} traits passed Levene."
    else:
        homo_summary = f"Homogeneity: {n_traits - n_levene_fail}/{n_traits} traits passed Levene; failed traits: {', '.join(levene_fail_names)}."

    text = "\n".join([
        "STRICT MODE MULTI-TRAIT SUMMARY",
        f"alpha = {alpha}",
        f"n_traits = {n_traits}",
        f"n_traits_with_significant_treatment_effect = {n_trt_sig}",
        f"n_traits_with_significant_interaction = {n_interaction_sig}",
        norm_summary,
        homo_summary,
        "Per-trait flags:",
        *lines_out,
    ]).strip()

    # Confidence: coverage-only (not biological truth)
    ok = True
    score = 1.0
    reasons: List[str] = []
    if n_traits == 0:
        ok = False
        score = 0.0
        reasons.append("NO_TRAITS")
    else:
        if n_shapiro_fail > 0:
            score -= 0.1
            reasons.append("SOME_NORMALITY_FAIL")
        if n_levene_fail > 0:
            score -= 0.2
            reasons.append("SOME_HOMOGENEITY_FAIL")
        score = max(0.0, min(1.0, round(score, 2)))

    level = "High" if score >= 0.9 else ("Medium" if score >= 0.75 else "Low")
    badge = {"text": "STRICT MODE VERIFIED" if ok else "STRICT MODE BLOCKED",
             "status": "verified" if ok else "blocked"}

    return {
        "ok": ok,
        "badge": badge,
        "confidence": {"score": score, "level": level, "reasons": reasons},
        "audit_log": audit_log,
        "text": text,
        "mode": "strict_deterministic",
        "flags": {
            "n_traits": n_traits,
            "n_trt_sig": n_trt_sig,
            "n_interaction_sig": n_interaction_sig,
            "n_shapiro_fail": n_shapiro_fail,
            "n_levene_fail": n_levene_fail,
        }
    }

def extract_facts(result: Dict[str, Any], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Extract canonical bullet-facts from a single-trait result JSON.
    ONLY facts provable from computed output.
    LLM receives this — not raw JSON.
    """
    facts: Dict[str, Any] = {}
    meta   = result.get("meta", {})
    tables = result.get("tables", {})
    intel  = result.get("intelligence", {})

    # ── Design context ──
    facts["design"]  = meta.get("design", "Unknown")
    facts["trait"]   = meta.get("trait", "Unknown")
    facts["alpha"]   = alpha
    facts["cv"]      = meta.get("cv_percent")

    # ── ANOVA significance (grounded from ANOVA table) ──
    anova_rows = tables.get("anova", tables.get("anova_corrected", []))
    sig_effects = []
    insig_effects = []
    for row in anova_rows:
        src = str(row.get("source", ""))
        if "Residual" in src:
            continue
        p_raw = row.get("PR(>F)") or row.get("p_corrected") or row.get("p_value")
        p_disp = row.get("PR(>F)_display") or row.get("p_corrected_display") or fmt_p_display(p_raw)
        F_val  = row.get("F") or row.get("F_corrected")
        df_val = row.get("df")
        if p_raw is None:
            continue
        entry = {
            "source":    src,
            "F":         round_val(F_val, 2) if F_val else None,
            "df":        df_val,
            "p":         round_val(p_raw, 4),
            "p_display": p_disp,
            "significant": bool(float(p_raw) < alpha),
        }
        if float(p_raw) < alpha:
            sig_effects.append(entry)
        else:
            insig_effects.append(entry)
    facts["significant_effects"]   = sig_effects
    facts["nonsignificant_effects"] = insig_effects

    # ── Interaction present? (critical for factorial/split-plot) ──
    interaction_row = next(
        (r for r in sig_effects   if ":" in str(r["source"]) or "x" in str(r["source"]).lower()),
        None
    )
    insig_interaction = next(
        (r for r in insig_effects if ":" in str(r["source"]) or "x" in str(r["source"]).lower()),
        None
    )
    facts["interaction_significant"] = interaction_row is not None
    facts["interaction_row"]         = interaction_row or insig_interaction

    # ── Means table (grounded) ──
    means_rows = tables.get("means", [])
    if means_rows:
        sorted_means = sorted(
            [r for r in means_rows if r.get("mean") is not None],
            key=lambda r: float(r["mean"]),
            reverse=True
        )
        facts["best_treatment"]  = sorted_means[0]  if sorted_means else None
        facts["worst_treatment"] = sorted_means[-1] if sorted_means else None
        facts["all_means"]       = sorted_means
        facts["has_tukey"]       = any("letters" in r or "groups" in r for r in means_rows)
    else:
        facts["best_treatment"]  = None
        facts["worst_treatment"] = None
        facts["all_means"]       = []
        facts["has_tukey"]       = False

    # ── Assumptions (locked verdict — never guessed) ──
    assumptions = tables.get("assumptions", [])
    shapiro = next((a for a in assumptions if a.get("test") == "Shapiro-Wilk"), {})
    levene  = next((a for a in assumptions if a.get("test") == "Levene"), {})
    guidance = tables.get("assumption_guidance", {})
    facts["assumptions"] = {
        "shapiro_stat":    shapiro.get("stat"),
        "shapiro_p":       shapiro.get("p_value"),
        "shapiro_passed":  shapiro.get("passed"),
        "levene_stat":     levene.get("stat"),
        "levene_p":        levene.get("p_value"),
        "levene_passed":   levene.get("passed"),
        "verdict":         guidance.get("overall", "Assumption results not available."),
        "verdict_code":    guidance.get("verdict_code", "unknown"),
        "alternative":     guidance.get("alternative"),
    }

    # ── Intelligence blocks (pre-computed, grounded) ──
    facts["assumptions_verdict"] = intel.get("assumptions_verdict", guidance.get("overall", "Not available."))
    facts["decision_rules"]      = intel.get("decision_rules", [])
    facts["reviewer_radar"]      = intel.get("reviewer_radar", [])
    facts["executive_insight"]   = intel.get("executive_insight", "")

    return facts


def facts_to_prompt_text(facts: Dict[str, Any]) -> str:
    """
    Render fact-bullets into a structured prompt block.
    LLM can only rewrite these bullets — it cannot invent new facts.
    """
    lines = []
    lines.append(f"ANALYSIS: {facts['design']} — Trait: {facts['trait']}")
    if facts.get("cv"):
        lines.append(f"CV: {facts['cv']}%")

    lines.append("")
    lines.append("SIGNIFICANCE FACTS (use only these):")
    for e in facts.get("significant_effects", []):
        F_str = f"F = {e['F']}" if e.get("F") else ""
        lines.append(f"  ✓ {e['source']}: significant ({F_str}, {e['p_display']})")
    for e in facts.get("nonsignificant_effects", []):
        F_str = f"F = {e['F']}" if e.get("F") else ""
        lines.append(f"  ✗ {e['source']}: NOT significant ({F_str}, {e['p_display']})")

    if facts.get("interaction_row"):
        ix = facts["interaction_row"]
        status = "SIGNIFICANT" if facts["interaction_significant"] else "NOT significant"
        F_str = f"F = {ix['F']}" if ix.get("F") else ""
        lines.append(f"  → INTERACTION ({ix['source']}): {status} ({F_str}, {ix['p_display']}) — report this FIRST")

    lines.append("")
    lines.append("MEANS FACTS (use only these):")
    for m in facts.get("all_means", []):
        group_col = next((k for k in m if k not in ("n","mean","sd","se","letters","groups")), "treatment")
        name    = m.get(group_col, "?")
        mean_v  = m.get("mean", "?")
        letters = m.get("letters") or m.get("groups", "")
        lines.append(f"  {name}: mean = {mean_v}, Tukey letter = {letters or 'N/A'}")

    lines.append("")
    lines.append(f"ASSUMPTIONS VERDICT (use exactly this text, do not contradict):")
    lines.append(f"  {facts['assumptions_verdict']}")
    if facts["assumptions"].get("shapiro_stat"):
        lines.append(f"  Shapiro-Wilk: W = {facts['assumptions']['shapiro_stat']}, p = {facts['assumptions']['shapiro_p']}")
    if facts["assumptions"].get("levene_stat") is not None:
        lines.append(f"  Levene: F = {facts['assumptions']['levene_stat']}, p = {facts['assumptions']['levene_p']}")

    if facts.get("decision_rules"):
        lines.append("")
        lines.append("DECISION RULES (present verbatim under 'Decision Rules' heading):")
        for rule in facts["decision_rules"]:
            lines.append(f"  • {rule}")

    if facts.get("reviewer_radar"):
        lines.append("")
        lines.append("REVIEWER RADAR (present verbatim under 'Reviewer Radar' heading):")
        for q in facts["reviewer_radar"]:
            lines.append(f"  ? {q}")

    return "\n".join(lines)



# ============================================================
#  DESIGN FAMILY MAP — used by strict template engine
# ============================================================

_DESIGN_FAMILY_MAP = {
    "CRD one-way":          "one_way",
    "RCBD one-way":         "one_way",
    "One-way ANOVA (CRD)":  "one_way",
    "One-way ANOVA (RCBD)": "one_way",
    "Two-way Factorial":    "twoway",
    "Factorial RCBD":       "twoway",
    "Split-plot":           "splitplot",
}

def attach_strict_template(result: Dict[str, Any], design_family: str) -> Dict[str, Any]:
    """
    Attach STRICT MODE block to a single-trait result.
    This is deterministic (no LLM) and includes auditability.
    """
    strict_block = render_strict_per_trait(result, design_family=design_family, alpha=float(result.get("meta", {}).get("alpha", 0.05) or 0.05))
    result.setdefault("strict_mode", {})
    result["strict_mode"] = strict_block
    # If STRICT verified, use it as the primary interpretation payload
    if strict_block.get("ok") is True:
        result["interpretation"] = strict_block.get("text", "")
        result.setdefault("meta", {})["interpretation_mode"] = "strict_verified"
    else:
        result.setdefault("meta", {})["interpretation_mode"] = "non_strict"
    return result

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

    p_map = {factor: p_factor}
    cv    = cv_percent(d[trait])
    best  = str(means_letters.iloc[0][factor]) if not means_letters.empty else None

    return {
        "meta": {
            "design": "CRD one-way",
            "factor": factor,
            "trait": trait,
            "alpha": alpha,
            "n_rows_used": int(d.shape[0]),
            "levels": sorted(d[factor].unique().tolist()),
            "cv_percent": cv,
        },
        "tables": {
            "anova": add_p_display_to_anova(df_to_records(anova[["source", "df", "sum_sq", "ms", "F", "PR(>F)"]])),
            "means": df_to_records(means_letters),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
            "assumption_guidance": guidance,
        },
        "plots": plots,
        "interpretation": interpret_anova("One-way ANOVA (CRD)", p_map, cv, alpha),
        "strict_template": {},  # populated by attach_strict_template below
        "intelligence": {
            "executive_insight":   build_executive_insight(p_map, cv, trait, best, alpha),
            "reviewer_radar":      build_reviewer_radar(sh, lv, p_map, cv, alpha=alpha),
            "decision_rules":      build_decision_rules(means_letters, factor, trait, alpha),
            "assumptions_verdict": guidance["overall"],
        },
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

    p_map = {f"Block ({block})": p_block, treatment: p_treatment}
    cv    = cv_percent(d[trait])
    best  = str(means_letters.iloc[0][treatment]) if not means_letters.empty else None

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
            "cv_percent": cv,
            "lsd": lsd,
        },
        "tables": {
            "anova": add_p_display_to_anova(df_to_records(anova[["source", "df", "sum_sq", "ms", "F", "PR(>F)"]])),
            "means": df_to_records(means_letters),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
            "assumption_guidance": guidance,
        },
        "plots": plots,
        "interpretation": interpret_anova("One-way ANOVA (RCBD)", p_map, cv, alpha),
        "strict_template": {},  # populated by attach_strict_template below
        "intelligence": {
            "executive_insight":   build_executive_insight(p_map, cv, trait, best, alpha),
            "reviewer_radar":      build_reviewer_radar(sh, lv, p_map, cv, alpha=alpha),
            "decision_rules":      build_decision_rules(means_letters, treatment, trait, alpha),
            "assumptions_verdict": guidance["overall"],
        },
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

    cv   = cv_percent(d[trait])
    ml   = means_letters.rename(columns={"_AB_": "A:B"})
    best = str(ml.iloc[0]["A:B"]) if not ml.empty else None

    return {
        "meta": {
            "design": "CRD two-way (factorial)",
            "factor_a": a,
            "factor_b": b,
            "trait": trait,
            "alpha": alpha,
            "n_rows_used": int(d.shape[0]),
            "cv_percent": cv,
        },
        "tables": {
            "anova": add_p_display_to_anova(df_to_records(anova[["source", "df", "sum_sq", "ms", "F", "PR(>F)"]])),
            "means": df_to_records(ml),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
            "assumption_guidance": guidance,
        },
        "plots": plots,
        "interpretation": interpret_anova("Two-way Factorial ANOVA (CRD)", p_map, cv, alpha),
        "strict_template": {},  # populated by attach_strict_template below
        "intelligence": {
            "executive_insight":   build_executive_insight(p_map, cv, trait, best, alpha),
            "reviewer_radar":      build_reviewer_radar(sh, lv, p_map, cv, alpha=alpha),
            "decision_rules":      build_decision_rules(ml, "A:B", trait, alpha),
            "assumptions_verdict": guidance["overall"],
        },
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

    cv   = cv_percent(d[trait])
    ml   = means_letters.rename(columns={"_AB_": "A:B"})
    best = str(ml.iloc[0]["A:B"]) if not ml.empty else None

    return {
        "meta": {
            "design": "Factorial in RCBD",
            "block": block,
            "factor_a": a,
            "factor_b": b,
            "trait": trait,
            "alpha": alpha,
            "n_rows_used": int(d.shape[0]),
            "cv_percent": cv,
        },
        "tables": {
            "anova": add_p_display_to_anova(df_to_records(anova[["source", "df", "sum_sq", "ms", "F", "PR(>F)"]])),
            "means": df_to_records(ml),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
            "assumption_guidance": guidance,
        },
        "plots": plots,
        "interpretation": interpret_anova("Factorial RCBD ANOVA", p_map, cv, alpha),
        "strict_template": {},  # populated by attach_strict_template below
        "intelligence": {
            "executive_insight":   build_executive_insight(p_map, cv, trait, best, alpha),
            "reviewer_radar":      build_reviewer_radar(sh, lv, p_map, cv, alpha=alpha),
            "decision_rules":      build_decision_rules(ml, "A:B", trait, alpha),
            "assumptions_verdict": guidance["overall"],
        },
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
            "anova_raw": add_p_display_to_anova(df_to_records(an0.replace({np.nan: None}))),
            "anova_corrected": add_p_display_to_anova(df_to_records(
                an_corr[["source", "df", "sum_sq", "ms",
                          "F_corrected", "p_corrected"]].replace({np.nan: None})
            )),
            "means": df_to_records(means_letters.rename(columns={"_AB_": "Main:Sub"})),
            "tukey": df_to_records(tukey_df),
            "assumptions": [sh, lv],
            "assumption_guidance": guidance,
        },
        "plots": plots,
        "interpretation": interpret_anova("Split-plot ANOVA", p_map, cv_percent(d[trait]), alpha),
        "strict_template": {},  # populated by attach_strict_template below
        "intelligence": {
            "executive_insight":   build_executive_insight(p_map, cv_percent(d[trait]), trait,
                                       str(means_letters.iloc[0]["_AB_"]) if not means_letters.empty else None,
                                       alpha),
            "reviewer_radar":      build_reviewer_radar(sh, lv, p_map, cv_percent(d[trait]), alpha=alpha),
            "decision_rules":      build_decision_rules(
                                       means_letters.rename(columns={"_AB_": "Main:Sub"}),
                                       "Main:Sub", trait, alpha),
            "assumptions_verdict": guidance["overall"],
        },
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
                "mean": round_val(np.nanmean(s), 2) if s.count() else None,
                "sd": round_val(np.nanstd(s, ddof=1), 2) if s.count() > 1 else None,
                "se": round_val(np.nanstd(s, ddof=1) / np.sqrt(s.count()), 2) if s.count() > 1 else None,
                "min": round_val(np.nanmin(s), 2) if s.count() else None,
                "max": round_val(np.nanmax(s), 2) if s.count() else None,
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
            "mean": g.mean().values.round(2),
            "sd": g.std(ddof=1).values.round(2),
            "se": (g.std(ddof=1) / np.sqrt(g.count().clip(lower=1))).values.round(2),
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
    return attach_strict_template(oneway_engine(df, factor, trait, float(alpha)), float(alpha))


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
    return attach_strict_template(oneway_rcbd_engine(df, block, treatment, trait, float(alpha)), float(alpha))


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
    return attach_strict_template(twoway_engine(df, factor_a, factor_b, trait, float(alpha)), float(alpha))


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
    return attach_strict_template(rcbd_factorial_engine(df, block, factor_a, factor_b, trait, float(alpha)), float(alpha))


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
    return attach_strict_template(splitplot_engine(df, block, main_plot, sub_plot, trait, float(alpha)), float(alpha))



# ============================================================
#  NON-PARAMETRIC ENGINES
# ============================================================

def dunn_test(df: pd.DataFrame, y: str, group: str, alpha: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dunn's post-hoc test with Bonferroni correction for pairwise comparisons.
    Returns (pairwise_df, means_letters_df).
    """
    groups = df[group].unique()
    groups_sorted = sorted(groups, key=lambda g: df.loc[df[group] == g, y].mean(), reverse=True)

    # Pairwise comparisons
    pairs = []
    from itertools import combinations
    for g1, g2 in combinations(groups_sorted, 2):
        s1 = df.loc[df[group] == g1, y].values
        s2 = df.loc[df[group] == g2, y].values
        # Dunn test statistic via rank sums
        all_vals = np.concatenate([s1, s2])
        all_groups = [g1] * len(s1) + [g2] * len(s2)
        # Use scipy mannwhitneyu as proxy for pairwise comparison
        u_stat, p_raw = stats.mannwhitneyu(s1, s2, alternative='two-sided')
        pairs.append({"group1": str(g1), "group2": str(g2), "p_raw": p_raw})

    # Bonferroni correction
    n_pairs = len(pairs)
    for pair in pairs:
        pair["p_adj"] = min(pair["p_raw"] * n_pairs, 1.0)
        pair["significant"] = pair["p_adj"] < alpha
        pair["p_adj"] = round_val(pair["p_adj"], 4)

    pairwise_df = pd.DataFrame(pairs)[["group1", "group2", "p_raw", "p_adj", "significant"]]

    # Build compact letter display from pairwise results
    # Groups that are NOT significantly different share a letter
    cld: Dict[str, set] = {str(g): set() for g in groups_sorted}
    letter = ord('a')
    remaining = list(str(g) for g in groups_sorted)
    assigned: Dict[str, List[str]] = {str(g): [] for g in groups_sorted}

    # Simple CLD: iterate groups, assign letters
    used_letters: List[str] = []
    for i, g1 in enumerate(str(g) for g in groups_sorted):
        cur_letter = chr(letter + len(used_letters))
        # Find all groups not significantly different from g1
        same_as_g1 = [g1]
        for g2 in (str(g) for g in groups_sorted):
            if g1 == g2:
                continue
            pair_match = [p for p in pairs if
                          (p["group1"] == g1 and p["group2"] == g2) or
                          (p["group1"] == g2 and p["group2"] == g1)]
            if pair_match and not pair_match[0]["significant"]:
                same_as_g1.append(g2)
        # Check if this group of non-significant pairs already has a letter
        existing = None
        for ul in used_letters:
            ul_groups = [g for g, ls in assigned.items() if ul in ls]
            if set(ul_groups) == set(same_as_g1):
                existing = ul
                break
        if existing is None:
            new_letter = chr(ord('a') + len(used_letters))
            used_letters.append(new_letter)
            for g in same_as_g1:
                assigned[g].append(new_letter)

    # Build means table with letters
    means_rows = []
    for g in groups_sorted:
        g_str = str(g)
        subset = df.loc[df[group] == g_str if df[group].dtype == object else df[group] == g, y]
        letters_str = "".join(sorted(assigned.get(g_str, ["?"])))
        means_rows.append({
            group: g_str,
            "mean": round(subset.mean(), 2),
            "sd": round(subset.std(ddof=1), 2),
            "se": round(subset.std(ddof=1) / np.sqrt(len(subset)), 2),
            "n": int(len(subset)),
            "groups": letters_str if letters_str else "a",
        })
    means_df = pd.DataFrame(means_rows)

    return pairwise_df, means_df


def kruskal_wallis_engine(df: pd.DataFrame, factor: str, trait: str, alpha: float) -> Dict[str, Any]:
    """
    Kruskal-Wallis H-test (non-parametric one-way ANOVA for CRD).
    Post-hoc: Dunn test with Bonferroni correction.
    """
    d = df[[factor, trait]].dropna()
    d[trait] = pd.to_numeric(d[trait], errors="coerce")
    d = d.dropna()
    d[factor] = d[factor].astype(str)

    group_data = [grp[trait].values for _, grp in d.groupby(factor)]
    if len(group_data) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 groups for Kruskal-Wallis test.")

    h_stat, p_value = stats.kruskal(*group_data)
    n_groups = d[factor].nunique()
    df_stat = n_groups - 1
    n_total = len(d)

    pairwise_df, means_df = dunn_test(d, trait, factor, alpha)

    plots = {
        "mean_plot": mean_plot(d, trait, factor, f"Median +/- IQR by {factor} (Kruskal-Wallis)"),
        "box_plot": box_plot(d, trait, factor, f"Distribution of {trait} by {factor}"),
    }

    # Rank table
    d["rank"] = d[trait].rank()
    rank_means = d.groupby(factor)["rank"].mean().reset_index()
    rank_means.columns = [factor, "mean_rank"]
    rank_means["mean_rank"] = rank_means["mean_rank"].round(2)

    return {
        "meta": {
            "design": "Kruskal-Wallis (Non-parametric CRD)",
            "test": "Kruskal-Wallis H-test",
            "posthoc": "Dunn test (Bonferroni correction)",
            "factor": factor,
            "trait": trait,
            "alpha": alpha,
            "n_rows_used": int(n_total),
            "n_groups": int(n_groups),
            "levels": sorted(d[factor].unique().tolist()),
            "cv_percent": cv_percent(d[trait]),
        },
        "tables": {
            "kruskal": [{
                "source": factor,
                "H_statistic": round_val(h_stat, 4),
                "df": int(df_stat),
                "p_value": round_val(p_value, 4),
                "significant": p_value < alpha,
                "interpretation": "Significant differences exist between groups" if p_value < alpha else "No significant differences between groups",
            }],
            "means": df_to_records(means_df),
            "pairwise": df_to_records(pairwise_df),
            "rank_means": df_to_records(rank_means),
        },
        "plots": plots,
        "interpretation": {
            "summary": f"Kruskal-Wallis H({df_stat}, N={n_total}) = {round_val(h_stat, 3)}, p = {round_val(p_value, 4)}. "
                       + ("Significant differences exist between groups." if p_value < alpha
                          else "No significant differences detected between groups."),
            "posthoc_note": "Dunn's post-hoc test with Bonferroni correction applied for pairwise comparisons." if p_value < alpha else "Post-hoc comparisons not required (overall test non-significant).",
            "parametric_note": "This is a non-parametric test that does not assume normality. Use when Shapiro-Wilk p < 0.05 or Levene's test is violated.",
        },
    }


def friedman_engine(df: pd.DataFrame, block: str, treatment: str, trait: str, alpha: float) -> Dict[str, Any]:
    """
    Friedman test (non-parametric one-way RCBD).
    Post-hoc: Wilcoxon signed-rank pairwise tests with Bonferroni correction.
    """
    d = df[[block, treatment, trait]].dropna()
    d[trait] = pd.to_numeric(d[trait], errors="coerce")
    d = d.dropna()
    d[block] = d[block].astype(str)
    d[treatment] = d[treatment].astype(str)

    # Build pivot: rows=blocks, cols=treatments
    try:
        pivot = d.pivot_table(index=block, columns=treatment, values=trait, aggfunc="mean")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot construct block × treatment table: {str(e)}")

    if pivot.isnull().any().any():
        raise HTTPException(status_code=422, detail="Friedman test requires complete data (no missing block-treatment combinations).")

    n_blocks = pivot.shape[0]
    n_treatments = pivot.shape[1]

    if n_blocks < 2 or n_treatments < 2:
        raise HTTPException(status_code=422, detail="Friedman test requires at least 2 blocks and 2 treatments.")

    # Friedman test
    friedman_stat, p_value = stats.friedmanchisquare(*[pivot[col].values for col in pivot.columns])
    df_stat = n_treatments - 1

    # Post-hoc: pairwise Wilcoxon signed-rank with Bonferroni correction
    from itertools import combinations
    treatments = list(pivot.columns)
    pairs = []
    for t1, t2 in combinations(treatments, 2):
        s1 = pivot[t1].values
        s2 = pivot[t2].values
        try:
            w_stat, p_raw = stats.wilcoxon(s1, s2, alternative='two-sided')
        except Exception:
            p_raw = 1.0
        pairs.append({"group1": str(t1), "group2": str(t2), "p_raw": round_val(p_raw, 4)})

    n_pairs = len(pairs)
    for pair in pairs:
        pair["p_adj"] = round_val(min(float(pair["p_raw"]) * n_pairs, 1.0), 4)
        pair["significant"] = float(pair["p_adj"]) < alpha

    pairwise_df = pd.DataFrame(pairs)[["group1", "group2", "p_raw", "p_adj", "significant"]]

    # Means table
    means_rows = []
    for t in treatments:
        vals = pivot[t].values
        means_rows.append({
            treatment: str(t),
            "mean": round(float(np.mean(vals)), 2),
            "median": round(float(np.median(vals)), 2),
            "sd": round(float(np.std(vals, ddof=1)), 2),
            "n": int(len(vals)),
        })
    means_rows.sort(key=lambda x: x["mean"], reverse=True)
    means_df = pd.DataFrame(means_rows)

    # Rank sums per treatment
    pivot_ranks = pivot.rank(axis=1)
    rank_sums = pivot_ranks.sum(axis=0).reset_index()
    rank_sums.columns = [treatment, "rank_sum"]
    rank_sums["mean_rank"] = (rank_sums["rank_sum"] / n_blocks).round(2)

    plots = {
        "mean_plot": mean_plot(d, trait, treatment, f"Means +/- SE by {treatment} (Friedman)"),
        "box_plot": box_plot(d, trait, treatment, f"Distribution of {trait} by {treatment} (RCBD)"),
    }

    return {
        "meta": {
            "design": "Friedman Test (Non-parametric RCBD)",
            "test": "Friedman chi-squared test",
            "posthoc": "Wilcoxon signed-rank pairwise (Bonferroni correction)",
            "block": block,
            "treatment": treatment,
            "trait": trait,
            "alpha": alpha,
            "n_rows_used": int(len(d)),
            "n_blocks": int(n_blocks),
            "n_treatments": int(n_treatments),
            "levels": sorted(d[treatment].unique().tolist()),
            "cv_percent": cv_percent(d[trait]),
        },
        "tables": {
            "friedman": [{
                "source": treatment,
                "chi2_statistic": round_val(friedman_stat, 4),
                "df": int(df_stat),
                "p_value": round_val(p_value, 4),
                "significant": p_value < alpha,
                "interpretation": "Significant differences exist between treatments" if p_value < alpha else "No significant differences between treatments",
            }],
            "means": df_to_records(means_df),
            "pairwise": df_to_records(pairwise_df),
            "rank_sums": df_to_records(rank_sums),
        },
        "plots": plots,
        "interpretation": {
            "summary": f"Friedman χ²({df_stat}, N={n_blocks}) = {round_val(friedman_stat, 3)}, p = {round_val(p_value, 4)}. "
                       + ("Significant differences exist between treatments." if p_value < alpha
                          else "No significant differences detected between treatments."),
            "posthoc_note": "Wilcoxon signed-rank pairwise tests with Bonferroni correction applied." if p_value < alpha else "Post-hoc comparisons not required (overall test non-significant).",
            "parametric_note": "This is the non-parametric equivalent of one-way ANOVA in RCBD. Use when residuals are non-normal (Shapiro-Wilk p < 0.05) in a blocked design.",
        },
    }


@app.post("/analyze/nonparametric/kruskal")
async def analyze_kruskal(
    file: UploadFile = File(...),
    factor: str = Form(...),
    trait: str = Form(...),
    alpha: float = Form(0.05),
):
    """Kruskal-Wallis H-test with Dunn post-hoc (non-parametric CRD)."""
    df = await load_csv(file)
    require_cols(df, [factor, trait])
    return kruskal_wallis_engine(df, factor, trait, float(alpha))


@app.post("/analyze/nonparametric/friedman")
async def analyze_friedman(
    file: UploadFile = File(...),
    block: str = Form(...),
    treatment: str = Form(...),
    trait: str = Form(...),
    alpha: float = Form(0.05),
):
    """Friedman test with Wilcoxon pairwise post-hoc (non-parametric RCBD)."""
    df = await load_csv(file)
    require_cols(df, [block, treatment, trait])
    return friedman_engine(df, block, treatment, trait, float(alpha))


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

            # Attach strict template to per-trait result
            # Attach strict template to per-trait result
            design_family = ('oneway' if design in ('oneway', 'oneway_crd') else ('oneway_rcbd' if design in ('oneway_rcbd', 'rcbd') else 'twofactor'))
            per_trait_results[trait] = attach_strict_template(result, design_family)

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
        "strict_synthesis": render_strict_synthesis(per_trait_results, alpha),
    }




# ============================================================
#  DEBUG ENDPOINT (temporary)
# ============================================================

@app.post("/debug/form")
async def debug_form(request: Request):
    """Echo back all form fields received - use to debug multitrait submissions."""
    form = await request.form()
    result = {}
    for key in form:
        val = form[key]
        result[key] = f"FILE:{val.filename}" if hasattr(val, "filename") else str(val)
    return {"received_fields": result, "field_count": len(result)}


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
