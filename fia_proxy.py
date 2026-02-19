"""
VivaSense + FIA AI Proxy — FastAPI Backend
==========================================
Deployment: Add this file to your existing Render project
Environment variable required: ANTHROPIC_API_KEY

Endpoints:
  POST /api/chat        — CSP 811 AI Tutor (Dr. Fayeun)
  POST /api/interpret   — VivaSense AI interpretation
  POST /api/followup    — VivaSense post-analysis chat
  GET  /api/health      — Health check

Install dependencies (add to requirements.txt):
  fastapi
  uvicorn
  httpx
  python-dotenv
"""

import os
import json
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="FIA AI Proxy", version="1.0.0")

# ── CORS: allow your Lovable domain ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fieldtoinsightacademy.com.ng",
        "https://www.fieldtoinsightacademy.com.ng",
        "http://localhost:3000",   # for local testing
        "http://localhost:5173",   # Vite dev server
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# ── Config ──────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL       = "claude-haiku-4-5-20251001"   # cost-optimised
SONNET_MODEL      = "claude-sonnet-4-20250514"     # higher quality

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")

ANTHROPIC_HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
}

# ── Rate limiting (simple in-memory — upgrade to Redis for scale) ──
from collections import defaultdict
from datetime import datetime, timedelta

request_counts: dict = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 20  # per IP

def is_rate_limited(ip: str) -> bool:
    now = datetime.now()
    cutoff = now - timedelta(minutes=1)
    request_counts[ip] = [t for t in request_counts[ip] if t > cutoff]
    if len(request_counts[ip]) >= MAX_REQUESTS_PER_MINUTE:
        return True
    request_counts[ip].append(now)
    return False


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════

CSP811_SYSTEM = """You are Dr. Fayeun, AI tutor for CSP 811 Biometrical Genetics at FUTA Nigeria, created by Prof. Lawrence Stephen Fayeun of the Field-to-Insight Academy (www.fieldtoinsightacademy.com.ng).

STYLE: Warm, precise, step-by-step. Use Nigerian crops (cassava, cowpea, maize, sorghum) as examples. Show formula → explain symbols → worked example. End longer answers with "— Dr. Fayeun". Invite follow-up.

EXAM TIPS (share when relevant): State assumptions. Show working. Include units. Check h²≤1. Interpret results for breeding decisions.

CORE CONTENT COVERS: variance components, heritability estimation, generation mean analysis, diallel analysis, ANOVA experimental designs, regression and path analysis, G×E interaction, stability analysis (AMMI/WAASB), selection indices (Smith-Hazel), machine learning in plant breeding (rrBLUP, GBLUP, LASSO, Random Forest, Bayesian methods).

For R code questions: provide complete runnable code with comments.
For ANOVA questions: always present the full table.
Link statistical results to breeding decisions."""


VIVASENSE_INTERPRET_SYSTEM = """You are Dr. Fayeun, Professor of Quantitative Genetics at FUTA and founder of Field-to-Insight Academy. You are interpreting statistical analysis results for a researcher who used VivaSense.

Write as you would supervise a postgraduate student or write in a peer-reviewed paper:

STRUCTURE YOUR RESPONSE AS:
1. **Overall Finding** — one sentence stating the key result
2. **Statistical Evidence** — reference specific F-value, p-value, means from the results
3. **Treatment/Genotype Comparisons** — interpret the post-hoc groupings specifically
4. **Assumptions** — comment on whether assumptions were met; flag any violations with advice
5. **Research Recommendation** — what the researcher should do or conclude
6. **Thesis/Paper Sentence** — give them one ready-to-use sentence they can copy into their write-up

RULES:
- Always reference the actual numbers provided — never speak generally
- Use Nigerian/West African crop context where the crop is mentioned
- Keep language accessible to final-year undergraduates and postgraduates
- If assumptions are violated, suggest the appropriate non-parametric alternative
- End with: "— Dr. Fayeun, VivaSense"

You are what separates a confused student from a confident researcher."""


VIVASENSE_FOLLOWUP_SYSTEM = """You are Dr. Fayeun, Professor of Quantitative Genetics at FUTA. A researcher has run a statistical analysis on VivaSense and you have already provided an interpretation. They are now asking follow-up questions.

You have full knowledge of their analysis results (provided below). Answer their specific question directly and concisely. If they ask about writing for a thesis, give them exact sentences. If they ask about assumption violations, give specific actionable advice. If they ask why a result is unexpected, reason through it with them.

Always ground your answer in THEIR specific numbers — not generic advice.
End with "— Dr. Fayeun" on detailed answers."""


# ══════════════════════════════════════════════════════════════
#  REQUEST MODELS
# ══════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    topic: Optional[str] = None   # e.g. "Topic 2: Heritability"
    stream: Optional[bool] = True

class InterpretRequest(BaseModel):
    analysis_type: str            # e.g. "One-way ANOVA"
    results: dict                 # full results JSON from VivaSense
    crop: Optional[str] = None    # e.g. "cowpea", "cassava"
    stream: Optional[bool] = True

class FollowupRequest(BaseModel):
    messages: List[Message]       # conversation history
    analysis_results: dict        # always include results for context
    stream: Optional[bool] = True


# ══════════════════════════════════════════════════════════════
#  HELPER: call Anthropic with streaming
# ══════════════════════════════════════════════════════════════

async def stream_anthropic(system: str, messages: list, model: str = HAIKU_MODEL, max_tokens: int = 1024):
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
            async with client.stream("POST", ANTHROPIC_URL,
                                     headers=ANTHROPIC_HEADERS,
                                     json=payload) as response:
                if response.status_code != 200:
                    error = await response.aread()
                    yield f"data: {json.dumps({'error': error.decode()})}\n\n"
                    return
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


async def call_anthropic(system: str, messages: list, model: str = HAIKU_MODEL, max_tokens: int = 1024) -> str:
    """Non-streaming call — returns full text."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(ANTHROPIC_URL, headers=ANTHROPIC_HEADERS, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code,
                                detail=response.json().get("error", {}).get("message", "API error"))
        data = response.json()
        return data["content"][0]["text"]


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 1 — CSP 811 AI TUTOR CHAT
# ══════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest):
    """
    CSP 811 Dr. Fayeun AI tutor chat.
    Called by the Lovable frontend instead of hitting Anthropic directly.
    """
    client_ip = request.client.host
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")

    system = CSP811_SYSTEM
    if body.topic:
        system += f"\n\n[Student is currently studying: {body.topic}]"

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream:
        return await stream_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=1024)
    else:
        text = await call_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=1024)
        return {"content": text}


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 2 — VIVASENSE AI INTERPRETATION
# ══════════════════════════════════════════════════════════════

@app.post("/api/interpret")
async def interpret(request: Request, body: InterpretRequest):
    """
    Generate Dr. Fayeun's interpretation of VivaSense analysis results.
    Called automatically after every analysis run.
    """
    client_ip = request.client.host
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")

    # Build the user message from results
    results_text = json.dumps(body.results, indent=2)
    crop_context = f"Crop/study context: {body.crop}" if body.crop else ""

    user_message = f"""Please interpret these {body.analysis_type} results for the researcher.

{crop_context}

Results:
{results_text}

Provide a complete interpretation following your structured format."""

    messages = [{"role": "user", "content": user_message}]

    # Use Sonnet for interpretation — higher quality for the key selling point
    if body.stream:
        return await stream_anthropic(VIVASENSE_INTERPRET_SYSTEM, messages,
                                      model=SONNET_MODEL, max_tokens=1200)
    else:
        text = await call_anthropic(VIVASENSE_INTERPRET_SYSTEM, messages,
                                    model=SONNET_MODEL, max_tokens=1200)
        return {"interpretation": text}


# ══════════════════════════════════════════════════════════════
#  ENDPOINT 3 — VIVASENSE POST-ANALYSIS FOLLOW-UP CHAT
# ══════════════════════════════════════════════════════════════

@app.post("/api/followup")
async def followup(request: Request, body: FollowupRequest):
    """
    Post-analysis chat — user asks follow-up questions about their results.
    Analysis results always included in context so Dr. Fayeun can reference specifics.
    """
    client_ip = request.client.host
    if is_rate_limited(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a moment.")

    # Inject results into system prompt so every reply is grounded in actual data
    results_context = json.dumps(body.analysis_results, indent=2)
    system = VIVASENSE_FOLLOWUP_SYSTEM + f"\n\nANALYSIS RESULTS IN CONTEXT:\n{results_context}"

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    if body.stream:
        return await stream_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=800)
    else:
        text = await call_anthropic(system, messages, model=HAIKU_MODEL, max_tokens=800)
        return {"content": text}


# ══════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    """Render will ping this to keep the service alive."""
    key_set = bool(ANTHROPIC_API_KEY)
    return {
        "status": "ok",
        "service": "FIA AI Proxy",
        "api_key_configured": key_set,
        "endpoints": ["/api/chat", "/api/interpret", "/api/followup"]
    }


# ══════════════════════════════════════════════════════════════
#  RUN (for local testing only — Render uses uvicorn directly)
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("proxy:app", host="0.0.0.0", port=8000, reload=True)
