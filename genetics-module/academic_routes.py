"""
VivaSense Academic Mentor — FastAPI Router
==========================================

Exposes POST /academic/interpret.

Architecture:
  Layer A  academic_validator.py  — rule engine (no AI, no I/O)
  Layer B  academic_interpretation.py  — Claude Haiku + validation loop + fallback
  Layer C  guided_writing.py  — sentence starters + examiner checkpoints

The endpoint is dependency-free except for the ANTHROPIC_API_KEY environment
variable.  If the key is absent at import time the router still loads, but
every request gets a 503 with a descriptive message so the frontend can show
a graceful "AI unavailable" state.

Rate limiting is deliberately left to the hosting layer (Render / Nginx).
This router does NOT rate-limit so that multi-trait batch calls from the
frontend are not throttled by the backend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from academic_schemas import AcademicInterpretRequest, AcademicInterpretationResponse

logger = logging.getLogger(__name__)

# ── API-key availability flag (set once at import time) ──────────────────────
_api_key_available: bool = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())

if _api_key_available:
    logger.info("AcademicMentor: ANTHROPIC_API_KEY present — AI interpretation enabled")
else:
    logger.warning("AcademicMentor: ANTHROPIC_API_KEY not set — deterministic fallback will be used")

# ── In-memory LRU Cache ───────────────────────────────────────────────────────
_CACHE_MAX_SIZE = 200
_cache_lock = threading.Lock()
_interp_cache: OrderedDict = OrderedDict()

def _get_cache_key(request: AcademicInterpretRequest) -> str:
    """Generate a deterministic hash key from the request payload."""
    req_str = json.dumps(request.model_dump(), sort_keys=True)
    return hashlib.sha256(req_str.encode("utf-8")).hexdigest()

# ── Router ────────────────────────────────────────────────────────────────────
router = APIRouter(tags=["Academic Mentor"])


@router.post(
    "/academic/interpret",
    response_model=AcademicInterpretationResponse,
    summary="Generate academic interpretation of analysis results",
    description=(
        "Accepts an analysis result from any module (anova, genetic_parameters, "
        "correlation, heatmap) and returns a three-layer academic interpretation:\n\n"
        "- **Layer A** — Validation report (rule violations detected in AI output)\n"
        "- **Layer B** — Structured interpretation (overall finding, statistical "
        "evidence, module-specific sections, examiner checkpoint)\n"
        "- **Layer C** — Guided writing support (sentence starters with ___ blanks, "
        "examiner checklist)\n\n"
        "Requires `ANTHROPIC_API_KEY` to be set in the deployment environment. "
        "Returns a deterministic fallback if AI generation fails validation after "
        "two repair attempts."
    ),
)
async def interpret(request: AcademicInterpretRequest) -> AcademicInterpretationResponse:
    """
    Three-pass academic interpretation pipeline:

    1. Format analysis data into a structured prompt
    2. Call Claude Haiku — generate interpretation
    3. Validate with AcademicValidator (Layer A)
    4. If blocked → repair pass (send violations back to Claude)
    5. If still blocked → deterministic fallback
    6. Build Layer C (guided writing) regardless of AI success
    7. Return AcademicInterpretationResponse
    """

    # ── Cache Check ───────────────────────────────────────────────────────────
    cache_key = _get_cache_key(request)
    with _cache_lock:
        if cache_key in _interp_cache:
            logger.info("AcademicMentor: cache hit for %s trait='%s'", request.module_type, request.trait)
            _interp_cache.move_to_end(cache_key)
            return _interp_cache[cache_key]

    # ── Lazy import to keep startup fast when key is absent ───────────────────
    try:
        from academic_interpretation import interpret_module
    except ImportError as exc:
        logger.error("academic_interpretation import failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Academic Mentor module failed to load: {exc}",
        )

    # ── Execute interpretation pipeline ───────────────────────────────────────
    try:
        result = await interpret_module(request)
        
        # Cache the successful result
        with _cache_lock:
            _interp_cache[cache_key] = result
            if len(_interp_cache) > _CACHE_MAX_SIZE:
                _interp_cache.popitem(last=False)
                
        return result

    except ValueError as exc:
        logger.warning("AcademicMentor bad request — %s: %s", request.module_type, exc)
        raise HTTPException(status_code=400, detail=str(exc))

    except RuntimeError as exc:
        logger.error("AcademicMentor runtime error — %s: %s", request.module_type, exc)
        raise HTTPException(status_code=422, detail=str(exc))

    except Exception as exc:
        logger.error(
            "AcademicMentor unexpected error — %s: %s",
            request.module_type, exc, exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(exc))
