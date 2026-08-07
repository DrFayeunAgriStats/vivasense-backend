"""
VivaSense pilot-access gate
===========================

Restricts the Genetics-only analysis surface to an explicit allowlist of
approved operators during the pilot validation window.

Why this exists separately from the free/pro mode gate
------------------------------------------------------
`profiles.plan` is currently ``'pro'`` for every account by deliberate design —
that is how the "free for 90 days, all features" pilot offer was implemented.
The ``X-VivaSense-Mode`` middleware therefore resolves to ``pro`` for everyone
and correctly implements "everyone gets everything", which is the opposite of
what pilot safety needs. This gate is layered *next to* that middleware and
keys off an identity allowlist that is **not** ``profiles.plan`` and is not
derived from it, so it keeps working regardless of what plan value every
account currently holds.

Nothing in this module reads or writes plan/mode state. The free/pro
middleware is untouched and continues to serve its monetization purpose.

Trust model — read this before relying on the gate
--------------------------------------------------
The backend has no authentication layer: it cannot verify a Supabase JWT and
holds no database connection, so it cannot read ``user_roles`` or
``profiles``. Identity therefore arrives as a request header
(``X-VivaSense-User``) and is checked against a server-side allowlist held in
the ``VIVASENSE_PILOT_ALLOWLIST`` environment variable.

Because the allowlist lives on the server, a pilot participant cannot grant
themselves access by editing localStorage — unlike the ``?mode=pro`` URL flag.
But a hand-crafted request can still assert any identity string. This is a
guardrail against *accidental* use of unvalidated capabilities by pilot
participants, not a security boundary against a motivated attacker. Closing
that gap requires verifying the Supabase JWT server-side, which is tracked
separately.

Fail-open on empty allowlist is deliberate: if ``VIVASENSE_PILOT_ALLOWLIST``
is unset the gate disables itself rather than locking every user out of a
running deployment. ``pilot_gate_status()`` reports this so startup can log it
loudly.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration surface
# ---------------------------------------------------------------------------

PILOT_ALLOWLIST_ENV = "VIVASENSE_PILOT_ALLOWLIST"
PILOT_USER_HEADER = "X-VivaSense-User"

ANALYZE_UPLOAD_PATH = "/genetics/analyze-upload"

# Paths whose entire capability is Genetics-only, regardless of payload shape.
PILOT_RESTRICTED_PATHS: Dict[str, str] = {
    "/analysis/genetic-parameters": "genetic_parameters",
    "/analysis/correlation": "correlation",
    "/genetics/correlation": "correlation",
    "/analysis/path-analysis": "path_analysis",
    "/analysis/path-analysis/preflight": "path_analysis",
    "/analysis/regression": "regression",
    "/genetics/trait-association/analyze": "trait_association",
}

# ---------------------------------------------------------------------------
# Payload-shape classification
#
# These two helpers are the single source of truth for "is this the validated
# design-aware Experimental Design shape, or the Genetics-only shape?".
# app_genetics.py imports them so the free/pro mode gate and this pilot gate
# classify identically and cannot drift apart.
# ---------------------------------------------------------------------------

FREE_ANOVA_DESIGN_TYPES = {
    "crd",
    "rcbd",
    "factorial",
    "factorial_rcbd",
    "split_plot_rcbd",
}

DESIGN_AWARE_ANOVA_FIELDS = (
    "design_type",
    "treatment_column",
    "factor_column",
    "factor_a_column",
    "factor_b_column",
    "factor_c_column",
    "main_plot_column",
    "sub_plot_column",
)


def payload_has_multi_environment_shape(payload: Mapping[str, Any]) -> bool:
    """True when the payload describes a multi-environment / G×E analysis."""
    body_mode = str(payload.get("mode") or "").strip().lower()
    body_env_col = payload.get("environment_column")
    has_environment_factor = isinstance(body_env_col, str) and body_env_col.strip() != ""
    return body_mode == "multi" or has_environment_factor


def payload_has_design_aware_anova_shape(payload: Mapping[str, Any]) -> bool:
    """True when the payload carries explicit Experimental Design structure."""
    design_type = str(payload.get("design_type") or "").strip().lower()
    if design_type in FREE_ANOVA_DESIGN_TYPES:
        return True

    for field_name in DESIGN_AWARE_ANOVA_FIELDS:
        field_value = payload.get(field_name)
        if isinstance(field_value, str) and field_value.strip() != "":
            return True

    return False


def classify_pilot_request(
    path: str,
    method: str = "POST",
    module_query: Optional[str] = None,
    payload: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Identify which pilot-restricted capability a request is asking for.

    Returns the capability name, or ``None`` when the request is outside the
    restricted surface and must pass through untouched.

    The ``/genetics/analyze-upload`` branch mirrors the free/pro middleware's
    classification exactly — same module resolution, same shape helpers — so
    the validated design-aware Experimental Design workflow continues to pass
    through here for the same reasons it passes through there.
    """
    capability = PILOT_RESTRICTED_PATHS.get(path)
    if capability is not None:
        return capability

    if path != ANALYZE_UPLOAD_PATH or method.upper() != "POST":
        return None

    payload = payload or {}

    # Body module takes priority; fallback to query; endpoint default is
    # genetic_parameters. Identical precedence to the mode gate and to the
    # analyze_upload route itself.
    body_module = str(payload.get("module") or "").strip().lower()
    actual_module = body_module or (module_query or "").strip().lower() or "genetic_parameters"

    if actual_module == "genetic_parameters":
        return "genetic_parameters"

    if actual_module == "anova":
        # Design-aware, single-environment ANOVA is the validated Experimental
        # Design workflow → not pilot-restricted.
        if payload_has_design_aware_anova_shape(payload) and not payload_has_multi_environment_shape(payload):
            return None
        return "anova_genetics_only"

    return None


# ---------------------------------------------------------------------------
# Allowlist resolution
# ---------------------------------------------------------------------------


def parse_allowlist(raw: Optional[str]) -> frozenset:
    """Parse a comma/whitespace-separated allowlist into normalized entries.

    Entries may be Supabase user UUIDs or email addresses. Matching is
    case-insensitive and tolerant of surrounding whitespace.
    """
    if not raw:
        return frozenset()
    entries = (
        item.strip().lower()
        for chunk in raw.split(",")
        for item in chunk.split()
    )
    return frozenset(e for e in entries if e)


def get_pilot_allowlist(env: Optional[Mapping[str, str]] = None) -> frozenset:
    """Read and parse the allowlist from the environment at call time."""
    source = os.environ if env is None else env
    return parse_allowlist(source.get(PILOT_ALLOWLIST_ENV))


def is_pilot_allowlisted(identity: Optional[str], allowlist: Iterable[str]) -> bool:
    """True when the asserted identity appears in the allowlist."""
    if not identity:
        return False
    normalized = identity.strip().lower()
    if not normalized:
        return False
    return normalized in set(allowlist)


def pilot_gate_status(env: Optional[Mapping[str, str]] = None) -> Tuple[bool, int]:
    """Return ``(enforcing, allowlist_size)`` for startup logging."""
    allowlist = get_pilot_allowlist(env)
    return (bool(allowlist), len(allowlist))


def pilot_denied_payload(capability: str) -> Dict[str, str]:
    """Body returned when the gate denies a request."""
    return {
        "error": "PILOT_RESTRICTED",
        "message": (
            "This analysis is limited to approved pilot operators during the "
            "validation window. Contact the VivaSense team for access."
        ),
        "capability": capability,
    }


# ---------------------------------------------------------------------------
# ASGI middleware
#
# Registered by app_genetics.py via app.middleware("http")(pilot_gate_middleware).
# The regression suite mounts this exact function, so the tests exercise the
# production code path rather than a reimplementation of it.
# ---------------------------------------------------------------------------


async def pilot_gate_middleware(request, call_next):
    """Block pilot-restricted capabilities for non-allowlisted identities."""
    from starlette.responses import JSONResponse  # local import keeps module import-light

    allowlist = get_pilot_allowlist()

    # No allowlist configured → gate disabled, request passes through.
    if not allowlist:
        return await call_next(request)

    payload: Dict[str, Any] = {}
    path = request.url.path
    method = request.method

    # Only the analyze-upload branch needs the body; everything else is
    # classified by path alone, so avoid reading bodies unnecessarily.
    if path == ANALYZE_UPLOAD_PATH and method.upper() == "POST":
        try:
            raw = await request.body()
            if raw:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
        except Exception:
            pass

    capability = classify_pilot_request(
        path,
        method,
        request.query_params.get("module"),
        payload,
    )

    if capability is not None:
        identity = request.headers.get(PILOT_USER_HEADER)
        if not is_pilot_allowlisted(identity, allowlist):
            logger.info(
                "pilot gate: denied capability=%s path=%s identity=%s",
                capability,
                path,
                identity or "<none>",
            )
            return JSONResponse(status_code=403, content=pilot_denied_payload(capability))

    return await call_next(request)
