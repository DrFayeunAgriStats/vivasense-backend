"""
Regression suite for the pilot-access gate (pilot_access.py).

These tests import and mount the *production* middleware function
(`pilot_access.pilot_gate_middleware`) and the production classifier
(`pilot_access.classify_pilot_request`). Nothing is reimplemented here — the
previous round's check reimplemented the gate in the harness and therefore
verified the reimplementation rather than the shipped behaviour.

`pilot_access` is deliberately dependency-free (stdlib + a local starlette
import inside the middleware), so it imports cleanly without the R engine,
pandas, or the full app.

Core guarantees under test
--------------------------
1. An allowlisted operator reaches every restricted capability unchanged.
2. A non-privileged user is blocked from the Genetics-only surface
   REGARDLESS of their plan / X-VivaSense-Mode value.
3. A non-privileged user still reaches the design-aware Experimental Design
   workflow REGARDLESS of their plan / X-VivaSense-Mode value.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import pilot_access
from pilot_access import (
    PILOT_USER_HEADER,
    classify_pilot_request,
    is_pilot_allowlisted,
    parse_allowlist,
    pilot_gate_middleware,
    pilot_gate_status,
)

ALLOWED_USER = "pilot.admin@fieldtoinsightacademy.com.ng"
ORDINARY_USER = "researcher@example.org"

# A design-aware, single-environment ANOVA payload — the validated
# Experimental Design workflow that must always pass through.
DESIGN_AWARE_PAYLOAD = {
    "module": "anova",
    "mode": "single",
    "design_type": "rcbd",
    "treatment_column": "Variety",
    "trait_columns": ["Yield"],
}

# A Genetics-only ANOVA payload — no design structure at all.
GENETICS_ONLY_ANOVA_PAYLOAD = {
    "module": "anova",
    "mode": "single",
    "genotype_column": "Genotype",
    "rep_column": "Rep",
    "trait_columns": ["Yield"],
}

GENETIC_PARAMETERS_PAYLOAD = {
    "module": "genetic_parameters",
    "mode": "single",
    "genotype_column": "Genotype",
    "rep_column": "Rep",
    "trait_columns": ["Yield"],
}


@pytest.fixture
def allowlist_env(monkeypatch):
    """Enable the gate with a single allowlisted identity."""
    monkeypatch.setenv(pilot_access.PILOT_ALLOWLIST_ENV, ALLOWED_USER)
    return ALLOWED_USER


@pytest.fixture
def client(allowlist_env):
    """A minimal app carrying the real production middleware."""
    app = FastAPI()
    app.middleware("http")(pilot_gate_middleware)

    @app.post("/genetics/analyze-upload")
    async def analyze_upload(payload: dict):  # noqa: ARG001
        return {"ok": True}

    @app.post("/analysis/genetic-parameters")
    async def genetic_parameters(payload: dict):  # noqa: ARG001
        return {"ok": True}

    @app.post("/analysis/correlation")
    async def correlation(payload: dict):  # noqa: ARG001
        return {"ok": True}

    @app.post("/analysis/path-analysis")
    async def path_analysis(payload: dict):  # noqa: ARG001
        return {"ok": True}

    @app.post("/analysis/regression")
    async def regression(payload: dict):  # noqa: ARG001
        return {"ok": True}

    @app.post("/analysis/descriptive-stats")
    async def descriptive_stats(payload: dict):  # noqa: ARG001
        return {"ok": True}

    with TestClient(app) as c:
        yield c


def _headers(identity=None, mode="pro"):
    """Build request headers. Default mode='pro' mirrors production, where
    every account currently holds plan='pro'."""
    headers = {"X-VivaSense-Mode": mode}
    if identity is not None:
        headers[PILOT_USER_HEADER] = identity
    return headers


# ---------------------------------------------------------------------------
# Guarantee 1 — allowlisted operator reaches everything
# ---------------------------------------------------------------------------

RESTRICTED_ENDPOINTS = [
    ("/analysis/genetic-parameters", {}),
    ("/analysis/correlation", {}),
    ("/analysis/path-analysis", {}),
    ("/analysis/regression", {}),
    ("/genetics/analyze-upload", GENETIC_PARAMETERS_PAYLOAD),
    ("/genetics/analyze-upload", GENETICS_ONLY_ANOVA_PAYLOAD),
]


@pytest.mark.parametrize("path,payload", RESTRICTED_ENDPOINTS)
def test_allowlisted_user_reaches_restricted_capabilities(client, path, payload):
    response = client.post(path, json=payload, headers=_headers(ALLOWED_USER))
    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True}


def test_allowlist_matching_is_case_and_whitespace_insensitive(client):
    response = client.post(
        "/analysis/genetic-parameters",
        json={},
        headers=_headers(f"  {ALLOWED_USER.upper()}  "),
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Guarantee 2 — non-privileged user blocked from the Genetics-only surface,
# regardless of plan / mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,payload", RESTRICTED_ENDPOINTS)
@pytest.mark.parametrize("mode", ["pro", "free"])
def test_ordinary_user_blocked_regardless_of_plan(client, path, payload, mode):
    response = client.post(path, json=payload, headers=_headers(ORDINARY_USER, mode=mode))
    assert response.status_code == 403, response.text
    body = response.json()
    assert body["error"] == "PILOT_RESTRICTED"
    assert "capability" in body


def test_missing_identity_is_blocked_even_with_pro_mode(client):
    response = client.post(
        "/genetics/analyze-upload",
        json=GENETIC_PARAMETERS_PAYLOAD,
        headers=_headers(None, mode="pro"),
    )
    assert response.status_code == 403
    assert response.json()["error"] == "PILOT_RESTRICTED"


def test_genetics_only_anova_shape_is_blocked_for_ordinary_user(client):
    """Plain genotype/rep ANOVA with no design structure is Genetics-only."""
    response = client.post(
        "/genetics/analyze-upload",
        json=GENETICS_ONLY_ANOVA_PAYLOAD,
        headers=_headers(ORDINARY_USER),
    )
    assert response.status_code == 403
    assert response.json()["capability"] == "anova_genetics_only"


def test_multi_environment_anova_is_blocked_for_ordinary_user(client):
    """Multi-environment/G×E is not the validated design-aware workflow."""
    payload = dict(DESIGN_AWARE_PAYLOAD, mode="multi", environment_column="Site")
    response = client.post(
        "/genetics/analyze-upload", json=payload, headers=_headers(ORDINARY_USER)
    )
    assert response.status_code == 403
    assert response.json()["capability"] == "anova_genetics_only"


def test_analyze_upload_defaults_to_genetic_parameters_when_module_absent(client):
    """Endpoint default is genetic_parameters — must not fall open."""
    response = client.post(
        "/genetics/analyze-upload",
        json={"mode": "single", "trait_columns": ["Yield"]},
        headers=_headers(ORDINARY_USER),
    )
    assert response.status_code == 403
    assert response.json()["capability"] == "genetic_parameters"


# ---------------------------------------------------------------------------
# Guarantee 3 — design-aware Experimental Design workflow always passes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["pro", "free"])
def test_design_aware_workflow_passes_for_ordinary_user(client, mode):
    response = client.post(
        "/genetics/analyze-upload",
        json=DESIGN_AWARE_PAYLOAD,
        headers=_headers(ORDINARY_USER, mode=mode),
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True}


@pytest.mark.parametrize(
    "design_type", ["crd", "rcbd", "factorial", "factorial_rcbd", "split_plot_rcbd"]
)
def test_every_design_aware_type_passes_for_ordinary_user(client, design_type):
    payload = dict(DESIGN_AWARE_PAYLOAD, design_type=design_type)
    response = client.post(
        "/genetics/analyze-upload", json=payload, headers=_headers(ORDINARY_USER)
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize(
    "field",
    [
        "treatment_column",
        "factor_column",
        "factor_a_column",
        "factor_b_column",
        "main_plot_column",
        "sub_plot_column",
    ],
)
def test_design_aware_marker_fields_pass_for_ordinary_user(client, field):
    """Any explicit design field marks the Experimental Design workflow."""
    payload = {"module": "anova", "mode": "single", field: "SomeColumn"}
    response = client.post(
        "/genetics/analyze-upload", json=payload, headers=_headers(ORDINARY_USER)
    )
    assert response.status_code == 200, response.text


def test_unrestricted_endpoint_passes_for_ordinary_user(client):
    """Descriptive stats is outside the pilot-restricted surface."""
    response = client.post(
        "/analysis/descriptive-stats", json={}, headers=_headers(ORDINARY_USER)
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Gate independence from plan/pro, and configuration behaviour
# ---------------------------------------------------------------------------


def test_gate_disabled_when_allowlist_unset(monkeypatch):
    """Empty allowlist fails open by design, so a deploy cannot lock everyone out."""
    monkeypatch.delenv(pilot_access.PILOT_ALLOWLIST_ENV, raising=False)

    app = FastAPI()
    app.middleware("http")(pilot_gate_middleware)

    @app.post("/analysis/genetic-parameters")
    async def genetic_parameters(payload: dict):  # noqa: ARG001
        return {"ok": True}

    with TestClient(app) as c:
        response = c.post(
            "/analysis/genetic-parameters", json={}, headers=_headers(ORDINARY_USER)
        )
    assert response.status_code == 200

    enforcing, size = pilot_gate_status()
    assert enforcing is False
    assert size == 0


def test_gate_status_reports_enforcing(monkeypatch):
    monkeypatch.setenv(pilot_access.PILOT_ALLOWLIST_ENV, f"{ALLOWED_USER}, second@x.org")
    enforcing, size = pilot_gate_status()
    assert enforcing is True
    assert size == 2


def test_classifier_never_consults_plan_or_mode():
    """The classifier signature carries no plan/mode input at all."""
    import inspect

    params = set(inspect.signature(classify_pilot_request).parameters)
    assert params == {"path", "method", "module_query", "payload"}
    assert not any("plan" in p or "pro" in p for p in params)


def test_parse_allowlist_handles_separators_and_blanks():
    parsed = parse_allowlist("  A@x.org , b@x.org\n\nC@x.org ,, ")
    assert parsed == frozenset({"a@x.org", "b@x.org", "c@x.org"})
    assert parse_allowlist("") == frozenset()
    assert parse_allowlist(None) == frozenset()


def test_is_pilot_allowlisted_rejects_blank_identities():
    allowlist = parse_allowlist(ALLOWED_USER)
    assert is_pilot_allowlisted(ALLOWED_USER, allowlist) is True
    assert is_pilot_allowlisted(None, allowlist) is False
    assert is_pilot_allowlisted("", allowlist) is False
    assert is_pilot_allowlisted("   ", allowlist) is False


def test_design_aware_classification_matches_mode_gate_semantics():
    """Direct classifier assertions mirroring the mode gate's shape logic."""
    assert classify_pilot_request("/genetics/analyze-upload", "POST", None, DESIGN_AWARE_PAYLOAD) is None
    assert (
        classify_pilot_request("/genetics/analyze-upload", "POST", None, GENETICS_ONLY_ANOVA_PAYLOAD)
        == "anova_genetics_only"
    )
    assert (
        classify_pilot_request("/genetics/analyze-upload", "POST", "genetic_parameters", {})
        == "genetic_parameters"
    )
    # Non-POST and unrelated paths are never restricted.
    assert classify_pilot_request("/genetics/analyze-upload", "GET", None, {}) is None
    assert classify_pilot_request("/health", "GET", None, {}) is None
