"""
Integration test for multitrait_upload_routes.py

Tests the full Python-side request/response cycle for all four scenarios.
The R engine (RGeneticsEngine.run_analysis) is replaced by a deterministic
mock that returns a realistic GeneticsResponse-shaped dict, so results are
reproducible without an R installation.

Run from inside genetics-module/:
    python test_upload_integration.py
"""

import base64
import io
import json
import sys
import types
import unittest
from unittest.mock import MagicMock

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Build a minimal app_genetics stub so multitrait_upload_routes can import it
# without starting FastAPI or touching the filesystem.
# ─────────────────────────────────────────────────────────────────────────────

def _make_genetics_response_dict(mode: str, trait_name: str, grand_mean: float):
    """Return a realistic GeneticsResponse-shaped dict (what R engine returns)."""
    r_val = grand_mean / 5
    return {
        "status": "SUCCESS",
        "mode": mode,
        "data_validation": {"is_valid": True, "warnings": []},
        "variance_warnings": {"is_valid": True, "warnings": []},
        "result": {
            "environment_mode": mode,
            "n_genotypes": 10,
            "n_reps": 3,
            "n_environments": 3 if mode == "multi" else None,
            "grand_mean": grand_mean,
            "variance_components": {
                "sigma2_genotype": round(r_val * 1.5, 4),
                "sigma2_error": round(r_val * 0.8, 4),
                "sigma2_ge": round(r_val * 0.4, 4) if mode == "multi" else None,
                "sigma2_phenotypic": round(r_val * 2.7, 4),
                "heritability_basis": "entry-mean",
            },
            "heritability": {
                "h2_broad_sense": 0.72,
                "interpretation_basis": "entry-mean",
                "formula": "σ²G / (σ²G + σ²E/r)",
            },
            "genetic_parameters": {
                "GCV": 14.3,
                "PCV": 18.9,
                "GAM": round(grand_mean * 0.24, 2),
                "GAM_percent": 24.0,
                "selection_intensity": 1.4,
            },
        },
        "interpretation": (
            f"[Mock] {trait_name}: Broad-sense heritability was high (H² = 0.72), "
            "indicating strong genetic control. GCV (14.3%) was lower than PCV (18.9%), "
            "suggesting moderate environmental influence. Genetic advance under selection "
            "was 24.0% of the mean, indicating good response to selection."
        ),
    }


def _setup_app_genetics_stub():
    """
    Insert a fake app_genetics module into sys.modules before importing
    multitrait_upload_routes, so the circular-import resolution works without
    actually loading FastAPI or R.
    """
    from pydantic import BaseModel, Field
    from typing import Any, Dict, List, Optional

    # Mirror only the classes used by multitrait_upload_schemas
    class GeneticsResult(BaseModel):
        environment_mode: str
        n_genotypes: int
        n_reps: int
        n_environments: Optional[int] = None
        grand_mean: float
        variance_components: Dict[str, Any]
        heritability: Dict[str, Any]
        genetic_parameters: Dict[str, Any]

    class GeneticsResponse(BaseModel):
        status: str
        mode: str
        data_validation: Dict[str, Any] = Field(default_factory=dict)
        variance_warnings: Dict[str, Any] = Field(default_factory=dict)
        result: Optional[GeneticsResult] = None
        interpretation: Optional[str] = None

    # Build the stub module
    stub = types.ModuleType("app_genetics")
    stub.GeneticsResponse = GeneticsResponse
    stub.GeneticsResult = GeneticsResult
    stub.r_engine = None  # will be replaced per-test
    sys.modules["app_genetics"] = stub
    return stub


# Patch FastAPI so the router decorator doesn't blow up
def _setup_fastapi_stub():
    fake_fastapi = types.ModuleType("fastapi")

    class FakeAPIRouter:
        def __init__(self, **kw): pass
        def post(self, *a, **kw):
            def decorator(fn): return fn
            return decorator
        def get(self, *a, **kw):
            def decorator(fn): return fn
            return decorator

    class FakeFile:
        @staticmethod
        def __call__(*a, **kw): return None
        # Also support File(...) at module-level default arg evaluation
        def __new__(cls, *a, **kw): return None

    class FakeUploadFile: pass
    class FakeHTTPException(Exception):
        def __init__(self, status_code, detail=""):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    fake_fastapi.APIRouter = FakeAPIRouter
    fake_fastapi.File = FakeFile
    fake_fastapi.UploadFile = FakeUploadFile
    fake_fastapi.HTTPException = FakeHTTPException
    sys.modules["fastapi"] = fake_fastapi

    fake_responses = types.ModuleType("fastapi.responses")
    fake_responses.JSONResponse = None
    sys.modules["fastapi.responses"] = fake_responses


# ─────────────────────────────────────────────────────────────────────────────
# CSV / XLSX fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def _xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def make_single_env_single_trait() -> bytes:
    """10 genotypes × 3 reps, one numeric trait. Balanced."""
    rows = []
    for g in range(1, 11):
        for r in range(1, 4):
            rows.append({
                "Genotype": f"G{g:02d}",
                "Rep": f"R{r}",
                "PlotID": g * 10 + r,   # numeric ID — should NOT be a trait
                "Yield_kg_ha": round(40 + g * 1.5 + r * 0.3, 2),
            })
    return _csv_bytes(pd.DataFrame(rows))


def make_multi_env_multi_trait() -> bytes:
    """10 genotypes × 3 environments × 2 reps, three trait columns. Excel."""
    rows = []
    for g in range(1, 11):
        for e in range(1, 4):
            for r in range(1, 3):
                rows.append({
                    "Genotype": f"G{g:02d}",
                    "Environment": f"E{e}",
                    "Rep": f"R{r}",
                    "Yield":       round(40 + g * 1.2 + e * 2 + r * 0.4, 2),
                    "PlantHeight": round(120 + g * 2.1 + e * 3, 1),
                    "DaysToFlower": round(60 + g * 0.5 + e * 1.2, 0),
                })
    return _xlsx_bytes(pd.DataFrame(rows))


def make_one_trait_fails() -> bytes:
    """
    Two traits: Yield is complete, BadTrait has only 4 valid values (< 6 minimum).
    """
    rows = []
    for g in range(1, 11):
        for r in range(1, 4):
            rows.append({
                "Genotype": f"G{g:02d}",
                "Rep": f"R{r}",
                "Yield":    round(40 + g * 1.5, 2),
                "BadTrait": round(10 + g, 2) if (g <= 1 and r <= 2) else None,
            })
    return _csv_bytes(pd.DataFrame(rows))


def make_missing_values() -> bytes:
    """
    10 genotypes × 3 reps. ~20% of Yield values missing, creating an
    unbalanced design. A second trait (PlantHeight) is complete.
    """
    import random
    random.seed(42)
    rows = []
    for g in range(1, 11):
        for r in range(1, 4):
            yield_val = None if random.random() < 0.20 else round(40 + g * 1.5 + r * 0.3, 2)
            rows.append({
                "Genotype":    f"G{g:02d}",
                "Rep":         f"R{r}",
                "Yield":       yield_val,
                "PlantHeight": round(120 + g * 2, 1),
            })
    return _csv_bytes(pd.DataFrame(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(
    label: str,
    file_bytes: bytes,
    file_type: str,
    genotype_column: str,
    rep_column: str,
    trait_columns: list,
    mode: str,
    environment_column: str = None,
    ag_stub=None,
):
    """
    Directly exercise Python-side upload logic:
      1. read_file / detect_columns / check_balance (preview path)
      2. build_observations + mocked r_engine.run_analysis (analysis path)
    Returns the UploadAnalysisResponse as a plain dict.
    """
    from multitrait_upload_routes import (
        read_file,
        detect_columns,
        build_observations,
        check_balance,
        _classify_heritability,
        _build_summary_row,
    )
    from multitrait_upload_schemas import (
        DatasetSummary,
        SummaryTableRow,
        TraitResult,
        UploadAnalysisResponse,
    )
    from app_genetics import GeneticsResponse

    print(f"\n{'='*60}")
    print(f"SCENARIO: {label}")
    print(f"{'='*60}")

    # ── Preview phase ──────────────────────────────────────────────────────
    df = read_file(file_bytes, file_type)
    detected = detect_columns(df)
    print(f"\n[PREVIEW]")
    print(f"  rows={len(df)}  cols={list(df.columns)}")
    print(f"  detected genotype : {detected.genotype}")
    print(f"  detected rep      : {detected.rep}")
    print(f"  detected env      : {detected.environment}")
    print(f"  detected traits   : {detected.traits}")

    # ── Analysis phase ─────────────────────────────────────────────────────
    n_genotypes = int(df[genotype_column].nunique())
    n_reps      = int(df[rep_column].nunique())
    n_envs      = int(df[environment_column].nunique()) if environment_column else None
    env_col     = environment_column if mode == "multi" else None

    summary_table = []
    trait_results = {}
    failed_traits = []

    for trait in trait_columns:
        try:
            warnings = check_balance(
                df=df,
                genotype_col=genotype_column,
                rep_col=rep_column,
                trait_col=trait,
                env_col=env_col,
            )

            obs = build_observations(
                df=df,
                genotype_col=genotype_column,
                rep_col=rep_column,
                trait_col=trait,
                env_col=env_col,
            )

            # ── Mock r_engine ──────────────────────────────────────────────
            mean_val = sum(o["trait_value"] for o in obs) / len(obs)
            result_dict = _make_genetics_response_dict(mode, trait, mean_val)
            validated = GeneticsResponse(**result_dict)

            trait_results[trait] = TraitResult(
                status="success",
                analysis_result=validated,
                data_warnings=warnings,
            )
            summary_table.append(_build_summary_row(trait, result_dict))

            print(f"\n  [TRAIT: {trait}] OK  obs={len(obs)}  mean={mean_val:.2f}  "
                  f"warnings={warnings or 'none'}")

        except Exception as exc:
            failed_traits.append(trait)
            trait_results[trait] = TraitResult(
                status="failed",
                analysis_result=None,
                error=str(exc),
            )
            summary_table.append(
                SummaryTableRow(trait=trait, status="failed", error=str(exc))
            )
            print(f"\n  [TRAIT: {trait}] FAIL  error: {exc}")

    response = UploadAnalysisResponse(
        summary_table=summary_table,
        trait_results=trait_results,
        dataset_summary=DatasetSummary(
            n_genotypes=n_genotypes,
            n_reps=n_reps,
            n_environments=n_envs,
            n_traits=len(trait_columns),
            mode=mode,
        ),
        failed_traits=failed_traits,
    )

    # Serialise via Pydantic to match what FastAPI sends over the wire
    response_json = json.loads(response.model_dump_json())

    print(f"\n[RESPONSE JSON]")
    print(json.dumps(response_json, indent=2))

    return response_json


# ─────────────────────────────────────────────────────────────────────────────
# Assertions
# ─────────────────────────────────────────────────────────────────────────────

def assert_eq(label, actual, expected):
    if actual != expected:
        print(f"  FAIL [{label}]: got {actual!r}, expected {expected!r}")
        return False
    print(f"  ok  [{label}]")
    return True


def assert_contains(label, container, item):
    if item not in container:
        print(f"  FAIL [{label}]: {item!r} not in {container!r}")
        return False
    print(f"  ok  [{label}]")
    return True


def check_response_shape(label: str, resp: dict, expected_traits: list,
                          expected_failed: list, check_warnings: bool = False):
    """Validate the shape of a response dict against known expectations."""
    print(f"\n[ASSERTIONS: {label}]")
    ok = True

    # Top-level keys
    for key in ("summary_table", "trait_results", "dataset_summary", "failed_traits"):
        ok &= assert_contains("has key", resp, key)

    ok &= assert_eq("failed_traits", sorted(resp["failed_traits"]), sorted(expected_failed))
    ok &= assert_eq("summary_table length", len(resp["summary_table"]), len(expected_traits))

    for t in expected_traits:
        ok &= assert_contains("trait in trait_results", resp["trait_results"], t)
        tr = resp["trait_results"][t]
        ok &= assert_contains("TraitResult has status", tr, "status")
        ok &= assert_contains("TraitResult has analysis_result", tr, "analysis_result")
        ok &= assert_contains("TraitResult has error", tr, "error")
        ok &= assert_contains("TraitResult has data_warnings", tr, "data_warnings")

        if t not in expected_failed:
            ok &= assert_eq(f"{t}.status", tr["status"], "success")
            ar = tr["analysis_result"]
            if ar is not None:
                ok &= assert_contains(f"{t} ar has result", ar, "result")
                ok &= assert_contains(f"{t} ar has interpretation", ar, "interpretation")
                res = ar["result"]
                if res is not None:
                    ok &= assert_contains(f"{t} result has variance_components", res, "variance_components")
                    ok &= assert_contains(f"{t} result has heritability", res, "heritability")
                    ok &= assert_contains(f"{t} result has genetic_parameters", res, "genetic_parameters")
                    vc = res["variance_components"]
                    ok &= assert_contains(f"{t} vc has sigma2_genotype", vc, "sigma2_genotype")
                    ok &= assert_contains(f"{t} vc has sigma2_error", vc, "sigma2_error")
                    ok &= assert_contains(f"{t} vc has sigma2_phenotypic", vc, "sigma2_phenotypic")
                    h = res["heritability"]
                    ok &= assert_contains(f"{t} heritability has h2_broad_sense", h, "h2_broad_sense")
                    gp = res["genetic_parameters"]
                    for param in ("GCV", "PCV", "GAM_percent"):
                        ok &= assert_contains(f"{t} gp has {param}", gp, param)
        else:
            ok &= assert_eq(f"{t}.status", tr["status"], "failed")
            ok &= assert_eq(f"{t}.analysis_result is null", tr["analysis_result"], None)
            ok &= assert_contains(f"{t}.error non-empty", tr["error"] or "", "")

    if check_warnings:
        # At least one trait should have data_warnings populated
        any_warnings = any(
            tr["data_warnings"]
            for tr in resp["trait_results"].values()
            if tr["status"] == "success"
        )
        ok &= assert_eq("some trait has data_warnings", any_warnings, True)

    # SummaryTableRow fields for successful traits
    for row in resp["summary_table"]:
        for field in ("trait", "status"):
            ok &= assert_contains(f"summary row has {field}", row, field)

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _setup_fastapi_stub()
    ag_stub = _setup_app_genetics_stub()

    all_ok = True

    # ── Scenario 1: single-env, single trait ─────────────────────────────────
    r1 = run_scenario(
        label="1 — Single-env, single trait (balanced CSV)",
        file_bytes=make_single_env_single_trait(),
        file_type="csv",
        genotype_column="Genotype",
        rep_column="Rep",
        trait_columns=["Yield_kg_ha"],
        mode="single",
        ag_stub=ag_stub,
    )
    all_ok &= check_response_shape(
        "Scenario 1", r1,
        expected_traits=["Yield_kg_ha"],
        expected_failed=[],
    )

    # Also verify PlotID was excluded from traits
    detected_traits_s1 = r1["summary_table"][0]["trait"]
    assert "PlotID" not in [row["trait"] for row in r1["summary_table"]], \
        "FAIL: PlotID should have been excluded from trait detection"
    print("  ok  [PlotID excluded from traits]")

    # ── Scenario 2: multi-env, multi-trait (balanced XLSX) ───────────────────
    r2 = run_scenario(
        label="2 — Multi-env, multi-trait (balanced XLSX)",
        file_bytes=make_multi_env_multi_trait(),
        file_type="xlsx",
        genotype_column="Genotype",
        rep_column="Rep",
        trait_columns=["Yield", "PlantHeight", "DaysToFlower"],
        mode="multi",
        environment_column="Environment",
        ag_stub=ag_stub,
    )
    all_ok &= check_response_shape(
        "Scenario 2", r2,
        expected_traits=["Yield", "PlantHeight", "DaysToFlower"],
        expected_failed=[],
    )
    # dataset_summary should record 3 environments
    all_ok &= assert_eq("n_environments=3", r2["dataset_summary"]["n_environments"], 3)

    # ── Scenario 3: one trait fails (< 6 valid obs) ──────────────────────────
    r3 = run_scenario(
        label="3 — One trait fails (BadTrait has only 4 valid values)",
        file_bytes=make_one_trait_fails(),
        file_type="csv",
        genotype_column="Genotype",
        rep_column="Rep",
        trait_columns=["Yield", "BadTrait"],
        mode="single",
        ag_stub=ag_stub,
    )
    all_ok &= check_response_shape(
        "Scenario 3", r3,
        expected_traits=["Yield", "BadTrait"],
        expected_failed=["BadTrait"],
    )

    # ── Scenario 4: missing values → unbalanced design warning ───────────────
    r4 = run_scenario(
        label="4 — Missing values (~20%) → unbalanced design warning",
        file_bytes=make_missing_values(),
        file_type="csv",
        genotype_column="Genotype",
        rep_column="Rep",
        trait_columns=["Yield", "PlantHeight"],
        mode="single",
        ag_stub=ag_stub,
    )
    all_ok &= check_response_shape(
        "Scenario 4", r4,
        expected_traits=["Yield", "PlantHeight"],
        expected_failed=[],
        check_warnings=True,  # expect unbalanced-design warnings on Yield
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ALL SCENARIOS PASSED" if all_ok else "SOME ASSERTIONS FAILED")
    print(f"{'='*60}\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
