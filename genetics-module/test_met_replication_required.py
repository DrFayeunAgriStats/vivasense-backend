"""
Server-side backstop: Replication is required for multi-environment analysis.

Replication is legitimately optional for CRD, which declares no blocking
structure. A multi-environment run is a different case — the combined model
fits environment:rep, so the replication structure within each environment must
be identified before the trial can be described.

Before this guard an empty rep_column silently dropped out of
build_observations (keep_cols filters falsy names), the R engine received
records with no rep column, and compute_multi_environment failed with an error
that said nothing about the missing field.

These tests exercise the route handler directly and stop at the guard, so no R
process is required. Cases mirror the directive:

    A. multi-environment + missing Replication  -> 400, replication-specific
    B. multi-environment + valid Replication    -> passes the guard unchanged
    C. CRD + missing Replication                -> unaffected, no new requirement

Run from inside genetics-module/:
    python -m pytest test_met_replication_required.py -v
"""

import asyncio
import base64
import io
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException

import app_genetics
import multitrait_upload_routes as routes
from multitrait_upload_schemas import UploadAnalysisRequest


# ── helpers ───────────────────────────────────────────────────────────────────

def _csv_b64(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return base64.b64encode(buf.getvalue().encode()).decode()


def _met_df() -> pd.DataFrame:
    """Sayo-shaped MET: 3 Locations x 3 Years x 3 Reps x 4 Genotypes."""
    rows = []
    for loc in ["Loc1", "Loc2", "Loc3"]:
        for year in [2023, 2024, 2025]:
            for rep in [1, 2, 3]:
                for gi, geno in enumerate(["G1", "G2", "G3", "G4"]):
                    rows.append({
                        "Location":   loc,
                        "Year":       year,
                        "Replication": rep,
                        "Genotype":   geno,
                        "Grain_Yield": 3.0 + gi * 0.2 + rep * 0.05,
                    })
    return pd.DataFrame(rows)


def _crd_df() -> pd.DataFrame:
    """Single-environment CRD: no blocking structure, repeated observations."""
    rows = []
    for gi, geno in enumerate(["G1", "G2", "G3", "G4"]):
        for obs in range(3):
            rows.append({
                "Genotype":    geno,
                "Grain_Yield": 3.0 + gi * 0.2 + obs * 0.05,
            })
    return pd.DataFrame(rows)


def _request(df: pd.DataFrame, **overrides) -> UploadAnalysisRequest:
    payload = dict(
        base64_content=_csv_b64(df),
        file_type="csv",
        genotype_column="Genotype",
        rep_column="Replication",
        environment_column=None,
        environment_factor_columns=["Location", "Year"],
        trait_columns=["Grain_Yield"],
        mode="multi",
        module="anova",
    )
    payload.update(overrides)
    return UploadAnalysisRequest(**payload)


class _StubEngine:
    """Stops execution once the guard has been passed.

    Reaching this means the request was NOT rejected — which is exactly what
    cases B and C assert. Raising a unique marker keeps the test off the R
    process without pretending to produce a statistical result.
    """

    class Reached(Exception):
        pass

    def run_analysis(self, *args, **kwargs):
        raise _StubEngine.Reached()


def _call(request: UploadAnalysisRequest):
    """Run the route handler with a stubbed engine.

    analyze_upload does a lazy `import app_genetics` and rejects with 503 when
    r_engine is None, which happens before any structural validation. Stubbing
    it on the real module — the same object the handler imports — lets the
    request reach the guard under test.
    """
    with patch.object(app_genetics, "r_engine", _StubEngine()):
        return asyncio.run(routes.analyze_upload(request))


def _assert_passes_validation(testcase, request: UploadAnalysisRequest) -> None:
    """Assert the request is NOT rejected and reaches trait analysis.

    analyze_single_trait catches Exception per trait, so the stub marker is
    recorded as a failed trait rather than propagating. That is the signal we
    want: a response came back at all (no HTTPException from the guard), and
    the trait got as far as the engine call.
    """
    response = _call(request)
    testcase.assertIsNotNone(response.dataset_summary)
    testcase.assertIn(
        request.trait_columns[0], response.failed_traits,
        "trait should have reached the stubbed engine",
    )


class TestReplicationRequiredForMultiEnvironment(unittest.TestCase):

    # ── A. multi-environment + missing Replication ───────────────────────────

    def test_a_multi_environment_missing_replication_is_rejected(self):
        """400 with a replication-specific message — not an opaque failure."""
        for empty in ("", "   ", None):
            with self.subTest(rep_column=repr(empty)):
                req = _request(_met_df(), rep_column=empty)
                with self.assertRaises(HTTPException) as ctx:
                    _call(req)
                self.assertEqual(ctx.exception.status_code, 400)
                detail = str(ctx.exception.detail)
                self.assertIn("Replication column is required", detail)
                self.assertIn("multi-environment", detail)

    def test_a_message_names_the_structural_problem(self):
        """The message must identify the missing field, not just say 'failed'."""
        req = _request(_met_df(), rep_column="")
        with self.assertRaises(HTTPException) as ctx:
            _call(req)
        detail = str(ctx.exception.detail).lower()
        self.assertIn("replication", detail)
        # Explains why, so the user can act on it.
        self.assertIn("environment", detail)

    def test_a_also_rejects_explicit_environment_column(self):
        """Not specific to constructed environments — explicit column too."""
        df = _met_df()
        df["Env"] = df["Location"] + "_" + df["Year"].astype(str)
        req = _request(
            df, rep_column="", environment_column="Env",
            environment_factor_columns=[],
        )
        with self.assertRaises(HTTPException) as ctx:
            _call(req)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Replication column is required", str(ctx.exception.detail))

    # ── B. multi-environment + valid Replication ─────────────────────────────

    def test_b_multi_environment_with_replication_passes_the_guard(self):
        """The existing valid workflow is untouched by this validation."""
        req = _request(_met_df())
        _assert_passes_validation(self, req)

    # ── C. CRD + missing Replication ─────────────────────────────────────────

    def test_c_crd_missing_replication_still_allowed(self):
        """No new requirement is imposed on CRD."""
        req = _request(
            _crd_df(),
            rep_column="",
            environment_column=None,
            environment_factor_columns=[],
            mode="single",
        )
        _assert_passes_validation(self, req)

    def test_c_single_environment_with_replication_still_allowed(self):
        """Single-environment RCBD is likewise unaffected."""
        df = _met_df()
        req = _request(
            df,
            environment_column=None,
            environment_factor_columns=[],
            mode="single",
        )
        _assert_passes_validation(self, req)

    def test_c_downgraded_multi_run_is_not_rejected(self):
        """A multi request downgraded for want of environment levels is single.

        Gating on effective_mode rather than the requested mode means such a run
        is treated as single-environment and keeps CRD's optional replication.
        """
        df = _crd_df()
        df["Env"] = "OnlyOne"
        req = _request(
            df,
            rep_column="",
            environment_column="Env",
            environment_factor_columns=[],
            mode="multi",
        )
        _assert_passes_validation(self, req)


if __name__ == "__main__":
    unittest.main(verbosity=2)
