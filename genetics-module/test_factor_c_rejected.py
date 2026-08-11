"""
FAC-08: a supplied third treatment factor is rejected, not silently dropped.

factor_c_column has always been accepted by the schema and read by nothing. The
R engine's factorial models top out at two treatment factors
(trait_value ~ genotype * factor), so a supplied third factor was discarded and
the researcher received a complete-looking two-factor ANOVA of an experiment
they did not run — the same failure class as a mis-mapped environment:
plausible output, wrong experiment, no warning.

Rejecting an unsupported input requires no scientific decision, so it ships
separately from the three-factor capability work (FAC-01..FAC-07), which is
blocked on biometrician sign-off. Nothing here touches the R engine, the
two-factor model, mean separation, or anything downstream of validation.

Run from inside genetics-module/:
    python -m pytest test_factor_c_rejected.py -v
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


def _factorial_df() -> pd.DataFrame:
    """Balanced two-factor factorial RCBD, with a spare column for Factor C."""
    rows = []
    for rep in ["R1", "R2", "R3"]:
        for irrigation in ["Full", "Deficit"]:
            for variety in ["V1", "V2", "V3"]:
                for spacing in ["Narrow", "Wide"]:
                    rows.append({
                        "Rep":        rep,
                        "Irrigation": irrigation,
                        "Variety":    variety,
                        "Spacing":    spacing,
                        "Yield":      3.0 + (0.4 if irrigation == "Full" else 0.0),
                    })
    return pd.DataFrame(rows)


def _splitplot_df() -> pd.DataFrame:
    rows = []
    for rep in ["B1", "B2", "B3"]:
        for tillage in ["Conventional", "NoTill"]:
            for fert in ["Low", "High"]:
                rows.append({
                    "Rep": rep, "Tillage": tillage, "Fert": fert,
                    "Yield": 10.0 + (2 if tillage == "NoTill" else 0),
                })
    return pd.DataFrame(rows)


def _met_df() -> pd.DataFrame:
    rows = []
    for loc in ["Loc1", "Loc2", "Loc3"]:
        for year in [2023, 2024]:
            for rep in [1, 2, 3]:
                for gi, geno in enumerate(["G1", "G2", "G3", "G4"]):
                    rows.append({
                        "Location": loc, "Year": year, "Replication": rep,
                        "Genotype": geno, "Yield": 3.0 + gi * 0.2,
                    })
    return pd.DataFrame(rows)


def _factorial_request(df: pd.DataFrame, **overrides) -> UploadAnalysisRequest:
    payload = dict(
        base64_content=_csv_b64(df),
        file_type="csv",
        genotype_column=None,
        rep_column="Rep",
        environment_column=None,
        factor_a_column="Irrigation",
        factor_b_column="Variety",
        trait_columns=["Yield"],
        mode="single",
        design_type="factorial",
        module="anova",
    )
    payload.update(overrides)
    return UploadAnalysisRequest(**payload)


class _StubEngine:
    """Marker: reaching run_analysis means validation was passed."""

    class Reached(Exception):
        pass

    def run_analysis(self, *args, **kwargs):
        raise _StubEngine.Reached()


def _call(request: UploadAnalysisRequest):
    """Run the handler with a stubbed engine (503 readiness guard precedes all validation)."""
    with patch.object(app_genetics, "r_engine", _StubEngine()):
        return asyncio.run(routes.analyze_upload(request))


def _assert_passes_validation(testcase, request: UploadAnalysisRequest) -> None:
    """No HTTPException, and the trait reached the stubbed engine."""
    response = _call(request)
    testcase.assertIsNotNone(response.dataset_summary)
    testcase.assertIn(request.trait_columns[0], response.failed_traits)


class TestFactorCRejected(unittest.TestCase):

    # ── Factor C supplied → clear, specific rejection ────────────────────────

    def test_factor_c_supplied_is_rejected(self):
        req = _factorial_request(_factorial_df(), factor_c_column="Spacing")
        with self.assertRaises(HTTPException) as ctx:
            _call(req)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_rejection_message_is_specific_not_generic(self):
        """Must name the limitation and the offending selection — not 'bad request'."""
        req = _factorial_request(_factorial_df(), factor_c_column="Spacing")
        with self.assertRaises(HTTPException) as ctx:
            _call(req)
        detail = str(ctx.exception.detail)
        self.assertIn("Three-factor factorial analysis isn't supported yet", detail)
        self.assertIn("two treatment factors", detail)
        # Names the actual column the researcher chose, so it is actionable.
        self.assertIn("Spacing", detail)
        # Not the generic column-existence error.
        self.assertNotIn("Columns not found in file", detail)

    def test_rejects_before_column_existence_check(self):
        """Presence alone triggers it — the answer must not depend on the
        column existing, or a typo'd third factor would return a misleading
        'columns not found' instead of 'not supported'."""
        req = _factorial_request(_factorial_df(), factor_c_column="NotAColumnAtAll")
        with self.assertRaises(HTTPException) as ctx:
            _call(req)
        self.assertIn("Three-factor factorial analysis isn't supported yet",
                      str(ctx.exception.detail))

    def test_whitespace_only_factor_c_is_not_a_selection(self):
        """Blank/whitespace is 'not supplied' — must not block a valid run."""
        for blank in ("", "   ", None):
            with self.subTest(factor_c_column=repr(blank)):
                req = _factorial_request(_factorial_df(), factor_c_column=blank)
                _assert_passes_validation(self, req)

    # ── Factor C absent → two-factor behaviour completely unchanged ──────────

    def test_two_factor_factorial_unchanged(self):
        req = _factorial_request(_factorial_df())
        _assert_passes_validation(self, req)

    def test_two_factor_factorial_crd_unchanged(self):
        req = _factorial_request(_factorial_df(), rep_column=None)
        _assert_passes_validation(self, req)

    # ── Boundary: Split-plot and MET paths unaffected ────────────────────────

    def test_split_plot_unaffected(self):
        req = UploadAnalysisRequest(
            base64_content=_csv_b64(_splitplot_df()),
            file_type="csv",
            rep_column="Rep",
            main_plot_column="Tillage",
            sub_plot_column="Fert",
            trait_columns=["Yield"],
            mode="single",
            design_type="split_plot_rcbd",
            module="anova",
        )
        _assert_passes_validation(self, req)

    def test_met_unaffected(self):
        req = UploadAnalysisRequest(
            base64_content=_csv_b64(_met_df()),
            file_type="csv",
            genotype_column="Genotype",
            rep_column="Replication",
            environment_column=None,
            environment_factor_columns=["Location", "Year"],
            trait_columns=["Yield"],
            mode="multi",
            module="anova",
        )
        _assert_passes_validation(self, req)

    def test_guard_is_design_agnostic_when_factor_c_present(self):
        """The field is honoured by no design, so its presence is rejected
        wherever it appears rather than only under design_type='factorial'."""
        req = UploadAnalysisRequest(
            base64_content=_csv_b64(_factorial_df()),
            file_type="csv",
            genotype_column="Variety",
            rep_column="Rep",
            trait_columns=["Yield"],
            mode="single",
            design_type="rcbd",
            factor_c_column="Spacing",
            module="anova",
        )
        with self.assertRaises(HTTPException) as ctx:
            _call(req)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Three-factor factorial analysis isn't supported yet",
                      str(ctx.exception.detail))


if __name__ == "__main__":
    unittest.main(verbosity=2)
