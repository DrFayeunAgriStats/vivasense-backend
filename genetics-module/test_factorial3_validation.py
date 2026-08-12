"""
Three-factor factorial — request validation (FAC-04, FAC-06, FAC-07, FAC-08).

Replaces test_factor_c_rejected.py, which asserted the blanket "three factors
not supported" rejection. That guard is deliberately retired here: left in
place it would fire before the new model logic could ever run, making the
capability unreachable. The tests below pin its removal directly — a valid
three-factor request must now SUCCEED, not merely stop hitting the old message.

What replaces it is dataset-specific:

  FAC-04  every A x B x C combination present, each with >= 3 observations
  FAC-06  failures reject outright, naming exactly what is wrong; never a
          silent downgrade to a two-factor analysis
  FAC-07  cell completeness also surfaced per trait by check_balance

Run from inside genetics-module/:
    python -m pytest test_factorial3_validation.py -v
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
from multitrait_upload_routes import check_balance
from multitrait_upload_schemas import UploadAnalysisRequest


# ── fixtures ──────────────────────────────────────────────────────────────────

def _csv_b64(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return base64.b64encode(buf.getvalue().encode()).decode()


def _abc_df(reps=("R1", "R2", "R3")) -> pd.DataFrame:
    """Complete 2 x 3 x 2 factorial in 3 blocks = 36 rows."""
    rows = []
    for rep in reps:
        for irrigation in ["Full", "Deficit"]:
            for variety in ["V1", "V2", "V3"]:
                for spacing in ["Narrow", "Wide"]:
                    rows.append({
                        "Rep": rep, "Irrigation": irrigation,
                        "Variety": variety, "Spacing": spacing,
                        "Yield": 10.0 + (1.5 if irrigation == "Full" else -1.5),
                    })
    return pd.DataFrame(rows)


def _abc_request(df: pd.DataFrame, **overrides) -> UploadAnalysisRequest:
    payload = dict(
        base64_content=_csv_b64(df),
        file_type="csv",
        genotype_column=None,
        rep_column="Rep",
        factor_a_column="Irrigation",
        factor_b_column="Variety",
        factor_c_column="Spacing",
        trait_columns=["Yield"],
        mode="single",
        design_type="factorial",
        module="anova",
    )
    payload.update(overrides)
    return UploadAnalysisRequest(**payload)


class _StubEngine:
    class Reached(Exception):
        pass

    def run_analysis(self, *args, **kwargs):
        raise _StubEngine.Reached()


def _call(request: UploadAnalysisRequest):
    with patch.object(app_genetics, "r_engine", _StubEngine()):
        return asyncio.run(routes.analyze_upload(request))


def _assert_passes_validation(testcase, request: UploadAnalysisRequest) -> None:
    """No HTTPException, and the trait reached the engine."""
    response = _call(request)
    testcase.assertIsNotNone(response.dataset_summary)
    testcase.assertIn(request.trait_columns[0], response.failed_traits)


class TestFacEightRetired(unittest.TestCase):
    """The blanket rejection is gone, not merely superseded."""

    def test_valid_three_factor_request_now_succeeds(self):
        _assert_passes_validation(self, _abc_request(_abc_df()))

    def test_old_blanket_message_is_never_produced(self):
        """Any rejection must be dataset-specific, never the retired text."""
        bad = _abc_df()
        bad = bad[~((bad["Irrigation"] == "Full") & (bad["Variety"] == "V3"))]
        with self.assertRaises(HTTPException) as ctx:
            _call(_abc_request(bad))
        self.assertNotIn("isn't supported yet", str(ctx.exception.detail))


class TestCellCompletenessAndReplication(unittest.TestCase):

    def test_incomplete_cells_rejected_naming_the_missing_combinations(self):
        df = _abc_df()
        df = df[~((df["Irrigation"] == "Full") & (df["Variety"] == "V3")
                  & (df["Spacing"] == "Wide"))]
        with self.assertRaises(HTTPException) as ctx:
            _call(_abc_request(df))
        detail = str(ctx.exception.detail)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Incomplete three-factor design", detail)
        self.assertIn("11 of 12", detail)
        self.assertIn("Full × V3 × Wide", detail)

    def test_insufficient_replication_rejected_naming_the_cells(self):
        df = _abc_df()
        mask = ((df["Irrigation"] == "Deficit") & (df["Variety"] == "V2")
                & (df["Spacing"] == "Narrow") & (df["Rep"] == "R3"))
        df = df[~mask]
        with self.assertRaises(HTTPException) as ctx:
            _call(_abc_request(df))
        detail = str(ctx.exception.detail)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Insufficient replication", detail)
        self.assertIn("Deficit × V2 × Narrow (n=2)", detail)

    def test_replication_message_states_the_proxy_limitation(self):
        """Passing the row-count check must not read as 'replication is valid'."""
        df = _abc_df()
        df = df[~((df["Irrigation"] == "Full") & (df["Variety"] == "V1")
                  & (df["Spacing"] == "Wide") & (df["Rep"] == "R1"))]
        with self.assertRaises(HTTPException) as ctx:
            _call(_abc_request(df))
        detail = str(ctx.exception.detail).lower()
        self.assertIn("structural check on row counts only", detail)
        self.assertIn("cannot tell independent replicates", detail)

    def test_two_replicates_everywhere_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            _call(_abc_request(_abc_df(reps=("R1", "R2"))))
        self.assertIn("Insufficient replication", str(ctx.exception.detail))

    def test_never_auto_downgrades_to_two_factor(self):
        """A failing three-factor request must reject, not quietly drop C."""
        df = _abc_df()
        df = df[~((df["Irrigation"] == "Full") & (df["Spacing"] == "Wide"))]
        with self.assertRaises(HTTPException):
            _call(_abc_request(df))


class TestStructuralPreconditions(unittest.TestCase):

    def test_factor_c_requires_factorial_design(self):
        with self.assertRaises(HTTPException) as ctx:
            _call(_abc_request(_abc_df(), design_type="rcbd",
                               genotype_column="Irrigation"))
        self.assertIn("factorial designs only", str(ctx.exception.detail))

    def test_factor_c_requires_first_two_factors(self):
        with self.assertRaises(HTTPException) as ctx:
            _call(_abc_request(_abc_df(), factor_b_column=None,
                               genotype_column="Variety"))
        self.assertIn("without both of the first two", str(ctx.exception.detail))


class TestTwoFactorUnchanged(unittest.TestCase):

    def test_two_factor_rcbd_unaffected(self):
        _assert_passes_validation(
            self, _abc_request(_abc_df(), factor_c_column=None))

    def test_two_factor_crd_unaffected(self):
        _assert_passes_validation(
            self, _abc_request(_abc_df(), factor_c_column=None, rep_column=None))

    def test_two_factor_with_two_reps_still_allowed(self):
        """The >=3 rule is a three-factor requirement, not a global one."""
        _assert_passes_validation(
            self, _abc_request(_abc_df(reps=("R1", "R2")), factor_c_column=None))


class TestBoundaryPathsUnaffected(unittest.TestCase):

    def test_split_plot_unaffected(self):
        rows = []
        for rep in ["B1", "B2", "B3"]:
            for till in ["Conv", "NoTill"]:
                for fert in ["Low", "High"]:
                    rows.append({"Rep": rep, "Till": till, "Fert": fert, "Yield": 10.0})
        req = UploadAnalysisRequest(
            base64_content=_csv_b64(pd.DataFrame(rows)), file_type="csv",
            rep_column="Rep", main_plot_column="Till", sub_plot_column="Fert",
            trait_columns=["Yield"], mode="single",
            design_type="split_plot_rcbd", module="anova",
        )
        _assert_passes_validation(self, req)

    def test_met_unaffected(self):
        rows = []
        for loc in ["L1", "L2", "L3"]:
            for year in [2023, 2024]:
                for rep in [1, 2, 3]:
                    for gi, g in enumerate(["G1", "G2", "G3", "G4"]):
                        rows.append({"Location": loc, "Year": year, "Replication": rep,
                                     "Genotype": g, "Yield": 3.0 + gi * 0.2})
        req = UploadAnalysisRequest(
            base64_content=_csv_b64(pd.DataFrame(rows)), file_type="csv",
            genotype_column="Genotype", rep_column="Replication",
            environment_column=None,
            environment_factor_columns=["Location", "Year"],
            trait_columns=["Yield"], mode="multi", module="anova",
        )
        _assert_passes_validation(self, req)


class TestCheckBalanceThreeFactor(unittest.TestCase):
    """FAC-07: cell completeness surfaced per trait, at the blocking level."""

    def test_complete_design_produces_no_completeness_warning(self):
        warnings = check_balance(
            df=_abc_df(), genotype_col="Irrigation", rep_col="Rep",
            trait_col="Yield", env_col=None, factor_col="Variety",
            design_type="factorial", factor_c_col="Spacing",
        )
        self.assertFalse(any("Incomplete three-factor" in w for w in warnings))

    def test_incomplete_design_is_flagged_not_silently_accepted(self):
        df = _abc_df()
        df = df[~((df["Irrigation"] == "Full") & (df["Variety"] == "V3")
                  & (df["Spacing"] == "Wide"))]
        warnings = check_balance(
            df=df, genotype_col="Irrigation", rep_col="Rep",
            trait_col="Yield", env_col=None, factor_col="Variety",
            design_type="factorial", factor_c_col="Spacing",
        )
        self.assertTrue(any("Incomplete three-factor structure" in w for w in warnings))
        self.assertTrue(any("11 of 12" in w for w in warnings))

    def test_unequal_replication_is_flagged(self):
        df = _abc_df()
        df = df[~((df["Irrigation"] == "Full") & (df["Variety"] == "V1")
                  & (df["Spacing"] == "Wide") & (df["Rep"] == "R1"))]
        warnings = check_balance(
            df=df, genotype_col="Irrigation", rep_col="Rep",
            trait_col="Yield", env_col=None, factor_col="Variety",
            design_type="factorial", factor_c_col="Spacing",
        )
        self.assertTrue(any("Unequal replication across three-factor cells" in w
                            for w in warnings))

    def test_two_factor_balance_path_untouched(self):
        warnings = check_balance(
            df=_abc_df(), genotype_col="Irrigation", rep_col="Rep",
            trait_col="Yield", env_col=None, factor_col="Variety",
            design_type="factorial", factor_c_col=None,
        )
        self.assertFalse(any("three-factor" in w for w in warnings))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestAnovaLabelMapping(unittest.TestCase):
    """Internal role names must never reach the researcher.

    This class runs the REAL R engine rather than the stub. Every other test
    here stops at validation, and the R-level tests read raw anova_table
    rownames — so the Python layer that translates genotype/factor/factor_c
    into the researcher's own column names was covered by nothing. A
    three-factor run consequently shipped "genotype:factor:factor_c" to the
    UI where it should have read "A×B×C". Slower than the rest of the file by
    an R subprocess, and worth it: this is the only test that sees the
    translated output.
    """

    @classmethod
    def setUpClass(cls):
        import app_genetics
        cls._saved = getattr(app_genetics, "r_engine", None)
        app_genetics.r_engine = app_genetics.RGeneticsEngine("vivasense_genetics.R")
        cls._app_genetics = app_genetics

    @classmethod
    def tearDownClass(cls):
        cls._app_genetics.r_engine = cls._saved

    def _sources(self, **overrides):
        req = _abc_request(_abc_df(), **overrides)
        response = asyncio.run(routes.analyze_upload(req))
        trait = response.trait_results[req.trait_columns[0]]
        self.assertEqual(trait.status, "success", msg=str(trait.error))
        return list(trait.analysis_result.result.anova_table.source)

    def test_three_factor_labels_use_researcher_column_names(self):
        sources = self._sources()
        self.assertEqual(
            sources,
            ["rep", "Irrigation", "Variety", "Spacing",
             "Irrigation×Variety", "Irrigation×Spacing", "Variety×Spacing",
             "Irrigation×Variety×Spacing", "Residuals"],
        )

    def test_no_internal_role_name_survives_to_the_client(self):
        for label in self._sources():
            for internal in ("genotype", "factor_c", "factor"):
                self.assertNotIn(internal, label,
                                 f"internal role name leaked in {label!r}")

    def test_two_factor_labels_unchanged(self):
        sources = self._sources(factor_c_column=None)
        self.assertEqual(
            sources,
            ["rep", "Irrigation", "Variety", "Irrigation×Variety", "Residuals"],
        )
