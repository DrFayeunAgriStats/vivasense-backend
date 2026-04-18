"""
API-level integration tests for the split_plot_rcbd path in upload_routes.py.

Tests the full route-handler cycle (UploadDatasetRequest → upload_dataset →
UploadDatasetResponse) without an R process.  All cases use non-genotype
factor names (Tillage × FertilizerRate, Irrigation × CropDensity) to
confirm the path is truly domain-neutral.

Run from inside genetics-module/:
    python -m pytest test_split_plot_upload_integration.py -v
"""

import asyncio
import base64
import io
import unittest

import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _csv_b64(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return base64.b64encode(buf.getvalue().encode()).decode()


def _tillage_fertilizer_df(n_reps: int = 3) -> pd.DataFrame:
    """Balanced split-plot: Block × Tillage (main) × FertilizerRate (sub)."""
    rows = []
    for rep in [f"B{i}" for i in range(1, n_reps + 1)]:
        for tillage in ["Conventional", "NoTill"]:
            for fert in ["Low", "Medium", "High"]:
                rows.append({
                    "Block":         rep,
                    "Tillage":       tillage,
                    "FertilizerRate": fert,
                    "Yield":         10.0 + (2 if tillage == "NoTill" else 0) + (3 if fert == "High" else 0),
                    "BiomassDW":     5.0  + (1 if tillage == "NoTill" else 0),
                })
    return pd.DataFrame(rows)


def _irrigation_density_df() -> pd.DataFrame:
    """Balanced split-plot: Rep × Irrigation (main) × CropDensity (sub)."""
    rows = []
    for rep in ["R1", "R2", "R3", "R4"]:
        for irr in ["Drip", "Sprinkler", "Rainfed"]:
            for density in ["Low", "High"]:
                rows.append({
                    "Rep":         rep,
                    "Irrigation":  irr,
                    "CropDensity": density,
                    "GrainYield":  7.0,
                })
    return pd.DataFrame(rows)


def _call(request):
    """Synchronously invoke the async upload_dataset route handler."""
    from upload_routes import upload_dataset
    return asyncio.run(upload_dataset(request))


def _make_request(**kwargs):
    from module_schemas import UploadDatasetRequest
    defaults = dict(
        file_type="csv",
        design_type="split_plot_rcbd",
        mode="single",
    )
    defaults.update(kwargs)
    return UploadDatasetRequest(**defaults)


# ── test cases ────────────────────────────────────────────────────────────────

class TestSplitPlotUploadHappyPath(unittest.TestCase):

    def setUp(self):
        self.df = _tillage_fertilizer_df(n_reps=3)
        self.b64 = _csv_b64(self.df)

    def _register(self, **extra):
        return _call(_make_request(
            base64_content=self.b64,
            rep_column="Block",
            main_plot_column="Tillage",
            sub_plot_column="FertilizerRate",
            **extra,
        ))

    def test_response_has_dataset_token(self):
        resp = self._register()
        self.assertTrue(resp.dataset_token, "dataset_token must be non-empty")

    def test_n_genotypes_is_none_for_split_plot(self):
        """Generic split-plot has no genotype column — n_genotypes must be None."""
        resp = self._register()
        self.assertIsNone(resp.n_genotypes)

    def test_n_reps_correct(self):
        resp = self._register()
        self.assertEqual(resp.n_reps, 3)

    def test_design_type_echoed(self):
        resp = self._register()
        self.assertEqual(resp.design_type, "split_plot_rcbd")

    def test_n_environments_is_none(self):
        resp = self._register()
        self.assertIsNone(resp.n_environments)

    def test_column_names_include_factor_columns(self):
        resp = self._register()
        for col in ("Block", "Tillage", "FertilizerRate", "Yield", "BiomassDW"):
            self.assertIn(col, resp.column_names)

    def test_n_rows_correct(self):
        """3 reps × 2 tillage × 3 fertilizer = 18 rows."""
        resp = self._register()
        self.assertEqual(resp.n_rows, 18)

    def test_second_non_genotype_fixture(self):
        """Irrigation × CropDensity dataset also registers correctly."""
        df = _irrigation_density_df()
        b64 = _csv_b64(df)
        resp = _call(_make_request(
            base64_content=b64,
            rep_column="Rep",
            main_plot_column="Irrigation",
            sub_plot_column="CropDensity",
        ))
        self.assertIsNone(resp.n_genotypes)
        self.assertEqual(resp.n_reps, 4)
        self.assertEqual(resp.n_rows, 4 * 3 * 2)

    def test_build_observations_end_to_end(self):
        """
        After registration, build_observations with the same params must
        emit only rep/main_plot/sub_plot/trait_value — no genotype key.
        """
        from multitrait_upload_routes import build_observations
        recs = build_observations(
            self.df,
            genotype_col=None,
            rep_col="Block",
            trait_col="Yield",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col="Tillage",
            sub_plot_col="FertilizerRate",
        )
        self.assertEqual(len(recs), 18)
        for rec in recs:
            self.assertIn("rep", rec)
            self.assertIn("main_plot", rec)
            self.assertIn("sub_plot", rec)
            self.assertIn("trait_value", rec)
            self.assertNotIn("genotype", rec)
            self.assertNotIn("environment", rec)
            self.assertNotIn("crd", rec)

    def test_check_balance_end_to_end_balanced(self):
        """Balanced Tillage × FertilizerRate dataset produces no warnings."""
        from multitrait_upload_routes import check_balance
        warnings = check_balance(
            self.df,
            genotype_col=None,
            rep_col="Block",
            trait_col="Yield",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col="Tillage",
            sub_plot_col="FertilizerRate",
        )
        self.assertEqual(warnings, [])

    def test_check_balance_end_to_end_unbalanced(self):
        """Dropping a subplot cell triggers a balance warning."""
        from multitrait_upload_routes import check_balance
        df = self.df.copy()
        # Remove all NoTill×High rows from B1
        mask = (df["Block"] == "B1") & (df["Tillage"] == "NoTill") & (df["FertilizerRate"] == "High")
        df = df[~mask]
        warnings = check_balance(
            df,
            genotype_col=None,
            rep_col="Block",
            trait_col="Yield",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col="Tillage",
            sub_plot_col="FertilizerRate",
        )
        self.assertTrue(len(warnings) > 0)

    def test_multi_trait_build_observations(self):
        """Both traits in the fixture can be built independently."""
        from multitrait_upload_routes import build_observations
        for trait in ("Yield", "BiomassDW"):
            recs = build_observations(
                self.df,
                genotype_col=None,
                rep_col="Block",
                trait_col=trait,
                env_col=None,
                design_type="split_plot_rcbd",
                main_plot_col="Tillage",
                sub_plot_col="FertilizerRate",
            )
            self.assertEqual(len(recs), 18, f"Expected 18 records for {trait}")


class TestSplitPlotUploadValidationRejections(unittest.TestCase):
    """
    Route-handler validates and rejects invalid split_plot_rcbd requests.
    """

    def setUp(self):
        self.df = _tillage_fertilizer_df()
        self.b64 = _csv_b64(self.df)

    def _expect_400(self, **kwargs):
        """Assert upload_dataset raises HTTPException with status_code 400."""
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            _call(_make_request(base64_content=self.b64, **kwargs))
        self.assertEqual(ctx.exception.status_code, 400)
        return ctx.exception.detail

    def test_genotype_column_as_third_factor_rejected(self):
        """Supplying genotype_column as a fourth column is rejected."""
        detail = self._expect_400(
            rep_column="Block",
            main_plot_column="Tillage",
            sub_plot_column="FertilizerRate",
            genotype_column="Yield",   # distinct fourth column — not valid
        )
        self.assertIn("third", detail.lower())

    def test_genotype_column_duplicating_main_plot_rejected(self):
        """genotype_column == main_plot_column triggers the general duplicate-column
        guard (which runs before design-specific checks)."""
        detail = self._expect_400(
            rep_column="Block",
            main_plot_column="Tillage",
            sub_plot_column="FertilizerRate",
            genotype_column="Tillage",  # same as main_plot → duplicate entry
        )
        self.assertIn("duplicate", detail.lower())

    def test_genotype_column_duplicating_sub_plot_rejected(self):
        """genotype_column == sub_plot_column triggers the general duplicate-column guard."""
        detail = self._expect_400(
            rep_column="Block",
            main_plot_column="Tillage",
            sub_plot_column="FertilizerRate",
            genotype_column="FertilizerRate",  # same as sub_plot → duplicate entry
        )
        self.assertIn("duplicate", detail.lower())

    def test_main_plot_equals_sub_plot_rejected(self):
        """main_plot_column == sub_plot_column triggers the general duplicate-column guard."""
        detail = self._expect_400(
            rep_column="Block",
            main_plot_column="Tillage",
            sub_plot_column="Tillage",    # duplicate of main_plot
        )
        self.assertIn("duplicate", detail.lower())

    def test_rep_equals_main_plot_rejected(self):
        """rep_column == main_plot_column triggers the general duplicate-column guard."""
        detail = self._expect_400(
            rep_column="Tillage",
            main_plot_column="Tillage",   # duplicate of rep
            sub_plot_column="FertilizerRate",
        )
        self.assertIn("duplicate", detail.lower())

    def test_environment_column_rejected_for_split_plot(self):
        """environment_column is not valid for split_plot_rcbd.
        Uses a column name that doesn't collide so the specific design check fires."""
        detail = self._expect_400(
            rep_column="Block",
            main_plot_column="Tillage",
            sub_plot_column="FertilizerRate",
            environment_column="Yield",   # unique — general dup check passes
        )
        self.assertIn("environment_column", detail)

    def test_factor_column_rejected_for_split_plot(self):
        """factor_column is not valid for split_plot_rcbd.
        Uses a column name that doesn't collide so the specific design check fires."""
        detail = self._expect_400(
            rep_column="Block",
            main_plot_column="Tillage",
            sub_plot_column="FertilizerRate",
            factor_column="Yield",        # unique — general dup check passes
        )
        self.assertIn("factor_column", detail)

    def test_multi_mode_rejected_for_split_plot(self):
        """mode='multi' is not valid for split_plot_rcbd."""
        detail = self._expect_400(
            rep_column="Block",
            main_plot_column="Tillage",
            sub_plot_column="FertilizerRate",
            mode="multi",
        )
        self.assertIn("single", detail.lower())

    def test_missing_rep_column_rejected(self):
        """Omitting rep_column for split_plot_rcbd is rejected."""
        detail = self._expect_400(
            main_plot_column="Tillage",
            sub_plot_column="FertilizerRate",
        )
        self.assertIn("rep_column", detail)

    def test_missing_main_plot_column_rejected(self):
        """Omitting main_plot_column for split_plot_rcbd is rejected."""
        detail = self._expect_400(
            rep_column="Block",
            sub_plot_column="FertilizerRate",
        )
        self.assertIn("main_plot_column", detail)

    def test_column_not_in_file_rejected(self):
        """Specifying a column name absent from the file is rejected."""
        detail = self._expect_400(
            rep_column="Block",
            main_plot_column="NonExistentColumn",
            sub_plot_column="FertilizerRate",
        )
        self.assertIn("NonExistentColumn", detail)


class TestNonSplitPlotDesignsUnaffected(unittest.TestCase):
    """
    Verify that the changes introduced for split_plot_rcbd do not break
    the standard RCBD registration path.
    """

    def test_standard_rcbd_still_requires_genotype_column(self):
        """Non-split-plot designs must still require genotype_column."""
        from fastapi import HTTPException
        df = pd.DataFrame({
            "Genotype": ["G1", "G2"] * 3,
            "Rep":      ["R1", "R1", "R2", "R2", "R3", "R3"],
            "Yield":    [5.0, 6.0, 5.1, 6.1, 4.9, 5.9],
        })
        b64 = _csv_b64(df)
        req = _make_request(
            base64_content=b64,
            file_type="csv",
            design_type="single",
            mode="single",
            rep_column="Rep",
            # genotype_column intentionally omitted
        )
        with self.assertRaises(HTTPException) as ctx:
            _call(req)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("genotype_column", ctx.exception.detail)

    def test_standard_rcbd_with_genotype_succeeds(self):
        """Standard RCBD with genotype_column registers correctly."""
        df = pd.DataFrame({
            "Genotype": ["G1", "G2", "G3"] * 3,
            "Rep":      ["R1"] * 3 + ["R2"] * 3 + ["R3"] * 3,
            "Yield":    [5.0, 6.0, 5.5] * 3,
        })
        b64 = _csv_b64(df)
        req = _make_request(
            base64_content=b64,
            file_type="csv",
            design_type="single",
            mode="single",
            genotype_column="Genotype",
            rep_column="Rep",
        )
        resp = _call(req)
        self.assertEqual(resp.n_genotypes, 3)
        self.assertEqual(resp.n_reps, 3)


if __name__ == "__main__":
    unittest.main()
