"""Strict balance preflight for classical multi-environment genetic parameters."""

import asyncio
import base64
import io
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException

import analysis_genetic_parameters_routes as gp_routes
import app_genetics
import multitrait_upload_routes as upload_routes
from module_schemas import ModuleRequest
from multitrait_upload_schemas import UploadAnalysisRequest


def _balanced_met(reps=("R1", "R2", "R3")) -> pd.DataFrame:
    rows = []
    for env_i, env in enumerate(("E1", "E2")):
        for rep_i, rep in enumerate(reps):
            for geno_i, geno in enumerate(("G1", "G2", "G3")):
                rows.append(
                    {
                        "Genotype": geno,
                        "Environment": env,
                        "Rep": rep,
                        "TraitA": 10 + env_i + rep_i + geno_i,
                        "TraitB": 20 + env_i + rep_i + geno_i,
                    }
                )
    return pd.DataFrame(rows)


def _single_environment() -> pd.DataFrame:
    rows = []
    for rep_i, rep in enumerate(("R1", "R2", "R3")):
        for geno_i, geno in enumerate(("G1", "G2", "G3")):
            rows.append({"Genotype": geno, "Rep": rep, "TraitA": 10 + rep_i + geno_i})
    return pd.DataFrame(rows)


def _csv_b64(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return base64.b64encode(buf.getvalue().encode()).decode()


class _Engine:
    def __init__(self):
        self.calls = 0

    def run_analysis(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("engine reached")


def _upload_request(df: pd.DataFrame, **overrides) -> UploadAnalysisRequest:
    payload = {
        "base64_content": _csv_b64(df),
        "file_type": "csv",
        "genotype_column": "Genotype",
        "rep_column": "Rep",
        "environment_column": "Environment",
        "environment_factor_columns": [],
        "trait_columns": ["TraitA"],
        "mode": "multi",
        "module": "genetic_parameters",
        "design_type": "rcbd",
    }
    payload.update(overrides)
    return UploadAnalysisRequest(**payload)


class TestBalancedMetValidator(unittest.TestCase):
    def validate(self, df, trait="TraitA"):
        return upload_routes.validate_balanced_met_variance_components(
            df, "Genotype", "Environment", "Rep", trait
        )

    def test_complete_balanced_met_passes(self):
        self.assertEqual(self.validate(_balanced_met()), [])

    def test_incomplete_genotype_environment_crossing_is_rejected(self):
        df = _balanced_met()
        df = df[~((df["Genotype"] == "G3") & (df["Environment"] == "E2"))]
        errors = self.validate(df)
        self.assertTrue(any("incomplete Genotype × Environment" in e for e in errors))

    def test_unequal_cell_counts_are_rejected(self):
        errors = self.validate(_balanced_met().drop(index=0))
        self.assertTrue(any("unequal observations" in e for e in errors))

    def test_one_replication_per_cell_is_rejected(self):
        errors = self.validate(_balanced_met(reps=("R1",)))
        self.assertTrue(any("at least 2" in e for e in errors))

    def test_duplicate_rep_cannot_disguise_missing_block(self):
        df = _balanced_met()
        missing = df[
            (df["Genotype"] == "G1")
            & (df["Environment"] == "E1")
            & (df["Rep"] == "R1")
        ].index[0]
        duplicate = df[
            (df["Genotype"] == "G1")
            & (df["Environment"] == "E1")
            & (df["Rep"] == "R2")
        ].iloc[[0]]
        df = pd.concat([df.drop(index=missing), duplicate], ignore_index=True)
        errors = self.validate(df)
        self.assertTrue(any("duplicate observations" in e for e in errors))
        self.assertTrue(any("do not contain each genotype exactly once" in e for e in errors))

    def test_inconsistent_rep_sets_are_rejected(self):
        df = _balanced_met()
        df.loc[(df["Environment"] == "E2") & (df["Rep"] == "R3"), "Rep"] = "R4"
        errors = self.validate(df)
        self.assertTrue(any("inconsistent replication labels" in e for e in errors))

    def test_missing_trait_value_that_breaks_balance_is_rejected(self):
        df = _balanced_met()
        df.loc[0, "TraitA"] = None
        self.assertTrue(any("unequal observations" in e for e in self.validate(df)))

    def test_non_numeric_trait_value_that_breaks_balance_is_rejected(self):
        df = _balanced_met()
        df["TraitA"] = df["TraitA"].astype(object)
        df.loc[0, "TraitA"] = "not measured"
        self.assertTrue(any("unequal observations" in e for e in self.validate(df)))

    def test_symmetric_missing_rep_remains_balanced(self):
        df = _balanced_met()
        df.loc[df["Rep"] == "R3", "TraitA"] = None
        self.assertEqual(self.validate(df), [])

    def test_trait_specific_environment_loss_is_rejected(self):
        df = _balanced_met()
        df.loc[df["Environment"] == "E2", "TraitA"] = None
        errors = self.validate(df)
        self.assertTrue(any("only 1 environment" in e for e in errors))

    def test_missing_structural_value_is_rejected(self):
        df = _balanced_met()
        df.loc[0, "Rep"] = None
        self.assertTrue(any("missing replication" in e for e in self.validate(df)))


class TestBalancedMetUploadRoute(unittest.TestCase):
    def call(self, request, engine):
        with patch.object(app_genetics, "r_engine", engine):
            return asyncio.run(upload_routes.analyze_upload(request))

    def test_invalid_genetic_parameters_rejects_whole_request_before_r(self):
        df = _balanced_met()
        df.loc[0, "TraitA"] = None
        df.loc[1, "TraitB"] = None
        engine = _Engine()
        request = _upload_request(df, trait_columns=["TraitA", "TraitB"])
        with self.assertRaises(HTTPException) as caught:
            self.call(request, engine)
        self.assertEqual(caught.exception.status_code, 400)
        detail = str(caught.exception.detail)
        self.assertIn("Trait 'TraitA'", detail)
        self.assertIn("Trait 'TraitB'", detail)
        self.assertEqual(engine.calls, 0)

    def test_valid_genetic_parameters_reaches_r(self):
        engine = _Engine()
        response = self.call(_upload_request(_balanced_met()), engine)
        self.assertEqual(engine.calls, 1)
        self.assertIn("TraitA", response.failed_traits)

    def test_constructed_location_year_structure_is_validated_after_resolution(self):
        df = _balanced_met().rename(columns={"Environment": "Location"})
        df["Year"] = 2025
        engine = _Engine()
        request = _upload_request(
            df,
            environment_column=None,
            environment_factor_columns=["Location", "Year"],
        )
        response = self.call(request, engine)
        self.assertEqual(engine.calls, 1)
        self.assertIn("TraitA", response.failed_traits)

    def test_unbalanced_met_anova_remains_warning_only(self):
        df = _balanced_met().drop(index=0)
        engine = _Engine()
        request = _upload_request(df, module="anova")
        response = self.call(request, engine)
        self.assertEqual(engine.calls, 1)
        self.assertIn("TraitA", response.failed_traits)

    def test_single_environment_genetic_parameters_is_unaffected(self):
        engine = _Engine()
        request = _upload_request(
            _single_environment(),
            mode="single",
            environment_column=None,
            trait_columns=["TraitA"],
        )
        response = self.call(request, engine)
        self.assertEqual(engine.calls, 1)
        self.assertIn("TraitA", response.failed_traits)


class TestBalancedMetModuleRoute(unittest.TestCase):
    def context(self, df):
        return {
            "base64_content": _csv_b64(df),
            "file_type": "csv",
            "genotype_column": "Genotype",
            "rep_column": "Rep",
            "environment_column": "Environment",
            "factor_column": None,
            "main_plot_column": None,
            "sub_plot_column": None,
            "mode": "multi",
            "design_type": "rcbd",
            "random_environment": False,
            "selection_intensity": 0.20,
        }

    def test_validation_precedes_cache_reuse(self):
        df = _balanced_met()
        df.loc[0, "TraitA"] = None
        engine = _Engine()
        request = ModuleRequest(dataset_token="token", trait_columns=["TraitA"])
        with (
            patch.object(app_genetics, "r_engine", engine),
            patch.object(gp_routes.dataset_cache, "get_dataset", return_value=self.context(df)),
            patch.object(
                gp_routes.dataset_cache,
                "get_analysis",
                side_effect=AssertionError("cache reached before validation"),
            ),
        ):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(gp_routes.analysis_genetic_parameters(request))
        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("Trait 'TraitA'", str(caught.exception.detail))
        self.assertEqual(engine.calls, 0)


if __name__ == "__main__":
    unittest.main()
