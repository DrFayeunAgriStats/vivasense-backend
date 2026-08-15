"""Regression coverage for token-based MET structure reconstruction."""

import asyncio
import base64
import io
import unittest
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException

import analysis_genetic_parameters_routes as gp_routes
import app_genetics
import dataset_cache
from environment_structure import (
    CONSTRUCTED_ENVIRONMENT_COLUMN,
    NESTED_REP_COLUMN,
    SOURCE_SUPPLIED,
    reconstruct_environment_structure,
)
from module_schemas import ModuleRequest, UploadDatasetRequest
from upload_routes import upload_dataset


def _csv_b64(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return base64.b64encode(buf.getvalue().encode()).decode()


def _met(*, explicit_environment=True, disjoint_reps=False) -> pd.DataFrame:
    rows = []
    for location in ("North", "South"):
        for year in (2024, 2025):
            environment = f"{location}-{year}"
            for rep in ("R1", "R2"):
                rep_label = f"{environment}-{rep}" if disjoint_reps else rep
                for genotype in ("G1", "G2"):
                    row = {
                        "Genotype": genotype,
                        "Location": location,
                        "Year": year,
                        "Rep": rep_label,
                        "Trait": len(rows) + 1,
                    }
                    if explicit_environment:
                        row["Environment"] = environment
                    rows.append(row)
    return pd.DataFrame(rows)


def _single() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Genotype": ["G1", "G2"] * 3,
            "Rep": ["R1", "R1", "R2", "R2", "R3", "R3"],
            "Trait": [1, 2, 2, 3, 3, 4],
        }
    )


class _RecordingEngine:
    def __init__(self):
        self.calls = []

    def run_analysis(self, *args, **kwargs):
        self.calls.append(kwargs)
        raise RuntimeError("recorded")


async def _register(
    df: pd.DataFrame,
    *,
    mode="multi",
    environment_column="Environment",
    environment_factor_columns=None,
) -> str:
    response = await upload_dataset(
        UploadDatasetRequest(
            base64_content=_csv_b64(df),
            file_type="csv",
            genotype_column="Genotype",
            rep_column="Rep",
            environment_column=environment_column,
            environment_factor_columns=environment_factor_columns or [],
            design_type="rcbd",
            mode=mode,
        )
    )
    return response.dataset_token


async def _analyse(token: str, engine: _RecordingEngine):
    with patch.object(app_genetics, "r_engine", engine):
        return await gp_routes.analysis_genetic_parameters(
            ModuleRequest(dataset_token=token, trait_columns=["Trait"])
        )


class TestMetTokenStructureProvenance(unittest.TestCase):
    def test_explicit_environment_token_round_trip(self):
        token = asyncio.run(_register(_met()))
        engine = _RecordingEngine()

        asyncio.run(_analyse(token, engine))

        self.assertEqual(len(engine.calls), 1)
        observations = engine.calls[0]["data"]
        self.assertEqual({row["environment"] for row in observations}, {
            "North-2024", "North-2025", "South-2024", "South-2025"
        })
        ctx = dataset_cache.get_dataset(token)
        self.assertEqual(ctx["environment_structure_recipe"]["environment_column"], "Environment")

    def test_location_year_environment_is_reconstructed_from_token_recipe(self):
        token = asyncio.run(
            _register(
                _met(explicit_environment=False),
                environment_column=None,
                environment_factor_columns=["Location", "Year"],
            )
        )
        engine = _RecordingEngine()

        asyncio.run(_analyse(token, engine))

        ctx = dataset_cache.get_dataset(token)
        self.assertIsNone(ctx["environment_column"])
        self.assertEqual(ctx["resolved_environment_column"], CONSTRUCTED_ENVIRONMENT_COLUMN)
        self.assertEqual(
            ctx["environment_structure_recipe"]["environment_factor_columns"],
            ["Location", "Year"],
        )
        observations = engine.calls[0]["data"]
        self.assertEqual(len({row["environment"] for row in observations}), 4)

    def test_disjoint_rep_labels_reconstruct_deterministic_nested_rep(self):
        token = asyncio.run(
            _register(
                _met(explicit_environment=False, disjoint_reps=True),
                environment_column=None,
                environment_factor_columns=["Location", "Year"],
            )
        )
        engine = _RecordingEngine()

        asyncio.run(_analyse(token, engine))

        ctx = dataset_cache.get_dataset(token)
        self.assertEqual(ctx["rep_column"], "Rep")
        self.assertEqual(ctx["resolved_rep_column"], NESTED_REP_COLUMN)
        self.assertEqual(ctx["environment_structure_recipe"]["rep_column"], "Rep")
        observations = engine.calls[0]["data"]
        self.assertEqual({row["rep"] for row in observations}, {"1", "2"})

    def test_explicit_environment_takes_precedence_over_factor_recipe(self):
        token = asyncio.run(
            _register(
                _met(),
                environment_column="Environment",
                environment_factor_columns=["Location", "Year"],
            )
        )
        ctx = dataset_cache.get_dataset(token)
        df = _met()

        structure = reconstruct_environment_structure(df, ctx)

        self.assertEqual(structure.source, SOURCE_SUPPLIED)
        self.assertEqual(structure.environment_column, "Environment")
        self.assertNotIn(CONSTRUCTED_ENVIRONMENT_COLUMN, df.columns)

    def test_legacy_synthetic_token_without_recipe_is_rejected_clearly(self):
        df = _met(explicit_environment=False)
        ctx = {
            "base64_content": _csv_b64(df),
            "file_type": "csv",
            "genotype_column": "Genotype",
            "rep_column": NESTED_REP_COLUMN,
            "environment_column": CONSTRUCTED_ENVIRONMENT_COLUMN,
            "factor_column": None,
            "main_plot_column": None,
            "sub_plot_column": None,
            "mode": "multi",
            "design_type": "rcbd",
            "random_environment": False,
            "selection_intensity": 0.20,
        }
        engine = _RecordingEngine()
        with (
            patch.object(app_genetics, "r_engine", engine),
            patch.object(gp_routes.dataset_cache, "get_dataset", return_value=ctx),
            patch.object(
                gp_routes.dataset_cache,
                "get_analysis",
                side_effect=AssertionError("cache must not be reached"),
            ),
        ):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(
                    gp_routes.analysis_genetic_parameters(
                        ModuleRequest(dataset_token="legacy", trait_columns=["Trait"])
                    )
                )

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("does not contain the original environment-resolution recipe", str(caught.exception.detail))
        self.assertEqual(engine.calls, [])

    def test_cache_cannot_bypass_reconstruction_and_balance_validation(self):
        df = _met(explicit_environment=False).drop(index=0)
        token = asyncio.run(
            _register(
                df,
                environment_column=None,
                environment_factor_columns=["Location", "Year"],
            )
        )
        engine = _RecordingEngine()
        with (
            patch.object(app_genetics, "r_engine", engine),
            patch.object(
                gp_routes.dataset_cache,
                "get_analysis",
                side_effect=AssertionError("cache reached before validation"),
            ),
        ):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(
                    gp_routes.analysis_genetic_parameters(
                        ModuleRequest(dataset_token=token, trait_columns=["Trait"])
                    )
                )

        self.assertEqual(caught.exception.status_code, 400)
        self.assertIn("Trait 'Trait'", str(caught.exception.detail))
        self.assertEqual(engine.calls, [])

    def test_single_environment_token_path_does_not_reconstruct(self):
        token = asyncio.run(
            _register(
                _single(), mode="single", environment_column=None
            )
        )
        engine = _RecordingEngine()
        with patch.object(
            gp_routes,
            "reconstruct_environment_structure",
            side_effect=AssertionError("single path changed"),
        ):
            asyncio.run(_analyse(token, engine))

        self.assertEqual(len(engine.calls), 1)
        self.assertNotIn("environment", engine.calls[0]["data"][0])


if __name__ == "__main__":
    unittest.main()
