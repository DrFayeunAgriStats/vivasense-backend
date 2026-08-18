"""Protected scientific regression tests for crop-protection factorial CRD.

Fixture provenance: ``Dorcas_Bioassay_Statistical_Analysis_FINAL_CORRECTED.xlsx``,
sheet ``Raw Data`` (header row 3), supplied 2026-08-18.  The CSV is a direct
selection of scientific columns; measurements are not altered.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from crop_protection_factorial import (
    FactorialCrdValidationError,
    analyze_factorial_crd,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "testdata"
    / "crop_protection"
    / "dorcas_bioassay_48.csv"
)
DOSES = [0.2, 0.4, 0.6, 0.8, 1.0]
FIXTURE_SHA256 = "a8d1ff26d59b316ae6e80ec2a328900d8153f419d0e5937cf84b14ea24870a25"


@pytest.fixture(scope="module")
def dorcas() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


@pytest.fixture(scope="module")
def transformed_result(dorcas: pd.DataFrame) -> dict:
    return analyze_factorial_crd(
        dorcas,
        treatment_column="Treatment",
        dose_column="Dose",
        replicate_column="Rep",
        response_column="AdtM72",
        display_column="Mort72_pct",
        control_level="C",
        expected_dose_series=DOSES,
    )


def _anova_by_source(result: dict) -> dict:
    return {row["source"]: row for row in result["anova"]}


def test_dorcas_fixture_is_bitwise_protected():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


def test_dorcas_design_recognition(dorcas: pd.DataFrame, transformed_result: dict):
    assert len(dorcas) == 48
    assert transformed_result["design"] == {
        "design_type": "factorial_crd",
        "treatment_levels": 3,
        "dose_levels": 5,
        "factorial_n": 45,
        "control_n": 3,
        "balanced": True,
    }
    assert set(dorcas.loc[dorcas.Treatment != "C", "Treatment"]) == {"CL", "CLTD", "TD"}
    assert len(transformed_result["factorial_rows_used"]) == 45
    assert len(transformed_result["rows_excluded"]) == 3


def test_factorial_crd_formula_omits_replicate(transformed_result: dict):
    assert transformed_result["model_formula"] == "response ~ treatment * dose"
    sources = set(_anova_by_source(transformed_result))
    assert sources == {"Treatment", "Dose", "Treatment:Dose", "Error"}
    assert "Rep" not in sources
    assert "Block" not in sources
    assert transformed_result["provenance"]["replicate_model_role"] == (
        "experimental_unit_identifier"
    )


def test_dorcas_degrees_of_freedom_are_scientifically_fixed(transformed_result: dict):
    anova = _anova_by_source(transformed_result)
    assert anova["Treatment"]["df"] == 2
    assert anova["Dose"]["df"] == 4
    assert anova["Treatment:Dose"]["df"] == 8
    assert anova["Error"]["df"] == 30
    assert sum(row["df"] for row in transformed_result["anova"]) == 44


def test_interaction_replication_is_three_not_treatment_total(transformed_result: dict):
    assert transformed_result["common_cell_n"] == 3
    assert {cell["n"] for cell in transformed_result["interaction"]["means"]} == {3}
    assert transformed_result["common_cell_n"] != 15


def test_interaction_se_is_sqrt_mse_over_cell_n(transformed_result: dict):
    expected = np.sqrt(transformed_result["residual_mean_square"] / 3)
    assert transformed_result["common_interaction_se"] == pytest.approx(expected, rel=1e-12)
    for cell in transformed_result["interaction"]["means"]:
        assert cell["se_inference_scale"] == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(5.392082568748735, rel=1e-12)


def test_tukey_letters_agree_with_independent_statsmodels_calculation(
    dorcas: pd.DataFrame, transformed_result: dict
):
    treated = dorcas.loc[dorcas.Treatment != "C"].copy()
    treated["cell"] = treated.Treatment + "|" + treated.Dose.astype(str)
    independent = pairwise_tukeyhsd(treated.AdtM72, treated.cell, alpha=0.05)
    letters = {
        f"{cell['treatment']}|{float(cell['dose'])}": set(cell["letter"])
        for cell in transformed_result["interaction"]["means"]
    }

    # Independently computed Tukey decisions and agricolae's CLD must encode the
    # same result for every one of the 105 cell pairs: significant pairs share
    # no letter; non-significant pairs share at least one.
    for left, right, reject in zip(
        independent._multicomp.pairindices[0],
        independent._multicomp.pairindices[1],
        independent.reject,
    ):
        left_name = independent.groupsunique[left]
        right_name = independent.groupsunique[right]
        share_letter = bool(letters[left_name] & letters[right_name])
        assert share_letter is not bool(reject), (left_name, right_name)

    assert letters["CL|0.6"] == {"a"}
    assert letters["TD|0.2"] == {"d"}


def test_transformed_inference_and_raw_display_are_kept_distinct(
    dorcas: pd.DataFrame, transformed_result: dict
):
    cell = next(
        row
        for row in transformed_result["interaction"]["means"]
        if row["treatment"] == "CL" and float(row["dose"]) == 0.2
    )
    source = dorcas[(dorcas.Treatment == "CL") & (dorcas.Dose == 0.2)]
    assert cell["mean_inference_scale"] == pytest.approx(source.AdtM72.mean())
    assert cell["mean_display_scale"] == pytest.approx(source.Mort72_pct.mean())
    assert cell["mean"] == pytest.approx(source.Mort72_pct.mean())
    assert transformed_result["provenance"]["inference_column"] == "AdtM72"
    assert transformed_result["provenance"]["display_column"] == "Mort72_pct"


def test_unequal_replication_returns_cell_n_and_no_common_se():
    rows = []
    for treatment_index, treatment in enumerate(["A", "B"]):
        for dose in [1.0, 2.0]:
            n = 2 if (treatment, dose) == ("B", 2.0) else 3
            for rep in range(1, n + 1):
                rows.append(
                    {
                        "Treatment": treatment,
                        "Dose": dose,
                        "Rep": rep,  # deliberately repeats in every cell
                        "Response": 5 * treatment_index + 2 * dose + rep,
                    }
                )
    rows.extend(
        [
            {"Treatment": "Control", "Dose": 0.0, "Rep": 1, "Response": 0.0},
            {"Treatment": "Control", "Dose": 0.0, "Rep": 2, "Response": 0.5},
        ]
    )
    result = analyze_factorial_crd(
        pd.DataFrame(rows),
        treatment_column="Treatment",
        dose_column="Dose",
        replicate_column="Rep",
        response_column="Response",
        control_level="Control",
        expected_dose_series=[1.0, 2.0],
    )

    assert result["design"]["balanced"] is False
    assert result["common_cell_n"] is None
    assert result["common_interaction_se"] is None
    assert sorted(cell["n"] for cell in result["interaction"]["means"]) == [2, 3, 3, 3]
    for cell in result["interaction"]["means"]:
        expected = np.sqrt(result["residual_mean_square"] / cell["n"])
        assert cell["se_inference_scale"] == pytest.approx(expected)
    assert any("unequal replication" in warning for warning in result["warnings"])
    assert "Rep" not in _anova_by_source(result)


def test_duplicate_experimental_unit_is_rejected(dorcas: pd.DataFrame):
    duplicate = pd.concat([dorcas, dorcas.iloc[[3]]], ignore_index=True)
    with pytest.raises(FactorialCrdValidationError, match="Duplicate Treatment"):
        analyze_factorial_crd(
            duplicate,
            treatment_column="Treatment",
            dose_column="Dose",
            replicate_column="Rep",
            response_column="AdtM72",
            control_level="C",
            expected_dose_series=DOSES,
        )
