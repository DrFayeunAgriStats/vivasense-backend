"""Scientific regression tests for Abbott correction and Bliss joint action."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from crop_protection_cotoxicity import (
    CotoxicityValidationError,
    analyze_bliss_joint_action,
    bliss_expected,
    bootstrap_bliss_excess,
)
from crop_protection_mortality import (
    MortalityResponseMapping,
    MortalityValidationError,
    prepare_mortality_responses,
    validate_cumulative_mortality,
)


DATA_DIR = Path(__file__).resolve().parent / "testdata" / "crop_protection"
AL_FIXTURE = DATA_DIR / "al_cl_alcl_joint_action.csv"
CLB_FIXTURE = DATA_DIR / "cl_b_clb_joint_action.csv"
DOSES = [0.2, 0.4, 0.6, 0.8, 1.0]

AL_MAPPINGS = [
    MortalityResponseMapping("Mort48_raw", 48, "h", "TAdTmRT48", "Mort48_abbott"),
    MortalityResponseMapping("Mort72_raw", 72, "h", "TAdTmrt72", "Mort72_abbott"),
    MortalityResponseMapping("Mort96_raw", 96, "h", "TAdtmrt96", "Mort96_abbott"),
]
CLB_MAPPINGS = [
    MortalityResponseMapping("Mort24 %", 24, "h", "AdtM24", "Abbott24 %"),
    MortalityResponseMapping("Mort48 %", 48, "h", "AdtM48", "Abbott48 %"),
    MortalityResponseMapping("Mort72 %", 72, "h", "AdtM72", "Abbott72 %"),
]


def _prepare_al() -> dict:
    return prepare_mortality_responses(
        pd.read_csv(AL_FIXTURE),
        treatment_column="Treatment", dose_column="Dose", replicate_column="Rep",
        control_level="C", mortality_responses=AL_MAPPINGS, floor_at_zero=True,
    )


def _prepare_clb() -> dict:
    return prepare_mortality_responses(
        pd.read_csv(CLB_FIXTURE),
        treatment_column="Treatment", dose_column="Dose numeric", replicate_column="Rep",
        control_level="C", mortality_responses=CLB_MAPPINGS, floor_at_zero=True,
        control_policy="deduplicate_identical_replicates",
    )


@pytest.fixture(scope="module")
def al_prepared() -> dict:
    return _prepare_al()


@pytest.fixture(scope="module")
def al_bliss(al_prepared: dict) -> dict:
    return analyze_bliss_joint_action(
        al_prepared["long_rows"], component_a_level="AL", component_b_level="CL",
        mixture_level="ALCL", expected_dose_series=DOSES,
        bootstrap_iterations=10_000, bootstrap_seed=20_260_818,
    )


@pytest.fixture(scope="module")
def clb_prepared() -> dict:
    return _prepare_clb()


@pytest.fixture(scope="module")
def clb_bliss(clb_prepared: dict) -> dict:
    return analyze_bliss_joint_action(
        clb_prepared["long_rows"], component_a_level="CL", component_b_level="B",
        mixture_level="CLB", expected_dose_series=DOSES,
        bootstrap_iterations=10_000, bootstrap_seed=20_260_818,
    )


def _cell(result: dict, time: float, dose: float) -> dict:
    return next(
        cell for cell in result["cells"]
        if cell["observation_time"] == time and cell["dose"] == dose
    )


def test_validation_fixtures_are_bitwise_protected():
    assert hashlib.sha256(AL_FIXTURE.read_bytes()).hexdigest() == (
        "359e77964d6fa22ad975bea60e2b1c5463efbe069b739df96ddcc4ecd3c08234"
    )
    assert hashlib.sha256(CLB_FIXTURE.read_bytes()).hexdigest() == (
        "52d551071072286b7120f0b5198e471bc9ac5fac3ece7c60512ab09233ee6081"
    )


def test_abbott_formula_and_time_specific_control_means(al_prepared: dict):
    by_time = {item["observation_time"]: item for item in al_prepared["responses"]}
    assert by_time[48]["control_mean_raw_mortality"] == pytest.approx(20.0)
    assert by_time[72]["control_mean_raw_mortality"] == pytest.approx(80 / 3)
    assert by_time[96]["control_mean_raw_mortality"] == pytest.approx(110 / 3)
    row = next(
        row for row in by_time[48]["rows"]
        if row["treatment"] == "CL" and row["dose"] == 0.2 and row["replicate"] == "1"
    )
    assert row["raw_mortality"] == 40
    assert row["raw_abbott_value"] == pytest.approx(25.0)


def test_control_has_no_abbott_value(al_prepared: dict):
    controls = [row for row in al_prepared["long_rows"] if row["treatment"] == "C"]
    assert controls
    assert all(row["raw_abbott_value"] is None for row in controls)
    assert all(row["display_abbott_value"] is None for row in controls)
    assert all(row["abbott_status"] == "reference_control" for row in controls)


def test_negative_abbott_floor_policy_is_explicit():
    df = pd.DataFrame({
        "Treatment": ["C", "C", "T"], "Dose": [0, 0, 1], "Rep": [1, 2, 1],
        "Mort": [20, 20, 10],
    })
    result = prepare_mortality_responses(
        df, treatment_column="Treatment", dose_column="Dose", replicate_column="Rep",
        control_level="C", mortality_responses=[MortalityResponseMapping("Mort", 24, "h")],
        floor_at_zero=True,
    )
    treated = next(row for row in result["long_rows"] if row["treatment"] == "T")
    assert treated["raw_abbott_value"] == pytest.approx(-12.5)
    assert treated["display_abbott_value"] == 0
    assert treated["floor_applied"] is True
    assert result["responses"][0]["provenance"]["floor_at_zero"] is True

    unbounded = prepare_mortality_responses(
        df, treatment_column="Treatment", dose_column="Dose", replicate_column="Rep",
        control_level="C", mortality_responses=[MortalityResponseMapping("Mort", 24, "h")],
        floor_at_zero=False,
    )
    extra_roles = [
        {**next(row for row in unbounded["long_rows"] if row["treatment"] == "T"),
         "treatment": role}
        for role in ["A", "B", "AB"]
    ]
    with pytest.raises(CotoxicityValidationError, match="zero-floor policy"):
        analyze_bliss_joint_action(
            extra_roles, component_a_level="A", component_b_level="B", mixture_level="AB",
            bootstrap_iterations=1000,
        )


def test_control_mortality_100_is_rejected():
    df = pd.DataFrame({
        "Treatment": ["C", "C", "T"], "Dose": [0, 0, 1], "Rep": [1, 2, 1],
        "Mort": [100, 100, 100],
    })
    with pytest.raises(MortalityValidationError, match="undefined"):
        prepare_mortality_responses(
            df, treatment_column="Treatment", dose_column="Dose", replicate_column="Rep",
            control_level="C", mortality_responses=[MortalityResponseMapping("Mort", 24, "h")],
        )


def test_supplied_abbott_is_verified_not_trusted(al_prepared: dict):
    assert all(
        response["supplied_correction_verification"]["verification_status"] == "matched"
        for response in al_prepared["responses"]
    )
    altered = pd.read_csv(AL_FIXTURE)
    altered.loc[altered.Treatment == "CL", "Mort48_abbott"] += 1
    result = prepare_mortality_responses(
        altered, treatment_column="Treatment", dose_column="Dose", replicate_column="Rep",
        control_level="C", mortality_responses=[AL_MAPPINGS[0]], floor_at_zero=True,
    )
    verification = result["responses"][0]["supplied_correction_verification"]
    assert verification["verification_status"] == "mismatch"
    assert verification["mismatch_count"] == 15
    assert verification["max_absolute_difference"] == pytest.approx(1.0)


def test_bliss_formula_excess_and_ratio_are_exact():
    assert bliss_expected(20, 50) == 60
    rows = []
    for treatment, values in {"A": [20, 20], "B": [50, 50], "AB": [75, 75]}.items():
        for rep, value in enumerate(values, 1):
            rows.append({
                "treatment": treatment, "dose": 1.0, "replicate": str(rep),
                "observation_time": 24, "time_unit": "h",
                "display_abbott_value": value, "abbott_status": "calculated",
            })
    cell = analyze_bliss_joint_action(
        rows, component_a_level="A", component_b_level="B", mixture_level="AB",
        bootstrap_iterations=1000, bootstrap_seed=1,
    )["cells"][0]
    assert cell["bliss_expected"] == 60
    assert cell["excess_observed_minus_expected"] == 15
    assert cell["observed_expected_ratio"] == 1.25


def test_zero_expected_mortality_returns_unavailable_ratio():
    rows = []
    for treatment in ["A", "B", "AB"]:
        for rep in [1, 2, 3]:
            rows.append({
                "treatment": treatment, "dose": 1, "replicate": str(rep),
                "observation_time": 24, "time_unit": "h", "display_abbott_value": 0,
                "abbott_status": "calculated",
            })
    cell = analyze_bliss_joint_action(
        rows, component_a_level="A", component_b_level="B", mixture_level="AB",
        bootstrap_iterations=1000, bootstrap_seed=2,
    )["cells"][0]
    assert cell["bliss_expected"] == 0
    assert cell["observed_expected_ratio"] is None
    assert cell["inference"] == "not_distinguishable_from_additivity"


def test_bootstrap_is_deterministic_and_stratified():
    kwargs = dict(iterations=5000, confidence_level=0.95, seed=42)
    first = bootstrap_bliss_excess([0, 10, 20], [30, 40, 50], [60, 70, 80], **kwargs)
    second = bootstrap_bliss_excess([0, 10, 20], [30, 40, 50], [60, 70, 80], **kwargs)
    assert first == second
    assert first["resampling"] == "independent_within_component_a_component_b_and_mixture"


def test_al_cl_alcl_48h_positive_but_inconclusive(al_bliss: dict):
    cell = _cell(al_bliss, 48, 0.2)
    assert cell["component_a"]["mean_corrected_mortality"] == 0
    assert cell["component_b"]["mean_corrected_mortality"] == 50
    assert cell["mixture"]["mean_corrected_mortality"] == pytest.approx(70.8333333333)
    assert cell["bliss_expected"] == 50
    assert cell["excess_observed_minus_expected"] == pytest.approx(20.8333333333)
    assert cell["observed_expected_ratio"] == pytest.approx(1.4166666667)
    assert cell["bootstrap_ci"]["low"] == pytest.approx(-4.1666666667)
    assert cell["bootstrap_ci"]["high"] == pytest.approx(45.8333333333)
    assert cell["descriptive_direction"] == "positive_deviation"
    assert cell["inference"] == "not_distinguishable_from_additivity"


def test_supported_positive_joint_action_requires_ci_above_zero():
    rows = []
    for treatment, value in {"A": 10, "B": 10, "AB": 50}.items():
        for rep in range(1, 5):
            rows.append({
                "treatment": treatment, "dose": 1, "replicate": str(rep),
                "observation_time": 24, "time_unit": "h",
                "display_abbott_value": value, "abbott_status": "calculated",
            })
    cell = analyze_bliss_joint_action(
        rows, component_a_level="A", component_b_level="B", mixture_level="AB",
        bootstrap_iterations=1000, bootstrap_seed=3,
    )["cells"][0]
    assert cell["bootstrap_ci"]["low"] > 0
    assert cell["inference"] == "supports_synergy_under_bliss"


def test_cl_b_clb_supports_negative_joint_action(clb_bliss: dict):
    cell = _cell(clb_bliss, 24, 0.2)
    assert cell["component_a"]["mean_corrected_mortality"] == pytest.approx(87.999876)
    assert cell["component_b"]["mean_corrected_mortality"] == 0
    assert cell["mixture"]["mean_corrected_mortality"] == pytest.approx(16.0187935)
    assert cell["bliss_expected"] == pytest.approx(87.999876)
    assert cell["excess_observed_minus_expected"] == pytest.approx(-71.9810825)
    assert cell["bootstrap_ci"]["high"] < 0
    assert cell["descriptive_direction"] == "negative_deviation"
    assert cell["inference"] == "supports_antagonism_under_bliss"


def test_cl_b_clb_48h_remains_strongly_negative(clb_bliss: dict):
    cell = _cell(clb_bliss, 48, 0.2)
    assert cell["bliss_expected"] == pytest.approx(96.0085275)
    assert cell["excess_observed_minus_expected"] == pytest.approx(-66.8502970)
    assert cell["bootstrap_ci"]["high"] < 0
    assert cell["inference"] == "supports_antagonism_under_bliss"


def test_ceiling_effect_prevents_synergy_claim(al_bliss: dict):
    cells = [_cell(al_bliss, 96, dose) for dose in DOSES]
    assert all(cell["bliss_expected"] == pytest.approx(100) for cell in cells)
    assert all(cell["ceiling_effect"] for cell in cells)
    assert all(cell["inference"] == "ceiling_limited" for cell in cells)
    assert all("cannot meaningfully distinguish" in cell["warnings"][0] for cell in cells)


def test_missing_matched_dose_is_unavailable_not_interpolated(al_prepared: dict):
    rows = [
        row for row in al_prepared["long_rows"]
        if not (row["treatment"] == "AL" and row["dose"] == 0.4 and row["observation_time"] == 48)
    ]
    result = analyze_bliss_joint_action(
        rows, component_a_level="AL", component_b_level="CL", mixture_level="ALCL",
        expected_dose_series=DOSES, bootstrap_iterations=1000, bootstrap_seed=4,
    )
    cell = _cell(result, 48, 0.4)
    assert cell["available"] is False
    assert cell["missing_roles"] == ["AL"]


def test_repeated_control_blocks_require_explicit_policy():
    with pytest.raises(MortalityValidationError, match="Repeated control blocks"):
        prepare_mortality_responses(
            pd.read_csv(CLB_FIXTURE), treatment_column="Treatment", dose_column="Dose numeric",
            replicate_column="Rep", control_level="C", mortality_responses=CLB_MAPPINGS,
            floor_at_zero=True,
        )


def test_repeated_control_blocks_do_not_inflate_n(clb_prepared: dict):
    control = clb_prepared["control"]
    assert control["control_rows_available"] == 9
    assert control["n_control"] == 3
    assert control["repeated_control_blocks_detected"] is True
    assert control["duplicates_removed"] == 6
    assert control["selection_rule"] == "deduplicate_identical_replicates"


def test_transformed_values_are_not_used_for_abbott_or_bliss():
    original = pd.read_csv(AL_FIXTURE)
    changed = original.copy()
    changed["TAdTmRT48"] = 9999
    base = prepare_mortality_responses(
        original, treatment_column="Treatment", dose_column="Dose", replicate_column="Rep",
        control_level="C", mortality_responses=[AL_MAPPINGS[0]], floor_at_zero=True,
    )
    altered = prepare_mortality_responses(
        changed, treatment_column="Treatment", dose_column="Dose", replicate_column="Rep",
        control_level="C", mortality_responses=[AL_MAPPINGS[0]], floor_at_zero=True,
    )
    base_values = [row["display_abbott_value"] for row in base["long_rows"]]
    altered_values = [row["display_abbott_value"] for row in altered["long_rows"]]
    assert base_values == altered_values
    assert altered["provenance"]["cotoxicity_scale"] == "abbott_corrected_percentage"


def test_cumulative_mortality_uses_raw_scale_and_flags_decreases(al_prepared: dict):
    audit = validate_cumulative_mortality(al_prepared["long_rows"], cumulative=True)
    assert audit["scale_checked"] == "raw_mortality_percentage"
    assert audit["decrease_count"] == 2
    assert audit["warnings"]
    assert all(
        decrease["from_raw_mortality"] > decrease["to_raw_mortality"]
        for decrease in audit["decreases"]
    )


def test_overall_summary_does_not_collapse_dose_specific_results(al_bliss: dict):
    summary_48 = next(row for row in al_bliss["time_summaries"] if row["observation_time"] == 48)
    assert summary_48["number_of_matched_doses"] == 5
    assert summary_48["number_positive"] == 4
    assert summary_48["number_additive"] == 1
    assert summary_48["number_inconclusive"] == 5
