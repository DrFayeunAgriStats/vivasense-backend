"""End-to-end regression tests for the Phase 3 crop-protection backend contract."""

import asyncio
import base64
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

from crop_protection_orchestration import orchestrate_bioassay
from crop_protection_routes import analyze_bioassay, router
from crop_protection_schemas import BioassayAnalysisRequest
from crop_protection_validation import BioassayValidationError


DATA_DIR = Path(__file__).resolve().parent / "testdata" / "crop_protection"
DOSES = [0.2, 0.4, 0.6, 0.8, 1.0]


def _request(*, responses, design=None, cotoxicity=None, correlations=None,
             regressions=None, options=None, dataset_content="x"):
    return BioassayAnalysisRequest.model_validate({
        "dataset": {"base64_content": dataset_content, "file_type": "csv"},
        "design": design or {
            "design_type": "crd", "treatment_column": "Treatment",
            "dose_column": "Dose", "replicate_column": "Rep",
            "control_treatment_level": "C", "expected_dose_series": DOSES,
        },
        "responses": responses,
        "cotoxicity": cotoxicity,
        "correlation_response_ids": correlations or [],
        "regression_response_ids": regressions or [],
        "options": options or {},
    })


def _mortality(response_id, raw, inference, time, corrected=None):
    return {
        "id": response_id, "type": "mortality", "raw_column": raw,
        "inference_column": inference, "observation_time": time,
        "time_unit": "h", "corrected_column": corrected,
        "abbott_correction": True, "cumulative": True,
    }


@pytest.fixture(scope="module")
def dorcas_result():
    df = pd.read_csv(DATA_DIR / "dorcas_bioassay_48.csv")
    request = _request(
        responses=[
            _mortality("mortality_72h", "Mort72_pct", "AdtM72", 72),
            {"id": "weight_loss", "type": "continuous", "raw_column": "WTL",
             "inference_column": "WTL"},
        ],
        correlations=["mortality_72h", "weight_loss"],
        regressions=["mortality_72h", "weight_loss"],
    )
    return orchestrate_bioassay(df, request)


def _joint_result(filename, responses, levels, *, dose_column="Dose", options=None):
    request = _request(
        design={"design_type": "crd", "treatment_column": "Treatment",
                "dose_column": dose_column, "replicate_column": "Rep",
                "control_treatment_level": "C", "expected_dose_series": DOSES},
        responses=responses,
        cotoxicity={"enabled": True, "method": "bliss",
                    "component_a_level": levels[0], "component_b_level": levels[1],
                    "mixture_level": levels[2],
                    "response_ids": [item["id"] for item in responses],
                    "bootstrap_iterations": 1000, "seed": 20_260_818},
        options=options,
    )
    return orchestrate_bioassay(pd.read_csv(DATA_DIR / filename), request)


@pytest.fixture(scope="module")
def al_result():
    return _joint_result(
        "al_cl_alcl_joint_action.csv",
        [_mortality("mortality_48h", "Mort48_raw", "TAdTmRT48", 48, "Mort48_abbott"),
         _mortality("mortality_72h", "Mort72_raw", "TAdTmrt72", 72, "Mort72_abbott"),
         _mortality("mortality_96h", "Mort96_raw", "TAdtmrt96", 96, "Mort96_abbott")],
        ("AL", "CL", "ALCL"),
    )


@pytest.fixture(scope="module")
def clb_result():
    return _joint_result(
        "cl_b_clb_joint_action.csv",
        [_mortality("mortality_24h", "Mort24 %", "AdtM24", 24, "Abbott24 %"),
         _mortality("mortality_48h", "Mort48 %", "AdtM48", 48, "Abbott48 %"),
         _mortality("mortality_72h", "Mort72 %", "AdtM72", 72, "Abbott72 %")],
        ("CL", "B", "CLB"), dose_column="Dose numeric",
        options={"control_policy": "deduplicate_identical_replicates"},
    )


def _cell(result, time, dose):
    return next(cell for block in result["cotoxicity"]["by_time"]
                if block["observation_time"] == time
                for cell in block["cells"] if cell["dose"] == dose)


def test_public_route_is_registered():
    assert any(route.path == "/crop-protection/bioassay/analyze" for route in router.routes)


def test_dorcas_partition_and_crd_model(dorcas_result):
    design = dorcas_result["design"]
    assert (design["total_rows"], design["control_rows"], design["factorial_rows"]) == (48, 3, 45)
    assert design["factorial_treatments"] == ["CL", "TD", "CLTD"]
    assert design["cells"] == 15 and design["cell_replication"] == 3 and design["balanced"]
    assert design["replicate_role"] == "experimental_unit_identifier"
    for response in dorcas_result["response_results"]:
        sources = {row["source"]: row for row in response["anova"]}
        assert "Rep" not in sources
        assert sources["Error"]["df"] == 30


def test_dorcas_scale_pairing_and_control_separation(dorcas_result):
    mortality = dorcas_result["response_results"][0]
    assert mortality["provenance"]["raw_column"] == "Mort72_pct"
    assert mortality["provenance"]["inference_column"] == "AdtM72"
    assert mortality["mortality_correction"]["abbott_applied"] is True
    assert mortality["mortality_correction"]["scales"] == {
        "raw": "percent_mortality",
        "inference": "explicit_transformed_column",
        "corrected": "abbott_percent",
    }
    assert mortality["mortality_correction"]["control_n"] == 3
    assert set(mortality["provenance"]["control_rows_used"]).isdisjoint(
        mortality["provenance"]["factorial_rows_used"])


def test_dorcas_interaction_priority_matches_p_value(dorcas_result):
    for response in dorcas_result["response_results"]:
        interaction = next(row for row in response["anova"] if row["source"] == "Treatment:Dose")
        expected = "interaction" if interaction["p_value"] < 0.05 else "main_effects"
        assert response["interpretation_metadata"]["interpretation_priority"] == expected


def test_dorcas_correlation_regression_and_diagnostics(dorcas_result):
    assert dorcas_result["correlation"][0]["n"] == 45
    assert dorcas_result["correlation"][0]["population"] == "treated_only"
    assert len(dorcas_result["regression"]) == 6
    assert all(row["control_included"] is False for row in dorcas_result["regression"])
    for response in dorcas_result["response_results"]:
        assert response["diagnostics"]["residual_normality"]["test"] == "Shapiro-Wilk"
        assert "Treatment" in response["diagnostics"]["homogeneity"]["grouping"]


def test_al_bliss_known_48h_cell(al_result):
    cell = _cell(al_result, 48, 0.2)
    assert cell["bliss_expected"] == pytest.approx(50.0)
    assert cell["mixture"]["mean_corrected_mortality"] == pytest.approx(70.8333333333)
    assert cell["excess_observed_minus_expected"] == pytest.approx(20.8333333333)
    assert cell["bootstrap_ci"]["low"] <= 0 <= cell["bootstrap_ci"]["high"]
    assert cell["inference"] == "not_distinguishable_from_additivity"


def test_al_factorial_abbott_and_ceiling_warning(al_result):
    assert all(result["mortality_correction"]["abbott_applied"]
               for result in al_result["response_results"])
    assert all(result["anova"] for result in al_result["response_results"])
    assert "co_toxicity_ceiling_effect" in {item["code"] for item in al_result["warnings"]}


def test_clb_repeated_controls_resolve_to_three(clb_result):
    assert clb_result["design"]["total_rows"] == 54
    assert clb_result["design"]["control_rows"] == 3
    assert clb_result["design"]["factorial_rows"] == 45
    assert clb_result["design"]["factorial_treatments"] == ["CL", "B", "CLB"]
    assert "repeated_control_blocks" in {item["code"] for item in clb_result["warnings"]}


def test_clb_known_antagonistic_cell(clb_result):
    cell = _cell(clb_result, 24, 0.2)
    assert cell["bliss_expected"] == pytest.approx(88, abs=0.05)
    assert cell["mixture"]["mean_corrected_mortality"] == pytest.approx(16, abs=0.05)
    assert cell["excess_observed_minus_expected"] == pytest.approx(-72, abs=0.1)
    assert cell["bootstrap_ci"]["high"] < 0
    assert cell["inference"] == "supports_antagonism_under_bliss"


def test_neutral_bliss_fixture_is_not_overclassified():
    response = _mortality("mortality", "Mortality", "Mortality", 24)
    request = _request(
        design={"design_type": "crd", "treatment_column": "Treatment",
                "dose_column": "Dose", "replicate_column": "Rep",
                "control_treatment_level": "C", "expected_dose_series": [0.5, 1.0]},
        responses=[response],
        cotoxicity={"enabled": True, "method": "bliss", "component_a_level": "A",
                    "component_b_level": "B", "mixture_level": "AB",
                    "response_ids": ["mortality"], "bootstrap_iterations": 1000, "seed": 1},
    )
    result = orchestrate_bioassay(pd.read_csv(DATA_DIR / "neutral_bliss_bioassay.csv"), request)
    for dose in [0.5, 1.0]:
        cell = _cell(result, 24, dose)
        assert cell["excess_observed_minus_expected"] == pytest.approx(0)
        assert cell["inference"] == "not_distinguishable_from_additivity"


def test_duplicate_treated_unit_is_validation_error():
    df = pd.read_csv(DATA_DIR / "neutral_bliss_bioassay.csv")
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    request = _request(
        design={"design_type": "crd", "treatment_column": "Treatment", "dose_column": "Dose",
                "replicate_column": "Rep", "control_treatment_level": "C",
                "expected_dose_series": [0.5, 1.0]},
        responses=[_mortality("mortality", "Mortality", "Mortality", 24)],
    )
    with pytest.raises(BioassayValidationError, match="Duplicate Treatment"):
        orchestrate_bioassay(df, request)


def test_unsupported_cotoxicity_method_returns_not_implemented():
    content = base64.b64encode((DATA_DIR / "neutral_bliss_bioassay.csv").read_bytes()).decode()
    request = _request(
        dataset_content=content,
        design={"design_type": "crd", "treatment_column": "Treatment", "dose_column": "Dose",
                "replicate_column": "Rep", "control_treatment_level": "C",
                "expected_dose_series": [0.5, 1.0]},
        responses=[_mortality("mortality", "Mortality", "Mortality", 24)],
        cotoxicity={"enabled": True, "method": "Sun-Johnson CTC", "component_a_level": "A",
                    "component_b_level": "B", "mixture_level": "AB",
                    "response_ids": ["mortality"], "bootstrap_iterations": 1000},
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(analyze_bioassay(request))
    assert error.value.status_code == 501
    assert error.value.detail["code"] == "not_implemented"


def test_endpoint_handles_sanitized_column_names():
    content = base64.b64encode((DATA_DIR / "cl_b_clb_joint_action.csv").read_bytes()).decode()
    request = _request(
        dataset_content=content,
        design={"design_type": "crd", "treatment_column": "Treatment",
                "dose_column": "Dose numeric", "replicate_column": "Rep",
                "control_treatment_level": "C", "expected_dose_series": DOSES},
        responses=[_mortality("mortality_24h", "Mort24 %", "AdtM24", 24, "Abbott24 %")],
        options={"control_policy": "deduplicate_identical_replicates"},
    )
    result = asyncio.run(analyze_bioassay(request)).model_dump()
    assert result["status"] == "success"
    assert result["design"]["control_rows"] == 3
    assert result["response_results"][0]["provenance"]["raw_column"] == "Mort24"


def test_response_order_and_warning_contract(dorcas_result):
    assert dorcas_result["result_order"][0:2] == ["design", "warnings"]
    for item in dorcas_result["warnings"]:
        assert set(item) == {"code", "severity", "response_id", "message", "details"}


def test_single_scale_response_does_not_duplicate_adapter_column(dorcas_result):
    weight = next(item for item in dorcas_result["response_results"]
                  if item["response_id"] == "weight_loss")
    assert weight["provenance"]["raw_column"] == weight["provenance"]["inference_column"] == "WTL"
    assert weight["anova"]
