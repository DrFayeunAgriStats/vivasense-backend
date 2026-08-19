"""Scientific regression tests for dynamic one-to-three-factor crop-protection CRD."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from crop_protection_factorial import analyze_factorial_crd
from crop_protection_orchestration import (
    UnsupportedBioassayAnalysis, _interpretation_priority, orchestrate_bioassay,
)
from crop_protection_schemas import BioassayAnalysisRequest

FIXTURE = Path(__file__).parent / "testdata" / "crop_protection" / "maize_three_factor_crd.csv"

def request(factors, *, regression=False, cotoxicity=False):
    return BioassayAnalysisRequest.model_validate({
        "dataset":{"base64_content":"x","file_type":"csv"},
        "design":{"design_type":"crd","factor_columns":factors,"replicate_column":"REP"},
        "responses":[{"id":"ovi58","type":"continuous","raw_column":"OVI58","inference_column":"OVI58"}],
        "regression_response_ids":["ovi58"] if regression else [],
        "correlation_response_ids":[],
        "cotoxicity": ({"enabled":True,"method":"bliss","component_a_level":"A",
                         "component_b_level":"B","mixture_level":"AB","response_ids":[]}
                        if cotoxicity else None), "options":{}})

@pytest.fixture(scope="module")
def maize(): return pd.read_csv(FIXTURE)

@pytest.fixture(scope="module")
def maize_result(maize):
    return analyze_factorial_crd(maize, factor_columns=["VRT","FORM","LVL"],
        replicate_column="REP", response_column="OVI58")

def test_maize_fixture_is_protected_and_complete(maize):
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == "7a4d77473cade1a4970ab98b1a4effcf5d7468a3434cfbea3d9f740acc9ed616"
    assert maize.shape == (96,19)
    assert {c:maize[c].nunique() for c in ["VRT","FORM","LVL","REP"]} == {"VRT":4,"FORM":2,"LVL":4,"REP":3}
    assert maize.groupby(["VRT","FORM","LVL"]).size().eq(3).all()

def test_three_factor_df_cells_rep_and_no_rep_source(maize_result):
    assert maize_result["design"]["cells"] == 32
    assert maize_result["design"]["cell_replication"] == 3
    assert maize_result["design"]["balanced"] is True
    assert {r["source"]:r["df"] for r in maize_result["anova"]} == {
        "VRT":3,"FORM":1,"LVL":3,"VRT × FORM":3,"VRT × LVL":9,
        "FORM × LVL":3,"VRT × FORM × LVL":9,"Error":64}
    assert "REP" not in {r["source"] for r in maize_result["anova"]}

def test_maize_ovi58_anova_matches_independent_statsmodels(maize, maize_result):
    independent = anova_lm(ols("OVI58 ~ C(VRT)*C(FORM)*C(LVL)", data=maize).fit(), typ=1)
    names = {"VRT":"C(VRT)","FORM":"C(FORM)","LVL":"C(LVL)",
      "VRT × FORM":"C(VRT):C(FORM)","VRT × LVL":"C(VRT):C(LVL)",
      "FORM × LVL":"C(FORM):C(LVL)","VRT × FORM × LVL":"C(VRT):C(FORM):C(LVL)","Error":"Residual"}
    for row in maize_result["anova"]:
        ref=independent.loc[names[row["source"]]]
        assert row["ss"] == pytest.approx(ref["sum_sq"], rel=1e-10)
        assert row["ms"] == pytest.approx(ref["sum_sq"]/ref["df"], rel=1e-10)
        if row["source"] != "Error":
            assert row["f_value"] == pytest.approx(ref["F"], rel=1e-10)
            assert row["p_value"] == pytest.approx(ref["PR(>F)"], rel=1e-10)

def test_full_cell_se_uses_three_replicates(maize_result):
    mse = next(r["ms"] for r in maize_result["anova"] if r["source"]=="Error")
    assert maize_result["common_interaction_se"] == pytest.approx(np.sqrt(mse/3))
    assert all(cell["se_inference_scale"] == pytest.approx(np.sqrt(mse/3)) for cell in maize_result["cell_means"])

def test_three_way_cld_agrees_with_independent_tukey_decision(maize, maize_result):
    labels = maize[["VRT","FORM","LVL"]].astype(str).agg("|".join,axis=1)
    tukey = pairwise_tukeyhsd(maize.OVI58, labels)
    cells={"|".join(c["factor_levels"].values()):c for c in maize_result["cell_means"]}
    checked=0
    for (a,b),reject in zip(zip(tukey._multicomp.pairindices[0],tukey._multicomp.pairindices[1]),tukey.reject):
        la,lb=tukey.groupsunique[a],tukey.groupsunique[b]
        share=bool(set(cells[la]["letter"]) & set(cells[lb]["letter"]))
        assert share is (not bool(reject)); checked+=1
    assert checked == 496

def test_one_factor_crd_model_tukey_and_rep_semantics(maize):
    one_factor = maize.loc[(maize["FORM"] == 1) & (maize["LVL"] == 1)].copy()
    result=analyze_factorial_crd(one_factor,factor_columns=["VRT"],replicate_column="REP",response_column="OVI58")
    assert [r["source"] for r in result["anova"]] == ["VRT","Error"]
    assert result["error_df"] == 8 and len(result["cell_means"]) == 4
    assert result["provenance"]["replicate_model_role"] == "experimental_unit_identifier"

def test_legacy_two_factor_contract_still_runs(maize):
    two_factor = maize.loc[maize["FORM"] == 1].copy()
    result=analyze_factorial_crd(two_factor,treatment_column="VRT",dose_column="LVL",
        replicate_column="REP",response_column="OVI58",control_level=None,expected_dose_series=[1,2,3,4])
    assert result["model_formula"] == "response ~ treatment * dose"
    assert [r["source"] for r in result["anova"]] == ["VRT","LVL","VRT:LVL","Error"]

def test_three_way_priority_is_first(maize):
    result=orchestrate_bioassay(maize,request([
      {"id":"v","column":"VRT"},{"id":"f","column":"FORM"},{"id":"l","column":"LVL"}]))
    meta=result["response_results"][0]["interpretation_metadata"]
    assert meta["interpretation_priority"] == "three_way_interaction"

@pytest.mark.parametrize("three_way,two_way,expected", [
    (0.20, 0.01, "two_way_interaction"),
    (0.20, 0.10, "main_effects"),
])
def test_three_factor_interpretation_priority_fallbacks(three_way, two_way, expected):
    rows = [
        {"source":"A × B", "p_value":two_way},
        {"source":"A × C", "p_value":0.50},
        {"source":"B × C", "p_value":0.50},
        {"source":"A × B × C", "p_value":three_way},
    ]
    priority, _, significant_two_way = _interpretation_priority(rows, 3, 0.05)
    assert priority == expected
    assert [row["source"] for row in significant_two_way] == (["A × B"] if two_way < 0.05 else [])

@pytest.mark.parametrize("kind",["regression","cotoxicity"])
def test_multifactor_pooling_is_blocked(maize,kind):
    req=request([{"id":"v","column":"VRT"},{"id":"f","column":"FORM"},
                 {"id":"l","column":"LVL","semantic_role":"dose"}],
                regression=kind=="regression",cotoxicity=kind=="cotoxicity")
    with pytest.raises(UnsupportedBioassayAnalysis,match="multifactor"):
        orchestrate_bioassay(maize,req)
