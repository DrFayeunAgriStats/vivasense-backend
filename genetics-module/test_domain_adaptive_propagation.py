import asyncio
import base64
import io

import pandas as pd

import app_genetics
import multitrait_upload_routes as routes
from academic_interpretation import detect_analysis_domain
from multitrait_upload_schemas import DatasetSummary, UploadAnalysisRequest, UploadAnalysisResponse


DR_M_COLUMNS = [
    "Rep",
    "Treatment",
    "1st_bloom",
    "2nd_bloom",
    "Sulfur_fungi",
    "Berry_set",
]


def test_dr_m_columns_detect_as_agronomy_for_anova():
    assert detect_analysis_domain(DR_M_COLUMNS, "anova") == "agronomy"


def test_dr_m_columns_detect_as_agronomy_for_genetic_parameters():
    assert detect_analysis_domain(DR_M_COLUMNS, "genetic_parameters") == "agronomy"


def test_plant_breeding_columns_remain_plant_breeding():
    columns = ["Rep", "Genotype", "Yield", "DTF"]
    assert detect_analysis_domain(columns, "genetic_parameters") == "plant_breeding"


def test_upload_analysis_response_defaults_to_general():
    response = UploadAnalysisResponse(
        summary_table=[],
        trait_results={},
        dataset_summary=DatasetSummary(n_reps=1, n_traits=0, mode="single"),
    )
    assert response.domain == "general"


def test_explicit_domain_override_wins_over_breeding_columns(monkeypatch):
    rows = []
    for rep in (1, 2, 3):
        for genotype, value in (("G1", 10 + rep), ("G2", 20 + rep)):
            rows.append({"Rep": rep, "Genotype": genotype, "Yield": value})
    frame = pd.DataFrame(rows)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)

    request = UploadAnalysisRequest(
        base64_content=base64.b64encode(buffer.getvalue().encode()).decode(),
        file_type="csv",
        genotype_column="Genotype",
        rep_column="Rep",
        trait_columns=["Yield"],
        mode="single",
        design_type="rcbd",
        module="anova",
        research_domain="agronomy",
    )
    monkeypatch.setattr(
        app_genetics,
        "r_engine",
        app_genetics.RGeneticsEngine("vivasense_genetics.R"),
    )
    response = asyncio.run(routes.analyze_upload(request))

    assert response.domain == "agronomy"
