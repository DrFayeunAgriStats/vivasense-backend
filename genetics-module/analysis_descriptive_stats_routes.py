import base64
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException

from module_schemas import ModuleRequest, DescriptiveResponse, TraitDescriptiveResult
import dataset_cache
from multitrait_upload_routes import read_file

from analysis_utils import compute_descriptive_stats
from interpretation_descriptive import (
    classify_cv_precision,
    generate_trait_flags,
    generate_trait_interpretation,
    generate_global_recommendation
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])

@router.post("/analysis/descriptive-stats", response_model=DescriptiveResponse, summary="Compute comprehensive descriptive statistics")
async def analyze_descriptive_stats(request: ModuleRequest):
    if not request.dataset_token:
        raise HTTPException(
            status_code=400,
            detail=(
                "dataset_token is required. Upload a file via POST /genetics/upload-preview "
                "and use the dataset_token returned in that response."
            ),
        )

    ctx = dataset_cache.get_dataset(request.dataset_token)
    if not ctx:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found or session expired. Re-upload to get a fresh token.",
        )
    
    try:
        df = read_file(base64.b64decode(ctx["base64_content"]), ctx["file_type"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}")

    missing = [c for c in request.trait_columns if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Trait columns not found in dataset: {missing}")

    summary_table = []
    reliable_traits = []
    caution_traits = []
    all_flags = set()

    for trait in request.trait_columns:
        series = df[trait]
        stats = compute_descriptive_stats(series)
        
        cv = stats.get("cv_percent")
        precision_class = classify_cv_precision(cv)
        n_valid = len(pd.to_numeric(series, errors="coerce").dropna())
        flags = generate_trait_flags(stats, n_valid)
        interp = generate_trait_interpretation(trait, stats, precision_class)
        
        if cv is not None and cv < 20:
            reliable_traits.append(trait)
        elif cv is not None and cv >= 20:
            caution_traits.append(trait)
            
        all_flags.update(flags)

        res = TraitDescriptiveResult(
            trait=trait, n=n_valid, mean=stats.get("grand_mean"), minimum=stats.get("min"),
            maximum=stats.get("max"), sd=stats.get("standard_deviation"), cv_percent=cv,
            median=stats.get("median"), skewness=stats.get("skewness"), kurtosis=stats.get("kurtosis"),
            missing_count=stats.get("missing_count", 0), zero_count=stats.get("zero_count", 0),
            precision_class=precision_class, flags=flags, interpretation=interp
        )
        summary_table.append(res)

    recommendation = generate_global_recommendation(reliable_traits, caution_traits, list(all_flags))

    return DescriptiveResponse(
        dataset_token=request.dataset_token,
        overview={"n_traits": len(request.trait_columns), "n_observations": len(df)},
        summary_table=summary_table,
        reliable_traits=reliable_traits,
        caution_traits=caution_traits,
        global_flags=list(all_flags),
        recommendation=recommendation
    )