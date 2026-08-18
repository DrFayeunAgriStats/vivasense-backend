"""HTTP boundary for Crop Protection Bioassay / Efficacy Analysis."""

import base64
import logging

from fastapi import APIRouter, HTTPException

from crop_protection_orchestration import UnsupportedBioassayAnalysis, orchestrate_bioassay
from crop_protection_schemas import BioassayAnalysisRequest, BioassayAnalysisResponse
from crop_protection_validation import BioassayValidationError
from multitrait_upload_routes import read_file


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/crop-protection", tags=["Crop Protection Analytics"])


def _resolve_request_columns(request: BioassayAnalysisRequest, mapping: dict[str, str]) -> None:
    def resolve(name: str | None) -> str | None:
        return mapping.get(name, name) if name else None

    design = request.design
    design.treatment_column = resolve(design.treatment_column)
    design.dose_column = resolve(design.dose_column)
    design.replicate_column = resolve(design.replicate_column)
    for response in request.responses:
        response.raw_column = resolve(response.raw_column)
        response.inference_column = resolve(response.inference_column)
        response.display_column = resolve(response.display_column)
        response.transformed_column = resolve(response.transformed_column)
        response.corrected_column = resolve(response.corrected_column)


@router.post("/bioassay/analyze", response_model=BioassayAnalysisResponse)
async def analyze_bioassay(request: BioassayAnalysisRequest):
    """Run the validated crop-protection factorial CRD workflow."""

    try:
        content = base64.b64decode(request.dataset.base64_content, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 dataset content.") from exc
    try:
        df, column_mapping = read_file(content, request.dataset.file_type)
        _resolve_request_columns(request, column_mapping)
        logger.info(
            "crop-protection bioassay: rows=%d responses=%s design=%s",
            len(df), [response.id for response in request.responses], request.design.design_type,
        )
        return BioassayAnalysisResponse(**orchestrate_bioassay(df, request))
    except UnsupportedBioassayAnalysis as exc:
        raise HTTPException(
            status_code=501,
            detail={"code": "not_implemented", "message": str(exc)},
        ) from exc
    except (BioassayValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Crop-protection bioassay analysis failed")
        raise HTTPException(status_code=500, detail="Bioassay analysis failed.") from exc
