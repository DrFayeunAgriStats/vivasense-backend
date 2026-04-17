"""
VivaSense – Shared upload endpoints for the module-based pipeline.

POST /upload/preview   — Parse uploaded file, detect columns, return a 5-row
                         preview.  Identical behaviour to /genetics/upload-preview;
                         provided here under the new /upload/* prefix so the
                         frontend can use a consistent base path.

POST /upload/dataset   — Accept base64 file + confirmed column mappings, parse
                         the file to validate it, store the context in
                         dataset_cache, and return a dataset_token.  That token
                         is then passed to every /analysis/* endpoint, allowing
                         each module to run independently without re-uploading
                         the file.

No genetics computation is performed here.  All analytics live in the
/analysis/* endpoints.
"""

import base64
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from multitrait_upload_routes import (
    detect_columns,
    read_file,
)
from multitrait_upload_schemas import (
    DetectedColumns,
    UploadPreviewResponse,
)
from module_schemas import UploadDatasetRequest, UploadDatasetResponse
import dataset_cache

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Upload"])


# ============================================================================
# POST /upload/preview
# ============================================================================

@router.post(
    "/upload/preview",
    response_model=UploadPreviewResponse,
    summary="Preview uploaded file and detect column types",
)
async def upload_preview_v2(file: UploadFile = File(...)):
    """
    Read a CSV or Excel file, auto-detect structural columns (genotype,
    replication, environment) and candidate trait columns, and return a
    5-row preview.

    Identical behaviour to POST /genetics/upload-preview.  Provided under
    the /upload/* prefix for the module-based frontend.
    """
    filename = file.filename or ""
    if filename.endswith(".csv"):
        file_type = "csv"
    elif filename.endswith((".xlsx", ".xls")):
        file_type = "xlsx"
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a .csv or .xlsx/.xls file.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        df = read_file(content, file_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detected = detect_columns(df)

    warnings: List[str] = []
    if detected.genotype is None:
        warnings.append("Could not detect genotype column — please specify manually")
    if detected.rep is None:
        warnings.append(
            "No replication column detected — CRD assumed "
            "(replication inferred from data)"
        )
    if not detected.traits:
        warnings.append(
            "No numeric trait columns detected. "
            "Verify that trait data is numeric and not labelled as an ID column."
        )

    mode_suggestion = "multi" if detected.environment is not None else "single"

    data_preview: List[Dict[str, Any]] = [
        {col: (None if pd.isna(val) else val) for col, val in row.items()}
        for row in df.head(5).to_dict(orient="records")
    ]

    return UploadPreviewResponse(
        detected_columns=detected,
        n_rows=len(df),
        n_columns=len(df.columns),
        data_preview=data_preview,
        mode_suggestion=mode_suggestion,
        column_names=list(df.columns),
        warnings=warnings,
    )


# ============================================================================
# POST /upload/dataset
# ============================================================================

@router.post(
    "/upload/dataset",
    response_model=UploadDatasetResponse,
    summary="Register confirmed dataset and receive a dataset token",
)
async def upload_dataset(request: UploadDatasetRequest):
    """
    Parse the uploaded file using the confirmed column mappings, validate the
    structure, store the context in dataset_cache, and return a dataset_token.

    The token is a short-lived in-memory handle (valid for the lifetime of the
    server process, max 50 entries in cache).  Pass it to any /analysis/*
    endpoint to run a module without re-uploading the file.

    No analysis is performed here.
    """
    # Decode
    try:
        file_bytes = base64.b64decode(request.base64_content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 content") from exc

    try:
        df = read_file(file_bytes, request.file_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Validate required columns exist (rep_column is optional — CRD allowed)
    required = [request.genotype_column]
    if request.rep_column:
        required.append(request.rep_column)
    if request.environment_column:
        required.append(request.environment_column)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Columns not found in file: {missing}",
        )

    # Gather dataset-level stats
    n_genotypes = int(df[request.genotype_column].nunique())
    if request.rep_column:
        n_reps = int(df[request.rep_column].nunique())
    else:
        # CRD: infer n_reps as maximum observations per genotype
        n_reps = int(df.groupby(request.genotype_column).size().max())
    n_envs: Optional[int] = (
        int(df[request.environment_column].nunique())
        if request.environment_column
        else None
    )

    # Store context
    token = dataset_cache.create_token()
    dataset_cache.put_dataset(token, {
        "base64_content":    request.base64_content,
        "file_type":         request.file_type,
        "genotype_column":   request.genotype_column,
        "rep_column":        request.rep_column,       # may be None for CRD
        "environment_column": request.environment_column,
        "mode":              request.mode,
        "random_environment": request.random_environment,
        "selection_intensity": request.selection_intensity,
    })

    logger.info(
        "upload/dataset: token=%s n_genotypes=%d n_reps=%d mode=%s",
        token, n_genotypes, n_reps, request.mode,
    )

    return UploadDatasetResponse(
        dataset_token=token,
        n_genotypes=n_genotypes,
        n_reps=n_reps,
        n_environments=n_envs,
        n_rows=len(df),
        column_names=list(df.columns),
        mode=request.mode,
    )
