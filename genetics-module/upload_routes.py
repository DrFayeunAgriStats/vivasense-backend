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
    suggest_experimental_design,
)
from multitrait_upload_schemas import (
    DetectedColumns,
    UploadPreviewResponse,
)
from module_schemas import UploadDatasetRequest, UploadDatasetResponse
import dataset_cache  # noqa: E402 — already imported above for upload/dataset

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

    suggested_design = suggest_experimental_design(list(df.columns))
    mode_suggestion = "multi" if detected.environment is not None else "single"

    data_preview: List[Dict[str, Any]] = [
        {col: (None if pd.isna(val) else val) for col, val in row.items()}
        for row in df.head(5).to_dict(orient="records")
    ]

    # Register dataset in cache using auto-detected defaults so the frontend
    # receives a usable dataset_token from the preview response — no separate
    # POST /upload/dataset call required before hitting /analysis/* endpoints.
    preview_token: Optional[str] = None
    try:
        b64 = base64.b64encode(content).decode("ascii")
        preview_token = dataset_cache.create_token()
        dataset_cache.put_dataset(preview_token, {
            "base64_content":     b64,
            "file_type":          file_type,
            "genotype_column":    detected.genotype.column if detected.genotype else None,
            "rep_column":         detected.rep.column if detected.rep else None,
            "environment_column": detected.environment.column if detected.environment else None,
            "factor_column":      None,
            "main_plot_column":   None,
            "sub_plot_column":    None,
            "mode":               mode_suggestion,
            "design_type":        mode_suggestion,
            "random_environment": False,
            "selection_intensity": 2.06,
        })
        logger.info("upload/preview: auto-registered dataset token %s", preview_token)
    except Exception as exc:
        logger.warning("upload/preview: failed to register dataset token — %s", exc)
        preview_token = None

    return UploadPreviewResponse(
        detected_columns=detected,
        n_rows=len(df),
        n_columns=len(df.columns),
        data_preview=data_preview,
        suggested_design=suggested_design,
        mode_suggestion=mode_suggestion,
        column_names=list(df.columns),
        warnings=warnings,
        dataset_token=preview_token,
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

    # Validate column mapping selections for the chosen design type.
    # Build the list of mapped structural columns (only those provided).
    mapped_columns: List[str] = []
    if request.genotype_column:
        mapped_columns.append(request.genotype_column)
    if request.rep_column:
        mapped_columns.append(request.rep_column)
    if request.environment_column:
        mapped_columns.append(request.environment_column)
    if request.factor_column:
        mapped_columns.append(request.factor_column)
    if request.main_plot_column:
        mapped_columns.append(request.main_plot_column)
    if request.sub_plot_column:
        mapped_columns.append(request.sub_plot_column)

    if len(set(mapped_columns)) != len(mapped_columns):
        raise HTTPException(
            status_code=400,
            detail="Duplicate column mappings are not allowed.",
        )

    if request.design_type == "split_plot_rcbd":
        # ── Generic split-plot RCBD validation ────────────────────────────────
        # The design is defined entirely by rep / main_plot / sub_plot roles.
        # genotype_column is NOT required. If supplied, it must not duplicate a
        # structural role column and must not be treated as a third treatment
        # factor in the model formula.
        if request.mode != "single":
            raise HTTPException(
                status_code=400,
                detail="split_plot_rcbd design must use single-environment mode.",
            )
        if request.environment_column is not None:
            raise HTTPException(
                status_code=400,
                detail="split_plot_rcbd design does not accept environment_column.",
            )
        if request.factor_column is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "split_plot_rcbd design does not accept factor_column. "
                    "The two treatment factors are specified via main_plot_column "
                    "and sub_plot_column."
                ),
            )
        if not request.rep_column or not request.main_plot_column or not request.sub_plot_column:
            raise HTTPException(
                status_code=400,
                detail=(
                    "split_plot_rcbd design requires rep_column, main_plot_column, "
                    "and sub_plot_column."
                ),
            )
        if request.main_plot_column == request.sub_plot_column:
            raise HTTPException(
                status_code=400,
                detail=(
                    "main_plot_column and sub_plot_column must be different columns. "
                    f"Both are currently set to '{request.main_plot_column}'."
                ),
            )
        if request.rep_column in (request.main_plot_column, request.sub_plot_column):
            raise HTTPException(
                status_code=400,
                detail=(
                    "rep_column must be distinct from main_plot_column and sub_plot_column."
                ),
            )
        # If genotype_column is supplied for split_plot_rcbd, validate its role.
        if request.genotype_column:
            if request.genotype_column in (
                request.main_plot_column, request.sub_plot_column
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"genotype_column '{request.genotype_column}' duplicates a "
                        "treatment factor column. Map genotypes as main_plot_column "
                        "(whole-plot factor) or sub_plot_column (subplot factor), "
                        "not as a separate additional term."
                    ),
                )
            # genotype_column is a distinct third column — not valid in the
            # standard two-factor split-plot model.
            raise HTTPException(
                status_code=400,
                detail=(
                    f"genotype_column '{request.genotype_column}' is a third "
                    "independent column separate from main_plot_column and "
                    "sub_plot_column. The generic split_plot_rcbd module supports "
                    "exactly two treatment factors (main_plot × sub_plot). "
                    "If genotype is one of your treatment factors, assign it as "
                    "main_plot_column or sub_plot_column."
                ),
            )

        required = [
            request.rep_column,
            request.main_plot_column,
            request.sub_plot_column,
        ]
    else:
        # All other designs require genotype_column
        if not request.genotype_column:
            raise HTTPException(
                status_code=400,
                detail=(
                    "genotype_column is required for this design type. "
                    "It may be omitted only for split_plot_rcbd."
                ),
            )
        required = [request.genotype_column]
        if request.rep_column:
            required.append(request.rep_column)
        if request.environment_column:
            required.append(request.environment_column)
        if request.factor_column:
            required.append(request.factor_column)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Columns not found in file: {missing}",
        )

    # Gather dataset-level stats
    if request.design_type == "split_plot_rcbd":
        # No genotype column for generic split_plot_rcbd
        n_genotypes: Optional[int] = None
        n_reps = int(df[request.rep_column].nunique())
    elif request.genotype_column:
        n_genotypes = int(df[request.genotype_column].nunique())
        if request.rep_column:
            n_reps = int(df[request.rep_column].nunique())
        else:
            # CRD: infer n_reps as maximum observations per genotype
            n_reps = int(df.groupby(request.genotype_column).size().max())
    else:
        n_genotypes = None
        n_reps = int(df[request.rep_column].nunique()) if request.rep_column else 1
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
        "main_plot_column":  request.main_plot_column,
        "sub_plot_column":   request.sub_plot_column,
        "environment_column": request.environment_column,
        "factor_column":     request.factor_column,    # may be None; factorial RCBD/CRD
        "mode":              request.mode,
        "design_type":       request.design_type,
        "random_environment": request.random_environment,
        "selection_intensity": request.selection_intensity,
    })

    logger.info(
        "upload/dataset: token=%s n_genotypes=%s n_reps=%d mode=%s design=%s",
        token, n_genotypes, n_reps, request.mode, request.design_type,
    )

    return UploadDatasetResponse(
        dataset_token=token,
        n_genotypes=n_genotypes,   # None for generic split_plot_rcbd
        n_reps=n_reps,
        n_environments=n_envs,
        n_rows=len(df),
        column_names=list(df.columns),
        mode=request.mode,
        design_type=request.design_type,
    )
