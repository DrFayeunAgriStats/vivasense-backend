"""
VivaSense Genetics - Multi-Trait Upload Response Schemas
"""

from pydantic import BaseModel, Field, validator
from typing import Any, Dict, List, Optional

from genetics_schemas import GeneticsResponse


# ============================================================================
# COLUMN DETECTION
# ============================================================================

class DetectedColumn(BaseModel):
    """A structural column detected by name-pattern matching."""
    column: str
    confidence: str  # "high" | "medium" | "low"


class DetectedColumns(BaseModel):
    """Columns detected in an uploaded file."""
    genotype: Optional[DetectedColumn] = None
    rep: Optional[DetectedColumn] = None
    environment: Optional[DetectedColumn] = None
    traits: List[str] = Field(
        default_factory=list,
        description="Candidate trait columns (numeric, not recognised as structural)"
    )


# ============================================================================
# UPLOAD PREVIEW
# ============================================================================

class UploadPreviewResponse(BaseModel):
    """
    Response from POST /genetics/upload-preview.

    Returned before analysis so the user can confirm/correct the column
    mapping. No genetics computation occurs at this stage.
    """
    detected_columns: DetectedColumns
    n_rows: int
    n_columns: int
    # data_preview uses Dict[str, Any] (typing.Any, not the built-in any).
    # Values may be None where the original cell was NaN/empty.
    data_preview: List[Dict[str, Any]] = Field(
        description="First 5 rows of the file. NaN cells are serialised as null."
    )
    mode_suggestion: str = Field(
        description="'single' or 'multi', inferred from environment column presence"
    )
    column_names: List[str]
    warnings: List[str] = Field(default_factory=list)


# ============================================================================
# UPLOAD ANALYSIS REQUEST
# ============================================================================

class UploadAnalysisRequest(BaseModel):
    """
    Request body for POST /genetics/analyze-upload.

    The file is sent as base64 to avoid a second multipart upload after the
    preview step. The user confirms column mapping in the frontend before
    calling this endpoint.
    """
    base64_content: str = Field(..., description="Base64-encoded file content")
    file_type: str = Field(..., description="'csv', 'xlsx', or 'xls'", pattern="^(csv|xlsx|xls)$")
    genotype_column: str
    rep_column: Optional[str] = Field(
        default=None,
        description=(
            "Replication/block column. Omit for CRD datasets — "
            "replication will be inferred from repeated observations."
        ),
    )
    environment_column: Optional[str] = None
    trait_columns: List[str] = Field(..., min_length=1)
    mode: str = Field(..., pattern="^(single|multi)$")
    random_environment: bool = False
    selection_intensity: float = 1.4
    module: Optional[str] = Field(
        default=None,
        description="Analysis module: 'anova' | 'genetic_parameters' | 'correlation' | 'heatmap'. "
                    "Can also be sent as a URL query parameter. Body value takes priority.",
    )


# ============================================================================
# UPLOAD ANALYSIS RESPONSE
# ============================================================================

class TraitResult(BaseModel):
    """
    Outcome for a single trait — present for both successes and failures.

    When status == 'success': analysis_result holds the full GeneticsResponse
    (same schema as POST /genetics/analyze) and error is None.

    When status == 'failed': analysis_result is None and error describes why.
    """
    status: Optional[str] = None  # "success" | "failed" — inferred when absent
    analysis_result: Optional[GeneticsResponse] = None
    error: Optional[str] = None
    data_warnings: List[str] = Field(
        default_factory=list,
        description="Balance or structure warnings (unequal reps, incomplete G×E, etc.)"
    )

    @validator("status", always=True, pre=False)
    def infer_status(cls, v, values):
        if v is not None:
            return v
        # Infer from analysis_result when the field is missing from the payload
        return "success" if values.get("analysis_result") is not None else "failed"


class SummaryTableRow(BaseModel):
    """One row in the cross-trait summary table."""
    trait: str
    grand_mean: Optional[float] = None
    h2: Optional[float] = None
    gcv: Optional[float] = None
    pcv: Optional[float] = None
    gam_percent: Optional[float] = None
    heritability_class: Optional[str] = None  # "high" | "moderate" | "low"
    status: str  # "success" | "failed"
    error: Optional[str] = None


class DatasetSummary(BaseModel):
    """Overall dataset statistics (computed from the uploaded file, not per-trait)."""
    n_genotypes: int
    n_reps: int
    n_environments: Optional[int] = None
    n_traits: int
    mode: str


class UploadAnalysisResponse(BaseModel):
    """
    Response from POST /genetics/analyze-upload.

    Outer contract (summary_table, dataset_summary, failed_traits) is stable.
    trait_results values are typed as TraitResult — each wraps a full
    GeneticsResponse (analysis_result) or is None when the trait failed.

    export_token: opaque UUID the backend stores alongside the full analysis
    result in result_cache.  The frontend must echo this field back verbatim
    when calling POST /genetics/download-results so the export endpoint can
    recover analysis_result objects that the frontend did not serialise.
    """
    summary_table: List[SummaryTableRow]
    trait_results: Dict[str, TraitResult]
    dataset_summary: DatasetSummary
    failed_traits: List[str] = Field(default_factory=list)
    export_token: Optional[str] = Field(
        default=None,
        description="Cache token — pass back to /download-results for full export",
    )
