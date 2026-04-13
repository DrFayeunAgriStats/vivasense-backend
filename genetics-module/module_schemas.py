"""
VivaSense – Pydantic schemas for the module-based analysis pipeline.

Covers:
  • Upload / dataset confirmation  (/upload/*)
  • Four analysis module responses  (/analysis/*)
  • Four export request bodies      (/export/*)

All analysis responses include dataset_token so the frontend can chain
calls without re-uploading the file.

Import hierarchy (leaf → root, no cycles):
  genetics_schemas  ←  module_schemas
  trait_relationships_schemas  ←  module_schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from genetics_schemas import AnovaTable, MeanSeparation


# ============================================================================
# UPLOAD / DATASET CONFIRMATION
# ============================================================================

class UploadDatasetRequest(BaseModel):
    """
    POST /upload/dataset — confirm column mapping and register a reusable
    dataset context.  Returns dataset_token used by all analysis modules.
    """
    base64_content: str = Field(..., description="Base64-encoded CSV or Excel file")
    file_type: str = Field(..., pattern="^(csv|xlsx|xls)$")
    genotype_column: str
    rep_column: str
    environment_column: Optional[str] = None
    mode: str = Field(default="single", pattern="^(single|multi)$")
    random_environment: bool = False
    selection_intensity: float = Field(default=2.06, ge=0.0)


class UploadDatasetResponse(BaseModel):
    """Returned by POST /upload/dataset."""
    dataset_token: str
    n_genotypes: int
    n_reps: int
    n_environments: Optional[int] = None
    n_rows: int
    column_names: List[str]
    mode: str


# ============================================================================
# SHARED MODULE REQUEST
# ============================================================================

class ModuleRequest(BaseModel):
    """Base request body for all /analysis/* endpoints."""
    dataset_token: str = Field(..., description="Token from POST /upload/dataset")
    trait_columns: List[str] = Field(..., min_length=1)


# ============================================================================
# ANOVA MODULE
# ============================================================================

class AnovaTraitResult(BaseModel):
    """Per-trait result slice returned by the ANOVA module."""
    trait: str
    status: str                                      # "success" | "failed"
    grand_mean: Optional[float] = None
    n_genotypes: Optional[int] = None
    n_reps: Optional[int] = None
    n_environments: Optional[int] = None
    anova_table: Optional[AnovaTable] = None
    descriptive_stats: Optional[Dict[str, Any]] = None
    assumption_tests: Optional[Dict[str, Any]] = None
    mean_separation: Optional[MeanSeparation] = None
    interpretation: Optional[str] = None
    data_warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class AnovaModuleResponse(BaseModel):
    """Response from POST /analysis/anova."""
    dataset_token: str
    mode: str
    trait_results: Dict[str, AnovaTraitResult]
    failed_traits: List[str] = Field(default_factory=list)


# ============================================================================
# GENETIC PARAMETERS MODULE
# ============================================================================

class GeneticParametersTraitResult(BaseModel):
    """Per-trait result slice returned by the Genetic Parameters module."""
    trait: str
    status: str
    grand_mean: Optional[float] = None
    variance_components: Optional[Dict[str, Any]] = None
    heritability: Optional[Dict[str, Any]] = None
    gcv: Optional[float] = None
    pcv: Optional[float] = None
    ga: Optional[float] = None
    gam: Optional[float] = None
    breeding_implication: Optional[str] = None
    interpretation: Optional[str] = None
    data_warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class GeneticParametersModuleResponse(BaseModel):
    """Response from POST /analysis/genetic-parameters."""
    dataset_token: str
    mode: str
    trait_results: Dict[str, GeneticParametersTraitResult]
    failed_traits: List[str] = Field(default_factory=list)


# ============================================================================
# CORRELATION MODULE
# ============================================================================

class CorrelationModuleResponse(BaseModel):
    """Response from POST /analysis/correlation."""
    dataset_token: str
    trait_names: List[str]
    r_matrix: List[List[Optional[float]]]
    p_matrix: List[List[Optional[float]]]
    n_observations: int
    method: str
    statistical_note: Optional[str] = None
    interpretation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


# ============================================================================
# HEATMAP MODULE
# ============================================================================

class HeatmapModuleResponse(BaseModel):
    """Response from POST /analysis/heatmap."""
    dataset_token: str
    matrix: List[List[Optional[float]]]
    labels: List[str]
    min_val: float
    max_val: float
    method: str
    interpretation: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)


# ============================================================================
# EXPORT REQUEST BODIES
# ============================================================================

class AnovaExportRequest(BaseModel):
    """POST /export/anova-word — accepts the AnovaModuleResponse directly."""
    dataset_token: Optional[str] = None
    mode: str = "single"
    trait_results: Dict[str, AnovaTraitResult]
    failed_traits: List[str] = Field(default_factory=list)


class GeneticParametersExportRequest(BaseModel):
    """POST /export/genetic-parameters-word."""
    dataset_token: Optional[str] = None
    mode: str = "single"
    trait_results: Dict[str, GeneticParametersTraitResult]
    failed_traits: List[str] = Field(default_factory=list)


class CorrelationExportRequest(CorrelationModuleResponse):
    """POST /export/correlation-word — extends CorrelationModuleResponse."""
    pass


class HeatmapExportRequest(HeatmapModuleResponse):
    """POST /export/heatmap-report — extends HeatmapModuleResponse."""
    pass
