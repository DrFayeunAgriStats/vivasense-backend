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

    rep_column is optional for standard designs.  When absent the system
    assumes a Completely Randomised Design (CRD) and infers replication from
    repeated observations per genotype.

    For split-plot RCBD the dataset must provide rep_column, main_plot_column,
    and sub_plot_column, and design_type must be set to "split_plot_rcbd".
    """
    base64_content: str = Field(..., description="Base64-encoded CSV or Excel file")
    file_type: str = Field(..., pattern="^(csv|xlsx|xls)$")
    genotype_column: str
    rep_column: Optional[str] = Field(
        default=None,
        description=(
            "Replication/block column. Leave null for CRD datasets — "
            "replication will be inferred from repeated observations."
        ),
    )
    main_plot_column: Optional[str] = Field(
        default=None,
        description="Main plot factor column for split-plot RCBD designs.",
    )
    sub_plot_column: Optional[str] = Field(
        default=None,
        description="Subplot factor column for split-plot RCBD designs.",
    )
    environment_column: Optional[str] = None
    factor_column: Optional[str] = Field(
        default=None,
        description=(
            "Second treatment factor for factorial designs (single-env only). "
            "When provided with rep_column the analysis uses a factorial RCBD model "
            "(trait ~ rep + genotype * factor). "
            "When provided without rep_column a factorial CRD model is used."
        ),
    )
    design_type: str = Field(
        default="single",
        pattern="^(single|multi|split_plot_rcbd)$",
        description="Experimental design type for the dataset.",
    )
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
    design_type: str


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

class DescriptiveStats(BaseModel):
    """Numerical descriptive statistics computed from trait observations."""
    grand_mean: Optional[float] = None
    standard_deviation: Optional[float] = None
    variance: Optional[float] = None
    standard_error: Optional[float] = None
    cv_percent: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    range: Optional[float] = None


class GenotypeDescriptiveStats(BaseModel):
    """Per-genotype descriptive statistics for a single trait."""
    genotype: str
    mean: Optional[float] = None
    se: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    n_reps: Optional[int] = None


class SummaryStats(BaseModel):
    """Structured summary of key descriptive statistics for interpretation."""
    grand_mean: Optional[float] = None
    cv_percent: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    range: Optional[float] = None
    standard_error: Optional[float] = None


class AnovaTraitResult(BaseModel):
    """Per-trait result slice returned by the ANOVA module."""
    trait: str
    status: str                                      # "success" | "failed"
    grand_mean: Optional[float] = None
    n_genotypes: Optional[int] = None
    n_reps: Optional[int] = None
    n_environments: Optional[int] = None
    anova_table: Optional[AnovaTable] = None
    descriptive_stats: Optional[DescriptiveStats] = None
    per_genotype_stats: Optional[List[GenotypeDescriptiveStats]] = None
    summary: Optional[SummaryStats] = None
    precision_level: Optional[str] = None  # "good" | "moderate" | "low"
    cv_interpretation_flag: Optional[str] = None  # "cv_available" | "cv_unavailable"
    ranking_caution: Optional[bool] = None
    selection_feasible: Optional[bool] = None
    genotype_significant: Optional[bool] = None
    environment_significant: Optional[bool] = None
    gxe_significant: Optional[bool] = None
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
    descriptive_stats: Optional[DescriptiveStats] = None
    variance_components: Optional[Dict[str, Any]] = None
    heritability: Optional[Dict[str, Any]] = None
    gcv: Optional[float] = None
    pcv: Optional[float] = None
    ga: Optional[float] = None
    gam: Optional[float] = None
    breeding_implication: Optional[str] = None
    interpretation: Optional[str] = None
    # ANOVA significance flags — forwarded from the ANOVA table so the export
    # layer can apply conditional GCV/PCV commentary without re-running the model.
    environment_significant: Optional[bool] = None
    gxe_significant: Optional[bool] = None
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


# ============================================================================
# TRAIT ASSOCIATION MODULE
# ============================================================================

class TraitAssociationModuleRequest(ModuleRequest):
    """Request body for POST /genetics/trait-association/analyze."""
    analysis_unit: str = Field(default="genotype_mean", pattern="^(genotype_mean|plot_level)$")
    alpha: float = Field(default=0.05, ge=0.0, le=1.0)
    gxe_significant: bool = Field(default=False)
    environment_context: str = Field(default="single_environment", pattern="^(single_environment|multi_environment)$")


class SignificantPair(BaseModel):
    """Details of a significant trait pair.

    The v1 beta response uses safer labels for UI display.
    Legacy fields are kept temporarily for backward compatibility.
    Future versions will require a pairwise n_matrix for full confidence grading.
    """
    trait_1: str
    trait_2: str
    r: float
    p_value: float
    direction: str  # "positive" | "negative" | "none"
    strength: str   # "very weak" | "weak" | "moderate" | "strong" | "very strong"
    confidence_status: str  # "limited_by_pairwise_n" or future full confidence grades
    selection_signal: str  # "exploratory only" | "potentially useful with validation" | "useful with validation"
    # Legacy compatibility fields for older clients
    confidence: Optional[str] = Field(None, description="Legacy field, mirrors confidence_status")
    selection_relevance: Optional[str] = Field(None, description="Legacy field, mirrors selection_signal")


class StrongestPair(BaseModel):
    """Strongest positive or negative pair."""
    trait_1: str
    trait_2: str
    r: float


class TraitAssociationSummary(BaseModel):
    """Statistical summary of the analysis."""
    num_traits: int
    num_significant_pairs: int
    strongest_positive_pair_label: Optional[str] = None
    strongest_negative_pair_label: Optional[str] = None


class TraitAssociationHeatmap(BaseModel):
    """Heatmap-ready matrix data."""
    matrix: Dict[str, Dict[str, Optional[float]]]
    type: str = "correlation_heatmap_ready"


class InterpretationPlaceholder(BaseModel):
    """Placeholder for future LLM interpretation."""
    status: str = "pending"
    message: str = "LLM interpretation not yet attached"


class TraitAssociationModuleResponse(BaseModel):
    """Response from POST /genetics/trait-association/analyze."""
    module: str = "trait_association_intelligence"
    analysis_unit: str
    n_observations: int
    alpha: float
    environment_context: str
    gxe_significant: bool
    trait_names: List[str]
    correlation_matrix: Dict[str, Dict[str, Optional[float]]]
    pvalue_matrix: Dict[str, Dict[str, Optional[float]]]
    significant_pairs: List[SignificantPair]
    strongest_positive_pair: Optional[StrongestPair] = None
    strongest_negative_pair: Optional[StrongestPair] = None
    risk_flags: List[str] = Field(default_factory=list)
    summary: TraitAssociationSummary
    heatmap: TraitAssociationHeatmap
    interpretation: Optional[str] = None
    interpretation_placeholder: InterpretationPlaceholder = Field(default_factory=InterpretationPlaceholder)
    dataset_token: str
    warnings: List[str] = Field(default_factory=list)
