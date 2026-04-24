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
    genotype_column: Optional[str] = Field(
        default=None,
        description=(
            "Observation-unit column (e.g. variety, accession, line). "
            "Required for CRD, RCBD, factorial RCBD, and multi-environment designs. "
            "Optional for generic split_plot_rcbd — the design is defined entirely by "
            "rep_column, main_plot_column, and sub_plot_column."
        ),
    )
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
    n_genotypes: Optional[int] = None   # None for generic split_plot_rcbd designs
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
    dataset_token: Optional[str] = Field(
        default=None,
        description=(
            "Token from POST /upload/dataset (or returned by POST /genetics/upload-preview). "
            "When null the endpoint returns 400 with a user-friendly message instead of "
            "a 422 Pydantic validation error."
        ),
    )
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
    median: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    missing_count: int = 0
    zero_count: int = 0


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
    design_type: Optional[str] = None   # e.g. "split_plot_rcbd"; used by export to skip inapplicable sections


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

class CorrelationStats(BaseModel):
    n_observations: int
    df: Optional[int] = None
    critical_r: Optional[float] = None
    r_matrix: List[List[Optional[float]]]
    p_matrix: List[List[Optional[float]]]
    p_adj_matrix: List[List[Optional[float]]]
    ci_lower_matrix: List[List[Optional[float]]]
    ci_upper_matrix: List[List[Optional[float]]]
    inference_approximate: bool = False
    inference_note: Optional[str] = None

class CorrelationModuleRequest(ModuleRequest):
    """Extends ModuleRequest with correlation-specific options."""
    method: str = Field(default="pearson", pattern="^(pearson|spearman)$")
    user_objective: str = Field(default="Field understanding", pattern="^(Field understanding|Genotype comparison|Breeding decision)$")

class CorrelationModuleResponse(BaseModel):
    """Response from POST /analysis/correlation."""
    dataset_token: str
    trait_names: List[str]
    method: str
    phenotypic: CorrelationStats
    between_genotype: CorrelationStats
    genotypic: Optional[CorrelationStats] = None
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


# ============================================================================
# DESCRIPTIVE STATISTICS MODULE
# ============================================================================

class TraitDescriptiveResult(BaseModel):
    trait: str
    n: int
    mean: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    sd: Optional[float] = None
    cv_percent: Optional[float] = None
    median: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    missing_count: int
    zero_count: int
    precision_class: str
    flags: List[str]
    interpretation: str

class DescriptiveResponse(BaseModel):
    dataset_token: Optional[str] = None
    overview: Dict[str, Any]
    summary_table: List[TraitDescriptiveResult]
    reliable_traits: List[str]
    caution_traits: List[str]
    global_flags: List[str]
    recommendation: str


# ============================================================================
# STABILITY ANALYSIS MODULE
# ============================================================================

class GenotypeStability(BaseModel):
    """Per-genotype Eberhart-Russell stability metrics."""
    genotype: str
    mean: float
    bi: float          # Regression coefficient (b_i)
    s2di: float        # Deviation from regression (S²d_i)
    rank: int
    stability_class: str  # "stable" | "responsive_favorable" | "responsive_poor" | "unpredictable"


# ── AMMI sub-schemas ──────────────────────────────────────────────────────────

class GenotypeIPCA(BaseModel):
    """Genotype IPCA scores from AMMI analysis."""
    genotype: str
    mean: float
    ipca1: float
    ipca2: Optional[float] = None
    ipca3: Optional[float] = None


class EnvironmentIPCA(BaseModel):
    """Environment IPCA scores from AMMI analysis."""
    environment: str
    mean: float
    ipca1: float
    ipca2: Optional[float] = None


class GenotypeASV(BaseModel):
    """AMMI Stability Value (Purchase et al. 2000). Lower ASV = more stable."""
    genotype: str
    asv: float
    rank: int
    stability_class: str   # "highly stable" | "stable" | "moderately stable" | "unstable"


class AMMIBiplotData(BaseModel):
    """Coordinates for AMMI biplot rendering."""
    genotypes: List[Dict[str, Any]]
    environments: List[Dict[str, Any]]


class AMMIResults(BaseModel):
    """Full AMMI analysis results."""
    variance_explained: List[float]       # % variance explained by each IPCA
    cumulative_variance: List[float]
    genotype_scores: List[GenotypeIPCA]
    environment_scores: List[EnvironmentIPCA]
    stability_measure: List[GenotypeASV]  # ASV per genotype
    biplot_data: AMMIBiplotData
    interpretation: str


# ── GGE Biplot sub-schemas ────────────────────────────────────────────────────

class GenotypePC(BaseModel):
    """Genotype principal-component scores from GGE biplot."""
    genotype: str
    mean: float
    pc1: float
    pc2: float


class EnvironmentPC(BaseModel):
    """Environment principal-component scores from GGE biplot."""
    environment: str
    mean: float
    pc1: float
    pc2: float


class MegaEnvironment(BaseModel):
    """A sector (mega-environment) in the which-won-where pattern."""
    id: int
    environments: List[str]
    best_genotype: str
    mean_yield: float


class WhichWonWhere(BaseModel):
    """Which-Won-Where pattern: mega-environment delineation."""
    mega_environments: List[MegaEnvironment]
    winning_genotypes: Dict[str, str]   # environment -> winning genotype
    interpretation: str


class GenotypeDistance(BaseModel):
    """Distance of a genotype from the ideal point in GGE Mean-vs-Stability view."""
    genotype: str
    distance_from_ideal: float
    rank: int


class MeanVsStability(BaseModel):
    """Mean performance vs stability from GGE biplot AEC view."""
    ideal_genotype: str
    ideal_coordinates: Dict[str, float]  # {"pc1": ..., "pc2": ...}
    genotype_distances: List[GenotypeDistance]
    interpretation: str


class GGEBiplotData(BaseModel):
    """Coordinates for GGE biplot rendering."""
    genotypes: List[Dict[str, Any]]
    environments: List[Dict[str, Any]]


class GGEResults(BaseModel):
    """Full GGE Biplot analysis results."""
    variance_explained: List[float]      # % variance by PC1, PC2
    cumulative_variance: float           # PC1 + PC2
    genotype_scores: List[GenotypePC]
    environment_scores: List[EnvironmentPC]
    which_won_where: Optional[WhichWonWhere] = None
    mean_vs_stability: Optional[MeanVsStability] = None
    biplot_data: GGEBiplotData
    interpretation: str


# ── Request / Response ────────────────────────────────────────────────────────

class StabilityRequest(BaseModel):
    """Request body for POST /analysis/stability."""
    dataset_token: str = Field(..., description="Token from POST /upload/dataset")
    trait_column: str = Field(..., description="Single trait column for stability analysis")
    methods: List[str] = Field(
        default=["eberhart-russell", "shukla"],
        description="Methods to compute: eberhart-russell, shukla, ammi, gge-biplot",
    )
    biplot_type: Optional[str] = Field(
        default="which-won-where",
        description="GGE biplot view: which-won-where | mean-stability | discriminativeness",
    )
    ammi_components: Optional[int] = Field(
        default=2,
        description="Number of IPCA axes to compute for AMMI",
    )


class StabilityResponse(BaseModel):
    """Response from POST /analysis/stability."""
    status: str
    trait: str
    methods_computed: List[str]
    n_genotypes: int
    n_environments: int
    genotype_stability: List[GenotypeStability]
    environment_means: Dict[str, float]
    grand_mean: float
    best_stable_genotypes: List[str]
    ammi_results: Optional[AMMIResults] = None
    gge_results: Optional[GGEResults] = None
    interpretation: str
    plot_data: Optional[Dict[str, Any]] = None


# ============================================================================
# BLUP MODULE
# ============================================================================

class GenotypeBLUP(BaseModel):
    """Per-genotype BLUP prediction."""
    genotype: str
    blup: float
    se: float
    reliability: float
    rank: int


class BLUPRequest(BaseModel):
    """Request body for POST /analysis/blup."""
    dataset_token: str = Field(..., description="Token from POST /upload/dataset")
    trait_column: str = Field(..., description="Trait column for BLUP analysis")
    random_effects: List[str] = Field(default=["genotype"], description="Columns to treat as random effects")
    fixed_effects: List[str] = Field(default=[], description="Columns to treat as fixed effects")


class BLUPResponse(BaseModel):
    """Response from POST /analysis/blup."""
    status: str
    trait: str
    model_type: str   # "single-environment" | "multi-environment"
    genotype_blups: List[GenotypeBLUP]
    best_genotypes: List[str]
    variance_components: Dict[str, float]
    interpretation: str


# ============================================================================
# PCA MODULE
# ============================================================================

class GenotypeScore(BaseModel):
    """Genotype coordinates on principal component axes."""
    genotype: str
    scores: List[float]


class BiplotData(BaseModel):
    """Data for rendering a PCA biplot."""
    loadings: Dict[str, List[float]]   # trait -> [PC1_loading, PC2_loading, ...]
    scores: List[GenotypeScore]


class PCARequest(BaseModel):
    """Request body for POST /analysis/pca."""
    dataset_token: str = Field(..., description="Token from POST /upload/dataset")
    trait_columns: List[str] = Field(..., min_length=2, description="≥2 trait columns for PCA")
    scale: bool = Field(default=True, description="Standardise traits before PCA")
    n_components: Optional[int] = Field(default=None, description="Number of PCs (default: all)")


class PCAResponse(BaseModel):
    """Response from POST /analysis/pca."""
    status: str
    n_traits: int
    n_genotypes: int
    variance_explained: List[float]
    cumulative_variance: List[float]
    loadings: Dict[str, List[float]]   # trait -> [PC1, PC2, ...] loadings
    scores: List[GenotypeScore]
    biplot_data: BiplotData
    interpretation: str


# ============================================================================
# CLUSTER ANALYSIS MODULE
# ============================================================================

class GenotypeCluster(BaseModel):
    """Cluster assignment for a single genotype."""
    genotype: str
    cluster_id: int
    silhouette_score: Optional[float] = None


class ClusterSummary(BaseModel):
    """Summary statistics for one cluster."""
    cluster_id: int
    size: int
    mean_per_trait: Dict[str, float]


class ClusterRequest(BaseModel):
    """Request body for POST /analysis/cluster."""
    dataset_token: str = Field(..., description="Token from POST /upload/dataset")
    trait_columns: List[str] = Field(..., min_length=2, description="≥2 trait columns for clustering")
    method: str = Field(default="ward", description="Linkage method: ward, complete, average, single")
    k: Optional[int] = Field(default=None, description="Number of clusters (None = auto-detect)")
    scale: bool = Field(default=True, description="Standardise traits before clustering")


class ClusterResponse(BaseModel):
    """Response from POST /analysis/cluster."""
    status: str
    n_genotypes: int
    n_traits: int
    method: str
    optimal_k: int
    cluster_assignments: List[GenotypeCluster]
    cluster_summary: List[ClusterSummary]
    silhouette_scores: List[float]
    dendrogram_data: Optional[Dict[str, Any]] = None
    interpretation: str


# ============================================================================
# NON-PARAMETRIC TESTS MODULE
# ============================================================================

class GroupMedian(BaseModel):
    """Summary statistics for a single group in non-parametric tests."""
    group_name: str
    median: float
    n: int
    rank_sum: Optional[float] = None
    mean_rank: Optional[float] = None


class PairwiseComparison(BaseModel):
    """Pairwise post-hoc comparison result (Dunn's test)."""
    group1: str
    group2: str
    p_value: float
    significant: bool
    adjustment: str


class NonparametricRequest(BaseModel):
    """Request body for POST /analysis/nonparametric."""
    dataset_token: str = Field(..., description="Token from POST /upload/dataset")
    trait_column: str = Field(..., description="Numeric trait column to test")
    group_column: str = Field(..., description="Factor column (e.g. genotype, treatment)")
    test_type: str = Field(
        default="kruskal-wallis",
        description="Test to run: kruskal-wallis | friedman | dunn",
    )
    block_column: Optional[str] = Field(
        default=None,
        description="Block/rep column for Friedman test (repeated measures)",
    )
    alpha: float = Field(default=0.05, ge=0.0, le=1.0)


class NonparametricResponse(BaseModel):
    """Response from POST /analysis/nonparametric."""
    status: str
    test_type: str
    trait: str
    group_column: str
    n_groups: int
    n_observations: int
    statistic: float
    statistic_name: str
    p_value: float
    df: int
    significant: bool
    group_medians: List[GroupMedian]
    posthoc_results: Optional[List[PairwiseComparison]] = None
    interpretation: str
    assumptions_met: Dict[str, Any]


# ============================================================================
# MANOVA MODULE
# ============================================================================

class UnivariateResult(BaseModel):
    """Per-trait univariate ANOVA result from MANOVA follow-up."""
    trait: str
    f_statistic: float
    p_value: float
    significant: bool
    eta_squared: Optional[float] = None


class MANOVARequest(BaseModel):
    """Request body for POST /analysis/manova."""
    dataset_token: str = Field(..., description="Token from POST /upload/dataset")
    trait_columns: List[str] = Field(..., min_length=2, description="≥2 trait columns for MANOVA")
    factor_column: str = Field(..., description="Independent variable column (e.g. genotype)")
    covariates: List[str] = Field(default=[], description="Optional covariate columns")
    test_statistic: str = Field(
        default="Wilks",
        description="MANOVA test statistic: Wilks | Pillai | Hotelling-Lawley | Roy",
    )
    alpha: float = Field(default=0.05, ge=0.0, le=1.0)


class MANOVAResponse(BaseModel):
    """Response from POST /analysis/manova."""
    status: str
    n_traits: int
    n_groups: int
    n_observations: int
    traits: List[str]
    factor: str
    test_statistic_name: str
    test_statistic_value: float
    f_statistic: float
    df_hypothesis: int
    df_error: int
    p_value: float
    significant: bool
    univariate_results: List[UnivariateResult]
    effect_sizes: Dict[str, float]
    interpretation: str
    assumptions_note: str


# ============================================================================
# PATH ANALYSIS MODULE
# ============================================================================

class PathCoefficient(BaseModel):
    """Direct effect of a predictor on the outcome trait."""
    predictor: str
    direct_effect: float
    std_error: float
    t_statistic: float
    p_value: float
    significant: bool


class CorrelationDecomp(BaseModel):
    """Decomposition of trait correlation into direct and indirect effects."""
    predictor: str
    total_correlation: float
    direct_effect: float
    indirect_effect: float
    percent_direct: float


class PathAnalysisRequest(BaseModel):
    """Request body for POST /analysis/path-analysis."""
    dataset_token: str = Field(..., description="Token from POST /upload/dataset")
    # Primary field names
    outcome_trait: Optional[str] = Field(default=None, description="Dependent variable (e.g. yield)")
    predictor_traits: Optional[List[str]] = Field(default=None, description="Independent variable columns")
    # Alias field names accepted from older frontend versions
    target_trait: Optional[str] = Field(default=None, description="Alias for outcome_trait")
    trait_columns: Optional[List[str]] = Field(default=None, description="Alias for predictor_traits")
    method: str = Field(
        default="correlation",
        description="Analysis method: correlation (phenotypic) | covariance",
    )
    standardize: bool = Field(default=True, description="Use standardised path coefficients")

    @property
    def resolved_outcome_trait(self) -> str:
        val = self.outcome_trait or self.target_trait
        if not val:
            raise ValueError("outcome_trait (or target_trait) is required")
        return val

    @property
    def resolved_predictor_traits(self) -> List[str]:
        val = self.predictor_traits or self.trait_columns
        if not val:
            raise ValueError("predictor_traits (or trait_columns) is required")
        return val


class PathAnalysisResponse(BaseModel):
    """Response from POST /analysis/path-analysis."""
    status: str
    outcome_trait: str
    predictor_traits: List[str]
    n_observations: int
    path_coefficients: List[PathCoefficient]
    correlation_decomposition: List[CorrelationDecomp]
    r_squared: float
    residual_effect: float
    indirect_effects: Dict[str, Dict[str, float]]
    interpretation: str
    path_diagram_data: Dict[str, Any]


# ============================================================================
# SELECTION INDEX MODULE
# ============================================================================

class GenotypeIndex(BaseModel):
    """Per-genotype selection index score and rank."""
    genotype: str
    index_value: float
    rank: int


class SelectionIndexRequest(BaseModel):
    """Request body for POST /analysis/selection-index."""
    dataset_token: str = Field(..., description="Token from POST /upload/dataset")
    trait_columns: List[str] = Field(..., min_length=2, description="≥2 trait columns for index")
    economic_weights: Dict[str, float] = Field(
        ..., description="Relative economic weight per trait (trait -> weight)"
    )
    genetic_parameters: Dict[str, Dict[str, float]] = Field(
        default={},
        description="Optional heritability (h2) and genetic variances per trait",
    )
    genetic_correlations: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Optional genetic correlation matrix between traits",
    )
    selection_intensity: float = Field(
        default=1.755,
        description="Selection intensity i (top 10% = 1.755, top 5% = 2.063)",
    )


class SelectionIndexResponse(BaseModel):
    """Response from POST /analysis/selection-index."""
    status: str
    traits: List[str]
    n_genotypes: int
    index_weights: Dict[str, float]
    genotype_scores: List[GenotypeIndex]
    expected_gain: Dict[str, float]
    total_merit: float
    selection_accuracy: float
    relative_efficiency: Dict[str, float]
    selected_genotypes: List[str]
    interpretation: str
