/**
 * Types for VivaSense Advanced Analysis modules:
 * /analysis/stability, /analysis/blup, /analysis/pca, /analysis/cluster
 */

// ── Stability (Eberhart-Russell + Shukla + AMMI + GGE) ─────────────────────

export type StabilityClass =
  | "stable"
  | "responsive_favorable"
  | "responsive_poor"
  | "unpredictable"
  | "responsive"
  | "highly stable"
  | "moderately stable"
  | "unstable"
  | string;

export type StabilityMethod = "eberhart-russell" | "shukla" | "ammi" | "gge-biplot";
export type GgeBiplotType = "which-won-where" | "mean-stability" | "discriminativeness";

export interface StabilityRow {
  genotype: string;
  mean: number;
  bi: number;
  s2di: number;
  shukla_variance?: number;
  rank: number;
  stability_class: StabilityClass;
}

// ── AMMI ───────────────────────────────────────────────────────────────────

export interface AmmiAnovaTable {
  source: string[];
  df: number[];
  sum_sq: number[];
  mean_sq: number[];
  f_value: number[];
  p_value: number[];
}

export interface GenotypeIPCA {
  genotype: string;
  mean: number;
  ipca1: number;
  ipca2: number;
  ipca3?: number;
}

export interface EnvironmentIPCA {
  environment: string;
  mean: number;
  ipca1: number;
  ipca2: number;
}

export interface GenotypeASV {
  genotype: string;
  mean?: number;
  asv: number;
  rank: number;
  stability_class: string;
}

export interface AmmiBiplotPoint {
  code?: string;
  genotype?: string;
  environment?: string;
  mean?: number;
  pc1?: number;
  pc2?: number;
  ipca1?: number;
  ipca2?: number;
}

export interface AmmiBiplotData {
  genotypes: AmmiBiplotPoint[];
  environments: AmmiBiplotPoint[];
}

export interface AMMIResults {
  anova_table: AmmiAnovaTable;
  variance_explained: number[];
  cumulative_variance: number[];
  genotype_scores: GenotypeIPCA[];
  environment_scores: EnvironmentIPCA[];
  stability_measure: GenotypeASV[];
  biplot_data: AmmiBiplotData;
  interpretation: string;
}

// ── GGE ────────────────────────────────────────────────────────────────────

export interface GenotypePC {
  genotype: string;
  mean: number;
  pc1: number;
  pc2: number;
}

export interface EnvironmentPC {
  environment: string;
  mean: number;
  pc1: number;
  pc2: number;
}

export interface MegaEnvironment {
  id: number;
  environments: string[];
  best_genotype: string;
  mean_yield: number;
}

export interface WhichWonWhere {
  mega_environments: MegaEnvironment[];
  winning_genotypes: Record<string, string>;
  interpretation: string;
}

export interface GenotypeDistance {
  genotype: string;
  distance_from_ideal: number;
  rank: number;
}

export interface MeanVsStability {
  ideal_genotype: string;
  ideal_coordinates: { pc1: number; pc2: number };
  genotype_distances: GenotypeDistance[];
  interpretation: string;
}

export interface GGEResults {
  variance_explained: number[];
  cumulative_variance: number;
  genotype_scores: GenotypePC[];
  environment_scores: EnvironmentPC[];
  which_won_where?: WhichWonWhere;
  mean_vs_stability?: MeanVsStability;
  biplot_data: { genotypes: GenotypePC[]; environments: EnvironmentPC[] };
  interpretation: string;
}

export interface StabilityResponse {
  status: "success" | "failed" | "error";
  trait: string;
  n_genotypes: number;
  n_environments: number;
  methods_computed?: string[];
  genotype_stability: StabilityRow[];
  environment_means: Record<string, number>;
  grand_mean: number;
  best_stable_genotypes: string[];
  ammi_results?: AMMIResults;
  gge_results?: GGEResults;
  interpretation: string;
  plot_data?: unknown;
}

export interface StabilityRequest {
  dataset_token: string;
  trait_column: string;
  methods?: StabilityMethod[];
  biplot_type?: GgeBiplotType;
  ammi_components?: number;
}

// ── BLUP ────────────────────────────────────────────────────────────────────

export interface BlupRow {
  genotype: string;
  blup: number;
  se: number;
  reliability: number;
  rank: number;
}

export interface BlupResponse {
  status: "success" | "failed";
  trait: string;
  model_type: "single-environment" | "multi-environment";
  genotype_blups: BlupRow[];
  best_genotypes: string[];
  variance_components: {
    genotype: number;
    residual: number;
    environment?: number;
  };
  interpretation: string;
}

export interface BlupRequest {
  dataset_token: string;
  trait_column: string;
  random_effects: string[];
  fixed_effects?: string[];
}

// ── PCA ─────────────────────────────────────────────────────────────────────

export interface PcaScore {
  genotype: string;
  scores: number[];
}

export interface PcaResponse {
  status: "success" | "failed";
  n_traits: number;
  n_genotypes: number;
  variance_explained: number[];
  cumulative_variance: number[];
  loadings: Record<string, number[]>;
  scores: PcaScore[];
  biplot_data?: {
    loadings: Record<string, number[]>;
    scores: PcaScore[];
  };
  interpretation: string;
}

export interface PcaRequest {
  dataset_token: string;
  trait_columns: string[];
  scale?: boolean;
  n_components?: number | null;
}

// ── Cluster ─────────────────────────────────────────────────────────────────

export interface ClusterAssignment {
  genotype: string;
  cluster_id: number;
  silhouette_score?: number;
}

export interface ClusterSummary {
  cluster_id: number;
  size: number;
  mean_per_trait: Record<string, number>;
}

export interface ClusterResponse {
  status: "success" | "failed";
  n_genotypes: number;
  n_traits: number;
  method: string;
  optimal_k: number;
  cluster_assignments: ClusterAssignment[];
  cluster_summary: ClusterSummary[];
  silhouette_scores: number[];
  dendrogram_data?: unknown;
  interpretation: string;
}

export interface ClusterRequest {
  dataset_token: string;
  trait_columns: string[];
  method?: "single" | "complete" | "average" | "ward";
  k?: number | null;
  scale?: boolean;
}

// ── Non-Parametric Tests ────────────────────────────────────────────────────

export type NonparametricTestType = "kruskal-wallis" | "friedman" | "dunn";

export interface NonparametricRequest {
  dataset_token: string;
  trait_column: string;
  group_column: string;
  test_type: NonparametricTestType;
  block_column?: string;
  alpha?: number;
}

export interface NonparametricGroupRow {
  group: string;
  n?: number;
  median?: number;
  mean_rank?: number;
}

export interface NonparametricPosthocRow {
  group_a: string;
  group_b: string;
  statistic?: number;
  p_value?: number;
  p_adjusted?: number;
  significant?: boolean;
}

export interface NonparametricResponse {
  status: "success" | "failed" | "error";
  test_type: string;
  trait?: string;
  group_column?: string;
  statistic?: number;
  p_value?: number;
  significant?: boolean;
  alpha?: number;
  df?: number;
  group_summary?: NonparametricGroupRow[];
  posthoc_results?: NonparametricPosthocRow[];
  assumptions_met?: Record<string, boolean | string>;
  interpretation?: string;
}

// ── MANOVA ──────────────────────────────────────────────────────────────────

export type ManovaTestStat = "Wilks" | "Pillai" | "Hotelling-Lawley" | "Roy";

export interface ManovaRequest {
  dataset_token: string;
  trait_columns: string[];
  factor_column: string;
  covariates?: string[];
  test_statistic?: ManovaTestStat;
  alpha?: number;
}

export interface ManovaTestRow {
  effect: string;
  test_statistic?: string;
  value?: number;
  f_statistic?: number;
  num_df?: number;
  den_df?: number;
  p_value?: number;
  significant?: boolean;
}

export interface ManovaUnivariateRow {
  trait: string;
  f_statistic?: number;
  p_value?: number;
  eta_squared?: number;
  significant?: boolean;
}

export interface ManovaCoordinate {
  label: string;
  group?: string;
  x: number;
  y: number;
}

export interface ManovaResponse {
  status: "success" | "failed" | "error";
  test_statistic?: string;
  statistic_value?: number;
  f_statistic?: number;
  p_value?: number;
  significant?: boolean;
  manova_table?: ManovaTestRow[];
  univariate_results?: ManovaUnivariateRow[];
  coordinates?: ManovaCoordinate[];
  assumptions_note?: string;
  interpretation?: string;
}

// ── Path Analysis ───────────────────────────────────────────────────────────

export type PathAnalysisMethod = "correlation" | "covariance";

export interface PathAnalysisRequest {
  dataset_token: string;
  outcome_trait: string;
  predictor_traits: string[];
  method?: PathAnalysisMethod;
  standardize?: boolean;
}

export interface PathCoefficientRow {
  predictor: string;
  direct_effect: number;
  std_error?: number;
  t_statistic?: number;
  p_value?: number;
  significant?: boolean;
}

export interface PathDecompositionRow {
  predictor: string;
  total_correlation?: number;
  direct_effect?: number;
  indirect_effect?: number;
  percent_direct?: number;
}

export interface PathDiagramNode { id: string; label?: string; type?: "predictor" | "outcome" | "residual"; }
export interface PathDiagramEdge { source: string; target: string; weight: number; significant?: boolean; }
export interface PathDiagramData { nodes?: PathDiagramNode[]; edges?: PathDiagramEdge[]; }

export interface PathAnalysisResponse {
  status: "success" | "failed" | "error";
  outcome_trait?: string;
  predictor_traits?: string[];
  n_observations?: number;
  r_squared?: number;
  residual_effect?: number;
  path_coefficients?: PathCoefficientRow[];
  correlation_decomposition?: PathDecompositionRow[];
  indirect_effects?: Record<string, Record<string, number>> | unknown;
  path_diagram_data?: PathDiagramData;
  interpretation?: string;
  // Legacy / alternate field names tolerated from earlier backend versions
  outcome?: string;
  n_predictors?: number;
  decomposition?: PathDecompositionRow[];
  [key: string]: unknown;
}

// ── Selection Index ─────────────────────────────────────────────────────────

export interface SelectionIndexRequest {
  dataset_token: string;
  trait_columns: string[];
  economic_weights: Record<string, number>;
  genetic_parameters?: Record<string, number>;
  genetic_correlations?: number[][];
  selection_intensity?: number;
}

export interface SelectionIndexWeight {
  trait: string;
  weight: number;
}

export interface SelectionRanking {
  genotype: string;
  index_value: number;
  rank: number;
  selected?: boolean;
}

export interface ExpectedGainRow {
  trait: string;
  expected_gain: number;
  percent_gain?: number;
}

export interface RelativeEfficiencyRow {
  trait: string;
  efficiency: number;
}

export interface SelectionIndexResponse {
  status: "success" | "failed" | "error";
  selection_accuracy?: number;
  total_merit?: number;
  n_selected?: number;
  top_genotype?: string;
  selection_intensity?: number;
  index_weights?: SelectionIndexWeight[];
  rankings?: SelectionRanking[];
  selected_genotypes?: string[];
  expected_gain?: ExpectedGainRow[];
  relative_efficiency?: RelativeEfficiencyRow[];
  interpretation?: string;
}
