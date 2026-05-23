/**
 * Types for VivaSense Genetics Upload & Correlation workflows.
 */

// ── Shared dataset context ──────────────────────────────────────────────────

export interface DatasetContext {
  file: File;
  base64Content: string;
  fileType: "csv" | "xlsx" | "xls";
  genotypeColumn: string;
  repColumn: string;
  environmentColumn: string | null;
  availableTraitColumns: string[];
  mode: "single" | "multi";
  /** Server-issued token from /genetics/upload-preview — required for stateful endpoints like /analysis/descriptive-stats */
  datasetToken?: string | null;
}

// ── Upload preview ──────────────────────────────────────────────────────────

export interface DetectedColumn {
  column: string;
  confidence: "high" | "medium" | "low";
}

export interface UploadPreviewResponse {
  detected_columns: {
    genotype: DetectedColumn | null;
    rep: DetectedColumn | null;
    environment: DetectedColumn | null;
    traits: string[];
  };
  n_rows: number;
  n_columns: number;
  data_preview: Record<string, unknown>[];
  mode_suggestion: "single" | "multi";
  column_names: string[];
  warnings: string[];
  /** Server-issued token used by stateful analysis endpoints */
  dataset_token?: string;
}

// ── Analyze-upload ──────────────────────────────────────────────────────────

export type AnalysisModule = "anova" | "genetic_parameters" | "correlation" | "heatmap";

export type AnovaDesignType = "crd" | "rcbd" | "factorial" | "factorial_rcbd" | "split_plot_rcbd";

export interface AnalyzeUploadRequest {
  base64_content: string;
  file_type: string;
  genotype_column: string;
  rep_column: string;
  environment_column: string | null;
  trait_columns: string[];
  mode: "single" | "multi";
  random_environment: boolean;
  selection_intensity: number;
  module: AnalysisModule;
  /** ANOVA design hint — backend routes ANOVA computation by this */
  design_type?: AnovaDesignType;
  /** Treatment column for CRD/RCBD */
  treatment_column?: string;
  /** Factor A column for factorial / factorial_rcbd */
  factor_a_column?: string;
  /** Factor B column for factorial / factorial_rcbd */
  factor_b_column?: string;
  /** Main-plot factor column for split_plot_rcbd */
  main_plot_column?: string;
  /** Subplot factor column for split_plot_rcbd */
  sub_plot_column?: string;
}

export interface TraitSummaryRow {
  trait: string;
  grand_mean: number;
  h2: number;
  gcv: number;
  pcv: number;
  gam_percent: number;
  heritability_class: string;
  status: string;
  error: string | null;
}

export interface VarianceComponents {
  sigma2_genotype: number;
  sigma2_error: number;
  sigma2_ge: number | null;
  sigma2_phenotypic: number;
}

export interface TraitAnalysisResult {
  status: string;
  analysis_result: {
    status: string;
    mode: string;
    data_validation: { is_valid: boolean; warnings: string[] };
    variance_warnings: { is_valid: boolean; warnings: string[] };
    result: {
      environment_mode: string;
      n_genotypes: number;
      n_reps: number;
      n_environments: number | null;
      grand_mean: number;
      variance_components: VarianceComponents;
      heritability: {
        h2_broad_sense: number;
        interpretation_basis: string;
      };
      genetic_parameters: {
        GCV: number;
        PCV: number;
        GAM: number;
        GAM_percent: number;
        selection_intensity: number;
      };
      anova_table?: Record<string, unknown>;
      mean_separation?: Record<string, unknown>;
    };
    interpretation: string;
  };
  error: string | null;
  data_warnings: string[];
}

export interface AnalyzeUploadResponse {
  summary_table: TraitSummaryRow[];
  trait_results: Record<string, TraitAnalysisResult>;
  dataset_summary: {
    n_genotypes: number;
    n_reps: number;
    n_environments: number | null;
    n_traits: number;
    mode: string;
  };
  failed_traits: string[];
}

// ── Correlation ─────────────────────────────────────────────────────────────

export interface CorrelationRequest {
  base64_content: string;
  file_type: string;
  genotype_column: string;
  rep_column: string;
  environment_column: string | null;
  trait_columns: string[];
}

export type CorrelationModeKey = "phenotypic" | "between_genotype" | "genotypic";

export interface CorrelationModeBlock {
  // Matrix shape (legacy / multi-trait)
  r_matrix?: number[][];
  p_matrix?: (number | null)[][];
  n_observations?: number | null;
  available: boolean;
  note?: string | null;
  fallback_used?: boolean;
  // Scalar pairwise shape (new backend, 2-trait pair)
  r?: number;
  rg?: number;
  p_value?: number;
  df?: number;
  critical_r?: number;
  ci_lower?: number;
  ci_upper?: number;
  n_genotypes?: number;
}

export interface CorrelationResponse {
  trait_names: string[];
  method?: string;
  // Legacy single-matrix fields (kept for back-compat)
  r_matrix?: number[][];
  p_matrix?: (number | null)[][];
  n_observations?: number | null;
  warnings: string[];
  statistical_note: string;
  interpretation: string;
  // New 3-mode fields (optional — present when backend supports it)
  phenotypic?: CorrelationModeBlock;
  between_genotype?: CorrelationModeBlock;
  genotypic?: CorrelationModeBlock;
  modes_available?: CorrelationModeKey[];
}
