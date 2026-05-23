/**
 * VivaSense Genetics Module — Types
 */

export type GeneticsMode = "auto" | "single_environment" | "multi_environment";

export interface GeneticsInputData {
  MSG: number;
  MSE: number;
  MSGE: number;
  replications: number;
  num_environments: number;
  trait_mean: number;
  selection_intensity: number;
}

export interface GeneticsRequestBody {
  mode: GeneticsMode;
  validate_only: boolean;
  data: GeneticsInputData;
}

export interface VarianceComponent {
  label: string;
  symbol: string;
  value: number;
  formula?: string;
}

export interface GeneticParameter {
  label: string;
  symbol: string;
  value: number;
  unit?: string;
  classification?: string;
}

export interface ClassificationEntry {
  parameter: string;
  value: number;
  category: string;
  reference?: string;
}

export interface GeneticsValidationResult {
  valid: boolean;
  errors?: string[];
  warnings?: string[];
}

export interface GeneticsAnalysisResult {
  computation_mode: string;
  estimation_basis: string;
  variance_components: VarianceComponent[];
  genetic_parameters: GeneticParameter[];
  classification_summary: ClassificationEntry[];
  interpretation_paragraph: string;
  breeding_implication: string;
  caution_note: string;
  anova_table?: Record<string, unknown>[] | Record<string, unknown>;
  mean_separation?: Record<string, unknown>[] | Record<string, unknown>;
  trait_results?: Record<string, unknown>;
  dataset_name?: string;
  [key: string]: unknown;
}
