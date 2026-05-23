/**
 * Types for VivaSense Descriptive Statistics module.
 * Mirrors backend POST /analysis/descriptive-stats response shape.
 */

export type PrecisionClass =
  | "Excellent"
  | "excellent"
  | "Good"
  | "good"
  | "Moderate"
  | "moderate"
  | "Poor"
  | "poor"
  | "High variability"
  | "high variability"
  | string;

export interface DescriptiveStatsRow {
  trait: string;
  n: number | null;
  mean: number | null;
  minimum: number | null;
  maximum: number | null;
  sd: number | null;
  cv_percent: number | null;
  median: number | null;
  skewness?: number | null;
  kurtosis?: number | null;
  missing_count: number | null;
  zero_count: number | null;
  precision_class: PrecisionClass | null;
  flags?: string[];
  interpretation?: string;
}

export interface DescriptiveStatsRequest {
  /** Preferred: dataset token returned by upload endpoint */
  dataset_token?: string | null;
  /** Fallback for legacy inline payloads */
  base64_content?: string;
  file_type?: string;
  trait_columns: string[];
  genotype_column?: string | null;
  rep_column?: string | null;
  expected_replication?: number | null;
}

export interface DescriptiveStatsOverview {
  n_traits?: number | null;
  n_observations?: number | null;
  [key: string]: unknown;
}

export interface DescriptiveStatsResponse {
  dataset_token?: string;
  /** New backend returns an object; older builds may return a string. */
  overview?: DescriptiveStatsOverview | string;
  summary_table: DescriptiveStatsRow[];
  reliable_traits?: string[];
  caution_traits?: string[];
  global_flags?: string[];
  recommendation?: string;
  // Tolerated extras
  warnings?: string[];
  [key: string]: unknown;
}
