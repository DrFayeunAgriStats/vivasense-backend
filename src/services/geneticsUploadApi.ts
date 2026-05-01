/**
 * VivaSense Genetics Upload API Client
 * =====================================
 * Calls the two multi-trait upload endpoints:
 *   POST /genetics/upload-preview   – file preview + column detection
 *   POST /genetics/analyze-upload   – trait-by-trait analysis
 *
 * In Lovable → Settings → Environment Variables, add:
 *   VITE_GENETICS_ENGINE_BASE = https://vivasense-genetics.onrender.com
 */

import { API_BASE } from "./apiConfig";
import { buildModeHeaders, guardProModule } from "./featureMode";
const ENGINE_BASE: string = API_BASE;

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface DetectedColumn {
  column: string;
  confidence: "high" | "medium" | "low";
}

export interface DetectedColumns {
  genotype: DetectedColumn | null;
  rep: DetectedColumn | null;
  environment: DetectedColumn | null;
  traits: string[];
}

export interface UploadPreviewResponse {
  detected_columns: DetectedColumns;
  n_rows: number;
  n_columns: number;
  data_preview: Record<string, unknown>[];
  mode_suggestion: "single" | "multi";
  column_names: string[];
  warnings: string[];
  /** Dataset token registered at preview time using auto-detected columns.
   *  Pass directly to /analysis/* endpoints. Superseded by a confirmed token
   *  from POST /upload/dataset once the user confirms column mapping. */
  dataset_token?: string | null;
}

export interface UploadAnalysisRequest {
  base64_content: string;
  file_type: "csv" | "xlsx" | "xls";
  genotype_column: string;
  rep_column: string;
  environment_column: string | null;
  trait_columns: string[];
  mode: "single" | "multi";
  random_environment?: boolean;
  selection_intensity: number;
  module?: "anova" | "genetic_parameters" | "correlation" | "heatmap";
  // Optional ANOVA-specific routing hints.
  design_type?: "crd" | "rcbd" | "factorial" | "factorial_rcbd" | "split_plot_rcbd";
  treatment_column?: string;
  factor_a_column?: string;
  factor_b_column?: string;
  main_plot_column?: string;
  sub_plot_column?: string;
}

export interface SummaryTableRow {
  trait: string;
  grand_mean?: number;
  h2?: number;
  gcv?: number;
  pcv?: number;
  gam_percent?: number;
  heritability_class?: "high" | "moderate" | "low";
  status: "success" | "failed";
  error?: string;
}

export interface DatasetSummary {
  n_genotypes: number;
  n_reps: number;
  n_environments?: number;
  n_traits: number;
  mode: string;
}

// ── Nested types that mirror GeneticsResult / GeneticsResponse in app_genetics.py ──

export interface AnovaTable {
  source: string[];
  df: number[];
  ss: (number | null)[];
  ms: (number | null)[];
  f_value: (number | null)[];
  p_value: (number | null)[];
}

export interface MeanSeparation {
  genotype: string[];
  mean: number[];
  se: (number | null)[];
  group: string[];
  test: string;
  alpha: number;
}

export interface GeneticsResult {
  environment_mode: string;
  n_genotypes: number;
  n_reps: number;
  n_environments: number | null;
  grand_mean: number;
  variance_components: Record<string, number | null>;
  heritability: {
    h2_broad_sense: number;
    interpretation_basis: string;
    formula?: string;
  };
  genetic_parameters: {
    GCV?: number;
    PCV?: number;
    GAM?: number;
    GAM_percent?: number;
    selection_intensity: number;
  };
  anova_table?: AnovaTable;
  mean_separation?: MeanSeparation;
}

export interface GeneticsResponse {
  status: string;
  mode: string;
  data_validation: Record<string, unknown>;
  variance_warnings: Record<string, unknown>;
  result: GeneticsResult | null;
  interpretation: string | null;
}

/** Matches TraitResult in multitrait_upload_schemas.py */
export interface TraitResult {
  status: "success" | "failed";
  analysis_result: GeneticsResponse | null; // null when status === "failed"
  error: string | null;
  data_warnings: string[];
}

export interface UploadAnalysisResponse {
  summary_table: SummaryTableRow[];
  trait_results: Record<string, TraitResult>;
  dataset_summary: DatasetSummary;
  failed_traits: string[];
}

// ─────────────────────────────────────────────────────────────────────────────
// API FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Upload a CSV/Excel file and get column detection + data preview.
 * No analysis is run — this is a fast pre-flight call.
 */
export async function previewUpload(file: File): Promise<UploadPreviewResponse> {
  const fd = new FormData();
  fd.append("file", file);

  const previewUrl = `${ENGINE_BASE}/genetics/upload-preview`;
  console.log("[geneticsUploadApi] POST", previewUrl);

  let response: Response;
  try {
    response = await fetch(previewUrl, {
      method: "POST",
      headers: buildModeHeaders(),
      body: fd,
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(
      `Network error reaching genetics engine: ${msg}. ` +
        "Verify VITE_GENETICS_ENGINE_BASE is set correctly."
    );
  }

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`Preview failed — ${detail}`);
  }

  return response.json() as Promise<UploadPreviewResponse>;
}

/**
 * Analyze all selected traits in a previously-uploaded file.
 * The file content is passed as base64 (avoids re-uploading).
 * One trait failing does not stop the others.
 */
export async function analyzeUpload(
  request: UploadAnalysisRequest
): Promise<UploadAnalysisResponse> {
  const selectedModule = request.module ?? "genetic_parameters";
  const hasEnvironmentFactor = (request.environment_column ?? "").trim().length > 0;
  const isCombinedAnova = selectedModule === "anova" && (request.mode === "multi" || hasEnvironmentFactor);

  if (selectedModule === "genetic_parameters") {
    guardProModule("genetic-parameters");
  } else if (isCombinedAnova) {
    guardProModule("combined-anova");
  }

  // Temporary debug log — remove after integration is confirmed working.
  console.log("[analyzeUpload] request fields:", {
    file_type: request.file_type,
    genotype_column: request.genotype_column,
    rep_column: request.rep_column,
    environment_column: request.environment_column,
    trait_columns: request.trait_columns,
    mode: request.mode,
    random_environment: request.random_environment,
    selection_intensity: request.selection_intensity,
    base64_content: request.base64_content
      ? `[base64, ${request.base64_content.length} chars]`
      : "(empty — file encoding failed)",
  });

  const analyzeUrl = `${ENGINE_BASE}/genetics/analyze-upload${request.module ? `?module=${encodeURIComponent(request.module)}` : ""}`;
  console.log("[geneticsUploadApi] POST", analyzeUrl);

  let response: Response;
  try {
    response = await fetch(analyzeUrl, {
      method: "POST",
      headers: buildModeHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Network error during analysis: ${msg}`);
  }

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`Analysis failed — ${detail}`);
  }

  const data = (await response.json()) as UploadAnalysisResponse;

  // Debug: log anova_table + mean_separation presence for each trait
  for (const [trait, tr] of Object.entries(data.trait_results ?? {})) {
    const result = tr.analysis_result?.result;
    console.log(`[analyzeUpload] trait="${trait}" status=${tr.status}`, {
      has_anova_table: result?.anova_table != null,
      anova_sources: result?.anova_table?.source,
      has_mean_separation: result?.mean_separation != null,
      mean_sep_genotypes: result?.mean_separation?.genotype,
      mean_sep_groups: result?.mean_separation?.group,
    });
  }

  return data;
}

/**
 * Generate and download a Word (.docx) report from completed analysis results.
 * Endpoint: POST /genetics/download-results  (alias: /genetics/export-word)
 *
 * The function triggers a browser file download directly — no return value.
 */
export async function exportWordReport(
  data: UploadAnalysisResponse,
  filename = "vivasense_genetics_report.docx"
): Promise<void> {
  guardProModule("export-word");
  // Normalise trait_results so every entry has the required "status" field.
  // Guards against state where status was dropped during result construction.
  const normalizedTraitResults: Record<string, TraitResult> = {};
  for (const [trait, tr] of Object.entries(data.trait_results)) {
    normalizedTraitResults[trait] = {
      status: tr.status ?? (tr.analysis_result != null ? "success" : "failed"),
      analysis_result: tr.analysis_result,
      error: tr.error,
      data_warnings: tr.data_warnings ?? [],
    };
  }
  const payload: UploadAnalysisResponse = {
    ...data,
    trait_results: normalizedTraitResults,
  };

  console.log("[exportWordReport] Download payload:", JSON.stringify(payload, null, 2));

  const exportUrl = `${ENGINE_BASE}/genetics/download-results`;
  console.log("[geneticsUploadApi] POST", exportUrl);

  let response: Response;
  try {
    response = await fetch(exportUrl, {
      method: "POST",
      headers: buildModeHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Network error during Word export: ${msg}`);
  }

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    const msg = typeof detail === "string" ? detail : JSON.stringify(detail);
    console.error("[exportWordReport] Server error", response.status, msg);
    throw new Error(`Word export failed (${response.status}) — ${msg}`);
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────

async function extractErrorDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    return JSON.stringify(body.detail ?? body);
  } catch {
    try {
      return await response.text();
    } catch {
      return `HTTP ${response.status} ${response.statusText}`;
    }
  }
}

/** Convert a File to base64 string (for analyzeUpload). */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // Strip the "data:...;base64," prefix
      resolve(result.split(",")[1]);
    };
    reader.onerror = () => reject(new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// DATASET CONTEXT
// Shared from the Upload File tab → Trait Relationships tab.
// MultiTraitUpload emits this once the user confirms column mapping.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Snapshot of a confirmed upload session.
 * Passed from MultiTraitUpload (via onDatasetReady) up to DataSourceTabs,
 * then down into TraitRelationships and DescriptiveStatsModule.
 */
export interface UploadDatasetContext {
  /** The original File object (kept for display purposes). */
  file: File;
  /** Pre-computed base64 string — avoids re-encoding when running correlation. */
  base64Content: string;
  fileType: "csv" | "xlsx" | "xls";
  genotypeColumn: string;
  repColumn: string;
  /** Defined only when mode === "multi". */
  environmentColumn?: string;
  /** All numeric columns detected in the file (not just the ones selected for heritability). */
  availableTraitColumns: string[];
  mode: "single" | "multi";
  /**
   * Token from POST /upload/dataset. Required by all /analysis/* module endpoints.
   * Null if dataset confirmation failed (module-based endpoints unavailable).
   */
  datasetToken: string | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// DATASET CONFIRMATION  (POST /upload/dataset)
// Step B in the module-based pipeline: register the confirmed dataset in the
// backend cache and receive a dataset_token for all /analysis/* endpoints.
// ─────────────────────────────────────────────────────────────────────────────

export interface ConfirmDatasetRequest {
  base64_content: string;
  file_type: "csv" | "xlsx" | "xls";
  genotype_column: string | null;
  rep_column: string | null;
  environment_column?: string | null;
  mode: "single" | "multi";
  design_type?: "single" | "multi";
  random_environment?: boolean;
  selection_intensity?: number;
}

export interface ConfirmDatasetResponse {
  dataset_token: string;
  n_genotypes: number | null;
  n_reps: number;
  n_environments: number | null;
  n_rows: number;
  column_names: string[];
  mode: string;
  design_type: string;
}

export async function confirmDataset(
  request: ConfirmDatasetRequest
): Promise<ConfirmDatasetResponse> {
  const url = `${ENGINE_BASE}/upload/dataset`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: buildModeHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Network error confirming dataset: ${msg}`);
  }
  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`Dataset confirmation failed — ${detail}`);
  }
  return response.json() as Promise<ConfirmDatasetResponse>;
}

// ─────────────────────────────────────────────────────────────────────────────
// DESCRIPTIVE STATISTICS  (POST /analysis/descriptive-stats)
// Requires a valid dataset_token from POST /upload/dataset.
// ─────────────────────────────────────────────────────────────────────────────

export interface TraitDescriptiveResult {
  trait: string;
  n: number;
  mean: number | null;
  minimum: number | null;
  maximum: number | null;
  sd: number | null;
  cv_percent: number | null;
  median: number | null;
  skewness: number | null;
  kurtosis: number | null;
  missing_count: number;
  zero_count: number;
  precision_class: string;
  flags: string[];
  interpretation: string;
}

export interface DescriptiveStatsResponse {
  dataset_token: string;
  overview: { n_traits: number; n_observations: number };
  summary_table: TraitDescriptiveResult[];
  reliable_traits: string[];
  caution_traits: string[];
  global_flags: string[];
  recommendation: string;
}

export async function runDescriptiveStats(request: {
  dataset_token: string;
  trait_columns: string[];
}): Promise<DescriptiveStatsResponse> {
  const url = `${ENGINE_BASE}/analysis/descriptive-stats`;
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: buildModeHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(request),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Network error running descriptive stats: ${msg}`);
  }
  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`Descriptive statistics failed — ${detail}`);
  }
  return response.json() as Promise<DescriptiveStatsResponse>;
}

/**
 * Export descriptive statistics results as a Word document.
 * Endpoint: POST /export/descriptive-stats-word
 *
 * Accepts two shapes:
 *   • DescriptiveStatsResponse — flat (from this module's handleExport)
 *   • Combined state object    — { response: DescriptiveStatsResponse, trait_columns, ... }
 *     as produced by some callers that store request + response together.
 *
 * Either way the payload sent to the backend is fully flat.
 * The nested "response" key is never forwarded.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function exportDescriptiveStats(currentData: DescriptiveStatsResponse | Record<string, any>): Promise<void> {
  guardProModule("export-word");
  const url = `${ENGINE_BASE}/export/descriptive-stats-word`;

  // Flatten: prefer fields from currentData.response if present, fall back to root.
  // This handles both the flat DescriptiveStatsResponse shape and the combined
  // { response: DescriptiveStatsResponse, trait_columns, genotype_column, ... } shape.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const r = (currentData as any).response;

  const payload = {
    dataset_token:    (currentData as any).dataset_token ?? null,
    overview:         r?.overview         ?? (currentData as any).overview         ?? null,
    summary_table:    r?.summary_table    ?? (currentData as any).summary_table    ?? [],
    reliable_traits:  r?.reliable_traits  ?? (currentData as any).reliable_traits  ?? [],
    caution_traits:   r?.caution_traits   ?? (currentData as any).caution_traits   ?? [],
    global_flags:     r?.global_flags     ?? (currentData as any).global_flags     ?? [],
    recommendation:   r?.recommendation   ?? (currentData as any).recommendation   ?? "",
    // extra context fields — backend ignores them, kept for traceability
    trait_columns:       (currentData as any).trait_columns        ?? [],
    genotype_column:     (currentData as any).genotype_column      ?? null,
    rep_column:          (currentData as any).rep_column           ?? null,
    expected_replication: (currentData as any).expected_replication ?? null,
  };

  console.log(
    "[exportDescriptiveStats] POST", url,
    "| payload keys:", Object.keys(payload),
    "| overview present:", !!payload.overview,
    "| summary_table rows:", payload.summary_table?.length ?? 0,
  );

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: buildModeHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Network error during descriptive stats export: ${msg}`);
  }

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    console.error("[exportDescriptiveStats] Server error", response.status, detail);
    throw new Error(`Descriptive stats export failed (${response.status}) — ${detail}`);
  }

  const blob = await response.blob();
  const blobUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = blobUrl;
  a.download = "vivasense_descriptive_stats_report.docx";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(blobUrl);
}

/** Infer file_type from File.name */
export function inferFileType(file: File): "csv" | "xlsx" | "xls" {
  const name = file.name.toLowerCase();
  if (name.endsWith(".csv")) return "csv";
  if (name.endsWith(".xls")) return "xls";
  return "xlsx";
}
