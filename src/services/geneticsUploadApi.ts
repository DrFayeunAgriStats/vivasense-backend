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

  const analyzeUrl = `${ENGINE_BASE}/genetics/analyze-upload`;
  console.log("[geneticsUploadApi] POST", analyzeUrl);

  let response: Response;
  try {
    response = await fetch(analyzeUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
  // Temporary debug log — remove after multi-env failure is resolved.
  console.log("[analyzeUpload] full response:", JSON.stringify(data, null, 2));
  return data;
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
 * then down into TraitRelationships so it can run correlation without
 * requiring a second file upload.
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
}

/** Infer file_type from File.name */
export function inferFileType(file: File): "csv" | "xlsx" | "xls" {
  const name = file.name.toLowerCase();
  if (name.endsWith(".csv")) return "csv";
  if (name.endsWith(".xls")) return "xls";
  return "xlsx";
}
