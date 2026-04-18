/**
 * VivaSense Genetics – Trait Relationships API Client
 * ====================================================
 * Calls POST /genetics/correlation.
 *
 * Uses the same VITE_GENETICS_ENGINE_BASE env variable as geneticsUploadApi.ts.
 */

import { API_BASE } from "./apiConfig";
const ENGINE_BASE: string = API_BASE;

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface CorrelationRequest {
  base64_content: string;
  file_type: "csv" | "xlsx" | "xls";
  genotype_column: string;
  rep_column: string;
  environment_column?: string;
  trait_columns: string[];
  mode: "single" | "multi";
  method?: "pearson" | "spearman";
  user_objective?: "Field understanding" | "Genotype comparison" | "Breeding decision";
}

/**
 * Mirrors CorrelationResponse in trait_relationships_schemas.py.
 * Three-mode: phenotypic, between_genotype, genotypic (VC-based, optional).
 */
export interface CorrelationStats {
  n_observations: number;
  df?: number;
  critical_r?: number;
  r_matrix: (number | null)[][];
  p_matrix: (number | null)[][];
  p_adj_matrix?: (number | null)[][];
  ci_lower_matrix?: (number | null)[][];
  ci_upper_matrix?: (number | null)[][];
  /** True for genotypic VC mode: p-values and CIs are Fisher-z approximations. */
  inference_approximate?: boolean;
  inference_note?: string | null;
}

export interface CorrelationResponse {
  trait_names: string[];
  method: string;
  /** Field-level co-variation (all observations). */
  phenotypic: CorrelationStats;
  /** Association among genotype means. NOT a true genetic correlation. */
  between_genotype: CorrelationStats;
  /** Variance-component genotypic correlation (bivariate REML). Null if unavailable. */
  genotypic: CorrelationStats | null;
  interpretation: string;
  warnings: string[];
  statistical_note: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// API FUNCTION
// ─────────────────────────────────────────────────────────────────────────────

export async function computeCorrelation(
  request: CorrelationRequest
): Promise<CorrelationResponse> {
  const correlationUrl = `${ENGINE_BASE}/genetics/correlation`;
  console.log("[traitRelationshipsApi] POST", correlationUrl);

  let response: Response;
  try {
    response = await fetch(correlationUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(
      `Network error during correlation analysis: ${msg}. ` +
        "Verify VITE_GENETICS_ENGINE_BASE is set correctly."
    );
  }

  if (!response.ok) {
    const detail = await _extractErrorDetail(response);
    throw new Error(`Correlation failed — ${detail}`);
  }

  return response.json() as Promise<CorrelationResponse>;
}

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────────────────

async function _extractErrorDetail(response: Response): Promise<string> {
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
