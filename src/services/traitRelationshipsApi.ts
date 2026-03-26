/**
 * VivaSense Genetics – Trait Relationships API Client
 * ====================================================
 * Calls POST /genetics/correlation.
 *
 * Uses the same VITE_GENETICS_ENGINE_BASE env variable as geneticsUploadApi.ts.
 */

const ENGINE_BASE: string =
  import.meta.env.VITE_GENETICS_ENGINE_BASE ||
  import.meta.env.VITE_GENETICS_API_BASE ||
  "https://vivasense-genetics.onrender.com";

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
}

/**
 * Mirrors CorrelationResponse in trait_relationships_schemas.py.
 *
 * r_matrix[i][j] — Pearson/Spearman r between trait_names[i] and trait_names[j].
 *   Diagonal = 1.0. null when a pair had < 3 complete genotype means.
 * p_matrix[i][j] — two-sided p-value from cor.test().
 *   Diagonal = 0.0. null when a pair had < 3 complete genotype means.
 */
export interface CorrelationResponse {
  trait_names: string[];
  n_observations: number;
  method: string;
  r_matrix: (number | null)[][];
  p_matrix: (number | null)[][];
  interpretation: string;
  warnings: string[];
}

// ─────────────────────────────────────────────────────────────────────────────
// API FUNCTION
// ─────────────────────────────────────────────────────────────────────────────

export async function computeCorrelation(
  request: CorrelationRequest
): Promise<CorrelationResponse> {
  let response: Response;
  try {
    response = await fetch(`${ENGINE_BASE}/genetics/correlation`, {
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
