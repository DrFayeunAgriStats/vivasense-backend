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
 * Per-mode statistics block.
 * All matrix fields are guaranteed to be arrays after normalization —
 * never undefined or null — so callers can index them safely.
 */
export interface CorrelationStats {
  n_observations: number;
  df?: number;
  critical_r?: number;
  r_matrix: (number | null)[][];
  p_matrix: (number | null)[][];
  p_adj_matrix: (number | null)[][];
  ci_lower_matrix: (number | null)[][];
  ci_upper_matrix: (number | null)[][];
  /** True for genotypic VC mode: p-values and CIs are Fisher-z approximations. */
  inference_approximate: boolean;
  inference_note: string | null;
}

/**
 * Normalized three-mode correlation response.
 *
 * phenotypic       – field-level co-variation (all observations)
 * between_genotype – association among genotype means; NOT a genetic parameter
 * genotypic        – variance-component REML estimate; null when unavailable
 */
export interface CorrelationResponse {
  trait_names: string[];
  method: string;
  phenotypic: CorrelationStats;
  between_genotype: CorrelationStats;
  genotypic: CorrelationStats | null;
  interpretation: string;
  warnings: string[];
  statistical_note: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// NORMALIZATION
// ─────────────────────────────────────────────────────────────────────────────

/** Return an n×n matrix of nulls — used as a safe fallback for missing data. */
function _nullMatrix(n: number): (null)[][] {
  return Array.from({ length: n }, () => Array(n).fill(null));
}

/**
 * Normalize a raw stats block from the server, guaranteeing every matrix
 * field is a proper array and every numeric field has a sensible default.
 * Accepts anything (raw JSON) and returns a fully-typed CorrelationStats.
 */
function _normalizeStats(raw: unknown, n: number): CorrelationStats {
  const src = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;

  const toMatrix = (v: unknown): (number | null)[][] => {
    if (Array.isArray(v) && v.length > 0 && Array.isArray(v[0])) {
      return v as (number | null)[][];
    }
    return _nullMatrix(n);
  };

  return {
    n_observations: typeof src.n_observations === "number" ? src.n_observations : 0,
    df: typeof src.df === "number" ? src.df : undefined,
    critical_r: typeof src.critical_r === "number" ? src.critical_r : undefined,
    r_matrix:        toMatrix(src.r_matrix),
    p_matrix:        toMatrix(src.p_matrix),
    p_adj_matrix:    toMatrix(src.p_adj_matrix),
    ci_lower_matrix: toMatrix(src.ci_lower_matrix),
    ci_upper_matrix: toMatrix(src.ci_upper_matrix),
    inference_approximate: src.inference_approximate === true,
    inference_note: typeof src.inference_note === "string" ? src.inference_note : null,
  };
}

/**
 * Normalize the raw server JSON into a fully typed, crash-safe CorrelationResponse.
 * All matrix fields are guaranteed to be arrays; genotypic is null when absent.
 */
function _normalizeResponse(raw: unknown): CorrelationResponse {
  const src = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;

  const traitNames: string[] = Array.isArray(src.trait_names)
    ? (src.trait_names as string[])
    : [];
  const n = traitNames.length;

  const normalized: CorrelationResponse = {
    trait_names:     traitNames,
    method:          typeof src.method === "string" ? src.method : "pearson",
    phenotypic:      _normalizeStats(src.phenotypic, n),
    between_genotype: _normalizeStats(src.between_genotype, n),
    genotypic:       src.genotypic != null ? _normalizeStats(src.genotypic, n) : null,
    interpretation:  typeof src.interpretation === "string" ? src.interpretation : "",
    warnings:        Array.isArray(src.warnings) ? (src.warnings as string[]) : [],
    statistical_note: typeof src.statistical_note === "string" ? src.statistical_note : "",
  };

  console.log("[traitRelationshipsApi] normalized payload", {
    traitNames: normalized.trait_names,
    n,
    phenotypic_n: normalized.phenotypic.n_observations,
    between_genotype_n: normalized.between_genotype.n_observations,
    genotypic_available: normalized.genotypic !== null,
    genotypic_n: normalized.genotypic?.n_observations ?? null,
    r_matrix_shape: `${normalized.phenotypic.r_matrix.length}×${normalized.phenotypic.r_matrix[0]?.length ?? 0}`,
  });

  return normalized;
}

// ─────────────────────────────────────────────────────────────────────────────
// API FUNCTION
// ─────────────────────────────────────────────────────────────────────────────

export async function computeCorrelation(
  request: CorrelationRequest
): Promise<CorrelationResponse> {
  const correlationUrl = `${ENGINE_BASE}/genetics/correlation`;
  console.log("[traitRelationshipsApi] POST", correlationUrl, {
    traits: request.trait_columns,
    method: request.method,
    objective: request.user_objective,
  });

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

  const raw = await response.json();
  console.log("[traitRelationshipsApi] raw response keys:", Object.keys(raw ?? {}));

  return _normalizeResponse(raw);
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
