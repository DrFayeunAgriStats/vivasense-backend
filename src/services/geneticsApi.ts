/**
 * VivaSense Genetics API Client — single-trait analysis
 * ======================================================
 * Calls the core genetics endpoints:
 *   POST /genetics/analyze   – single-trait ANOVA + heritability
 *   POST /genetics/validate  – data validation pre-flight
 *   GET  /health             – service health check
 */

import { API_BASE } from "./apiConfig";
const ENGINE_BASE: string = API_BASE;

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface ObservationRecord {
  genotype: string;
  rep: string;
  trait_value: number;
  environment?: string;
}

export interface GeneticsRequest {
  data: ObservationRecord[];
  mode: "single" | "multi";
  trait_name?: string;
  random_environment?: boolean;
}

export interface GeneticsResult {
  environment_mode: string;
  n_genotypes: number;
  n_reps: number;
  n_environments?: number;
  grand_mean: number;
  variance_components: Record<string, number | null | string>;
  heritability: Record<string, unknown>;
  genetic_parameters: Record<string, number | null>;
}

export interface GeneticsResponse {
  status: string;
  mode: string;
  data_validation: Record<string, unknown>;
  variance_warnings: Record<string, unknown>;
  result: GeneticsResult | null;
  interpretation: string | null;
}

export interface ValidationResponse {
  is_valid: boolean;
  warnings: Record<string, unknown>;
}

// ─────────────────────────────────────────────────────────────────────────────
// API FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Run single-trait genetic parameter analysis.
 * Endpoint: POST /genetics/analyze  (replaces old /analysis/anova)
 */
export async function analyzeGenetics(
  request: GeneticsRequest
): Promise<GeneticsResponse> {
  let response: Response;
  try {
    response = await fetch(`${ENGINE_BASE}/genetics/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Network error reaching genetics engine: ${msg}`);
  }

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`Analysis failed — ${detail}`);
  }

  return response.json() as Promise<GeneticsResponse>;
}

/**
 * Validate observation data before analysis.
 * Endpoint: POST /genetics/validate  (replaces old /analysis/validate)
 */
export async function validateGenetics(
  request: GeneticsRequest
): Promise<ValidationResponse> {
  let response: Response;
  try {
    response = await fetch(`${ENGINE_BASE}/genetics/validate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Network error reaching genetics engine: ${msg}`);
  }

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`Validation failed — ${detail}`);
  }

  return response.json() as Promise<ValidationResponse>;
}

/**
 * Health check — confirm the genetics engine is reachable.
 * Endpoint: GET /health
 */
export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${ENGINE_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
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
