/**
 * VivaSense Academic Mentor API Client
 * =====================================
 * POST /academic/interpret — on-demand per-trait academic interpretation.
 *
 * Option A architecture: called only when the user explicitly clicks
 * "Get Academic Interpretation". Never called automatically on analysis.
 */

import { API_BASE } from "./apiConfig";

const ENGINE_BASE: string = API_BASE;

// ─────────────────────────────────────────────────────────────────────────────
// REQUEST TYPES
// ─────────────────────────────────────────────────────────────────────────────

export type AcademicModuleType =
  | "anova"
  | "genetic_parameters"
  | "correlation"
  | "heatmap";

export interface AcademicInterpretRequest {
  module_type: AcademicModuleType;
  trait?: string | null;
  /** Full analysis result dict from the matching /analysis/* or /genetics/* response */
  analysis_result: Record<string, unknown>;
  crop_context?: string | null;
  include_writing_support?: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// RESPONSE TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface ValidationViolation {
  rule_id: string;
  severity: "block" | "warn";
  excerpt: string;
  message: string;
}

export interface ValidationResult {
  passed: boolean;
  blocked: boolean;
  violations: ValidationViolation[];
  warning_count: number;
  block_count: number;
}

export interface SentenceStarter {
  purpose: string;
  template: string;
  values_to_fill: string[];
  hint?: string | null;
}

export interface GuidedWritingBlock {
  module_type: string;
  trait?: string | null;
  sentence_starters: SentenceStarter[];
  examiner_checkpoint: string[];
  scope_statement: string;
  caution_note?: string | null;
  supervisor_prompt: string;
}

export interface AcademicInterpretationResponse {
  module_type: string;
  trait?: string | null;

  // Core interpretation
  overall_finding: string;
  statistical_evidence: string;
  module_sections: Record<string, string>;

  // Fixed sections
  scope_statement: string;
  examiner_checkpoint: string[];
  closing: string;
  research_writing_referral: string;

  // Layer C — guided writing
  guided_writing?: GuidedWritingBlock | null;

  // Layer A — validator
  validator_result?: ValidationResult | null;

  // Provenance
  fallback_used: boolean;
  ai_generated: boolean;
  raw_ai_text?: string | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// API FUNCTION
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Request an academic interpretation for one trait's analysis result.
 *
 * Pass the raw analysis_result object from /genetics/analyze-upload
 * (or any /analysis/* endpoint). The backend handles normalisation.
 *
 * @throws Error with a human-readable message on network or server failure.
 */
export async function getAcademicInterpretation(
  request: AcademicInterpretRequest
): Promise<AcademicInterpretationResponse> {
  const url = `${ENGINE_BASE}/academic/interpret`;

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...request,
        include_writing_support: request.include_writing_support ?? true,
      }),
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    throw new Error(`Network error reaching Academic Mentor: ${msg}`);
  }

  if (response.status === 503) {
    throw new Error(
      "Academic Mentor is not available on this deployment. " +
        "Ask the administrator to set ANTHROPIC_API_KEY in Render environment variables."
    );
  }

  if (!response.ok) {
    const detail = await extractErrorDetail(response);
    throw new Error(`Academic interpretation failed — ${detail}`);
  }

  return response.json() as Promise<AcademicInterpretationResponse>;
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
