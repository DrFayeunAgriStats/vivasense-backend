/**
 * VivaSense Genetics API Service
 * Backend: https://vivasense-backend-r-production.up.railway.app
 */

import type {
  GeneticsRequestBody,
  GeneticsValidationResult,
  GeneticsAnalysisResult,
} from "@/types/genetics";

import { GENETICS_API_BASE } from "@/config/vivasense";

const API_BASE = GENETICS_API_BASE;

function logComponentRequest(componentName: string, url: string) {
  console.log(`[COMPONENT] ${componentName} -> ${url}`);
}

export async function checkGeneticsHealth(): Promise<{ status: string }> {
  const url = `${API_BASE}/health`;
  logComponentRequest("VivaSenseGenetics", url);
  try {
    const res = await fetch(url);
    return res.json();
  } catch {
    return { status: "unreachable" };
  }
}

export async function validateGeneticsInput(
  body: GeneticsRequestBody
): Promise<GeneticsValidationResult> {
  const url = `${API_BASE}/genetics/validate`;
  logComponentRequest("GeneticsForm", url);

  // Map frontend mode values to backend-expected "single" | "multi"
  const modeMap: Record<string, string> = {
    auto: body.data.num_environments > 1 ? "multi" : "single",
    single_environment: "single",
    multi_environment: "multi",
  };
  const mappedMode = modeMap[body.mode] ?? body.mode;

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, mode: mappedMode, validate_only: true }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Validation failed (${res.status})`);
  }
  return res.json();
}

export async function runGeneticsAnalysis(
  body: GeneticsRequestBody
): Promise<GeneticsAnalysisResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);
  const url = `${API_BASE}/genetics/analyze`;

  // Map frontend mode values to backend-expected "single" | "multi"
  const modeMap: Record<string, string> = {
    auto: body.data.num_environments > 1 ? "multi" : "single",
    single_environment: "single",
    multi_environment: "multi",
  };
  const mappedMode = modeMap[body.mode] ?? body.mode;

  try {
    logComponentRequest("GeneticsForm", url);
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, mode: mappedMode, validate_only: false }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const detail = Array.isArray(err.detail)
        ? err.detail.map((d: any) => d.msg || JSON.stringify(d)).join("; ")
        : err.detail || `Analysis failed (${res.status})`;
      throw new Error(detail);
    }
    return res.json();
  } finally {
    clearTimeout(timeout);
  }
}
