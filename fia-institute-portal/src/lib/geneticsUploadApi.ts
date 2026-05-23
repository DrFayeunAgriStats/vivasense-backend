/**
 * API functions for VivaSense Genetics upload, analysis & correlation endpoints.
 */

import type {
  UploadPreviewResponse,
  AnalyzeUploadRequest,
  AnalyzeUploadResponse,
  CorrelationRequest,
  CorrelationResponse,
  AnalysisModule,
} from "@/types/geneticsUpload";

import { GENETICS_API_BASE } from "@/config/vivasense";

const BASE = GENETICS_API_BASE;

export async function uploadPreview(file: File): Promise<UploadPreviewResponse> {
  const url = `${BASE}/genetics/upload-preview`;
  console.log("[MODULE] upload-preview");
  console.log("[REQUEST] upload-preview", url);
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(url, { method: "POST", body: fd });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload preview failed (${res.status})`);
  }
  return res.json();
}

export async function analyzeUpload(body: AnalyzeUploadRequest): Promise<AnalyzeUploadResponse> {
  const url = `${BASE}/genetics/analyze-upload?module=${body.module}`;
  console.log("[MODULE]", body.module);
  console.log("[REQUEST] analyze-upload", url);
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Analysis failed (${res.status})`);
  }
  return res.json();
}

export async function computeAnova(body: {
  dataset_token?: string | null;
  trait_columns: string[];
}): Promise<Record<string, unknown>> {
  const url = `${BASE}/analysis/anova`;
  console.log("[MODULE] anova");
  console.log("[REQUEST] anova", url, body);
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || err.error || `ANOVA failed (${res.status})`);
  }
  return res.json();
}

export async function computeCorrelation(body: CorrelationRequest): Promise<CorrelationResponse> {
  const url = `${BASE}/genetics/correlation`;
  console.log("[MODULE] correlation");
  console.log("[REQUEST] correlation", url);
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errorText = await res.text();
    console.error("[CORRELATION ERROR] Status:", res.status, "Body:", errorText);
    let detail = `Correlation failed (${res.status})`;
    try {
      const parsed = JSON.parse(errorText);
      detail = parsed.detail || detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

export async function downloadReport(
  module: AnalysisModule,
  payload: Record<string, unknown>
): Promise<Blob> {
  const url = `${BASE}/genetics/download-results?module=${module}`;
  console.log("[MODULE]", module);
  console.log("[REQUEST] download-results", url);
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, module }),
  });
  if (!res.ok) {
    throw new Error(`Download failed (${res.status})`);
  }
  return res.blob();
}

export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.includes(",") ? result.split(",")[1] : result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
