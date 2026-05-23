/**
 * API client for VivaSense Descriptive Statistics endpoints.
 *
 * Routes (backend):
 *   POST /analysis/descriptive-stats
 *   POST /export/descriptive-stats-word   (optional)
 */

import { GENETICS_API_BASE } from "@/config/vivasense";
import type {
  DescriptiveStatsRequest,
  DescriptiveStatsResponse,
} from "@/types/descriptiveStats";

const BASE = GENETICS_API_BASE;

export async function computeDescriptiveStats(
  body: DescriptiveStatsRequest
): Promise<DescriptiveStatsResponse> {
  const url = `${BASE}/analysis/descriptive-stats`;
  console.log("[MODULE] descriptive-stats");
  console.log("[REQUEST] descriptive-stats", url, {
    traits: body.trait_columns,
    genotype: body.genotype_column,
    rep: body.rep_column,
    expected_rep: body.expected_replication,
  });
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errorText = await res.text();
    console.error("[DESCRIPTIVE-STATS ERROR]", res.status, errorText);
    let detail = `Descriptive stats failed (${res.status})`;
    try {
      const parsed = JSON.parse(errorText);
      detail = parsed.detail || parsed.error || detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function exportDescriptiveStatsWord(
  payload: DescriptiveStatsRequest | { response: DescriptiveStatsResponse } | Record<string, unknown>
): Promise<Blob> {
  const url = `${BASE}/export/descriptive-stats-word`;
  console.log("[MODULE] descriptive-stats");
  console.log("[REQUEST] export-descriptive-stats-word", url, payload);
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorText = await res.text().catch(() => "");
    console.error("[DESCRIPTIVE-STATS EXPORT ERROR]", res.status, errorText);
    let detail = `Export failed (${res.status})`;
    try {
      const parsed = JSON.parse(errorText);
      detail = parsed.detail || parsed.error || detail;
    } catch {
      /* keep default */
    }
    throw new Error(detail);
  }
  return res.blob();
}
