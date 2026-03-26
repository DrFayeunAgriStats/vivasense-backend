/**
 * CorrelationHeatmap
 * ==================
 * Renders an n×n CSS/HTML table heatmap from a CorrelationResponse.
 *
 * Colour scale:  red (r = −1) → white (r = 0) → emerald (r = +1)
 * Cell content:  coefficient (2 d.p.) + significance stars
 * Significance:  *** p < 0.001 · ** p < 0.01 · * p < 0.05 · ns ≥ 0.05
 *
 * No external chart libraries.  Colour is computed via HSL interpolation.
 */

import React from "react";
import { CorrelationResponse } from "@/services/traitRelationshipsApi";

interface CorrelationHeatmapProps {
  data: CorrelationResponse;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Map a correlation coefficient to a CSS background colour string.
 *   Positive r → emerald hue (hsl 160°)
 *   Negative r → red hue    (hsl   0°)
 *   Magnitude drives lightness: 0→100% (white), 1→48% (full colour)
 */
function rToBackground(r: number | null): string {
  if (r === null) return "rgb(229,231,235)"; // Tailwind gray-200 — null/insufficient data
  const abs = Math.min(Math.abs(r), 1);
  const lightness = Math.round(100 - abs * 52); // 100% at 0 → 48% at ±1
  if (r > 0) return `hsl(160,58%,${lightness}%)`;
  if (r < 0) return `hsl(0,70%,${lightness}%)`;
  return "rgb(255,255,255)";
}

/** Use white text when the background is dark enough (|r| > 0.55). */
function textClass(r: number | null, isDiagonal: boolean): string {
  if (isDiagonal) return "text-gray-400";
  if (r === null) return "text-gray-400";
  return Math.abs(r) > 0.55 ? "text-white" : "text-gray-800";
}

/** Significance stars from a p-value. */
function sigStars(p: number | null, isDiagonal: boolean): string {
  if (isDiagonal) return "—";
  if (p === null) return "?";
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "ns";
}

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

export function CorrelationHeatmap({ data }: CorrelationHeatmapProps) {
  const { trait_names, r_matrix, p_matrix, n_observations, method } = data;
  const n = trait_names.length;

  // Shorten very long trait names for display (full name shown in title tooltip)
  const labels = trait_names.map((t) => (t.length > 15 ? t.slice(0, 13) + "…" : t));

  // Cell size — shrinks for larger matrices so the heatmap fits more easily
  const cellPx = n <= 6 ? 60 : n <= 10 ? 52 : 44;

  return (
    <div className="overflow-x-auto">
      <table className="border-collapse" style={{ fontSize: "0.72rem" }}>
        {/* Column headers — rotated 45° for space efficiency */}
        <thead>
          <tr>
            {/* Empty corner cell matching the row-label column */}
            <th style={{ width: 130 }} />
            {labels.map((label, j) => (
              <th
                key={j}
                style={{
                  width: cellPx,
                  height: 88,
                  verticalAlign: "bottom",
                  paddingBottom: 4,
                }}
              >
                <div
                  style={{
                    writingMode: "vertical-rl",
                    transform: "rotate(180deg)",
                    maxHeight: 84,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    fontWeight: 500,
                    color: "#4B5563",
                  }}
                  title={trait_names[j]}
                >
                  {label}
                </div>
              </th>
            ))}
          </tr>
        </thead>

        {/* Rows */}
        <tbody>
          {trait_names.map((rowTrait, i) => {
            const r_row = r_matrix[i] ?? [];
            const p_row = p_matrix[i] ?? [];

            return (
              <tr key={i}>
                {/* Row label */}
                <td
                  className="pr-2 font-medium text-gray-600 whitespace-nowrap text-right"
                  style={{
                    maxWidth: 128,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                  title={rowTrait}
                >
                  {labels[i]}
                </td>

                {/* Cells */}
                {trait_names.map((colTrait, j) => {
                  const isDiag = i === j;
                  const r = isDiag ? null : (r_row[j] ?? null);
                  const p = isDiag ? null : (p_row[j] ?? null);
                  const bg = rToBackground(isDiag ? null : r);
                  const tc = textClass(r, isDiag);

                  const tooltip = isDiag
                    ? rowTrait
                    : `${rowTrait} × ${colTrait}` +
                      (r !== null ? `\nr = ${r.toFixed(3)}` : "\nr = ?") +
                      (p !== null ? `\np = ${p < 0.001 ? "<0.001" : p.toFixed(4)}` : "\np = ?");

                  return (
                    <td
                      key={j}
                      title={tooltip}
                      className="border border-white text-center align-middle select-none"
                      style={{
                        backgroundColor: bg,
                        width: cellPx,
                        height: cellPx,
                        padding: 2,
                      }}
                    >
                      {isDiag ? (
                        <span className={`font-medium ${tc}`}>—</span>
                      ) : (
                        <div className={`leading-tight ${tc}`}>
                          <div className="font-semibold">
                            {r !== null ? r.toFixed(2) : "?"}
                          </div>
                          <div style={{ opacity: 0.8 }}>
                            {sigStars(p, false)}
                          </div>
                        </div>
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>

      {/* Legend row */}
      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-500">
        <span>
          {method === "spearman" ? "Spearman rank" : "Pearson"} r ·{" "}
          {n_observations} genotype means
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-3 w-6 rounded"
            style={{ background: "hsl(160,58%,48%)" }}
          />
          Strong positive
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-3 w-6 rounded"
            style={{ background: "hsl(0,70%,48%)" }}
          />
          Strong negative
        </span>
        <span>
          *** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05 · ns ≥0.05
        </span>
      </div>
    </div>
  );
}
