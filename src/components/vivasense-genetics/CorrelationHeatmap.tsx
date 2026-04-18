/**
 * CorrelationHeatmap
 * ==================
 * Publication-quality correlation heatmap for the VivaSense Genetics module.
 *
 * Colour scale : Red (r = −1) → White (r = 0) → Blue (r = +1)  [RdBu diverging]
 * Cell content : coefficient (2 d.p.) + significance stars
 * Significance : *** p < 0.001 · ** p < 0.01 · * p < 0.05 · ns ≥ 0.05
 *               For genotypic VC mode: "≈" prefix on all significance labels.
 * Diagonal    : Self-correlations (r = 1.0) shown as light gray with trait initial.
 *
 * Crash safety:
 *   flattenMatrix() always returns exactly n² cells or [].
 *   The SVG grid renders ONLY when cells.length === n*n; otherwise shows
 *   a "No heatmap data available" placeholder.
 *   Every cell access is guarded against undefined.
 *
 * Export:
 *   "Export PNG"  — high-DPI Canvas (3×) → download
 *   "Export SVG"  — SVG serialisation → download
 */

import React, { useCallback, useRef } from "react";
import { CorrelationResponse, CorrelationStats } from "@/services/traitRelationshipsApi";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface HeatmapCell {
  x: string;
  y: string;
  value: number | null;
  pValue: number | null;
  isDiagonal: boolean;
}

export type HeatmapMode = "phenotypic" | "between_genotype" | "genotypic";

interface CorrelationHeatmapProps {
  data: CorrelationResponse;
  mode?: HeatmapMode;
  exportRef?: React.RefObject<{ exportPng: () => void; exportSvg: () => void }>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Data helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Resolve which stats block to use, with graceful fallback. */
function _resolveStats(
  data: CorrelationResponse,
  mode: HeatmapMode
): CorrelationStats | null {
  if (mode === "phenotypic") return data.phenotypic ?? null;
  if (mode === "between_genotype") return data.between_genotype ?? null;
  // genotypic VC: fall back to between_genotype if null
  return data.genotypic ?? data.between_genotype ?? null;
}

/**
 * Flatten the n×n r_matrix + p_matrix into a flat array of HeatmapCell.
 * Returns exactly n² items or [] when data is missing/malformed.
 * Never throws.
 */
function flattenMatrix(data: CorrelationResponse, mode: HeatmapMode): HeatmapCell[] {
  const traitNames = Array.isArray(data.trait_names) ? data.trait_names : [];
  const n = traitNames.length;

  if (n === 0) {
    console.warn("CorrelationHeatmap: trait_names is empty");
    return [];
  }

  const stats = _resolveStats(data, mode);
  if (!stats) {
    console.warn(`CorrelationHeatmap: no stats for mode "${mode}"`);
    return [];
  }

  const { r_matrix, p_matrix } = stats;

  if (
    !Array.isArray(r_matrix) || r_matrix.length === 0 ||
    !Array.isArray(p_matrix) || p_matrix.length === 0
  ) {
    console.warn(`CorrelationHeatmap: matrices missing or empty for mode "${mode}"`);
    return [];
  }

  const cells: HeatmapCell[] = [];
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const isDiagonal = row === col;
      const rRow = Array.isArray(r_matrix[row]) ? r_matrix[row] : [];
      const pRow = Array.isArray(p_matrix[row]) ? p_matrix[row] : [];
      cells.push({
        x: traitNames[col],
        y: traitNames[row],
        value:  isDiagonal ? 1 : (rRow[col] ?? null),
        pValue: isDiagonal ? null : (pRow[col] ?? null),
        isDiagonal,
      });
    }
  }

  console.log(
    `CorrelationHeatmap: flattenMatrix mode="${mode}" → ${n}×${n} = ${cells.length} cells,`,
    `sample:`, cells[1] ?? cells[0] ?? "none"
  );
  return cells;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour scale (Red → White → Blue)
// ─────────────────────────────────────────────────────────────────────────────

function rToRgb(r: number | null, isDiagonal: boolean): string {
  if (r === null) return "#D1D5DB";
  if (isDiagonal) return "#F9FAFB";

  const c = Math.max(-1, Math.min(1, r));
  if (c > 0) {
    if (c <= 0.5) {
      const t = c / 0.5;
      return `rgb(${Math.round(247 - t * 102)},${Math.round(247 - t * 56)},${Math.round(247 - t * 28)})`;
    }
    const t = (c - 0.5) / 0.5;
    return `rgb(${Math.round(145 - t * 76)},${Math.round(191 - t * 74)},${Math.round(219 - t * 39)})`;
  }
  if (c < 0) {
    if (c >= -0.5) {
      const t = (-c) / 0.5;
      return `rgb(${Math.round(241 + t * 6)},${Math.round(163 + t * 84)},${Math.round(64 + t * 183)})`;
    }
    const t = (-c - 0.5) / 0.5;
    return `rgb(${Math.round(215 + t * 26)},${Math.round(48 + t * 115)},${Math.round(39 + t * 25)})`;
  }
  return "#F7F7F7";
}

function labelColor(r: number | null, isDiagonal: boolean): string {
  if (r === null || isDiagonal) return "#6B7280";
  return Math.abs(r) < 0.6 ? "#1F2937" : "#FFFFFF";
}

function sigStars(p: number | null, isDiagonal: boolean, isApprox = false): string {
  if (isDiagonal) return "";
  if (p === null) return "?";
  const prefix = isApprox ? "≈" : "";
  if (p < 0.001) return `${prefix}***`;
  if (p < 0.01)  return `${prefix}**`;
  if (p < 0.05)  return `${prefix}*`;
  return "ns";
}

// ─────────────────────────────────────────────────────────────────────────────
// Export utilities
// ─────────────────────────────────────────────────────────────────────────────

function exportAsSvg(svgEl: SVGSVGElement, filename = "correlation_heatmap.svg") {
  const svg = new XMLSerializer().serializeToString(svgEl);
  const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  a.click();
  URL.revokeObjectURL(url);
}

function exportAsPng(svgEl: SVGSVGElement, filename = "correlation_heatmap.png", scale = 3) {
  const { width, height } = svgEl.getBoundingClientRect();
  const svg = new XMLSerializer().serializeToString(svgEl);
  const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml;charset=utf-8" }));
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = width * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(scale, scale);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    URL.revokeObjectURL(url);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const pngUrl = URL.createObjectURL(blob);
      Object.assign(document.createElement("a"), { href: pngUrl, download: filename }).click();
      URL.revokeObjectURL(pngUrl);
    }, "image/png");
  };
  img.src = url;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour legend
// ─────────────────────────────────────────────────────────────────────────────

function ColorLegend() {
  const stops = Array.from({ length: 31 }, (_, i) => rToRgb((i / 30) * 2 - 1, false));
  return (
    <div className="flex items-center gap-3 text-xs text-gray-500 mt-2">
      <div className="flex items-center gap-1.5 flex-1">
        <span className="font-semibold text-red-700 w-8">−1.0</span>
        <div
          className="h-4 flex-1 rounded"
          style={{ background: `linear-gradient(to right, ${stops.join(", ")})` }}
          aria-label="Diverging colour scale: red negative, white zero, blue positive"
        />
        <span className="font-semibold text-blue-700 w-8 text-right">+1.0</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tooltip
// ─────────────────────────────────────────────────────────────────────────────

interface TooltipState { cell: HeatmapCell | null; x: number; y: number; }

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export function CorrelationHeatmap({ data, mode = "phenotypic" }: CorrelationHeatmapProps) {
  const traitNames = Array.isArray(data.trait_names) ? data.trait_names : [];
  const n = traitNames.length;

  const stats      = _resolveStats(data, mode);
  const isApprox   = stats?.inference_approximate === true;
  const n_obs      = stats?.n_observations ?? 0;
  const method     = data.method ?? "pearson";

  // Flatten — guaranteed to return n² cells or []
  const cells = flattenMatrix(data, mode);
  const hasData = cells.length === n * n && n > 0;

  console.log(`CorrelationHeatmap: mode="${mode}" n=${n} hasData=${hasData} cells=${cells.length}`);

  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = React.useState<TooltipState>({ cell: null, x: 0, y: 0 });

  const labelW  = 130;
  const headerH = 96;
  const cellSize = n <= 6 ? 64 : n <= 10 ? 54 : 44;
  const totalW  = labelW + n * cellSize;
  const totalH  = headerH + n * cellSize;

  const shortName = (t: string) => t.length > 14 ? t.slice(0, 12) + "…" : t;

  const handleExportSvg = useCallback(() => { if (svgRef.current) exportAsSvg(svgRef.current); }, []);
  const handleExportPng = useCallback(() => { if (svgRef.current) exportAsPng(svgRef.current, "correlation_heatmap_300dpi.png", 3); }, []);

  const handleMouseEnter = (cell: HeatmapCell, evt: React.MouseEvent) =>
    setTooltip({ cell, x: evt.clientX, y: evt.clientY });
  const handleMouseLeave = () => setTooltip({ cell: null, x: 0, y: 0 });
  const handleMouseMove  = (evt: React.MouseEvent) => {
    if (tooltip.cell) setTooltip((p) => ({ ...p, x: evt.clientX, y: evt.clientY }));
  };

  return (
    <div className="space-y-4">
      {/* Export buttons */}
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-400">
          {hasData ? "Hover cells for exact values" : "No data to export"}
        </p>
        <div className="flex gap-2">
          <button
            type="button" onClick={handleExportSvg} disabled={!hasData}
            className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-xs font-medium text-gray-600 shadow-sm hover:bg-gray-50 disabled:opacity-40"
            title="Download SVG"
          >
            <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 1v9M4 6l4 4 4-4M2 12h12v2H2z"/>
            </svg>
            SVG
          </button>
          <button
            type="button" onClick={handleExportPng} disabled={!hasData}
            className="inline-flex items-center gap-1.5 rounded-lg border border-blue-300 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 shadow-sm hover:bg-blue-100 disabled:opacity-40"
            title="Download PNG (300 DPI)"
          >
            <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 1v9M4 6l4 4 4-4M2 12h12v2H2z"/>
            </svg>
            PNG (300 DPI)
          </button>
        </div>
      </div>

      {/* Heatmap or empty state */}
      <div className="overflow-x-auto rounded-xl border border-gray-100 bg-white p-4 shadow-sm">
        {!hasData ? (
          <div className="flex flex-col items-center justify-center py-12 text-center text-sm text-gray-400">
            <svg className="h-8 w-8 mb-2 text-gray-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>
              <rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>
            </svg>
            <p className="font-medium text-gray-500">No heatmap data available</p>
            <p className="text-xs mt-1 max-w-xs">
              {n === 0
                ? "No trait names returned from the analysis."
                : `Matrix data is missing or malformed for the "${mode}" mode.`}
            </p>
          </div>
        ) : (
          <svg
            ref={svgRef}
            width={totalW + 16}
            height={totalH + 16}
            viewBox={`0 0 ${totalW + 16} ${totalH + 16}`}
            xmlns="http://www.w3.org/2000/svg"
            style={{ fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif" }}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          >
            <rect x="0" y="0" width={totalW + 16} height={totalH + 16} fill="#ffffff" />

            {/* Column headers */}
            {traitNames.map((trait, j) => (
              <g key={`col-${j}`} transform={`translate(${labelW + j * cellSize + cellSize / 2}, ${headerH})`}>
                <text transform="rotate(-55)" textAnchor="start" dominantBaseline="middle"
                  fontSize={11} fill="#374151" fontWeight="500">
                  {shortName(trait)}
                </text>
              </g>
            ))}

            {/* Rows */}
            {traitNames.map((rowTrait, i) => (
              <g key={`row-${i}`} transform={`translate(0, ${headerH + i * cellSize})`}>
                <text x={labelW - 8} y={cellSize / 2} textAnchor="end"
                  dominantBaseline="middle" fontSize={11} fill="#374151" fontWeight="500">
                  {shortName(rowTrait)}
                </text>

                {traitNames.map((_colTrait, j) => {
                  const cell = cells[i * n + j];
                  // Guard: skip undefined cell (should not happen when hasData=true)
                  if (!cell) return null;

                  const bg    = rToRgb(cell.isDiagonal ? null : cell.value, cell.isDiagonal);
                  const fg    = labelColor(cell.isDiagonal ? null : cell.value, cell.isDiagonal);
                  const stars = sigStars(cell.pValue, cell.isDiagonal, isApprox);
                  const cx    = labelW + j * cellSize;

                  return (
                    <g
                      key={`cell-${i}-${j}`}
                      transform={`translate(${cx}, 0)`}
                      onMouseEnter={(e) => handleMouseEnter(cell, e)}
                      onMouseLeave={handleMouseLeave}
                      style={{ cursor: cell.isDiagonal ? "default" : "pointer" }}
                    >
                      <rect x={1} y={1} width={cellSize - 2} height={cellSize - 2}
                        fill={bg} rx={3} ry={3} />
                      {cell.isDiagonal ? (
                        <text x={cellSize / 2} y={cellSize / 2} textAnchor="middle"
                          dominantBaseline="middle" fontSize={10} fill="#9CA3AF" fontStyle="italic">
                          {rowTrait.charAt(0)}
                        </text>
                      ) : (
                        <>
                          <text x={cellSize / 2} y={cellSize / 2 - 5} textAnchor="middle"
                            dominantBaseline="middle" fontSize={cellSize < 55 ? 9 : 11}
                            fontWeight="600" fill={fg}>
                            {cell.value !== null ? cell.value.toFixed(2) : "?"}
                          </text>
                          <text x={cellSize / 2} y={cellSize / 2 + 8} textAnchor="middle"
                            dominantBaseline="middle" fontSize={8} fill={fg} opacity={0.85}>
                            {stars}
                          </text>
                        </>
                      )}
                    </g>
                  );
                })}
              </g>
            ))}
          </svg>
        )}

        {/* Colour legend — shown regardless of data presence */}
        <div className="mt-3 px-1">
          <ColorLegend />
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-400">
            <span>
              {mode === "genotypic"
                ? `rg (VC-based) · ${n_obs} genotypes`
                : `${method === "spearman" ? "Spearman ρ" : "Pearson r"} · ${n_obs} ${mode === "phenotypic" ? "observations" : "genotype means"}`}
            </span>
            {isApprox ? (
              <span className="text-amber-600">≈approx. p · *** &lt;0.001 · ** &lt;0.01 · * &lt;0.05 · ns ≥0.05</span>
            ) : (
              <span>*** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05 · ns ≥0.05</span>
            )}
          </div>
          {isApprox && (
            <p className="mt-1 text-xs text-amber-600">
              ⚠ Genotypic VC: p-values and CIs are approximate (Fisher z on n_genotypes). Interpret cautiously.
            </p>
          )}
          {mode === "genotypic" && data.genotypic === null && (
            <p className="mt-1 text-xs text-amber-600">
              ⚠ Genotypic VC unavailable — showing between-genotype association as fallback.
            </p>
          )}
        </div>
      </div>

      {/* Hover tooltip */}
      {tooltip.cell && !tooltip.cell.isDiagonal && (
        <div
          className="pointer-events-none fixed z-50 rounded-lg border border-gray-200 bg-white px-3 py-2 shadow-xl text-xs"
          style={{ left: tooltip.x + 14, top: tooltip.y - 10 }}
        >
          <p className="font-semibold text-gray-800 mb-1">
            {tooltip.cell.y} × {tooltip.cell.x}
          </p>
          <p>
            <span className="text-gray-500">{mode === "genotypic" ? "rg = " : "r = "}</span>
            <span className="font-mono font-bold text-gray-800">
              {tooltip.cell.value !== null ? tooltip.cell.value.toFixed(4) : "—"}
            </span>
          </p>
          <p>
            <span className="text-gray-500">{isApprox ? "≈p = " : "p = "}</span>
            <span className="font-mono text-gray-700">
              {tooltip.cell.pValue !== null
                ? tooltip.cell.pValue < 0.001 ? "<0.001" : tooltip.cell.pValue.toFixed(4)
                : "—"}
            </span>{" "}
            <span className="text-gray-400">{sigStars(tooltip.cell.pValue, false, isApprox)}</span>
          </p>
          {isApprox && tooltip.cell.pValue !== null && (
            <p className="text-amber-500 mt-1">approx. inference</p>
          )}
        </div>
      )}
    </div>
  );
}
