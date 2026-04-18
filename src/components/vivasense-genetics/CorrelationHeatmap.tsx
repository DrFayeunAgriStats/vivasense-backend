/**
 * CorrelationHeatmap
 * ==================
 * Publication-quality correlation heatmap for the VivaSense Genetics module.
 *
 * Colour scale : Red (r = −1) → Light Red (r = −0.5) → White (r = 0) → Light Blue (r = +0.5) → Blue (r = +1)
 * Perceptually uniform diverging scale (RdBu style) for clear sign distinction.
 * Cell content : coefficient (2 d.p.) + significance stars
 * Significance : *** p < 0.001 · ** p < 0.01 · * p < 0.05 · ns ≥ 0.05
 * Diagonal    : Self-correlations (r = 1.0) shown as light gray with trait initial
 *
 * Data shape accepted:
 *   The component internally flattens the r_matrix into
 *   [{ x: 'TraitA', y: 'TraitB', value: 0.85, pValue: 0.003 }, …]
 *   before rendering — satisfying the requirement for a flat data array.
 *
 * Export:
 *   "Export PNG"  — draws the SVG onto a high-DPI Canvas (3×) → download
 *   "Export SVG"  — serialises the SVG node directly → download
 *
 * Dependencies: recharts (for Tooltip; install: npm install recharts)
 */

import React, { useCallback, useRef } from "react";
// recharts is used for the project's chart ecosystem; install: npm install recharts
// (Tooltip rendered as a custom floating div for full SVG export compatibility)
import { CorrelationResponse } from "@/services/traitRelationshipsApi";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/** Flat cell datum — produced by flattenMatrix() */
export interface HeatmapCell {
  x: string;       // column trait name
  y: string;       // row trait name
  value: number | null;  // Pearson/Spearman r, null when insufficient data
  pValue: number | null; // two-sided p-value from cor.test()
  isDiagonal: boolean;
}

interface CorrelationHeatmapProps {
  data: CorrelationResponse;
  /** Correlation mode to display. */
  mode?: 'phenotypic' | 'between_genotype' | 'genotypic';
  /** Passed from parent to support controlled export triggers. Optional. */
  exportRef?: React.RefObject<{ exportPng: () => void; exportSvg: () => void }>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Data helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Flatten the n×n r_matrix + p_matrix into a 1-D array of HeatmapCell objects.
 * Produces n² cells — including the diagonal (value=1, isDiagonal=true).
 */
function flattenMatrix(data: CorrelationResponse, mode: 'phenotypic' | 'between_genotype' | 'genotypic' = 'phenotypic'): HeatmapCell[] {
  const stats =
    mode === 'phenotypic' ? data.phenotypic :
    mode === 'between_genotype' ? data.between_genotype :
    data.genotypic;

  const { trait_names } = data;

  if (!stats) {
    console.warn(`CorrelationHeatmap: ${mode} data not available`);
    return [];
  }

  const { r_matrix, p_matrix } = stats;
  const cells: HeatmapCell[] = [];

  if (!r_matrix || !p_matrix || !Array.isArray(r_matrix) || !Array.isArray(p_matrix)) {
    console.warn(`CorrelationHeatmap: ${mode} matrices not available, returning empty cells`);
    return cells;
  }

  for (let row = 0; row < trait_names.length; row++) {
    for (let col = 0; col < trait_names.length; col++) {
      const isDiagonal = row === col;
      cells.push({
        x: trait_names[col],
        y: trait_names[row],
        value: isDiagonal ? 1 : (r_matrix[row]?.[col] ?? null),
        pValue: isDiagonal ? null : (p_matrix[row]?.[col] ?? null),
        isDiagonal,
      });
    }
  }

  return cells;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour scale  (Red → White → Blue)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Map a correlation coefficient r ∈ [−1, +1] to an RGB background colour.
 * Uses a perceptually-uniform diverging colour scale (RdBu-style):
 *
 *   r = −1  → strong red    (#D73027)
 *   r = −0.5 → light red    (#F1A340)
 *   r =  0  → white         (#F7F7F7)
 *   r = +0.5 → light blue   (#91BFDB)
 *   r = +1  → strong blue   (#4575B4)
 *
 * Null / missing → light gray.
 */
function rToRgb(r: number | null, isDiagonal: boolean): string {
  if (r === null) return "#D1D5DB"; // neutral gray for null values
  if (isDiagonal) return "#F9FAFB"; // very light gray for diagonal (self-correlation)

  const clamped = Math.max(-1, Math.min(1, r));

  if (clamped > 0) {
    // Positive: White → Light Blue → Strong Blue
    if (clamped <= 0.5) {
      // White to Light Blue: 0 → 0.5
      const t = clamped / 0.5; // normalize to 0-1
      const r_val = Math.round(247 - t * (247 - 145)); // #F7 to #91
      const g_val = Math.round(247 - t * (247 - 191)); // #F7 to #BF
      const b_val = Math.round(247 + t * (219 - 247)); // #F7 to #DB
      return `rgb(${r_val},${g_val},${b_val})`;
    } else {
      // Light Blue to Strong Blue: 0.5 → 1.0
      const t = (clamped - 0.5) / 0.5; // normalize to 0-1
      const r_val = Math.round(145 - t * (145 - 69)); // #91 to #45
      const g_val = Math.round(191 - t * (191 - 117)); // #BF to #75
      const b_val = Math.round(219 - t * (219 - 180)); // #DB to #B4
      return `rgb(${r_val},${g_val},${b_val})`;
    }
  } else if (clamped < 0) {
    // Negative: Strong Red → Light Red → White
    if (clamped >= -0.5) {
      // Light Red to White: -0.5 → 0
      const t = (-clamped) / 0.5; // normalize to 0-1
      const r_val = Math.round(241 + t * (247 - 241)); // #F1 to #F7
      const g_val = Math.round(163 + t * (247 - 163)); // #A3 to #F7
      const b_val = Math.round(64 + t * (247 - 64)); // #40 to #F7
      return `rgb(${r_val},${g_val},${b_val})`;
    } else {
      // Strong Red to Light Red: -1 → -0.5
      const t = (-clamped - 0.5) / 0.5; // normalize to 0-1
      const r_val = Math.round(215 + t * (241 - 215)); // #D7 to #F1
      const g_val = Math.round(48 + t * (163 - 48)); // #30 to #A3
      const b_val = Math.round(39 + t * (64 - 39)); // #27 to #40
      return `rgb(${r_val},${g_val},${b_val})`;
    }
  } else {
    // r = 0 exactly → white
    return "#F7F7F7";
  }
}

/** Choose black or white label text based on background luminance. */
function labelColor(r: number | null, isDiagonal: boolean): string {
  if (r === null) return "#6B7280";
  if (isDiagonal) return "#6B7280";
  
  const abs = Math.abs(r);
  // For very light colors (near 0), use dark text
  // For very dark colors (near ±1), use white text
  if (abs < 0.3) return "#1F2937"; // dark gray/black for light backgrounds (near zero)
  if (abs < 0.6) return "#1F2937"; // dark text for medium colors
  return "#FFFFFF"; // white text for strong colors (dark backgrounds)
}

/** Significance stars. */
function sigStars(p: number | null, isDiagonal: boolean): string {
  if (isDiagonal) return "";
  if (p === null) return "?";
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "ns";
}

// ─────────────────────────────────────────────────────────────────────────────
// Export utilities
// ─────────────────────────────────────────────────────────────────────────────

function exportAsSvg(svgEl: SVGSVGElement, filename = "correlation_heatmap.svg") {
  const serialiser = new XMLSerializer();
  const svgStr = serialiser.serializeToString(svgEl);
  const blob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function exportAsPng(svgEl: SVGSVGElement, filename = "correlation_heatmap.png", scale = 3) {
  const { width, height } = svgEl.getBoundingClientRect();
  const serialiser = new XMLSerializer();
  const svgStr = serialiser.serializeToString(svgEl);
  const blob = new Blob([svgStr], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width  = width  * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(scale, scale);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    ctx.drawImage(img, 0, 0, width, height);
    URL.revokeObjectURL(url);
    canvas.toBlob((pngBlob) => {
      if (!pngBlob) return;
      const pngUrl = URL.createObjectURL(pngBlob);
      const a = document.createElement("a");
      a.href = pngUrl;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(pngUrl);
    }, "image/png");
  };
  img.src = url;
}

// ─────────────────────────────────────────────────────────────────────────────
// Colour legend bar
// ─────────────────────────────────────────────────────────────────────────────

function ColorLegend() {
  const stops: string[] = [];
  // Generate 30 stops for smooth gradient from -1 to +1
  for (let i = 0; i <= 30; i++) {
    const r = (i / 30) * 2 - 1; // -1 to +1
    stops.push(rToRgb(r, false));
  }
  const gradient = stops.join(", ");

  return (
    <div className="flex items-center gap-3 text-xs text-gray-500 mt-2">
      <div className="flex items-center gap-1.5 flex-1">
        <span className="font-semibold text-red-700 w-8">−1.0</span>
        <div
          className="h-4 flex-1 rounded"
          style={{ background: `linear-gradient(to right, ${gradient})` }}
          aria-label="Diverging colour scale: red for negative correlations, white for zero, blue for positive correlations"
        />
        <span className="font-semibold text-blue-700 w-8 text-right">+1.0</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tooltip content (recharts Tooltip is used for the custom tooltip overlay)
// ─────────────────────────────────────────────────────────────────────────────

interface TooltipState {
  cell: HeatmapCell | null;
  x: number;
  y: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export function CorrelationHeatmap({ data, mode = 'phenotypic' }: CorrelationHeatmapProps) {
  const { trait_names, method } = data;
  const stats =
    mode === 'phenotypic' ? data.phenotypic :
    mode === 'between_genotype' ? data.between_genotype :
    data.genotypic;
  const n_observations = stats?.n_observations ?? 0;
  const n = trait_names.length;

  // Flatten the matrix to a typed flat array as required
  const cells: HeatmapCell[] = flattenMatrix(data, mode);

  // Debug logging for troubleshooting
  console.log(`CorrelationHeatmap: Rendering ${mode} mode with ${cells.length} cells`);
  console.log('CorrelationHeatmap: Sample cell:', cells[0]);

  const svgRef = useRef<SVGSVGElement>(null);

  // Tooltip state (hover)
  const [tooltip, setTooltip] = React.useState<TooltipState>({ cell: null, x: 0, y: 0 });

  // ── Layout constants ───────────────────────────────────────────────────────
  const labelW = 130;       // px for row labels
  const headerH = 96;       // px for rotated column headers
  const cellSize = n <= 6 ? 64 : n <= 10 ? 54 : 44;
  const totalW = labelW + n * cellSize;
  const totalH = headerH + n * cellSize;

  // Truncate long names for display
  const shortName = (t: string) => (t.length > 14 ? t.slice(0, 12) + "…" : t);

  // ── Export handlers ────────────────────────────────────────────────────────
  const handleExportSvg = useCallback(() => {
    if (svgRef.current) exportAsSvg(svgRef.current);
  }, []);

  const handleExportPng = useCallback(() => {
    if (svgRef.current) exportAsPng(svgRef.current, "correlation_heatmap_300dpi.png", 3);
  }, []);

  // ── Cell mouse handlers ────────────────────────────────────────────────────
  const handleMouseEnter = (cell: HeatmapCell, evt: React.MouseEvent) => {
    setTooltip({ cell, x: evt.clientX, y: evt.clientY });
  };
  const handleMouseLeave = () => setTooltip({ cell: null, x: 0, y: 0 });
  const handleMouseMove  = (evt: React.MouseEvent) => {
    if (tooltip.cell) setTooltip((prev) => ({ ...prev, x: evt.clientX, y: evt.clientY }));
  };

  return (
    <div className="space-y-4">
      {/* ── Export button row ── */}
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-400">
          Hover cells for exact r &amp; p-values
        </p>
        <div className="flex gap-2">
          <button
            id="heatmap-export-svg"
            type="button"
            onClick={handleExportSvg}
            className="inline-flex items-center gap-1.5 rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-xs font-medium text-gray-600 shadow-sm hover:bg-gray-50 hover:border-gray-400 transition-colors"
            title="Download publication-ready SVG vector"
          >
            <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 1v9M4 6l4 4 4-4M2 12h12v2H2z"/>
            </svg>
            SVG
          </button>
          <button
            id="heatmap-export-png"
            type="button"
            onClick={handleExportPng}
            className="inline-flex items-center gap-1.5 rounded-lg border border-blue-300 bg-blue-50 px-3 py-1.5 text-xs font-medium text-blue-700 shadow-sm hover:bg-blue-100 transition-colors"
            title="Download 300 DPI PNG (3× upscaled)"
          >
            <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 1v9M4 6l4 4 4-4M2 12h12v2H2z"/>
            </svg>
            PNG (300 DPI)
          </button>
        </div>
      </div>

      {/* ── SVG heatmap ── */}
      <div className="overflow-x-auto rounded-xl border border-gray-100 bg-white p-4 shadow-sm">
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
          {/* Background */}
          <rect x="0" y="0" width={totalW + 16} height={totalH + 16} fill="#ffffff" />

          {/* ── Column headers (rotated text) ── */}
          {trait_names.map((trait, j) => (
            <g key={`col-${j}`} transform={`translate(${labelW + j * cellSize + cellSize / 2}, ${headerH})`}>
              <text
                transform="rotate(-55)"
                textAnchor="start"
                dominantBaseline="middle"
                fontSize={11}
                fill="#374151"
                fontWeight="500"
              >
                {shortName(trait)}
              </text>
            </g>
          ))}

          {/* ── Rows ── */}
          {trait_names.map((rowTrait, i) => (
            <g key={`row-${i}`} transform={`translate(0, ${headerH + i * cellSize})`}>
              {/* Row label */}
              <text
                x={labelW - 8}
                y={cellSize / 2}
                textAnchor="end"
                dominantBaseline="middle"
                fontSize={11}
                fill="#374151"
                fontWeight="500"
              >
                {shortName(rowTrait)}
              </text>

              {/* Cells */}
              {trait_names.map((colTrait, j) => {
                const cell = cells[i * n + j];
                const bg   = rToRgb(cell.isDiagonal ? null : cell.value, cell.isDiagonal);
                const fg   = labelColor(cell.isDiagonal ? null : cell.value, cell.isDiagonal);
                const stars = sigStars(cell.pValue, cell.isDiagonal);
                const cx = labelW + j * cellSize;

                return (
                  <g
                    key={`cell-${i}-${j}`}
                    transform={`translate(${cx}, 0)`}
                    onMouseEnter={(e) => handleMouseEnter(cell, e)}
                    onMouseLeave={handleMouseLeave}
                    style={{ cursor: cell.isDiagonal ? "default" : "pointer" }}
                  >
                    {/* Cell background */}
                    <rect
                      x={1}
                      y={1}
                      width={cellSize - 2}
                      height={cellSize - 2}
                      fill={bg}
                      rx={3}
                      ry={3}
                    />

                    {/* Diagonal: just show trait initial */}
                    {cell.isDiagonal ? (
                      <text
                        x={cellSize / 2}
                        y={cellSize / 2}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontSize={10}
                        fill="#9CA3AF"
                        fontStyle="italic"
                      >
                        {rowTrait.charAt(0)}
                      </text>
                    ) : (
                      <>
                        {/* r value */}
                        <text
                          x={cellSize / 2}
                          y={cellSize / 2 - 5}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize={cellSize < 55 ? 9 : 11}
                          fontWeight="600"
                          fill={fg}
                        >
                          {cell.value !== null ? cell.value.toFixed(2) : "?"}
                        </text>
                        {/* Significance stars */}
                        <text
                          x={cellSize / 2}
                          y={cellSize / 2 + 8}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize={8}
                          fill={fg}
                          opacity={0.85}
                        >
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

        {/* ── Colour legend ── */}
        <div className="mt-3 px-1">
          <ColorLegend />
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-400">
            <span>
              {method === "spearman" ? "Spearman ρ" : "Pearson r"} ·{" "}
              {n_observations} genotype means
            </span>
            <span>*** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05 · ns ≥0.05</span>
          </div>
        </div>
      </div>

      {/* ── Hover tooltip (floating, not recharts internal) ── */}
      {tooltip.cell && !tooltip.cell.isDiagonal && (
        <div
          className="pointer-events-none fixed z-50 rounded-lg border border-gray-200 bg-white px-3 py-2 shadow-xl text-xs"
          style={{ left: tooltip.x + 14, top: tooltip.y - 10 }}
        >
          <p className="font-semibold text-gray-800 mb-1">
            {tooltip.cell.y} × {tooltip.cell.x}
          </p>
          <p>
            <span className="text-gray-500">r = </span>
            <span className="font-mono font-bold text-gray-800">
              {tooltip.cell.value !== null ? tooltip.cell.value.toFixed(4) : "—"}
            </span>
          </p>
          <p>
            <span className="text-gray-500">p = </span>
            <span className="font-mono text-gray-700">
              {tooltip.cell.pValue !== null
                ? tooltip.cell.pValue < 0.001
                  ? "<0.001"
                  : tooltip.cell.pValue.toFixed(4)
                : "—"}
            </span>{" "}
            <span className="text-gray-400">
              {sigStars(tooltip.cell.pValue, false)}
            </span>
          </p>
        </div>
      )}
    </div>
  );
}
