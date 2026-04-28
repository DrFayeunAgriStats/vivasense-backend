/**
 * FieldLayoutGenerator — Version 1
 * ==================================
 * Client-side only. No backend calls, no Word export.
 * Supports CRD and RCBD with seeded Fisher-Yates randomization.
 * Outputs an SVG preview and a PNG download.
 */

import React, { useCallback, useRef, useState } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type Design = "CRD" | "RCBD";

interface PlotCell {
  plotNumber: number;
  treatmentIndex: number;
  blockNumber: number; // 1-based; all cells share block 1 in CRD (no block structure)
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

// 20 visually distinct colours — one per treatment slot.
const TREATMENT_COLORS: string[] = [
  "#4CAF50", "#2196F3", "#FF5722", "#9C27B0", "#FF9800",
  "#00BCD4", "#F44336", "#3F51B5", "#8BC34A", "#E91E63",
  "#009688", "#FFC107", "#673AB7", "#CDDC39", "#795548",
  "#607D8B", "#FF4081", "#00E676", "#40C4FF", "#FFAB40",
];

// SVG pixel geometry (actual field dimensions are shown as text labels).
const CELL_W = 84;
const CELL_H = 64;
const SVG_MARGIN = 24;
const BLOCK_GAP = 10;      // extra vertical gap between RCBD blocks
const BLOCK_LABEL_W = 58;  // left-column width reserved for "Block N" text
const TITLE_H = 28;        // vertical space for the title row

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/** Seeded LCG — deterministic, no dependencies, no floating-point overflow. */
function seededPRNG(seed: number): () => number {
  let s = (seed >>> 0) || 1;
  return function (): number {
    s = ((s * 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function fisherYates<T>(arr: T[], rand: () => number): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

/** Lighten a CSS hex colour toward white by `factor` (0 = original, 1 = white). */
function lighten(hex: string, factor: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgb(${Math.round(r + (255 - r) * factor)},${Math.round(
    g + (255 - g) * factor
  )},${Math.round(b + (255 - b) * factor)})`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Layout generation
// ─────────────────────────────────────────────────────────────────────────────

function generateLayout(
  design: Design,
  nTreatments: number,
  nReps: number,
  seed: number
): PlotCell[][] {
  const rand = seededPRNG(seed);
  const indices = Array.from({ length: nTreatments }, (_, i) => i);

  if (design === "RCBD") {
    // Each row = one replication block; treatments shuffled within the block.
    let plotNum = 1;
    return Array.from({ length: nReps }, (_, rep) =>
      fisherYates(indices, rand).map((ti) => ({
        plotNumber: plotNum++,
        treatmentIndex: ti,
        blockNumber: rep + 1,
      }))
    );
  }

  // CRD: pool all treatment × rep combinations, shuffle globally, arrange in grid.
  const pool: number[] = [];
  for (let r = 0; r < nReps; r++)
    for (let t = 0; t < nTreatments; t++)
      pool.push(t);

  const shuffled = fisherYates(pool, rand);
  const nCols = Math.ceil(Math.sqrt(shuffled.length));
  const nRows = Math.ceil(shuffled.length / nCols);
  const grid: PlotCell[][] = [];
  let plotNum = 1;

  for (let row = 0; row < nRows; row++) {
    const cells: PlotCell[] = [];
    for (let col = 0; col < nCols; col++) {
      const idx = row * nCols + col;
      if (idx < shuffled.length)
        cells.push({ plotNumber: plotNum++, treatmentIndex: shuffled[idx], blockNumber: 1 });
    }
    grid.push(cells);
  }
  return grid;
}

// ─────────────────────────────────────────────────────────────────────────────
// SVG layout component
// ─────────────────────────────────────────────────────────────────────────────

interface LayoutSVGProps {
  layout: PlotCell[][];
  design: Design;
  treatmentNames: string[];
  nTreatments: number;
  nReps: number;
  plotWidth: number;
  plotLength: number;
}

const LayoutSVG = React.forwardRef<SVGSVGElement, LayoutSVGProps>(
  ({ layout, design, treatmentNames, nTreatments, nReps, plotWidth, plotLength }, ref) => {
    const nRows = layout.length;
    const nCols = Math.max(...layout.map((row) => row.length));
    const leftPad = design === "RCBD" ? BLOCK_LABEL_W : SVG_MARGIN;

    const totalW = leftPad + nCols * CELL_W + SVG_MARGIN;
    const totalH =
      TITLE_H +
      SVG_MARGIN +
      nRows * CELL_H +
      (design === "RCBD" ? (nRows - 1) * BLOCK_GAP : 0) +
      SVG_MARGIN;

    const designLabel =
      design === "CRD"
        ? "Completely Randomized Design"
        : "Randomized Complete Block Design";

    return (
      <svg
        ref={ref}
        width={totalW}
        height={totalH}
        xmlns="http://www.w3.org/2000/svg"
        style={{ fontFamily: "ui-sans-serif, system-ui, sans-serif" }}
      >
        {/* White background — required for correct PNG export */}
        <rect width={totalW} height={totalH} fill="white" />

        {/* Title */}
        <text
          x={totalW / 2}
          y={TITLE_H - 8}
          textAnchor="middle"
          fontSize={11}
          fontWeight="600"
          fill="#374151"
        >
          {designLabel} — {nTreatments}T × {nReps}R (plot {plotWidth}×{plotLength} m)
        </text>

        {/* Rows */}
        {layout.map((row, rowIdx) => {
          const blockGap = design === "RCBD" ? rowIdx * BLOCK_GAP : 0;
          const y = TITLE_H + SVG_MARGIN + rowIdx * CELL_H + blockGap;

          return (
            <g key={rowIdx}>
              {/* Dashed separator between RCBD blocks */}
              {design === "RCBD" && rowIdx > 0 && (
                <line
                  x1={leftPad}
                  y1={y - BLOCK_GAP / 2}
                  x2={leftPad + nCols * CELL_W}
                  y2={y - BLOCK_GAP / 2}
                  stroke="#D1D5DB"
                  strokeWidth={1}
                  strokeDasharray="5,4"
                />
              )}

              {/* Block label (RCBD only) */}
              {design === "RCBD" && (
                <text
                  x={leftPad - 6}
                  y={y + CELL_H / 2}
                  textAnchor="end"
                  dominantBaseline="middle"
                  fontSize={9}
                  fontWeight="500"
                  fill="#6B7280"
                >
                  Block {rowIdx + 1}
                </text>
              )}

              {/* Plot cells */}
              {row.map((cell, colIdx) => {
                const x = leftPad + colIdx * CELL_W;
                const baseColor = TREATMENT_COLORS[cell.treatmentIndex % TREATMENT_COLORS.length];
                const fillColor = lighten(baseColor, 0.72);
                const rawLabel = treatmentNames[cell.treatmentIndex] ?? `T${cell.treatmentIndex + 1}`;
                const label = rawLabel.length > 9 ? rawLabel.slice(0, 8) + "…" : rawLabel;

                return (
                  <g key={colIdx}>
                    <rect
                      x={x + 2}
                      y={y + 2}
                      width={CELL_W - 4}
                      height={CELL_H - 4}
                      fill={fillColor}
                      stroke={baseColor}
                      strokeWidth={1.5}
                      rx={3}
                    />
                    {/* Plot number — small, top-left corner */}
                    <text x={x + 7} y={y + 14} fontSize={8} fill="#9CA3AF">
                      {cell.plotNumber}
                    </text>
                    {/* Treatment label — centred */}
                    <text
                      x={x + CELL_W / 2}
                      y={y + CELL_H / 2 + 2}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontSize={11}
                      fontWeight="700"
                      fill="#1F2937"
                    >
                      {label}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
    );
  }
);
LayoutSVG.displayName = "LayoutSVG";

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export function FieldLayoutGenerator() {
  const svgRef = useRef<SVGSVGElement>(null);

  // ── Form state ────────────────────────────────────────────────────────────
  const [design, setDesign] = useState<Design>("RCBD");
  const [nTreatments, setNTreatments] = useState(4);
  const [nReps, setNReps] = useState(3);
  const [treatmentNames, setTreatmentNames] = useState<string[]>(["T1", "T2", "T3", "T4"]);
  const [plotWidth, setPlotWidth] = useState(5);
  const [plotLength, setPlotLength] = useState(10);
  const [aisleWidth, setAisleWidth] = useState(1);
  const [seed, setSeed] = useState(42);

  // ── Snapshot — frozen copy of inputs at last Generate click ───────────────
  // Prevents the live summary / SVG from changing mid-edit.
  const [layout, setLayout] = useState<PlotCell[][] | null>(null);
  const [snap, setSnap] = useState({
    design: "RCBD" as Design,
    nTreatments: 4,
    nReps: 3,
    names: ["T1", "T2", "T3", "T4"] as string[],
    plotW: 5,
    plotL: 10,
    aisleW: 1,
    seed: 42,
  });

  // ── Treatment name helpers ────────────────────────────────────────────────
  const handleNTreatmentsChange = (raw: number) => {
    const n = Math.max(2, Math.min(20, raw));
    setNTreatments(n);
    setTreatmentNames((prev) => {
      const copy = [...prev];
      while (copy.length < n) copy.push(`T${copy.length + 1}`);
      return copy.slice(0, n);
    });
  };

  const handleNameChange = (idx: number, value: string) =>
    setTreatmentNames((prev) => {
      const copy = [...prev];
      copy[idx] = value;
      return copy;
    });

  // ── Validation ────────────────────────────────────────────────────────────
  const errors: string[] = [];
  if (nTreatments < 2 || nTreatments > 20) errors.push("Treatments must be 2 – 20.");
  if (nReps < 2 || nReps > 10) errors.push("Replications must be 2 – 10.");
  if (plotWidth <= 0) errors.push("Plot width must be greater than 0.");
  if (plotLength <= 0) errors.push("Plot length must be greater than 0.");
  const labelSet = new Set(treatmentNames.map((n) => n.trim()).filter(Boolean));
  if (labelSet.size < nTreatments) errors.push("Treatment labels must be unique and non-empty.");
  const canGenerate = errors.length === 0;

  // ── Field dimension estimates (shown live in summary) ─────────────────────
  const totalPlots = nTreatments * nReps;
  const crdCols = Math.ceil(Math.sqrt(totalPlots));
  const crdRows = Math.ceil(totalPlots / crdCols);
  const fieldW =
    design === "RCBD"
      ? nTreatments * plotWidth + Math.max(0, nTreatments - 1) * aisleWidth
      : crdCols * plotWidth + Math.max(0, crdCols - 1) * aisleWidth;
  const fieldL =
    design === "RCBD"
      ? nReps * plotLength + Math.max(0, nReps - 1) * aisleWidth
      : crdRows * plotLength + Math.max(0, crdRows - 1) * aisleWidth;

  // ── Handlers ──────────────────────────────────────────────────────────────
  const handleGenerate = () => {
    if (!canGenerate) return;
    setLayout(generateLayout(design, nTreatments, nReps, seed));
    setSnap({
      design,
      nTreatments,
      nReps,
      names: [...treatmentNames],
      plotW: plotWidth,
      plotL: plotLength,
      aisleW: aisleWidth,
      seed,
    });
  };

  const handleExportPNG = useCallback(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;
    const svgString = new XMLSerializer().serializeToString(svgEl);
    const blob = new Blob([svgString], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const svgW = parseInt(svgEl.getAttribute("width") ?? "800", 10);
    const svgH = parseInt(svgEl.getAttribute("height") ?? "600", 10);
    const scale = 2; // 2× for retina-quality PNG
    const img = new window.Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = svgW * scale;
      canvas.height = svgH * scale;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      const link = document.createElement("a");
      link.download = `field-layout-${snap.design.toLowerCase()}-seed${snap.seed}.png`;
      link.href = canvas.toDataURL("image/png");
      link.click();
    };
    img.src = url;
  }, [snap.design, snap.seed]);

  const handleExportCSV = useCallback(() => {
    if (!layout) return;

    function csvCell(value: string): string {
      return /[",\n\r]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
    }
    function row(...cells: string[]): string {
      return cells.map(csvCell).join(",");
    }

    const lines: string[] = [
      // Metadata rows — preserved as the first lines so the seed is always
      // recoverable from the file without parsing the filename.
      row(`# VivaSense Field Layout Data Collection Sheet`),
      row(`# Design: ${snap.design}`, `Treatments: ${snap.nTreatments}`, `Replications: ${snap.nReps}`, `Seed: ${snap.seed}`),
      row(`# Plot size: ${snap.plotW} x ${snap.plotL} m`),
      row(), // blank separator
      // Column headers
      row(
        "Plot No.", "Replication", "Treatment Code", "Treatment Name",
        "Row", "Column", "Plot Width", "Plot Length", "Unit",
        "Trait 1", "Trait 2", "Trait 3", "Remarks",
      ),
    ];

    // For CRD there is no block structure — derive rep number as the
    // sequential occurrence of each treatment across the global grid.
    const repCount: Record<number, number> = {};

    layout.forEach((rowCells, rowIdx) => {
      rowCells.forEach((cell, colIdx) => {
        let replication: number;
        if (snap.design === "RCBD") {
          replication = cell.blockNumber;
        } else {
          repCount[cell.treatmentIndex] = (repCount[cell.treatmentIndex] ?? 0) + 1;
          replication = repCount[cell.treatmentIndex];
        }
        lines.push(
          row(
            String(cell.plotNumber),
            String(replication),
            `T${cell.treatmentIndex + 1}`,
            snap.names[cell.treatmentIndex] ?? `T${cell.treatmentIndex + 1}`,
            String(rowIdx + 1),
            String(colIdx + 1),
            String(snap.plotW),
            String(snap.plotL),
            "m",
            "", "", "", "", // Trait 1, Trait 2, Trait 3, Remarks — left blank for field entry
          ),
        );
      });
    });

    const csv = lines.join("\r\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.download = `VivaSense_DataCollection_${snap.design}_${snap.nTreatments}T_${snap.nReps}R_Seed${snap.seed}.csv`;
    link.href = url;
    link.click();
    URL.revokeObjectURL(url);
  }, [layout, snap]);

  // ─────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">

      {/* ── Design + Seed ─────────────────────────────────────────────────── */}
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">
            Experimental Design
          </label>
          <div className="flex gap-2">
            {(["CRD", "RCBD"] as Design[]).map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setDesign(d)}
                className={[
                  "flex-1 rounded-lg border px-4 py-2 text-sm font-medium transition-colors",
                  design === d
                    ? "border-emerald-600 bg-emerald-600 text-white"
                    : "border-gray-300 bg-white text-gray-600 hover:border-emerald-400",
                ].join(" ")}
              >
                {d}
              </button>
            ))}
          </div>
          <p className="mt-1 text-xs text-gray-400">
            {design === "CRD"
              ? "All plots globally randomized — no blocking"
              : "Treatments randomized within each replication block"}
          </p>
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">
            Randomization Seed
          </label>
          <input
            type="number"
            min={1}
            value={seed}
            onChange={(e) => setSeed(Math.max(1, parseInt(e.target.value) || 1))}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
          />
          <p className="mt-1 text-xs text-gray-400">
            Same seed always produces the same randomization
          </p>
        </div>
      </div>

      {/* ── Treatment and replication counts ─────────────────────────────── */}
      <div className="grid gap-4 sm:grid-cols-2">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">
            Number of Treatments <span className="text-red-500">*</span>
          </label>
          <input
            type="number"
            min={2}
            max={20}
            value={nTreatments}
            onChange={(e) => handleNTreatmentsChange(parseInt(e.target.value) || 2)}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
          />
          <p className="mt-1 text-xs text-gray-400">2 – 20</p>
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">
            Number of Replications <span className="text-red-500">*</span>
          </label>
          <input
            type="number"
            min={2}
            max={10}
            value={nReps}
            onChange={(e) => setNReps(Math.max(2, Math.min(10, parseInt(e.target.value) || 2)))}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
          />
          <p className="mt-1 text-xs text-gray-400">2 – 10</p>
        </div>
      </div>

      {/* ── Plot dimensions ───────────────────────────────────────────────── */}
      <div className="grid gap-4 sm:grid-cols-3">
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">
            Plot Width (m) <span className="text-red-500">*</span>
          </label>
          <input
            type="number"
            min={0.1}
            step={0.5}
            value={plotWidth}
            onChange={(e) => setPlotWidth(Math.max(0.1, parseFloat(e.target.value) || 0.1))}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
          />
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">
            Plot Length (m) <span className="text-red-500">*</span>
          </label>
          <input
            type="number"
            min={0.1}
            step={0.5}
            value={plotLength}
            onChange={(e) => setPlotLength(Math.max(0.1, parseFloat(e.target.value) || 0.1))}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
          />
        </div>
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-1">
            Aisle Width (m)
          </label>
          <input
            type="number"
            min={0}
            step={0.5}
            value={aisleWidth}
            onChange={(e) => setAisleWidth(Math.max(0, parseFloat(e.target.value) || 0))}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
          />
        </div>
      </div>

      {/* ── Treatment labels ─────────────────────────────────────────────── */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Treatment Labels
        </label>
        <div className="grid gap-2 sm:grid-cols-4">
          {treatmentNames.map((name, i) => (
            <div key={i} className="flex items-center gap-1.5">
              <span
                className="h-3 w-3 shrink-0 rounded-full"
                style={{ backgroundColor: TREATMENT_COLORS[i % TREATMENT_COLORS.length] }}
              />
              <input
                type="text"
                value={name}
                maxLength={12}
                placeholder={`T${i + 1}`}
                onChange={(e) => handleNameChange(i, e.target.value)}
                className="min-w-0 flex-1 rounded border border-gray-300 px-2 py-1 text-xs focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
              />
            </div>
          ))}
        </div>
      </div>

      {/* ── Validation errors ────────────────────────────────────────────── */}
      {errors.length > 0 && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 space-y-1">
          {errors.map((err, i) => (
            <p key={i} className="flex items-center gap-1.5 text-xs text-red-700">
              <span aria-hidden="true">✕</span> {err}
            </p>
          ))}
        </div>
      )}

      {/* ── Live field size estimate ─────────────────────────────────────── */}
      {canGenerate && (
        <p className="text-xs text-gray-400">
          Estimated field: {fieldW.toFixed(1)} m wide × {fieldL.toFixed(1)} m long
          {" "}({totalPlots} plots total)
        </p>
      )}

      {/* ── Generate ─────────────────────────────────────────────────────── */}
      <button
        type="button"
        onClick={handleGenerate}
        disabled={!canGenerate}
        className="w-full rounded-lg bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        Generate Field Layout
      </button>

      {/* ── Results ─────────────────────────────────────────────────────── */}
      {layout !== null && (
        <div className="space-y-5">

          {/* Summary box */}
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
            <p className="text-sm font-semibold text-emerald-800 mb-3">Layout Summary</p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <SummaryItem label="Design" value={snap.design} />
              <SummaryItem label="Treatments" value={String(snap.nTreatments)} />
              <SummaryItem label="Replications" value={String(snap.nReps)} />
              <SummaryItem label="Total Plots" value={String(snap.nTreatments * snap.nReps)} />
              <SummaryItem label="Plot Size" value={`${snap.plotW} × ${snap.plotL} m`} />
              <SummaryItem label="Field Width" value={`${fieldW.toFixed(1)} m`} />
              <SummaryItem label="Field Length" value={`${fieldL.toFixed(1)} m`} />
              <SummaryItem label="Seed" value={String(snap.seed)} />
            </div>
          </div>

          {/* SVG preview */}
          <div className="overflow-x-auto rounded-lg border border-gray-200 bg-white p-3">
            <LayoutSVG
              ref={svgRef}
              layout={layout}
              design={snap.design}
              treatmentNames={snap.names}
              nTreatments={snap.nTreatments}
              nReps={snap.nReps}
              plotWidth={snap.plotW}
              plotLength={snap.plotL}
            />
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-2">
            {snap.names.map((name, i) => (
              <span
                key={i}
                className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium"
                style={{
                  backgroundColor: lighten(TREATMENT_COLORS[i % TREATMENT_COLORS.length], 0.72),
                  color: "#1a1a1a",
                  border: `1.5px solid ${TREATMENT_COLORS[i % TREATMENT_COLORS.length]}`,
                }}
              >
                {name}
              </span>
            ))}
          </div>

          {/* Export buttons */}
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={handleExportPNG}
              className="inline-flex items-center gap-2 rounded-lg border border-emerald-600 px-5 py-2 text-sm font-medium text-emerald-700 hover:bg-emerald-50 transition-colors"
            >
              <span aria-hidden="true">↓</span> Export as PNG
            </button>
            <button
              type="button"
              onClick={handleExportCSV}
              className="inline-flex items-center gap-2 rounded-lg border border-emerald-600 bg-emerald-600 px-5 py-2 text-sm font-medium text-white hover:bg-emerald-700 transition-colors"
            >
              <span aria-hidden="true">↓</span> Download Data Collection Excel
            </button>
          </div>

        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function SummaryItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-sm font-semibold text-gray-800">{value}</p>
    </div>
  );
}
