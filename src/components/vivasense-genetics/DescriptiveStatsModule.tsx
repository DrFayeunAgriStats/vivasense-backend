/**
 * DescriptiveStatsModule
 * ======================
 * Module-based descriptive statistics analysis.
 *
 * Pipeline:
 *   1. Receives datasetContext (incl. dataset_token from POST /upload/dataset)
 *   2. User selects traits from availableTraitColumns
 *   3. On submit → POST /analysis/descriptive-stats { dataset_token, trait_columns }
 *   4. Renders summary table + interpretation + recommendation
 *
 * Guardrails:
 *   - Run button disabled when token is null, traits unselected, or loading
 *   - Friendly validation messages surfaced to user
 *   - Backend errors shown clearly; never silently swallowed
 */

import React, { useState } from "react";
import {
  runDescriptiveStats,
  DescriptiveStatsResponse,
  TraitDescriptiveResult,
  UploadDatasetContext,
} from "@/services/geneticsUploadApi";

interface DescriptiveStatsModuleProps {
  datasetContext: UploadDatasetContext | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function fmt(val: number | null | undefined, decimals = 3): string {
  if (val == null) return "—";
  return val.toFixed(decimals);
}

function precisionBadge(cls: string) {
  const map: Record<string, string> = {
    excellent: "bg-emerald-100 text-emerald-800",
    good: "bg-green-100 text-green-800",
    moderate: "bg-yellow-100 text-yellow-800",
    low: "bg-red-100 text-red-800",
    poor: "bg-red-100 text-red-800",
  };
  const color = map[cls.toLowerCase()] ?? "bg-gray-100 text-gray-600";
  return (
    <span className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${color}`}>
      {cls}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait row (expandable)
// ─────────────────────────────────────────────────────────────────────────────

function TraitRow({
  row,
  isCaution,
}: {
  row: TraitDescriptiveResult;
  isCaution: boolean;
}) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <tr
        className={`cursor-pointer hover:bg-gray-50 transition-colors ${isCaution ? "bg-amber-50/40" : ""}`}
        onClick={() => setOpen((o) => !o)}
      >
        <td className="px-4 py-2.5 font-medium text-gray-800 text-sm">
          <span className="mr-1 text-gray-400 text-xs">{open ? "▼" : "▶"}</span>
          {row.trait}
        </td>
        <td className="px-4 py-2.5 text-sm text-gray-600 text-right">{row.n}</td>
        <td className="px-4 py-2.5 text-sm text-gray-600 text-right">{fmt(row.mean)}</td>
        <td className="px-4 py-2.5 text-sm text-gray-600 text-right">{fmt(row.sd)}</td>
        <td className="px-4 py-2.5 text-sm text-gray-600 text-right">
          {row.cv_percent != null ? `${fmt(row.cv_percent, 1)}%` : "—"}
        </td>
        <td className="px-4 py-2.5 text-sm text-gray-600 text-right">{fmt(row.median)}</td>
        <td className="px-4 py-2.5 text-sm text-right">{precisionBadge(row.precision_class)}</td>
        <td className="px-4 py-2.5 text-sm text-gray-500 text-right">
          {row.missing_count > 0 && (
            <span className="text-amber-600 font-medium">{row.missing_count}</span>
          )}
          {row.missing_count === 0 && <span className="text-gray-300">0</span>}
        </td>
      </tr>

      {open && (
        <tr>
          <td colSpan={8} className="px-4 pb-4 pt-1 bg-gray-50/70">
            <div className="rounded-lg border border-gray-200 bg-white p-4 space-y-3">
              {/* Full stats */}
              <div>
                <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
                  Full Statistics
                </p>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                  {[
                    ["Min", fmt(row.minimum)],
                    ["Max", fmt(row.maximum)],
                    ["Skewness", fmt(row.skewness, 3)],
                    ["Kurtosis", fmt(row.kurtosis, 3)],
                    ["Missing", String(row.missing_count)],
                    ["Zeros", String(row.zero_count)],
                  ].map(([label, val]) => (
                    <div key={label} className="rounded bg-gray-50 border border-gray-100 px-2 py-1.5">
                      <p className="text-gray-400 text-[10px] uppercase">{label}</p>
                      <p className="font-medium text-gray-700">{val}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Flags */}
              {row.flags.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
                    Flags
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {row.flags.map((f) => (
                      <span
                        key={f}
                        className="inline-block rounded-full bg-amber-100 text-amber-800 text-xs px-2 py-0.5"
                      >
                        {f}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Interpretation */}
              {row.interpretation && (
                <div className="rounded-lg bg-emerald-50 border border-emerald-100 p-3">
                  <p className="text-xs font-semibold text-emerald-700 mb-1">Interpretation</p>
                  <p className="text-sm text-gray-700 leading-relaxed">{row.interpretation}</p>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Results panel
// ─────────────────────────────────────────────────────────────────────────────

function ResultsPanel({
  results,
  onReset,
}: {
  results: DescriptiveStatsResponse;
  onReset: () => void;
}) {
  const cautionSet = new Set(results.caution_traits);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-gray-800">Descriptive Statistics</h3>
          <p className="text-sm text-gray-500">
            {results.overview.n_traits} trait{results.overview.n_traits !== 1 ? "s" : ""} ·{" "}
            {results.overview.n_observations} observations
          </p>
        </div>
        <button
          onClick={onReset}
          className="text-xs text-gray-500 hover:text-gray-700 underline"
        >
          Run again
        </button>
      </div>

      {/* Reliable / Caution */}
      <div className="grid sm:grid-cols-2 gap-3">
        {results.reliable_traits.length > 0 && (
          <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3">
            <p className="text-xs font-semibold text-emerald-700 mb-2">Reliable Traits (CV &lt; 20%)</p>
            <div className="flex flex-wrap gap-1">
              {results.reliable_traits.map((t) => (
                <span key={t} className="text-xs bg-emerald-100 text-emerald-800 rounded-full px-2 py-0.5">
                  {t}
                </span>
              ))}
            </div>
          </div>
        )}
        {results.caution_traits.length > 0 && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
            <p className="text-xs font-semibold text-amber-700 mb-2">Use with Caution (CV ≥ 20%)</p>
            <div className="flex flex-wrap gap-1">
              {results.caution_traits.map((t) => (
                <span key={t} className="text-xs bg-amber-100 text-amber-800 rounded-full px-2 py-0.5">
                  {t}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Summary table */}
      <div className="rounded-lg border border-gray-200 overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              {["Trait", "N", "Mean", "SD", "CV%", "Median", "Precision", "Missing"].map(
                (h) => (
                  <th
                    key={h}
                    className="px-4 py-2.5 text-left text-xs font-semibold text-gray-600 uppercase tracking-wide"
                  >
                    {h}
                  </th>
                )
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {results.summary_table.map((row) => (
              <TraitRow key={row.trait} row={row} isCaution={cautionSet.has(row.trait)} />
            ))}
          </tbody>
        </table>
      </div>

      {/* Global flags */}
      {results.global_flags.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
          <p className="text-sm font-semibold text-amber-800 mb-2">Global Flags</p>
          <ul className="space-y-1">
            {results.global_flags.map((f, i) => (
              <li key={i} className="text-sm text-amber-700 flex items-start gap-1.5">
                <span className="mt-0.5 shrink-0">⚠</span>
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Recommendation */}
      {results.recommendation && (
        <div className="rounded-lg border border-blue-200 bg-blue-50 p-4">
          <p className="text-sm font-semibold text-blue-800 mb-1">Recommendation</p>
          <p className="text-sm text-blue-700 leading-relaxed">{results.recommendation}</p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export function DescriptiveStatsModule({ datasetContext }: DescriptiveStatsModuleProps) {
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<DescriptiveStatsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const token = datasetContext?.datasetToken ?? null;
  const availableTraits = datasetContext?.availableTraitColumns ?? [];

  const canRun = token !== null && selectedTraits.length > 0 && !loading;

  const toggleTrait = (t: string) =>
    setSelectedTraits((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]
    );

  const selectAll = () => setSelectedTraits([...availableTraits]);
  const clearAll = () => setSelectedTraits([]);

  const handleRun = async () => {
    if (!canRun) return;

    // Runtime guard: token must be present before the request is sent.
    // This catches any case where the run button is triggered programmatically
    // before dataset confirmation has completed.
    if (!token) {
      setError(
        "Preview completed, but dataset confirmation is still required. " +
        "Re-upload your file and confirm the column mapping to receive a dataset token."
      );
      return;
    }

    console.log(
      "[DescriptiveStats] sending request — dataset_token:", token,
      "| traits:", selectedTraits,
    );
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const data = await runDescriptiveStats({
        dataset_token: token,
        trait_columns: selectedTraits,
      });
      console.log("[DescriptiveStats] response received — n_traits:", data.overview.n_traits);
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Descriptive statistics failed");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResults(null);
    setError(null);
  };

  // ── Case 1: no dataset confirmed yet ──────────────────────────────────────
  if (!datasetContext) {
    return (
      <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 p-8 text-center">
        <p className="text-sm font-medium text-gray-600">No dataset loaded</p>
        <p className="mt-1 text-xs text-gray-400">
          Upload a file in the Multi-Trait File tab to get started.
        </p>
      </div>
    );
  }

  // ── Case 2: dataset loaded but token missing (upload/dataset failed) ───────
  if (token === null) {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-6">
        <p className="text-sm font-semibold text-amber-800">Dataset confirmation required</p>
        <p className="mt-1 text-sm text-amber-700">
          Please confirm dataset before running analysis. Re-upload your file and confirm
          the column mapping to receive a dataset token.
        </p>
      </div>
    );
  }

  // ── Case 3: results ready ─────────────────────────────────────────────────
  if (results) {
    return <ResultsPanel results={results} onReset={handleReset} />;
  }

  // ── Case 4: setup UI ──────────────────────────────────────────────────────
  return (
    <div className="space-y-5">
      {/* Dataset info badge */}
      <div className="flex items-center gap-2 rounded-lg bg-emerald-50 border border-emerald-200 px-3 py-2">
        <span className="h-2 w-2 rounded-full bg-emerald-500 shrink-0" />
        <p className="text-xs text-emerald-700">
          Dataset confirmed ·{" "}
          <span className="font-medium">{datasetContext.file.name}</span>{" "}
          · {availableTraits.length} numeric columns available
        </p>
      </div>

      {/* Trait selection */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-semibold text-gray-700">
            Select traits to analyse
          </label>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={selectAll}
              className="text-xs text-emerald-600 hover:underline"
            >
              Select all
            </button>
            <span className="text-gray-300">|</span>
            <button
              type="button"
              onClick={clearAll}
              className="text-xs text-gray-500 hover:underline"
            >
              Clear
            </button>
          </div>
        </div>

        {availableTraits.length === 0 ? (
          <p className="text-sm text-gray-400 italic">No numeric trait columns detected.</p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {availableTraits.map((t) => {
              const selected = selectedTraits.includes(t);
              return (
                <button
                  key={t}
                  type="button"
                  onClick={() => toggleTrait(t)}
                  className={[
                    "rounded-full border px-3 py-1 text-sm font-medium transition-colors",
                    selected
                      ? "bg-emerald-600 border-emerald-600 text-white"
                      : "bg-white border-gray-300 text-gray-600 hover:border-emerald-400 hover:text-emerald-700",
                  ].join(" ")}
                >
                  {t}
                </button>
              );
            })}
          </div>
        )}

        {selectedTraits.length > 0 && (
          <p className="mt-1.5 text-xs text-gray-400">
            {selectedTraits.length} trait{selectedTraits.length !== 1 ? "s" : ""} selected
          </p>
        )}
      </div>

      {/* Validation hint */}
      {selectedTraits.length === 0 && (
        <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
          Select at least one trait to run descriptive statistics.
        </p>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3">
          <p className="text-sm font-medium text-red-700">Analysis failed</p>
          <p className="mt-0.5 text-xs font-mono text-red-600 break-all">
            {error.replace(/^Descriptive statistics failed — /, "")}
          </p>
        </div>
      )}

      {/* Run button */}
      <button
        type="button"
        onClick={handleRun}
        disabled={!canRun}
        className={[
          "w-full rounded-lg px-4 py-2.5 text-sm font-semibold transition-colors",
          canRun
            ? "bg-emerald-600 text-white hover:bg-emerald-700"
            : "bg-gray-100 text-gray-400 cursor-not-allowed",
        ].join(" ")}
      >
        {loading ? "Running…" : "Run Descriptive Statistics"}
      </button>
    </div>
  );
}
