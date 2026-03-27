import React, { useState } from "react";
import {
  UploadAnalysisResponse,
  SummaryTableRow,
  TraitResult,
} from "@/services/geneticsUploadApi";

interface ResultsDisplayProps {
  results: UploadAnalysisResponse;
  onReset: () => void;
}

export function ResultsDisplay({ results, onReset }: ResultsDisplayProps) {
  const { summary_table, dataset_summary, failed_traits } = results;
  const successCount = summary_table.filter((r) => r.status === "success").length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-800">Analysis Complete</h3>
          <p className="text-sm text-gray-500">
            {successCount} of {summary_table.length} trait
            {summary_table.length !== 1 ? "s" : ""} analysed successfully
            {" · "}
            {dataset_summary.n_genotypes} genotypes
            {dataset_summary.n_environments
              ? ` · ${dataset_summary.n_environments} environments`
              : ` · ${dataset_summary.n_reps} reps`}
          </p>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="shrink-0 rounded-lg border border-gray-300 px-4 py-1.5 text-sm text-gray-600 hover:bg-gray-50"
        >
          New Upload
        </button>
      </div>

      {/* Failed traits warning */}
      {failed_traits.length > 0 && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          <p className="font-medium">Failed traits: {failed_traits.join(", ")}</p>
          <p className="mt-0.5 text-red-500">
            These traits had insufficient data or errors. See details below.
          </p>
        </div>
      )}

      {/* Summary table */}
      <div className="overflow-x-auto rounded-xl border border-gray-200">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <Th>Trait</Th>
              <Th>Mean</Th>
              <Th>H² (Heritability)</Th>
              <Th>GCV %</Th>
              <Th>PCV %</Th>
              <Th>GAM %</Th>
              <Th>Class</Th>
              <Th>Details</Th>
            </tr>
          </thead>
          <tbody>
            {summary_table.map((row, i) => (
              <SummaryRow key={row.trait} row={row} isEven={i % 2 === 0} traitResult={results.trait_results[row.trait]} />
            ))}
          </tbody>
        </table>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-xs text-gray-500">
        <span className="font-medium text-gray-600">Heritability class:</span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-500" /> High ≥ 0.60
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-yellow-400" /> Moderate 0.30–0.59
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-red-400" /> Low &lt; 0.30
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary row with expandable details
// ─────────────────────────────────────────────────────────────────────────────

function SummaryRow({
  row,
  isEven,
  traitResult,
}: {
  row: SummaryTableRow;
  isEven: boolean;
  // TraitResult is always present (backend emits it for both success and failure).
  traitResult: TraitResult | undefined;
}) {
  const [expanded, setExpanded] = useState(false);

  const bg = isEven ? "bg-white" : "bg-gray-50/50";

  if (row.status === "failed") {
    const errorMsg = row.error ?? traitResult?.error ?? "unknown error";
    return (
      <tr className={bg}>
        <td className="px-4 py-3 font-medium text-gray-700">{row.trait}</td>
        <td colSpan={7} className="px-4 py-3">
          <p className="text-xs font-semibold text-red-600 mb-0.5">Failed</p>
          <p className="font-mono text-xs text-red-500 break-all whitespace-pre-wrap">{errorMsg}</p>
        </td>
      </tr>
    );
  }

  // Only show the expand button when analysis_result is present.
  const hasDetails = traitResult?.analysis_result != null;

  return (
    <>
      <tr className={bg}>
        <td className="px-4 py-3 font-medium text-gray-800">{row.trait}</td>
        <Td>{row.grand_mean != null ? row.grand_mean.toFixed(2) : "—"}</Td>
        <Td>
          <HeritabilityCell h2={row.h2} />
        </Td>
        <Td>{row.gcv != null ? row.gcv.toFixed(1) : "—"}</Td>
        <Td>{row.pcv != null ? row.pcv.toFixed(1) : "—"}</Td>
        <Td>{row.gam_percent != null ? row.gam_percent.toFixed(1) : "—"}</Td>
        <td className="px-4 py-3">
          <ClassBadge cls={row.heritability_class} />
        </td>
        <td className="px-4 py-3">
          {hasDetails && (
            <button
              type="button"
              onClick={() => setExpanded((p) => !p)}
              className="text-xs text-emerald-600 hover:underline"
            >
              {expanded ? "Hide ▲" : "Show ▼"}
            </button>
          )}
        </td>
      </tr>
      {expanded && traitResult && (
        <tr className={bg}>
          <td colSpan={8} className="px-6 pb-4">
            <TraitDetails traitResult={traitResult} />
          </td>
        </tr>
      )}
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Expanded trait detail
// ─────────────────────────────────────────────────────────────────────────────

function TraitDetails({ traitResult }: { traitResult: TraitResult }) {
  // analysis_result is the GeneticsResponse — same shape as POST /genetics/analyze.
  // It is nested one level down inside TraitResult, not the TraitResult itself.
  const ar = traitResult.analysis_result;
  const vc = ar?.result?.variance_components;
  const interpretation = ar?.interpretation;
  const warnings = traitResult.data_warnings;

  return (
    <div className="mt-2 space-y-3 text-sm">
      {/* Balance / structure warnings surfaced by the backend */}
      {warnings.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 space-y-0.5">
          <p className="text-xs font-semibold text-amber-700">Data warnings</p>
          {warnings.map((w) => (
            <p key={w} className="text-xs text-amber-600 flex items-start gap-1">
              <span className="mt-0.5 shrink-0">⚠</span> {w}
            </p>
          ))}
        </div>
      )}

      {/* Variance components grid */}
      {vc && (
        <div className="grid gap-2 sm:grid-cols-3">
          {Object.entries(vc)
            .filter(([, v]) => typeof v === "number" && v !== null)
            .map(([key, val]) => (
              <div key={key} className="rounded-lg bg-gray-100 px-3 py-2">
                <p className="text-xs text-gray-500 font-mono">{key}</p>
                <p className="font-semibold text-gray-800">{(val as number).toFixed(4)}</p>
              </div>
            ))}
        </div>
      )}

      {/* Interpretation paragraph from R engine */}
      {interpretation && (
        <div className="rounded-lg bg-emerald-50 border border-emerald-100 p-3">
          <p className="text-xs font-semibold text-emerald-700 mb-1">Interpretation</p>
          <p className="text-gray-700 leading-relaxed whitespace-pre-line">{interpretation}</p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide whitespace-nowrap">
      {children}
    </th>
  );
}

function Td({ children }: { children: React.ReactNode }) {
  return <td className="px-4 py-3 text-gray-700 whitespace-nowrap">{children}</td>;
}

function HeritabilityCell({ h2 }: { h2?: number }) {
  if (h2 == null) return <span className="text-gray-300">—</span>;
  const color =
    h2 >= 0.6 ? "text-emerald-700" : h2 >= 0.3 ? "text-yellow-700" : "text-red-600";
  return <span className={`font-semibold ${color}`}>{h2.toFixed(3)}</span>;
}

function ClassBadge({ cls }: { cls?: string }) {
  if (!cls) return null;
  const styles: Record<string, string> = {
    high: "bg-emerald-100 text-emerald-700 border-emerald-200",
    moderate: "bg-yellow-100 text-yellow-700 border-yellow-200",
    low: "bg-red-100 text-red-600 border-red-200",
  };
  return (
    <span
      className={`inline-block rounded-full border px-2 py-0.5 text-xs font-medium capitalize ${styles[cls] ?? "bg-gray-100 text-gray-600"}`}
    >
      {cls}
    </span>
  );
}
