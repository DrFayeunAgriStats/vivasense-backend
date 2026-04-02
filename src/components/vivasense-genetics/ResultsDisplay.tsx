import React, { useState } from "react";
import {
  AnovaTable,
  MeanSeparation,
  UploadAnalysisResponse,
  SummaryTableRow,
  TraitResult,
  exportWordReport,
} from "@/services/geneticsUploadApi";

interface ResultsDisplayProps {
  results: UploadAnalysisResponse;
  onReset: () => void;
}

// Human-readable labels for ANOVA source terms produced by R
const ANOVA_LABELS: Record<string, string> = {
  rep: "Replication",
  genotype: "Genotype",
  environment: "Environment",
  "environment:rep": "Rep(Environment)",
  "genotype:environment": "G×E Interaction",
  Residuals: "Error",
};

function fmtP(p: number | null): { text: string; stars: string } {
  if (p === null) return { text: "—", stars: "" };
  if (p < 0.001) return { text: p.toExponential(2), stars: "***" };
  if (p < 0.01) return { text: p.toFixed(4), stars: "**" };
  if (p < 0.05) return { text: p.toFixed(4), stars: "*" };
  return { text: p.toFixed(4), stars: "ns" };
}

export function ResultsDisplay({ results, onReset }: ResultsDisplayProps) {
  const { summary_table, dataset_summary, failed_traits } = results;
  const successCount = summary_table.filter((r) => r.status === "success").length;
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const handleDownload = async () => {
    setDownloading(true);
    setDownloadError(null);
    try {
      await exportWordReport(results);
    } catch (err) {
      setDownloadError(err instanceof Error ? err.message : "Download failed");
    } finally {
      setDownloading(false);
    }
  };

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
        <div className="flex gap-2 shrink-0">
          <button
            type="button"
            onClick={handleDownload}
            disabled={downloading || successCount === 0}
            className="rounded-lg border border-emerald-600 bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {downloading ? "Generating…" : "Download Report (.docx)"}
          </button>
          <button
            type="button"
            onClick={onReset}
            className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm text-gray-600 hover:bg-gray-50"
          >
            New Upload
          </button>
        </div>
      </div>

      {/* Download error */}
      {downloadError && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          <p className="font-medium">Download failed</p>
          <p className="mt-0.5">{downloadError}</p>
        </div>
      )}

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
              <SummaryRow
                key={row.trait}
                row={row}
                isEven={i % 2 === 0}
                traitResult={results.trait_results[row.trait]}
              />
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
          <td colSpan={8} className="px-6 pb-5 pt-1">
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
  const ar = traitResult.analysis_result;
  const result = ar?.result;
  const warnings = traitResult.data_warnings;

  return (
    <div className="mt-2 space-y-4 text-sm">
      {/* Balance / structure warnings */}
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

      {/* ANOVA Table */}
      {result?.anova_table && (
        <AnovaTableSection at={result.anova_table} />
      )}

      {/* Mean Separation */}
      {result?.mean_separation && (
        <MeanSeparationSection ms={result.mean_separation} />
      )}

      {/* Variance components grid */}
      {result?.variance_components && (
        <div>
          <p className="text-xs font-semibold text-gray-600 mb-1.5">Variance Components</p>
          <div className="grid gap-2 sm:grid-cols-3">
            {Object.entries(result.variance_components)
              .filter(([, v]) => typeof v === "number" && v !== null)
              .map(([key, val]) => (
                <div key={key} className="rounded-lg bg-gray-100 px-3 py-2">
                  <p className="text-xs text-gray-500 font-mono">{key}</p>
                  <p className="font-semibold text-gray-800">{(val as number).toFixed(4)}</p>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Interpretation */}
      {ar?.interpretation && (
        <div className="rounded-lg bg-emerald-50 border border-emerald-100 p-3">
          <p className="text-xs font-semibold text-emerald-700 mb-1">Interpretation</p>
          <p className="text-gray-700 leading-relaxed whitespace-pre-line">{ar.interpretation}</p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ANOVA Table section
// ─────────────────────────────────────────────────────────────────────────────

function AnovaTableSection({ at }: { at: AnovaTable }) {
  return (
    <div>
      <p className="text-xs font-semibold text-gray-600 mb-1.5">ANOVA Table</p>
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full text-xs">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              {["Source", "df", "SS", "MS", "F-value", "P-value"].map((h) => (
                <th key={h} className="px-3 py-2 text-left font-semibold text-gray-500 whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {at.source.map((src, i) => {
              const { text: pText, stars } = fmtP(at.p_value[i]);
              const isError = src === "Residuals";
              return (
                <tr key={src} className={isError ? "bg-gray-50/60" : "bg-white"}>
                  <td className="px-3 py-1.5 font-medium text-gray-700">
                    {ANOVA_LABELS[src] ?? src}
                  </td>
                  <td className="px-3 py-1.5 text-gray-600">{at.df[i]}</td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {at.ss[i] != null ? at.ss[i]!.toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {at.ms[i] != null ? at.ms[i]!.toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {at.f_value[i] != null ? at.f_value[i]!.toFixed(3) : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {pText}
                    {stars && (
                      <span className="ml-1 font-semibold text-emerald-700">{stars}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-1 text-xs text-gray-400">
        *** p&lt;0.001 · ** p&lt;0.01 · * p&lt;0.05 · ns p≥0.05
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Mean Separation section
// ─────────────────────────────────────────────────────────────────────────────

function MeanSeparationSection({ ms }: { ms: MeanSeparation }) {
  return (
    <div>
      <p className="text-xs font-semibold text-gray-600 mb-1.5">
        Mean Separation — {ms.test} (α = {ms.alpha})
      </p>
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full text-xs">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">Rank</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">Genotype</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">Mean</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">SE</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">Group</th>
            </tr>
          </thead>
          <tbody>
            {ms.genotype.map((geno, i) => (
              <tr key={geno} className={i % 2 === 0 ? "bg-white" : "bg-gray-50/50"}>
                <td className="px-3 py-1.5 text-gray-400">{i + 1}</td>
                <td className="px-3 py-1.5 font-medium text-gray-800">{geno}</td>
                <td className="px-3 py-1.5 text-gray-700">{ms.mean[i].toFixed(2)}</td>
                <td className="px-3 py-1.5 text-gray-600">
                  {ms.se[i] != null ? ms.se[i]!.toFixed(2) : "—"}
                </td>
                <td className="px-3 py-1.5">
                  <span className="inline-block rounded bg-emerald-100 px-1.5 py-0.5 font-mono text-emerald-800 font-semibold">
                    {ms.group[i]}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-1 text-xs text-gray-400">
        Means sharing the same letter are not significantly different (Tukey HSD).
      </p>
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
