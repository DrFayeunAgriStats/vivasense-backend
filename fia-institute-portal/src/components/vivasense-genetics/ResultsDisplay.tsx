import React, { useEffect, useMemo, useState } from "react";
import {
  AnovaTable,
  MeanSeparation,
  InteractionMeansData,
  UploadAnalysisResponse,
  SummaryTableRow,
  TraitResult,
  exportWordReport,
} from "@/services/geneticsUploadApi";
import { DomainKey, getDomainTerms } from "./domainTerms";
import { AcademicInterpretationPanel } from "./AcademicInterpretationPanel";
import { WordExportPreviewModal } from "@/components/export/WordExportPreviewModal";
import { buildScientificInterpretationSections, downloadCsv } from "@/components/vivasense/advanced/shared";
import { safeArray, logDebug, safeNumber } from "@/utils/normalizeModuleData";
import type { PreviewSection } from "@/utils/normalizeModuleData";
import {
  getVivaSenseMode,
  ProFeatureError,
  VIVASENSE_MODE_CHANGED_EVENT,
  VivaSenseMode,
} from "@/services/featureMode";
import { ProFeatureModal } from "./ProFeatureModal";
import {
  DEFAULT_SELECTION_INTENSITY,
  selectionIntensityDisclosure,
} from "./selectionIntensity";
import { SplitPlotResults } from "./SplitPlotResults";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface ResultsDisplayProps {
  results: UploadAnalysisResponse;
  onReset: () => void;
  domain?: DomainKey;
  module?: "anova" | "genetic_parameters";
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

const VARIANCE_SYMBOL_MAP: Record<string, { symbol: string; label: string }> = {
  sigma2_genotype: { symbol: "σ²g", label: "Genotypic variance" },
  sigma2_ge: { symbol: "σ²ge", label: "GxE interaction variance" },
  sigma2_gxe: { symbol: "σ²ge", label: "GxE interaction variance" },
  sigma2_error: { symbol: "σ²e", label: "Error variance" },
  sigma2_phenotypic: { symbol: "σ²p", label: "Phenotypic variance" },
  sigma2_phenotype: { symbol: "σ²p", label: "Phenotypic variance" },
  h2: { symbol: "H²", label: "Heritability, broad-sense" },
  gcv: { symbol: "GCV%", label: "Genotypic coefficient of variation" },
  pcv: { symbol: "PCV%", label: "Phenotypic coefficient of variation" },
  ga: { symbol: "GA", label: "Genetic advance" },
  gam: { symbol: "GAM%", label: "Genetic advance as % of mean" },
};

const SPLIT_PLOT_VARIANCE_PARAMS: Array<{ key: string; label: string; fmt: number }> = [
  { key: "sigma2_whole_plot_error", label: "Whole-plot variance (σ²)", fmt: 4 },
  { key: "sigma2_subplot_error",    label: "Subplot variance (σ²)",    fmt: 4 },
  { key: "cv_A",                    label: "CV-A — Main-plot (%)",     fmt: 2 },
  { key: "cv_B",                    label: "CV-B — Subplot (%)",       fmt: 2 },
  { key: "n_main_plot_levels",      label: "Main-plot levels",         fmt: 0 },
  { key: "n_sub_plot_levels",       label: "Subplot levels",           fmt: 0 },
];

function fmtP(p: number | null): { text: string; stars: string } {
  if (p === null) return { text: "—", stars: "" };
  if (p < 0.001) return { text: p.toExponential(2), stars: "***" };
  if (p < 0.01) return { text: p.toFixed(4), stars: "**" };
  if (p < 0.05) return { text: p.toFixed(4), stars: "*" };
  return { text: p.toFixed(4), stars: "ns" };
}

function fmtSafeFixed(val: any, digits: number): string {
  return typeof val === "number" && !isNaN(val) ? val.toFixed(digits) : "—";
}

function buildReportPreview(results: UploadAnalysisResponse): {
  sections: PreviewSection[];
  warnings: string[];
  notes: string[];
} {
  const { summary_table, dataset_summary, failed_traits, trait_results } = results;
  const sections: PreviewSection[] = [];
  const warnings: string[] = [];

  // Defensive: ensure summary_table is array
  const safeSummary = safeArray<SummaryTableRow>(summary_table);
  const successRows = safeSummary.filter((r: SummaryTableRow) => r?.status === "success");
  const failedTraits = safeArray<string>(failed_traits);

  logDebug("ResultsDisplay:preview", {
    total_traits: safeSummary.length,
    success_traits: successRows.length,
    failed_traits: failedTraits.length,
    has_dataset_summary: !!dataset_summary,
  });

  // Dataset overview section — defensive number formatting
  const nGenotypes = safeNumber(dataset_summary?.n_genotypes);
  const nReps = safeNumber(dataset_summary?.n_reps);
  const nEnvironments = safeNumber(dataset_summary?.n_environments);
  const nTraits = safeNumber(dataset_summary?.n_traits);

  sections.push({
    title: "Dataset Overview",
    rows: [
      { label: "Genotypes", value: String(nGenotypes || "—") },
      { label: "Replicates", value: String(nReps || "—") },
      ...(nEnvironments > 0
        ? [{ label: "Environments", value: String(nEnvironments) }]
        : []),
      { label: "Traits analysed", value: String(nTraits || "—") },
      { label: "Mode", value: String(dataset_summary?.mode ?? "single") },
    ],
  });

  // Per-trait summary — defensive property access
  if (successRows.length > 0) {
    sections.push({
      title: "Trait Summary (Genetic Parameters)",
      rows: successRows.map((r: SummaryTableRow) => ({
        label: String(r?.trait ?? "Unknown"),
        value: [
          typeof r?.grand_mean === "number" && !isNaN(r.grand_mean) ? `Mean: ${fmtSafeFixed(r.grand_mean, 2)}` : null,
          typeof r?.h2 === "number" && !isNaN(r.h2) ? `H²: ${fmtSafeFixed(r.h2, 3)}` : null,
          r?.gam_class ? `(GAM class: ${r.gam_class})` : null,
          typeof r?.gcv === "number" && !isNaN(r.gcv) ? `GCV: ${fmtSafeFixed(r.gcv, 1)}%` : null,
          typeof r?.gam_percent === "number" && !isNaN(r.gam_percent) ? `GAM: ${fmtSafeFixed(r.gam_percent, 1)}%` : null,
        ]
          .filter(Boolean)
          .join("  ·  "),
      })),
      note: "H² = broad-sense heritability; GCV = genotypic coefficient of variation; GAM = genetic advance as % of mean",
    });
  }

  // ANOVA availability per trait — defensive access
  const anovaRows: { label: string; value: string }[] = [];
  for (const row of successRows) {
    if (!row || !row.trait) continue;
    const tr = trait_results?.[row.trait];
    const at = tr?.analysis_result?.result?.anova_table;
    const sources = safeArray<string>(at?.source);
    anovaRows.push({
      label: row.trait,
      value: sources.length > 0 ? `${sources.length} sources (${sources.join(", ")})` : "Not available",
    });
  }
  if (anovaRows.length > 0) {
    sections.push({ title: "ANOVA Table Availability", rows: anovaRows });
  }

  // Warnings
  if (failedTraits.length > 0) {
    warnings.push(`Failed traits (excluded from report): ${failedTraits.join(", ")}`);
  }

  const notes = [
    "The Word report includes full ANOVA tables, mean separation (Tukey HSD), variance components, and interpretation for each trait.",
    "Figures and significance stars follow standard agricultural biometrics conventions.",
  ];

  logDebug("ResultsDisplay:preview", {
    n_sections: sections.length,
    n_warnings: warnings.length,
    success_traits: successRows.length,
  });

  return { sections, warnings, notes };
}

export function ResultsDisplay({
  results,
  onReset,
  domain,
  module = "genetic_parameters",
}: ResultsDisplayProps) {
  const effectiveDomain = domain ?? "plant_breeding";
  const domainTerms = getDomainTerms(domain);
  const isPlantBreeding = effectiveDomain === "plant_breeding";
  const isAgronomy = effectiveDomain === "agronomy";
  const isGeneral = effectiveDomain === "general";
  const isAnovaModule = module === "anova";
  const showH2Column = !isAnovaModule && (isPlantBreeding || module === "genetic_parameters");
  const showBreedingMetrics = !isAnovaModule && isPlantBreeding;
  const showClassColumn = !isAnovaModule && isPlantBreeding;
  const totalSummaryColumns =
    3 +
    (isAnovaModule ? 1 : 0) +
    (showH2Column ? 1 : 0) +
    (showBreedingMetrics ? 3 : 0) +
    (showClassColumn ? 1 : 0);
  const { summary_table, dataset_summary, failed_traits } = results;
  const successCount = summary_table.filter((r: SummaryTableRow) => r.status === "success").length;
  const anovaTypeWarning =
    results.anova_type_warning ??
    summary_table
      .map((row: SummaryTableRow) => results.trait_results[row.trait]?.analysis_result?.anova_type_warning)
      .find((value: unknown): value is string => typeof value === "string" && value.length > 0) ??
    null;
  const detectedSelectionIntensity =
    summary_table
      .map((row: SummaryTableRow) => results.trait_results[row.trait]?.analysis_result?.result?.genetic_parameters?.selection_intensity)
      .find((value: unknown): value is number => typeof value === "number") ?? DEFAULT_SELECTION_INTENSITY;

  const subtitleLine = useMemo(() => {
    if (!isAnovaModule) return null;
    const firstResult = Object.values(results.trait_results)[0]?.analysis_result?.result;
    const isSP = firstResult?.design === "split_plot_rcbd" || firstResult?.main_plot_mean_separation != null;
    if (isSP) {
      const vc = firstResult?.variance_components as Record<string, unknown> | null | undefined;
      const nMain =
        (typeof vc?.n_main_plot_levels === "number" ? vc.n_main_plot_levels as number : undefined) ??
        firstResult?.main_plot_mean_separation?.genotype?.length;
      const nSub =
        (typeof vc?.n_sub_plot_levels === "number" ? vc.n_sub_plot_levels as number : undefined) ??
        firstResult?.mean_separation?.genotype?.length;
      if (nMain && nSub) {
        return `${nMain} main-plot levels × ${nSub} subplot levels · ${dataset_summary.n_reps} reps`;
      }
      return `split-plot design · ${dataset_summary.n_reps} reps`;
    }
    return `${dataset_summary.n_genotypes ?? ""} treatments · ${dataset_summary.n_reps} reps`.trim();
  }, [isAnovaModule, results.trait_results, dataset_summary]);

  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showProModal, setShowProModal] = useState(false);
  const [mode, setMode] = useState<VivaSenseMode>(() => getVivaSenseMode());
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");

  const compiledInterpretation = useMemo(() => {
    return summary_table
      .filter((row: SummaryTableRow) => row.status === "success")
      .map((row: SummaryTableRow) => {
        const interpretation = results.trait_results[row.trait]?.analysis_result?.interpretation?.trim();
        if (!interpretation) return null;
        return `${row.trait}\n${interpretation}`;
      })
      .filter((value): value is string => Boolean(value))
      .join("\n\n");
  }, [results.trait_results, summary_table]);

  useEffect(() => {
    const syncMode = () => setMode(getVivaSenseMode());
    window.addEventListener(VIVASENSE_MODE_CHANGED_EVENT, syncMode);
    window.addEventListener("storage", syncMode);
    return () => {
      window.removeEventListener(VIVASENSE_MODE_CHANGED_EVENT, syncMode);
      window.removeEventListener("storage", syncMode);
    };
  }, []);

  const isPro = mode === "pro";

  const handleDownload = async () => {
    if (!isPro) {
      setShowProModal(true);
      return;
    }
    setDownloading(true);
    setDownloadError(null);
    try {
      await exportWordReport(results, "vivasense_genetics_report.docx", domain);
    } catch (err) {
      if (err instanceof ProFeatureError) {
        setShowProModal(true);
        return;
      }
      setDownloadError(err instanceof Error ? err.message : "Download failed");
    } finally {
      setDownloading(false);
    }
  };

  const handleCopyInterpretation = async () => {
    try {
      await navigator.clipboard.writeText(compiledInterpretation);
      setCopyState("copied");
      window.setTimeout(() => setCopyState("idle"), 2000);
    } catch {
      setCopyState("failed");
      window.setTimeout(() => setCopyState("idle"), 2500);
    }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-3xl border border-gray-200 bg-white p-5 shadow-sm lg:p-6">
        <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-gray-500">Results workspace</p>
            <h3 className="mt-1 text-xl font-semibold text-gray-800">Analysis Complete</h3>
            <p className="mt-1 text-sm text-gray-500 leading-relaxed">
            {successCount} of {summary_table.length} trait
            {summary_table.length !== 1 ? "s" : ""} analysed successfully
            {" · "}
            {subtitleLine ?? (
              <>
                {dataset_summary.n_genotypes} {domainTerms.treatment_plural}
                {dataset_summary.n_environments
                  ? ` · ${dataset_summary.n_environments} environments`
                  : ` · ${dataset_summary.n_reps} reps`}
              </>
            )}
</p>
            <p className="mt-2 text-xs text-gray-500">
              Tables, interpretation blocks, and downloadable exports are formatted for scientific review and reporting.
            </p>
          </div>
          <div className="flex flex-wrap gap-2.5 shrink-0">
            <button
              type="button"
              onClick={() =>
                downloadCsv(
                  "vivasense_trait_summary.csv",
                  summary_table.map((row: SummaryTableRow) => ({
                    trait: row.trait,
                    status: row.status,
                    grand_mean: row.grand_mean ?? null,
                    h2: row.h2 ?? null,
                    gcv: row.gcv ?? null,
                    pcv: row.pcv ?? null,
                    gam_percent: row.gam_percent ?? null,
                    class: row.gam_class ?? null,
                    error: row.error ?? null,
                  })) as unknown as Record<string, unknown>[]
                )
              }
              className="rounded-lg border border-gray-300 px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 transition-colors"
            >
              Export Summary CSV
            </button>
            <button
              type="button"
              onClick={() => setShowPreview(true)}
              disabled={successCount === 0}
              className="rounded-lg border border-gray-300 px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Preview Report
            </button>
            <button
              type="button"
              onClick={handleCopyInterpretation}
              disabled={!compiledInterpretation}
              className="rounded-lg border border-gray-300 px-4 py-2 text-sm text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {copyState === "copied" ? "Interpretation copied" : copyState === "failed" ? "Copy failed" : "Copy Interpretation"}
            </button>
            <button
              type="button"
              onClick={handleDownload}
              disabled={downloading || successCount === 0}
              className="rounded-lg border border-emerald-600 bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {downloading
                ? "Generating…"
                : isPro
                ? "Download Word Report"
                : "🔒 Pro · Download Word Report"}
            </button>
            <button
              type="button"
              onClick={onReset}
              className="rounded-lg border border-gray-300 px-4 py-2 text-sm text-gray-600 hover:bg-gray-50"
            >
              New Upload
            </button>
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-blue-100 bg-blue-50 px-4 py-3 text-xs text-blue-700 leading-relaxed">
        <span className="font-semibold">How this works: </span>
        {isAnovaModule
          ? "VivaSense ANOVA partitions total variation into treatment, block, and error components. Mean separation (Tukey HSD / Fisher LSD) identifies which treatment levels differ significantly — expand any trait row to see full output."
          : (
            <>
              VivaSense Genetics uses ANOVA internally to partition phenotypic variance into genetic,
              environmental, and error components. Heritability (H²) and genetic advance (GAM) are
              derived from those variance components. Mean separation (Tukey HSD) identifies which
              genotypes perform significantly differently — expand any trait row below to see the full
              statistical output.
              <p className="mt-1 text-[11px] text-blue-700">
                {selectionIntensityDisclosure(detectedSelectionIntensity)}
              </p>
            </>
          )
        }
      </div>

      {/* Download error */}
      {downloadError && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          <p className="font-medium">Download failed</p>
          <p className="mt-0.5">{downloadError}</p>
        </div>
      )}

      <div className="rounded-2xl border border-gray-200 bg-gray-50/70 px-4 py-3 text-sm text-gray-600 leading-relaxed">
        <span className="font-semibold text-gray-800">Citation-ready note: </span>
        Report the design, response traits, significance thresholds, and interpretation limits alongside these outputs. VivaSense supports reproducible reporting, but scientific conclusions should remain aligned with accepted statistical and domain-specific principles.
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

      {anovaTypeWarning && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-700">
          <p className="font-medium">
            Note: Type I sums of squares used for this analysis. Install the car package for Type III SS (recommended for unbalanced designs).
          </p>
        </div>
      )}

      {/* ── Word Export Preview Modal ──────────────────────────────────────── */}
      {showPreview && (() => {
        const { sections, warnings, notes } = buildReportPreview(results);
        return (
          <WordExportPreviewModal
            moduleName={isAnovaModule ? "ANOVA" : "ANOVA & Genetic Parameters"}
            reportTitle="Multi-Trait Genetics Report"
            datasetSummary={`${dataset_summary.n_genotypes} ${isAnovaModule ? "treatments" : "genotypes"} · ${dataset_summary.n_traits} traits · ${dataset_summary.mode} environment`}
            sections={sections}
            warnings={warnings}
            notes={notes}
            canExport={successCount > 0}
            onExport={handleDownload}
            onClose={() => setShowPreview(false)}
          />
        );
      })()}

      {/* ── SECTION: Trait Summary ─────────────────────────────────────────── */}
      <div>
        <SectionLabel>Trait Summary</SectionLabel>
        <div className="overflow-x-auto rounded-xl border border-gray-200">
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <Th>Trait</Th>
                <Th>Mean</Th>
                {isAnovaModule ? (
                  <Th>CV %</Th>
                ) : (
                  <>
                    {showH2Column && <Th>{isPlantBreeding ? domainTerms.h2_label : "Repeatability"}</Th>}
                    {showBreedingMetrics && <Th>{domainTerms.gcv_label}</Th>}
                    {showBreedingMetrics && <Th>{domainTerms.pcv_label}</Th>}
                    {showBreedingMetrics && <Th>{domainTerms.gam_label}</Th>}
                    {showClassColumn && <Th>Class</Th>}
                  </>
                )}
                <Th>Details</Th>
              </tr>
            </thead>
            <tbody>
              {summary_table.map((row: SummaryTableRow, i: number) => (
                <SummaryRow
                  key={row.trait}
                  row={row}
                  isEven={i % 2 === 0}
                  isFirst={i === 0}
                  traitResult={results.trait_results[row.trait]}
                  domainTerms={domainTerms}
                  showH2Column={showH2Column}
                  showBreedingMetrics={showBreedingMetrics}
                  showClassColumn={showClassColumn}
                  totalColumns={totalSummaryColumns}
                  module={module}
                />
              ))}
            </tbody>
          </table>
        </div>

        {/* Variance Components / Treatment Variance Summary Tab Label */}
        <div className="mt-4">
          <h4 className="text-base font-semibold text-gray-700">
            {isPlantBreeding ? "Variance Components" : "Treatment Variance Summary"}
          </h4>
        </div>

        {/* Conditional variance components output */}
        {isPlantBreeding && !isAnovaModule ? (
          <BreedingSummaryPanel summary={(results as unknown as { breeding_summary?: string }).breeding_summary ?? ""} />
        ) : (
          <ManagementRecommendationsPanel
            rows={summary_table}
            traitResults={results.trait_results}
            treatmentLabel={domainTerms.treatment}
          />
        )}

        {/* Legend for Plant Breeding only */}
        {isPlantBreeding && (
          <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-500">
            <span className="font-medium text-gray-600">{domainTerms.h2_label} class:</span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-500" /> High ≥ 0.60
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-yellow-400" /> Moderate 0.30–0.59
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-red-400" /> Low &lt; 0.30
            </span>
            <span className="w-full text-[11px] text-gray-500">
              Broad-sense heritability (H²): the proportion of total phenotypic variance attributable to genotypic differences. Estimated as σ²g / σ²p using ANOVA variance components.
            </span>
          </div>
        )}
      </div>

      <ProFeatureModal
        open={showProModal}
        onClose={() => setShowProModal(false)}
        onActivated={() => setMode(getVivaSenseMode())}
      />
    </div>
  );
}

const BreedingSummaryPanel = ({ summary }: { summary: string }) => {
  if (!summary) return null;
  const paras = summary.split("\n\n").filter(Boolean);
  return (
    <div
      style={{
        background: "var(--color-background-secondary)",
        border: "0.5px solid var(--color-border-tertiary)",
        borderLeft: "3px solid #1D9E75",
        borderRadius: 8,
        padding: "1rem 1.25rem",
        marginTop: 16,
      }}
    >
      <div style={{ fontWeight: 500, marginBottom: 12, fontSize: 14 }}>
        Breeding Strategy Summary
      </div>
      {paras.map((p, i) => (
        <p
          key={i}
          style={{
            fontSize: 13,
            lineHeight: 1.7,
            color: "var(--color-text-secondary)",
            marginBottom: i < paras.length - 1 ? 12 : 0,
          }}
        >
          {p}
        </p>
      ))}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Summary row with expandable details
// ─────────────────────────────────────────────────────────────────────────────

function SummaryRow({
  row,
  isEven,
  isFirst,
  traitResult,
  domainTerms,
  showH2Column,
  showBreedingMetrics,
  showClassColumn,
  totalColumns,
  module,
}: {
  row: SummaryTableRow;
  isEven: boolean;
  isFirst: boolean;
  traitResult: TraitResult | undefined;
  domainTerms: ReturnType<typeof getDomainTerms>;
  showH2Column: boolean;
  showBreedingMetrics: boolean;
  showClassColumn: boolean;
  totalColumns: number;
  module?: string;
}) {
  const [expanded, setExpanded] = useState(isFirst);
  const bg = isEven ? "bg-white" : "bg-gray-50/50";
  const isAnovaRow = module === "anova";

  const cvPercent = isAnovaRow
    ? (() => {
        const ds = (traitResult?.analysis_result?.result as unknown as Record<string, unknown> | undefined)
          ?.descriptive_stats as Record<string, unknown> | null | undefined;
        const v = ds?.cv_percent;
        return typeof v === "number" ? v : null;
      })()
    : null;

  // Show first sentence of interpretation as a hint below the trait name when collapsed
  const interpHint = (() => {
    const interp = (traitResult?.analysis_result as unknown as Record<string, unknown> | undefined)
      ?.interpretation as string | undefined;
    if (!interp || expanded) return null;
    const first = interp.split(/\.\s/)[0];
    return first ? `${first}.` : null;
  })();

  if (row.status === "failed") {
    const errorMsg = row.error ?? traitResult?.error ?? "unknown error";
    return (
      <tr className={bg}>
        <td className="px-4 py-3 font-medium text-gray-700">{row.trait}</td>
        <td colSpan={totalColumns - 1} className="px-4 py-3">
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
        <td className="px-4 py-3">
          <p className="font-medium text-gray-800 leading-tight">{row.trait}</p>
          {interpHint && (
            <p className="mt-0.5 text-xs text-gray-400 leading-snug line-clamp-1 max-w-xs">{interpHint}</p>
          )}
        </td>
        <Td>{fmtSafeFixed(row.grand_mean, 2)}</Td>
        {isAnovaRow ? (
          <Td>{cvPercent != null ? `${cvPercent.toFixed(1)}%` : "—"}</Td>
        ) : (
          <>
            {showH2Column && (
              <Td>
                <HeritabilityCell h2={row.h2} />
              </Td>
            )}
            {showBreedingMetrics && <Td>{fmtSafeFixed(row.gcv, 1)}</Td>}
            {showBreedingMetrics && <Td>{fmtSafeFixed(row.pcv, 1)}</Td>}
            {showBreedingMetrics && <Td>{fmtSafeFixed(row.gam_percent, 1)}</Td>}
            {showClassColumn && (
              <td className="px-4 py-3">
                <ClassBadge cls={row.gam_class} />
              </td>
            )}
          </>
        )}
        <td className="px-4 py-3">
          {hasDetails && (
            <button
              type="button"
              onClick={() => setExpanded((p: boolean) => !p)}
              className="text-xs text-emerald-600 hover:underline"
            >
              {expanded ? "Hide ▲" : "Show ▼"}
            </button>
          )}
        </td>
      </tr>
      {expanded && traitResult && (
        <tr className={bg}>
          <td colSpan={totalColumns} className="px-6 pb-5 pt-1">
            <TraitDetails traitResult={traitResult} traitName={row.trait} domainTerms={domainTerms} module={module} />
          </td>
        </tr>
      )}
    </>
  );
}

function ManagementRecommendationsPanel({
  rows,
  traitResults,
  treatmentLabel,
}: {
  rows: SummaryTableRow[];
  traitResults: Record<string, TraitResult>;
  treatmentLabel: string;
}) {
  const recommendations = rows
    .filter((row) => row.status === "success")
    .map((row) => {
      const meanSeparation = traitResults[row.trait]?.analysis_result?.result?.mean_separation;
      if (!meanSeparation || meanSeparation.genotype.length === 0 || meanSeparation.mean.length === 0) {
        return null;
      }

      return {
        trait: row.trait,
        topName: meanSeparation.genotype[0],
        topMean: meanSeparation.mean[0],
        topGroup: meanSeparation.group[0] ?? "—",
        totalRanked: meanSeparation.genotype.length,
      };
    })
    .filter(
      (item): item is {
        trait: string;
        topName: string;
        topMean: number;
        topGroup: string;
        totalRanked: number;
      } => item !== null
    );

  if (recommendations.length === 0) {
    return null;
  }

  return (
    <div
      style={{
        background: "var(--color-background-secondary)",
        border: "0.5px solid var(--color-border-tertiary)",
        borderLeft: "3px solid #1D9E75",
        borderRadius: 8,
        padding: "1rem 1.25rem",
        marginTop: 16,
      }}
    >
      <div style={{ fontWeight: 500, marginBottom: 12, fontSize: 14 }}>
        Management Recommendations
      </div>
      <div className="space-y-3">
        {recommendations.map((item) => (
          <div key={item.trait} className="rounded-lg border border-emerald-100 bg-emerald-50/60 px-3 py-2">
            <p className="text-sm font-medium text-gray-800">{item.trait}</p>
            <p className="mt-1 text-xs text-gray-600">
              Mean separation ranking: 1 of {item.totalRanked}
            </p>
            <p className="mt-1 text-sm text-gray-700">
              Top {treatmentLabel.toLowerCase()}: <span className="font-semibold text-emerald-800">{item.topName}</span>
              {" · "}
              Mean {fmtSafeFixed(item.topMean, 2)}
              {" · "}
              Group {item.topGroup}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Expanded trait detail — three named sections
// ─────────────────────────────────────────────────────────────────────────────

function TraitDetails({
  traitResult,
  traitName,
  domainTerms,
  module,
}: {
  traitResult: TraitResult;
  traitName: string;
  domainTerms: ReturnType<typeof getDomainTerms>;
  module?: string;
}) {
  const [showAnovaDetails, setShowAnovaDetails] = useState(true);
  const [showAcademic, setShowAcademic] = useState(false);
  const [showProModal, setShowProModal] = useState(false);
  const isPro = getVivaSenseMode() === "pro";
  const isAnovaModule = module === "anova";

  const ar = traitResult.analysis_result;
  const result = ar?.result;
  const warnings = traitResult.data_warnings;
  const intensityDisclosure = selectionIntensityDisclosure(
    result?.genetic_parameters?.selection_intensity ?? DEFAULT_SELECTION_INTENSITY
  );
  const interpretationSections = buildScientificInterpretationSections(ar?.interpretation ?? "");

  const isSplitPlot = result?.design === "split_plot_rcbd" || result?.main_plot_mean_separation != null;
  const interactionMeans = result?.interaction_means ?? null;
  const assumptionTests = (result?.assumption_tests ?? null) as Record<string, { statistic: number; p_value: number; conclusion: string }> | null;

  const intSig = isSplitPlot ? isInteractionSignificant(result?.anova_table) : false;
  const mpLabel = (result?.main_plot_mean_separation as MeanSeparation | null | undefined)?.treatment_label ?? "Main-Plot Factor";
  const spLabel = (result?.mean_separation as MeanSeparation | null | undefined)?.treatment_label ?? "Sub-Plot Factor";

  return (
    <div className="mt-3 space-y-5 text-sm">
      {/* Data warnings */}
      {warnings.length > 0 && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 space-y-0.5">
          <p className="text-xs font-semibold text-amber-700">Data warnings</p>
          {warnings.map((w: string) => (
            <p key={w} className="text-xs text-amber-600 flex items-start gap-1">
              <span className="mt-0.5 shrink-0">⚠</span> {w}
            </p>
          ))}
        </div>
      )}

      {/* Interaction significance banner — shown first when significant */}
      {isSplitPlot && intSig && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          <span className="font-semibold">Significant A×B interaction detected.</span>{" "}
          Main effects should be interpreted with caution — inspect the interaction means table and line plot below to understand the nature of the interaction.
        </div>
      )}

      {/* ── SECTION: Statistical Analysis ──────────────────────────────────── */}
      <div className="space-y-3">
        <SubSectionLabel>Statistical Analysis</SubSectionLabel>

        {/* ANOVA Table */}
        {result?.anova_table ? (
          <div>
            <button
              type="button"
              onClick={() => setShowAnovaDetails((p: boolean) => !p)}
              className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 mb-1"
            >
              <span className={`transition-transform ${showAnovaDetails ? "rotate-90" : ""}`}>▶</span>
              {showAnovaDetails ? "Hide" : "Show"} ANOVA table
              <span className="text-gray-400">(statistical details)</span>
            </button>
            {showAnovaDetails && (
              isSplitPlot
                ? <SplitPlotResults
                    anovaTable={result.anova_table}
                    cvMainPlot={(result as unknown as Record<string, number>).cv_main_plot_pct}
                    cvSubPlot={(result as unknown as Record<string, number>).cv_sub_plot_pct}
                  />
                : <AnovaTableSection at={result.anova_table} />
            )}
          </div>
        ) : (
          <p className="text-xs text-gray-400 italic">ANOVA table not available for this trait.</p>
        )}

        {/* Sub-plot / primary mean separation */}
        {result?.mean_separation ? (
          <div>
            {isSplitPlot && (
              <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">
                Sub-Plot Factor — Mean Separation ({spLabel})
              </p>
            )}
            <MeanSeparationSection
              ms={result.mean_separation}
              domainTerms={domainTerms}
              factorLabel={isSplitPlot ? spLabel : undefined}
              module={module}
            />
          </div>
        ) : (
          <p className="text-xs text-gray-400 italic">
            Mean separation (LSD) not available — insufficient degrees of freedom or singular model.
          </p>
        )}

        {/* Main-plot mean separation (split-plot only) */}
        {isSplitPlot && result?.main_plot_mean_separation && (
          <div>
            <p className="text-[11px] font-semibold text-gray-500 uppercase tracking-wide mb-1">
              Main-Plot Factor — Mean Separation ({mpLabel})
            </p>
            <MeanSeparationSection
              ms={result.main_plot_mean_separation}
              domainTerms={domainTerms}
              factorLabel={mpLabel}
              module={module}
            />
          </div>
        )}
      </div>

      {/* Interaction means + line plot (split-plot only) */}
      {isSplitPlot && interactionMeans && (
        <div className="space-y-3">
          <SubSectionLabel>
            A×B Interaction Means{intSig ? " — Significant" : ""}
          </SubSectionLabel>
          <InteractionMeansTable data={interactionMeans} mpLabel={mpLabel} spLabel={spLabel} traitName={traitName} />
          <InteractionLinePlot data={interactionMeans} mpLabel={mpLabel} spLabel={spLabel} traitName={traitName} />
        </div>
      )}

      {/* Assumption diagnostics */}
      {assumptionTests && Object.keys(assumptionTests).length > 0 && (
        <AssumptionTestsPanel tests={assumptionTests} />
      )}

      {/* ── SECTION: Variance Parameters ───────────────────────────────────── */}
      {result?.variance_components && (
        <div className="space-y-2">
          <SubSectionLabel>{isAnovaModule ? "Experimental Parameters" : domainTerms.variance_module}</SubSectionLabel>
          {isSplitPlot ? (
            <div className="rounded-lg border border-gray-200 overflow-hidden text-sm">
              <table className="w-full">
                <tbody>
                  {SPLIT_PLOT_VARIANCE_PARAMS.map(({ key, label, fmt }) => {
                    const val = (result.variance_components as Record<string, unknown>)[key];
                    if (typeof val !== "number") return null;
                    const display = fmt === 0 ? Math.round(val).toString() : val.toFixed(fmt);
                    return (
                      <tr key={key} className="border-b border-gray-100 last:border-0 even:bg-gray-50">
                        <td className="px-3 py-1.5 text-gray-600">{label}</td>
                        <td className="px-3 py-1.5 font-semibold text-gray-800 text-right font-mono">{display}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="grid gap-2 sm:grid-cols-3">
              {Object.entries(result.variance_components)
                .filter(([, v]) => typeof v === "number" && v !== null)
                .map(([key, val]) => (
                  <div key={key} className="rounded-lg bg-gray-100 px-3 py-2">
                    {(() => {
                      const mapped = VARIANCE_SYMBOL_MAP[key] ?? {
                        symbol: key,
                        label: key.replace(/_/g, " "),
                      };
                      return (
                        <p className="text-sm text-gray-800 leading-relaxed">
                          <strong>{mapped.symbol}</strong>
                          <span className="mx-1 text-gray-500">|</span>
                          <span>{mapped.label}</span>
                          <span className="mx-1 text-gray-500">|</span>
                          <span className="font-semibold">{fmtSafeFixed(val, 4)}</span>
                        </p>
                      );
                    })()}
                  </div>
                ))}
            </div>
          )}
        </div>
      )}

      {/* ── SECTION: Interpretation ────────────────────────────────────────── */}
      {ar?.interpretation && (
        <div>
          <SubSectionLabel>AI-assisted academic interpretation</SubSectionLabel>
          <div className="rounded-lg bg-emerald-50 border border-emerald-100 p-3 space-y-3">
            <p className="text-xs font-medium text-emerald-800">{intensityDisclosure}</p>
            {interpretationSections.length > 0 ? (
              interpretationSections.map((section) => (
                <div key={`${section.title}-${section.content.slice(0, 24)}`} className="rounded-lg border border-emerald-100 bg-white/75 px-3 py-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-emerald-800">{section.title}</p>
                  <p className="mt-2 text-sm text-gray-700 leading-relaxed">{section.content}</p>
                </div>
              ))
            ) : (
              <p className="text-gray-700 leading-relaxed">{ar.interpretation}</p>
            )}
            {warnings.length > 0 && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-amber-800">Cautionary note</p>
                <p className="mt-1 text-sm text-amber-800 leading-relaxed">
                  {warnings[0]}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── SECTION: Academic Interpretation (Option A — on demand) ────────── */}
      {ar != null && (
        <div>
          {!showAcademic ? (
            <button
              type="button"
              onClick={() => {
                if (!isPro) {
                  setShowProModal(true);
                  return;
                }
                setShowAcademic(true);
              }}
              className="mt-1 inline-flex items-center gap-2 rounded-lg border border-violet-300 bg-white px-3 py-1.5 text-xs font-medium text-violet-700 hover:bg-violet-50 transition-colors"
            >
              <span>🎓</span>
              {isPro ? "Get Academic Interpretation" : "🔒 Pro · Get Academic Interpretation"}
            </button>
          ) : (
            <AcademicInterpretationPanel
              traitName={traitName}
              moduleType="anova"
              analysisResult={ar as unknown as Record<string, unknown>}
              onClose={() => setShowAcademic(false)}
            />
          )}
          <ProFeatureModal
            open={showProModal}
            onClose={() => setShowProModal(false)}
          />
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Interaction significance helper
// ─────────────────────────────────────────────────────────────────────────────

function isInteractionSignificant(at: AnovaTable | undefined): boolean {
  if (!at) return false;
  const idx = at.source.findIndex((s) => /:|×/i.test(s) && !/error|residual/i.test(s));
  if (idx < 0) return false;
  const p = at.p_value[idx];
  return p !== null && p !== undefined && p < 0.05;
}

// ─────────────────────────────────────────────────────────────────────────────
// Interaction means table (A×B cell means, wide-format display)
// ─────────────────────────────────────────────────────────────────────────────

const LINE_COLORS = ["#059669", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

function InteractionMeansTable({
  data,
  mpLabel,
  spLabel,
  traitName,
}: {
  data: InteractionMeansData;
  mpLabel: string;
  spLabel: string;
  traitName: string;
}) {
  const cm = data.cell_means;
  if (!cm || cm.main_plot.length === 0) return null;

  const mpLevels = [...new Set(cm.main_plot)];
  const spLevels = [...new Set(cm.sub_plot)];

  const cellValue = (mp: string, sp: string): string => {
    const idx = cm.main_plot.findIndex((m, i) => m === mp && cm.sub_plot[i] === sp);
    return idx >= 0 ? cm.trait_value[idx].toFixed(2) : "—";
  };

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200">
      <table className="min-w-full text-xs">
        <thead className="bg-gray-50 border-b border-gray-200">
          <tr>
            <th className="px-3 py-2 text-left font-semibold text-gray-500">{mpLabel} \ {spLabel}</th>
            {spLevels.map((sp) => (
              <th key={sp} className="px-3 py-2 text-left font-semibold text-gray-500">{sp}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {mpLevels.map((mp, i) => (
            <tr key={mp} className={i % 2 === 0 ? "bg-white" : "bg-gray-50/50"}>
              <td className="px-3 py-1.5 font-medium text-gray-700">{mp}</td>
              {spLevels.map((sp) => (
                <td key={sp} className="px-3 py-1.5 text-gray-600">{cellValue(mp, sp)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="px-3 py-1.5 text-[10px] text-gray-400">Mean {traitName} per A×B combination.</p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Interaction line plot (recharts, one line per main-plot level)
// ─────────────────────────────────────────────────────────────────────────────

function InteractionLinePlot({
  data,
  mpLabel,
  spLabel,
  traitName,
}: {
  data: InteractionMeansData;
  mpLabel: string;
  spLabel: string;
  traitName: string;
}) {
  const cm = data.cell_means;
  if (!cm || cm.main_plot.length === 0) return null;

  const mpLevels = [...new Set(cm.main_plot)];
  const spLevels = [...new Set(cm.sub_plot)];

  // Convert long format → recharts [{sub_plot: "S1", "Main A": 12.3, ...}]
  const chartData = spLevels.map((sp) => {
    const row: Record<string, string | number> = { sub_plot: sp };
    mpLevels.forEach((mp) => {
      const idx = cm.main_plot.findIndex((m, i) => m === mp && cm.sub_plot[i] === sp);
      row[mp] = idx >= 0 ? cm.trait_value[idx] : 0;
    });
    return row;
  });

  return (
    <div>
      <p className="text-[11px] text-gray-500 mb-1">
        Interaction plot — non-parallel lines indicate a significant A×B interaction.
      </p>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={chartData} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="sub_plot" tick={{ fontSize: 11, fill: "#6b7280" }} label={{ value: spLabel, position: "insideBottom", offset: -2, fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11, fill: "#6b7280" }} label={{ value: traitName, angle: -90, position: "insideLeft", offset: 8, fontSize: 11 }} />
          <Tooltip contentStyle={{ fontSize: 12 }} />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {mpLevels.map((mp, i) => (
            <Line
              key={mp}
              type="monotone"
              dataKey={mp}
              stroke={LINE_COLORS[i % LINE_COLORS.length]}
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Assumption diagnostics panel (Shapiro-Wilk + Bartlett)
// ─────────────────────────────────────────────────────────────────────────────

function AssumptionTestsPanel({
  tests,
}: {
  tests: Record<string, { statistic: number; p_value: number; conclusion: string }>;
}) {
  const sw = tests.shapiro_wilk;
  const bt = tests.bartlett;
  if (!sw && !bt) return null;

  const card = (
    label: string,
    t: { statistic: number; p_value: number; conclusion: string },
    statSymbol: string
  ) => {
    const ok = t.p_value >= 0.05;
    return (
      <div
        key={label}
        className={`rounded-lg border px-3 py-2 text-xs ${ok ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-amber-200 bg-amber-50 text-amber-800"}`}
      >
        <p className="font-semibold">{label}</p>
        <p className="mt-0.5">{t.conclusion}</p>
        <p className="mt-0.5 text-[10px] opacity-70">
          {statSymbol} = {t.statistic.toFixed(4)} · p = {t.p_value.toFixed(4)}
        </p>
      </div>
    );
  };

  return (
    <div className="space-y-2">
      <SubSectionLabel>Assumption Diagnostics</SubSectionLabel>
      {sw && card("Shapiro-Wilk — Residual Normality", sw, "W")}
      {bt && card("Bartlett — Homogeneity of Variance", bt, "K²")}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ANOVA Table section (shown only when expanded)
// ─────────────────────────────────────────────────────────────────────────────

function AnovaTableSection({ at }: { at: AnovaTable }) {
  return (
    <div>
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
            {at.source.map((src: string, i: number) => {
              const { text: pText, stars } = fmtP(at.p_value[i]);
              const isError = src === "Residuals";
              return (
                <tr key={src} className={isError ? "bg-gray-50/60" : "bg-white"}>
                  <td className="px-3 py-1.5 font-medium text-gray-700">
                    {ANOVA_LABELS[src] ?? src}
                  </td>
                  <td className="px-3 py-1.5 text-gray-600">{at.df[i]}</td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {fmtSafeFixed(at.ss[i], 2)}
                  </td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {fmtSafeFixed(at.ms[i], 2)}
                  </td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {fmtSafeFixed(at.f_value[i], 3)}
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

function MeanSeparationSection({
  ms,
  domainTerms,
  factorLabel,
  module,
}: {
  ms: MeanSeparation;
  domainTerms: ReturnType<typeof getDomainTerms>;
  factorLabel?: string;
  module?: string;
}) {
  const isAnovaModule = module === "anova";
  // Identify the top group letter for row highlighting
  const topGroup = ms.group[0] ?? "a";
  const columnHeader = factorLabel
    ?? (isAnovaModule ? (ms.treatment_label ?? "Treatment") : domainTerms.treatment);

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full text-xs">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">Rank</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">{columnHeader}</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">Mean</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">SE</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">Group</th>
            </tr>
          </thead>
          <tbody>
            {ms.genotype.map((geno: string, i: number) => {
              const isTop = ms.group[i] === topGroup;
              return (
                <tr key={geno} className={i % 2 === 0 ? "bg-white" : "bg-gray-50/50"}>
                  <td className="px-3 py-1.5 text-gray-400">{i + 1}</td>
                  <td className={`px-3 py-1.5 font-medium ${isTop ? "text-emerald-800" : "text-gray-700"}`}>
                    {geno}
                  </td>
                  <td className="px-3 py-1.5 text-gray-700">{ms.mean[i] != null ? ms.mean[i].toFixed(2) : "—"}</td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {ms.se[i] != null ? ms.se[i]!.toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5">
                    <span
                      className={`inline-block rounded-full border px-2 py-0.5 text-xs font-medium capitalize ${
                        isTop
                          ? "bg-emerald-100 text-emerald-800"
                          : "bg-gray-100 text-gray-600"
                      }`}
                    >
                      {ms.group[i]}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {/* Statistical significance note */}
      <div className="rounded-md bg-emerald-50 border border-emerald-100 px-3 py-2 text-xs text-emerald-800">
        {factorLabel ?? domainTerms.treatments} sharing the same letter are{" "}
        <em>not</em> significantly different at α&nbsp;=&nbsp;{ms.alpha} ({ms.test}).
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Small helpers
// ─────────────────────────────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h4 className="text-sm font-semibold text-gray-700 border-b border-gray-200 pb-1 mb-3">
      {children}
    </h4>
  );
}

function SubSectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">{children}</p>
  );
}

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
    High: "bg-emerald-100 text-emerald-700 border-emerald-200",
    Medium: "bg-yellow-100 text-yellow-700 border-yellow-200",
    Low: "bg-red-100 text-red-600 border-red-200",
  };
  return (
    <span
      className={`inline-block rounded-full border px-2 py-0.5 text-xs font-medium capitalize ${styles[cls] ?? "bg-gray-100 text-gray-600"}`}
    >
      {cls}
    </span>
  );
}
