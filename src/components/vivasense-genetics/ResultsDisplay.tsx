import React, { useEffect, useState } from "react";
import {
  AnovaTable,
  MeanSeparation,
  UploadAnalysisResponse,
  SummaryTableRow,
  TraitResult,
  exportWordReport,
} from "@/services/geneticsUploadApi";
import { DomainKey, getDomainTerms } from "./domainTerms";
import { AcademicInterpretationPanel } from "./AcademicInterpretationPanel";
import { WordExportPreviewModal } from "@/components/export/WordExportPreviewModal";
import { safeArray, logDebug, safeNumber } from "@/utils/normalizeModuleData";
import type { PreviewSection } from "@/utils/normalizeModuleData";
import { buildAnovaPreview, buildGeneticParametersPreview } from "@/utils/previewBuilders";
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

interface ResultsDisplayProps {
  results: UploadAnalysisResponse;
  onReset: () => void;
  domain?: DomainKey;
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

function fmtP(p: number | null): { text: string; stars: string } {
  if (p === null) return { text: "—", stars: "" };
  if (p < 0.001) return { text: p.toExponential(2), stars: "***" };
  if (p < 0.01) return { text: p.toFixed(4), stars: "**" };
  if (p < 0.05) return { text: p.toFixed(4), stars: "*" };
  return { text: p.toFixed(4), stars: "ns" };
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
          r?.grand_mean != null ? `Mean: ${(r.grand_mean as number).toFixed(2)}` : null,
          r?.h2 != null ? `H²: ${(r.h2 as number).toFixed(3)}` : null,
          r?.gam_class ? `(GAM class: ${r.gam_class})` : null,
          r?.gcv != null ? `GCV: ${(r.gcv as number).toFixed(1)}%` : null,
          r?.gam_percent != null ? `GAM: ${(r.gam_percent as number).toFixed(1)}%` : null,
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
    const tr = trait_results?.[row.trait] as any;
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

export function ResultsDisplay({ results, onReset, domain }: ResultsDisplayProps) {
  const domainTerms = getDomainTerms(domain);
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
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showProModal, setShowProModal] = useState(false);
  const [mode, setMode] = useState<VivaSenseMode>(() => getVivaSenseMode());

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
      console.log("[DEBUG] domain at export:", domain);
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
            {dataset_summary.n_genotypes} {domainTerms.treatment_plural}
            {dataset_summary.n_environments
              ? ` · ${dataset_summary.n_environments} environments`
              : ` · ${dataset_summary.n_reps} reps`}
          </p>
        </div>
        <div className="flex gap-2 shrink-0">
          <button
            type="button"
            onClick={() => setShowPreview(true)}
            disabled={successCount === 0}
            className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            Preview Report
          </button>
          <button
            type="button"
            onClick={handleDownload}
            disabled={downloading || successCount === 0}
            className="rounded-lg border border-emerald-600 bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {downloading
              ? "Generating…"
              : isPro
              ? "Download Report (.docx)"
              : "🔒 Pro · Download Report (.docx)"}
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

      {/* Workflow info banner */}
      <div className="rounded-lg border border-blue-100 bg-blue-50 px-4 py-3 text-xs text-blue-700 leading-relaxed">
        <span className="font-semibold">How this works: </span>
        VivaSense Genetics uses ANOVA internally to partition phenotypic variance into genetic,
        environmental, and error components. Heritability (H²) and genetic advance (GAM) are
        derived from those variance components. Mean separation (Tukey HSD) identifies which
        genotypes perform significantly differently — expand any trait row below to see the full
        statistical output.
        <p className="mt-1 text-[11px] text-blue-700">
          {selectionIntensityDisclosure(detectedSelectionIntensity)}
        </p>
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
            moduleName="ANOVA & Genetic Parameters"
            reportTitle="Multi-Trait Genetics Report"
            datasetSummary={`${dataset_summary.n_genotypes} genotypes · ${dataset_summary.n_traits} traits · ${dataset_summary.mode} environment`}
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
                <Th>{domainTerms.h2_label}</Th>
                <Th>{domainTerms.gcv_label}</Th>
                <Th>{domainTerms.pcv_label}</Th>
                <Th>{domainTerms.gam_label}</Th>
                <Th>Class</Th>
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
                />
              ))}
            </tbody>
          </table>
        </div>

        {/* Legend */}
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
      </div>

      <ProFeatureModal
        open={showProModal}
        onClose={() => setShowProModal(false)}
        onActivated={() => setMode(getVivaSenseMode())}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Summary row with expandable details
// ─────────────────────────────────────────────────────────────────────────────

function SummaryRow({
  row,
  isEven,
  isFirst,
  traitResult,
  domainTerms,
}: {
  row: SummaryTableRow;
  isEven: boolean;
  isFirst: boolean;
  traitResult: TraitResult | undefined;
  domainTerms: ReturnType<typeof getDomainTerms>;
}) {
  const [expanded, setExpanded] = useState(isFirst);
  const bg = isEven ? "bg-white" : "bg-gray-50/50";

  // Show first sentence of interpretation as a hint below the trait name when collapsed
  const interpHint = (() => {
    const interp = (traitResult?.analysis_result as Record<string, unknown> | undefined)
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
        <td className="px-4 py-3">
          <p className="font-medium text-gray-800 leading-tight">{row.trait}</p>
          {interpHint && (
            <p className="mt-0.5 text-xs text-gray-400 leading-snug line-clamp-1 max-w-xs">{interpHint}</p>
          )}
        </td>
        <Td>{row.grand_mean != null ? row.grand_mean.toFixed(2) : "—"}</Td>
        <Td>
          <HeritabilityCell h2={row.h2} />
        </Td>
        <Td>{row.gcv != null ? row.gcv.toFixed(1) : "—"}</Td>
        <Td>{row.pcv != null ? row.pcv.toFixed(1) : "—"}</Td>
        <Td>{row.gam_percent != null ? row.gam_percent.toFixed(1) : "—"}</Td>
        <td className="px-4 py-3">
          <ClassBadge cls={row.gam_class} />
        </td>
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
          <td colSpan={8} className="px-6 pb-5 pt-1">
            <TraitDetails traitResult={traitResult} traitName={row.trait} domainTerms={domainTerms} />
          </td>
        </tr>
      )}
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Expanded trait detail — three named sections
// ─────────────────────────────────────────────────────────────────────────────

function TraitDetails({
  traitResult,
  traitName,
  domainTerms,
}: {
  traitResult: TraitResult;
  traitName: string;
  domainTerms: ReturnType<typeof getDomainTerms>;
}) {
  const [showAnovaDetails, setShowAnovaDetails] = useState(true);
  const [showAcademic, setShowAcademic] = useState(false);
  const [showProModal, setShowProModal] = useState(false);
  const isPro = getVivaSenseMode() === "pro";

  const ar = traitResult.analysis_result;
  const result = ar?.result;
  const warnings = traitResult.data_warnings;
  const intensityDisclosure = selectionIntensityDisclosure(
    result?.genetic_parameters?.selection_intensity ?? DEFAULT_SELECTION_INTENSITY
  );

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

      {/* ── SECTION: Statistical Analysis ──────────────────────────────────── */}
      <div className="space-y-3">
        <SubSectionLabel>Statistical Analysis</SubSectionLabel>

        {/* ANOVA Table — shown expanded by default */}
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
            {showAnovaDetails && <AnovaTableSection at={result.anova_table} />}
          </div>
        ) : (
          <p className="text-xs text-gray-400 italic">ANOVA table not available for this trait.</p>
        )}

        {/* Mean Separation — primary output */}
        {result?.mean_separation ? (
          <MeanSeparationSection ms={result.mean_separation} domainTerms={domainTerms} />
        ) : (
          <p className="text-xs text-gray-400 italic">
            Mean separation (Tukey HSD) not available — insufficient degrees of freedom or singular model.
          </p>
        )}
      </div>

      {/* ── SECTION: Genetic Parameters ────────────────────────────────────── */}
      {result?.variance_components && (
        <div className="space-y-2">
          <SubSectionLabel>{domainTerms.variance_module}</SubSectionLabel>
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
                        <span className="font-semibold">{(val as number).toFixed(4)}</span>
                      </p>
                    );
                  })()}
                </div>
              ))}
          </div>
        </div>
      )}

      {/* ── SECTION: Interpretation ────────────────────────────────────────── */}
      {ar?.interpretation && (
        <div>
          <SubSectionLabel>Interpretation</SubSectionLabel>
          <div className="rounded-lg bg-emerald-50 border border-emerald-100 p-3">
            <p className="mb-2 text-xs font-medium text-emerald-800">{intensityDisclosure}</p>
            {ar.interpretation
              .split(/\n\s*\n/)
              .map((para: string) => para.trim())
              .filter(Boolean)
              .map((para: string, idx: number) => (
                <p
                  key={`interp-para-${idx}`}
                  className={idx === 0 ? "text-gray-700 leading-relaxed" : "mt-3 text-gray-700 leading-relaxed"}
                >
                  {para}
                </p>
              ))}
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

function MeanSeparationSection({ ms, domainTerms }: { ms: MeanSeparation; domainTerms: ReturnType<typeof getDomainTerms> }) {
  // Identify the top group letter for selection guidance
  const topGroup = ms.group[0] ?? "a";

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="min-w-full text-xs">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">Rank</th>
              <th className="px-3 py-2 text-left font-semibold text-gray-500">{domainTerms.treatment}</th>
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
                  <td className="px-3 py-1.5 text-gray-700">{ms.mean[i].toFixed(2)}</td>
                  <td className="px-3 py-1.5 text-gray-600">
                    {ms.se[i] != null ? ms.se[i]!.toFixed(2) : "—"}
                  </td>
                  <td className="px-3 py-1.5">
                    <span
                      className={`inline-block rounded px-1.5 py-0.5 font-mono font-semibold ${
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
      {/* Selection guidance */}
      <div className="rounded-md bg-emerald-50 border border-emerald-100 px-3 py-2 text-xs text-emerald-800">
        <span className="font-semibold">{ms.test} (α = {ms.alpha}) — </span>
        {domainTerms.treatments} sharing the same letter are <em>not</em> significantly different.
        {" "}{domainTerms.treatments} in group <strong className="font-mono">&apos;{topGroup}&apos;</strong> are
        the top performers — considered a {domainTerms.top_performer}.
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
