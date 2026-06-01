/**
 * MultiTraitUpload
 * ================
 * Orchestrates the full upload workflow:
 *   idle → previewing → confirming → analyzing → results
 *
 * Usage:
 *   <MultiTraitUpload />
 */

import React, { useState } from "react";
import { FileUpload } from "./FileUpload";
import { ColumnMappingConfirm } from "./ColumnMappingConfirm";
import { ResultsDisplay } from "./ResultsDisplay";
import { FieldLayoutGenerator } from "@/components/vivasense/FieldLayoutGenerator";
import {
  previewUpload,
  analyzeUpload,
  confirmDataset,
  fileToBase64,
  inferFileType,
  UploadPreviewResponse,
  UploadAnalysisResponse,
  UploadDatasetContext,
} from "@/services/geneticsUploadApi";
import { ProFeatureError } from "@/services/featureMode";
import { ProFeatureModal } from "./ProFeatureModal";
import { logAnalysis } from "../../lib/analysisLogger";

type Step = "idle" | "confirming" | "analyzing" | "results";

const FIELD_LAYOUT_SESSION_KEY = "vivasense:field-layout-generator:open";

function inferScientificChecks(preview: UploadPreviewResponse) {
  const warnings = preview.warnings ?? [];
  const rowCount = preview.n_rows ?? 0;
  const traitCount = preview.detected_columns.traits?.length ?? 0;
  const mode = preview.mode_suggestion;
  const environmentDetected = !!preview.detected_columns.environment?.column;
  const repDetected = !!preview.detected_columns.rep?.column;
  const genotypeConfidence = preview.detected_columns.genotype?.confidence ?? "low";
  const repConfidence = preview.detected_columns.rep?.confidence ?? "low";
  const envConfidence = preview.detected_columns.environment?.confidence ?? "low";

  const items = [
    {
      title: "Structure detection",
      severity: genotypeConfidence === "high" ? "ok" : genotypeConfidence === "medium" ? "review" : "attention",
      recommendation:
        genotypeConfidence === "high"
          ? "Treatment or genotype column was detected confidently."
          : "Review detected structural variables before analysis.",
    },
  ];

  if (!repDetected && mode === "single") {
    items.push({
      title: "Replication detection",
      severity: "review",
      recommendation: "No replication column was confidently detected. Confirm whether the dataset is CRD or map the block column manually.",
    });
  } else if (repDetected) {
    items.push({
      title: "Replication detection",
      severity: repConfidence === "high" ? "ok" : "review",
      recommendation: `Replication column detected${repConfidence === "high" ? " with high confidence." : ". Please verify before continuing."}`,
    });
  }

  if (mode === "multi") {
    items.push({
      title: "Environment structure",
      severity: environmentDetected && envConfidence === "high" ? "ok" : "review",
      recommendation: environmentDetected
        ? "Likely multi-environment trial structure detected. Confirm environment mapping before G×E analyses."
        : "A multi-environment structure is suspected, but the environment column needs confirmation.",
    });
  }

  if (warnings.length > 0) {
    items.push({
      title: "Validation warnings",
      severity: warnings.length > 2 ? "attention" : "review",
      recommendation: warnings[0],
    });
  }

  if (rowCount < 12) {
    items.push({
      title: "Dataset size",
      severity: "review",
      recommendation: "Small datasets can limit model stability and mean-separation reliability. Proceed with caution.",
    });
  }

  if (traitCount === 0) {
    items.push({
      title: "Trait detection",
      severity: "attention",
      recommendation: "No numeric trait columns were detected. Inspect numeric formatting before analysis.",
    });
  }

  return items;
}

function summarizeProceedingAdvice(preview: UploadPreviewResponse) {
  const warnings = preview.warnings ?? [];
  if (warnings.length === 0) return "Analysis can proceed safely after variable confirmation.";
  if (warnings.length <= 2) return "Analysis can proceed, but review the validation notes below before running models.";
  return "Analysis can proceed conditionally, but the dataset should be reviewed carefully before interpreting outputs.";
}

export interface FileStatusInfo {
  state: "none" | "invalid" | "loaded";
  filename?: string;
  n_rows?: number;
  n_columns?: number;
}

interface MultiTraitUploadProps {
  /**
   * Called once the user confirms column mapping, before analysis begins.
   * Provides the dataset context needed by the Trait Relationships tab so
   * it can run correlation without requiring a second file upload.
   */
  onDatasetReady?: (ctx: UploadDatasetContext) => void;
  /**
   * Called whenever the upload state changes so the parent (DataSourceTabs)
   * can update the top status bar without reaching into this component's state.
   */
  onFileStatus?: (info: FileStatusInfo) => void;
}

export function MultiTraitUpload({ onDatasetReady, onFileStatus }: MultiTraitUploadProps = {}) {
  const [step, setStep] = useState<Step>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<UploadPreviewResponse | null>(null);
  const [results, setResults] = useState<UploadAnalysisResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [datasetToken, setDatasetToken] = useState<string | null>(null);
  const [researchDomain, setResearchDomain] = useState<"plant_breeding" | "agronomy" | "general">("plant_breeding");
  const [showProModal, setShowProModal] = useState(false);
  // State 1B: file parsed successfully but contained zero valid rows.
  const [invalidFile, setInvalidFile] = useState(false);
  // Latches true after first valid upload; never reset — used to detect replacements.
  const [datasetWasLoaded, setDatasetWasLoaded] = useState(false);
  // Shown at the top of the config step when the user uploads a second file.
  const [replacementNotice, setReplacementNotice] = useState(false);
  const [showFieldLayoutGenerator, setShowFieldLayoutGenerator] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }
    try {
      return window.sessionStorage.getItem(FIELD_LAYOUT_SESSION_KEY) === "open";
    } catch {
      return false;
    }
  });

  const toggleFieldLayoutGenerator = () => {
    setShowFieldLayoutGenerator((current) => {
      const next = !current;
      if (typeof window !== "undefined") {
        try {
          window.sessionStorage.setItem(
            FIELD_LAYOUT_SESSION_KEY,
            next ? "open" : "closed"
          );
        } catch {
          // Ignore storage failures and keep the in-memory toggle working.
        }
      }
      return next;
    });
  };

  // ── Step 1: File selected → preview fetched ──────────────────────────────

  const handlePreviewReady = async (f: File, p: UploadPreviewResponse) => {
    // State 1B: preview succeeded but file has no usable rows.
    if (p.n_rows === 0) {
      setInvalidFile(true);
      onFileStatus?.({ state: "invalid" });
      return;
    }
    setInvalidFile(false);

    // Show replacement notice when a dataset was already loaded before.
    if (datasetWasLoaded) {
      setReplacementNotice(true);
    }
    setDatasetWasLoaded(true);

    onFileStatus?.({
      state: "loaded",
      filename: f.name,
      n_rows: p.n_rows,
      n_columns: p.n_columns,
    });

    setFile(f);
    setPreview(p);
    if (p.dataset_token) {
      setDatasetToken(p.dataset_token);
      // Emit context immediately using auto-detected columns so 
      // /analysis/descriptive-stats can be called before explicit confirmation
      try {
        const base64Content = await fileToBase64(f);
        onDatasetReady?.({
          file: f,
          base64Content,
          fileType: inferFileType(f),
          genotypeColumn: p.detected_columns.genotype?.column || "",
          repColumn: p.detected_columns.rep?.column || "",
          environmentColumn: p.mode_suggestion === "multi" ? (p.detected_columns.environment?.column || undefined) : undefined,
          availableTraitColumns: p.detected_columns.traits || [],
          mode: p.mode_suggestion,
          datasetToken: p.dataset_token,
        });
      } catch (err) {
        console.warn("[MultiTraitUpload] Failed to generate initial dataset context", err);
      }
    }
    setStep("confirming");
  };

  // ── Step 2: Column mapping confirmed → run analysis ──────────────────────

  const handleConfirm = async (mapping: {
    genotypeColumn: string;
    repColumn: string;
    environmentColumn: string;
    selectedTraits: string[];
    mode: "single" | "multi";
    randomEnvironment: boolean;
    selectionIntensity: number;
    numericFactorOverrides: string[];
    research_domain: "plant_breeding" | "agronomy" | "general";
    design_type?: string;
  }) => {
    if (!file) return;
    setResearchDomain(mapping.research_domain);
    setStep("analyzing");
    setAnalysisError(null);
    const startedAt = Date.now();

    try {
      const base64_content = await fileToBase64(file);

      // Always call POST /upload/dataset with the user's confirmed column mapping.
      // Never shortcut via the preview token — that token was registered with
      // auto-detected defaults and may not reflect the user's actual selections.
      // Module endpoints (/analysis/*) require a token from this confirmation step.
      console.log("Selected Design:", mapping.design_type || mapping.mode);
      console.log("[MultiTraitUpload] calling POST /upload/dataset with confirmed mapping…");
      let finalToken: string | null = datasetToken; // Fallback to preview token
      try {
        const confirmed = await confirmDataset({
          base64_content,
          file_type: inferFileType(file),
          genotype_column: mapping.genotypeColumn || null,
          rep_column: mapping.repColumn || null,
          environment_column:
            mapping.mode === "multi" ? mapping.environmentColumn || null : null,
          numeric_factor_columns: mapping.numericFactorOverrides ?? [],
          mode: mapping.mode,
          design_type: (mapping.design_type || "rcbd") as "crd" | "rcbd" | "factorial" | "split_plot_rcbd",
          random_environment: mapping.randomEnvironment,
          selection_intensity: mapping.selectionIntensity,
        });
        finalToken = confirmed.dataset_token;
        setDatasetToken(finalToken);
        console.log(
          "[MultiTraitUpload] POST /upload/dataset succeeded — dataset_token:",
          finalToken,
          "| n_rows:", confirmed.n_rows,
          "| n_reps:", confirmed.n_reps,
        );
      } catch (tokenErr) {
        console.warn(
          "[MultiTraitUpload] POST /upload/dataset failed — module endpoints (descriptive stats, etc.) will be unavailable:",
          tokenErr
        );
      }

      // Share dataset context with sibling components (Trait Relationships,
      // Descriptive Stats) after dataset confirmation.
      // finalToken falls back to preview token if /upload/dataset failed above.
      const ctx: UploadDatasetContext = {
        file,
        base64Content: base64_content,
        fileType: inferFileType(file),
        genotypeColumn: mapping.genotypeColumn,
        repColumn: mapping.repColumn,
        environmentColumn:
          mapping.mode === "multi" ? mapping.environmentColumn || undefined : undefined,
        availableTraitColumns: preview?.detected_columns.traits ?? mapping.selectedTraits,
        mode: mapping.mode,
        datasetToken: finalToken,
        research_domain: mapping.research_domain,
      };
      console.log(
        "[MultiTraitUpload] sharing dataset context — datasetToken:",
        ctx.datasetToken,
        "| availableTraitColumns:", ctx.availableTraitColumns,
      );
      onDatasetReady?.(ctx);

      console.log("Selected Design (analyze):", mapping.design_type || mapping.mode);
      const data = await analyzeUpload({
        base64_content,
        file_type: inferFileType(file),
        genotype_column: mapping.genotypeColumn,
        rep_column: mapping.repColumn,
        environment_column: mapping.mode === "multi" ? (mapping.environmentColumn || null) : null,
        numeric_factor_columns: mapping.numericFactorOverrides ?? [],
        trait_columns: mapping.selectedTraits.length > 0 ? mapping.selectedTraits : [],
        mode: mapping.mode,
        random_environment: mapping.randomEnvironment,
        selection_intensity: mapping.selectionIntensity,
        research_domain: mapping.research_domain,
      });

      await logAnalysis({
        analysis_type: "genetics",
        design_type: (mapping.design_type || mapping.mode) as "crd" | "rcbd" | "factorial" | "split_plot_rcbd" | "met" | undefined,
        trait_count: mapping.selectedTraits.length,
        dataset_rows: preview?.n_rows,
        success: true,
        duration_ms: Date.now() - startedAt,
      });

      setResults(data);
      setStep("results");
    } catch (err) {
      await logAnalysis({
        analysis_type: "genetics",
        design_type: (mapping.design_type || mapping.mode) as "crd" | "rcbd" | "factorial" | "split_plot_rcbd" | "met" | undefined,
        trait_count: mapping.selectedTraits.length,
        dataset_rows: preview?.n_rows,
        success: false,
        error_message: err instanceof Error ? err.message : "Analysis failed",
        duration_ms: Date.now() - startedAt,
      });

      if (err instanceof ProFeatureError) {
        setShowProModal(true);
        setAnalysisError(null);
        setStep("confirming");
        return;
      }
      setAnalysisError(err instanceof Error ? err.message : "Analysis failed");
      setStep("confirming"); // go back so user can retry
    }
  };

  // ── Reset ─────────────────────────────────────────────────────────────────

  const reset = () => {
    setStep("idle");
    setFile(null);
    setPreview(null);
    setResults(null);
    setAnalysisError(null);
    setDatasetToken(null);
    setResearchDomain("plant_breeding");
    setShowProModal(false);
    setInvalidFile(false);
    setReplacementNotice(false);
    onFileStatus?.({ state: "none" });
    // datasetWasLoaded intentionally kept — next upload will be a replacement.
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="w-full">
      <StepIndicator current={step} />

      <div className="mt-4">
        {step === "idle" && (
          <>
            <FileUpload
              previewFn={previewUpload}
              onPreviewStart={() => setInvalidFile(false)}
              onPreviewReady={handlePreviewReady}
            />
            {invalidFile && (
              <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                Uploaded file contains no valid data. Please check the format or try a different file.
              </div>
            )}
          </>
        )}

        {(step === "confirming" || step === "analyzing") && preview && (
          <>
            <UploadScientificSummary preview={preview} file={file} />

            {replacementNotice && (
              <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-700">
                New file loaded — please reconfigure your analysis.
              </div>
            )}
            {analysisError && (
              <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                <p className="font-medium">Analysis error</p>
                {/* Strip the "Analysis failed — " prefix so the backend detail
                    message is shown directly. Falls back to full message. */}
                <p className="mt-0.5 font-mono text-xs break-all">
                  {analysisError.replace(/^Analysis failed — /, "")}
                </p>
              </div>
            )}
            <ColumnMappingConfirm
              preview={preview}
              onConfirm={handleConfirm}
              onBack={reset}
              loading={step === "analyzing"}
            />
          </>
        )}

        {step === "results" && results && (
          <ResultsDisplay results={results} onReset={reset} domain={researchDomain} />
        )}
      </div>

      <section className="mt-8 rounded-xl border border-gray-200 bg-white p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-base font-semibold text-gray-800">Field Layout Generator</h3>
            <p className="mt-1 text-sm text-gray-500">
              Generate a field plan for trial setup and data collection.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700">
              Free Tool
            </span>
            <button
              type="button"
              onClick={toggleFieldLayoutGenerator}
              aria-expanded={showFieldLayoutGenerator}
              className="inline-flex items-center rounded-full border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-700 transition hover:border-emerald-300 hover:text-emerald-700"
            >
              {showFieldLayoutGenerator ? "Hide Generator" : "Open Generator"}
            </button>
          </div>
        </div>

        {showFieldLayoutGenerator ? (
          <div className="mt-4">
            <FieldLayoutGenerator />
          </div>
        ) : (
          <p className="mt-4 rounded-lg border border-dashed border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-600">
            Open the generator when you need a randomized CRD or RCBD field plan for trial setup.
          </p>
        )}
      </section>

      <ProFeatureModal
        open={showProModal}
        onClose={() => setShowProModal(false)}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Step indicator
// ─────────────────────────────────────────────────────────────────────────────

const STEPS: { label: string; helper: string }[] = [
  { label: "Ingest dataset", helper: "Upload CSV or Excel" },
  { label: "Preview structure", helper: "Detect likely design" },
  { label: "Confirm mapping", helper: "Verify factors and traits" },
  { label: "Validate readiness", helper: "Review scientific checks" },
  { label: "Configure model", helper: "Set analysis options" },
  { label: "Run analysis", helper: "Execute statistical workflow" },
  { label: "Review outputs", helper: "Interpret findings" },
  { label: "Export report", helper: "Prepare publication outputs" },
];

function StepIndicator({ current }: { current: Step }) {
  const activeIdx = current === "idle" ? 0 : current === "confirming" ? 3 : current === "analyzing" ? 5 : 6;

  return (
    <nav aria-label="Progress" className="rounded-3xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-gray-500">Workflow Progress</p>
          <p className="mt-1 text-sm text-gray-600">Each stage confirms methodological readiness before advancing to the next analysis action.</p>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-8">
      {STEPS.map((step, idx) => {
        const done = idx < activeIdx;
        const active = idx === activeIdx;
        return (
          <div
            key={step.label}
            className={[
              "rounded-2xl border px-3 py-3 transition-colors",
              done
                ? "border-emerald-200 bg-emerald-50"
                : active
                ? "border-emerald-300 bg-white shadow-sm"
                : "border-gray-200 bg-gray-50/70",
            ].join(" ")}
          >
            <div className="flex items-center gap-3">
              <div
                className={[
                  "flex h-9 w-9 items-center justify-center rounded-full text-sm font-bold transition-colors",
                  done
                    ? "bg-emerald-600 text-white"
                    : active
                    ? "bg-emerald-100 text-emerald-700 ring-2 ring-emerald-500"
                    : "bg-white text-gray-500 border border-gray-200",
                ].join(" ")}
              >
                {done ? "✓" : active ? "●" : "○"}
              </div>
              <div>
                <p className={[
                  "text-sm font-semibold",
                  active ? "text-emerald-700" : done ? "text-emerald-800" : "text-gray-600",
                ].join(" ")}>{step.label}</p>
                <p className="text-xs text-gray-500 leading-relaxed">{step.helper}</p>
              </div>
            </div>
          </div>
        );
      })}
      </div>
    </nav>
  );
}

function UploadScientificSummary({
  preview,
  file,
}: {
  preview: UploadPreviewResponse;
  file: File | null;
}) {
  const checks = inferScientificChecks(preview);
  const proceedText = summarizeProceedingAdvice(preview);

  return (
    <div className="mb-5 space-y-4">
      <div className="rounded-3xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-gray-500">Dataset preview</p>
            <h3 className="mt-1 text-lg font-semibold text-gray-900">Scientific intake summary</h3>
            <p className="mt-1 text-sm text-gray-600 leading-relaxed">
              {file?.name ? `${file.name} uploaded.` : "Dataset uploaded."} VivaSense detected a likely {preview.mode_suggestion === "multi" ? "multi-environment trial" : "single-environment"} structure with {preview.detected_columns.traits.length} numeric trait{preview.detected_columns.traits.length !== 1 ? "s" : ""}.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm sm:min-w-[280px]">
            <SummaryMetric label="Rows" value={preview.n_rows.toLocaleString()} />
            <SummaryMetric label="Variables" value={String(preview.n_columns)} />
            <SummaryMetric label="Design guess" value={preview.mode_suggestion === "multi" ? "Likely MET" : preview.detected_columns.rep?.column ? "Likely RCBD" : "Likely CRD"} />
            <SummaryMetric label="Replication" value={preview.detected_columns.rep?.column ? "Detected" : "Review needed"} />
          </div>
        </div>
      </div>

      <div className="rounded-3xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="max-w-3xl">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-gray-500">Scientific validation</p>
            <h4 className="mt-1 text-base font-semibold text-gray-900">Validation panel</h4>
            <p className="mt-1 text-sm text-gray-600 leading-relaxed">{proceedText}</p>
          </div>
          <div className="rounded-full border border-gray-200 bg-gray-50 px-3 py-1 text-xs font-medium text-gray-700">
            {preview.warnings.length === 0 ? "Proceed safely" : preview.warnings.length <= 2 ? "Proceed with review" : "Proceed conditionally"}
          </div>
        </div>
        <div className="mt-4 grid gap-3 lg:grid-cols-2">
          {checks.map((check) => (
            <div key={`${check.title}-${check.recommendation}`} className="rounded-2xl border border-gray-200 bg-gray-50 px-4 py-3">
              <div className="flex items-center gap-2">
                <span className={[
                  "inline-flex rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide",
                  check.severity === "ok"
                    ? "bg-emerald-100 text-emerald-700"
                    : check.severity === "review"
                    ? "bg-amber-100 text-amber-700"
                    : "bg-red-100 text-red-700",
                ].join(" ")}>{check.severity === "ok" ? "ready" : check.severity}</span>
                <p className="text-sm font-semibold text-gray-800">{check.title}</p>
              </div>
              <p className="mt-2 text-sm text-gray-600 leading-relaxed">{check.recommendation}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function SummaryMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-gray-200 bg-gray-50 px-3 py-3">
      <p className="text-[11px] font-semibold uppercase tracking-wide text-gray-500">{label}</p>
      <p className="mt-1 text-sm font-semibold text-gray-800 leading-snug">{value}</p>
    </div>
  );
}
