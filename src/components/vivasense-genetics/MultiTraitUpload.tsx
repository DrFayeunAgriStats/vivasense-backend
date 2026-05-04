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
import { FieldLayoutGenerator } from "./FieldLayoutGenerator";
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

type Step = "idle" | "confirming" | "analyzing" | "results";

const FIELD_LAYOUT_SESSION_KEY = "vivasense:field-layout-generator:open";

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
    research_domain: "plant_breeding" | "agronomy" | "general";
  }) => {
    if (!file) return;
    setResearchDomain(mapping.research_domain);
    setStep("analyzing");
    setAnalysisError(null);

    try {
      const base64_content = await fileToBase64(file);

      // Always call POST /upload/dataset with the user's confirmed column mapping.
      // Never shortcut via the preview token — that token was registered with
      // auto-detected defaults and may not reflect the user's actual selections.
      // Module endpoints (/analysis/*) require a token from this confirmation step.
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
          mode: mapping.mode,
          design_type: mapping.mode === "multi" ? "multi" : "single",
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

      const data = await analyzeUpload({
        base64_content,
        file_type: inferFileType(file),
        genotype_column: mapping.genotypeColumn,
        rep_column: mapping.repColumn,
        environment_column: mapping.mode === "multi" ? (mapping.environmentColumn || null) : null,
        trait_columns: mapping.selectedTraits.length > 0 ? mapping.selectedTraits : [],
        mode: mapping.mode,
        random_environment: mapping.randomEnvironment,
        selection_intensity: mapping.selectionIntensity,
        research_domain: mapping.research_domain,
      });
      setResults(data);
      setStep("results");
    } catch (err) {
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
      {/* Step indicator — always visible */}
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
              Generate a CRD or RCBD field plan for trial setup and data collection before analysis.
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

const STEPS: { key: Step | "idle"; label: string }[] = [
  { key: "idle", label: "Upload" },
  { key: "confirming", label: "Map Columns" },
  { key: "analyzing", label: "Analyze" },
  { key: "results", label: "Results" },
];

function StepIndicator({ current }: { current: Step }) {
  const currentIdx = STEPS.findIndex((s) => s.key === current);
  // -1 shouldn't happen but clamp to 0 so "idle" fallback still shows step 1 active
  const activeIdx = currentIdx < 0 ? 0 : currentIdx;

  return (
    <nav aria-label="Progress" className="rounded-2xl border border-gray-100 bg-gradient-to-r from-white via-gray-50 to-white p-3">
      <div className="flex items-center gap-0">
      {STEPS.map((step, idx) => {
        const done = idx < activeIdx;
        const active = idx === activeIdx;
        return (
          <React.Fragment key={step.key}>
            <div className="flex flex-col items-center">
              <div
                className={[
                  "flex h-8 w-8 items-center justify-center rounded-full text-xs font-bold transition-colors",
                  done
                    ? "bg-emerald-600 text-white"
                    : active
                    ? "bg-emerald-100 text-emerald-700 ring-2 ring-emerald-500"
                    : "bg-gray-100 text-gray-500",
                ].join(" ")}
              >
                {done ? "✓" : idx + 1}
              </div>
              <span
                className={[
                  "mt-1 text-xs",
                  active ? "font-semibold text-emerald-700" : "text-gray-400",
                ].join(" ")}
              >
                {step.label}
              </span>
            </div>
            {idx < STEPS.length - 1 && (
              <div
                className={[
                  "h-0.5 flex-1 mx-1 mb-4 transition-colors",
                  idx < activeIdx ? "bg-emerald-500" : "bg-gray-200",
                ].join(" ")}
              />
            )}
          </React.Fragment>
        );
      })}
      </div>
    </nav>
  );
}
