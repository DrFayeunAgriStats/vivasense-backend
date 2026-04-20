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

type Step = "idle" | "confirming" | "analyzing" | "results";

interface MultiTraitUploadProps {
  /**
   * Called once the user confirms column mapping, before analysis begins.
   * Provides the dataset context needed by the Trait Relationships tab so
   * it can run correlation without requiring a second file upload.
   */
  onDatasetReady?: (ctx: UploadDatasetContext) => void;
}

export function MultiTraitUpload({ onDatasetReady }: MultiTraitUploadProps = {}) {
  const [step, setStep] = useState<Step>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<UploadPreviewResponse | null>(null);
  const [results, setResults] = useState<UploadAnalysisResponse | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // ── Step 1: File selected → preview fetched ──────────────────────────────

  const handlePreviewReady = (f: File, p: UploadPreviewResponse) => {
    setFile(f);
    setPreview(p);
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
  }) => {
    if (!file) return;
    setStep("analyzing");
    setAnalysisError(null);

    try {
      const base64_content = await fileToBase64(file);

      // Always call POST /upload/dataset with the user's confirmed column mapping.
      // Never shortcut via the preview token — that token was registered with
      // auto-detected defaults and may not reflect the user's actual selections.
      // Module endpoints (/analysis/*) require a token from this confirmation step.
      console.log("[MultiTraitUpload] calling POST /upload/dataset with confirmed mapping…");
      let datasetToken: string | null = null;
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
          selection_intensity: 2.06,
        });
        datasetToken = confirmed.dataset_token;
        console.log(
          "[MultiTraitUpload] POST /upload/dataset succeeded — dataset_token:",
          datasetToken,
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
      // datasetToken is null only if /upload/dataset failed above.
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
        datasetToken,
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
        selection_intensity: 2.06,
      });
      setResults(data);
      setStep("results");
    } catch (err) {
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
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="w-full">
      {/* Step indicator */}
      {step !== "idle" && (
        <StepIndicator current={step} />
      )}

      <div className="mt-4">
        {(step === "idle") && (
          <FileUpload
            previewFn={previewUpload}
            onPreviewStart={() => {}}
            onPreviewReady={handlePreviewReady}
          />
        )}

        {(step === "confirming" || step === "analyzing") && preview && (
          <>
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
          <ResultsDisplay results={results} onReset={reset} />
        )}
      </div>
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

  return (
    <nav aria-label="Progress" className="flex items-center gap-0">
      {STEPS.map((step, idx) => {
        const done = idx < currentIdx;
        const active = idx === currentIdx;
        return (
          <React.Fragment key={step.key}>
            <div className="flex flex-col items-center">
              <div
                className={[
                  "flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-colors",
                  done
                    ? "bg-emerald-600 text-white"
                    : active
                    ? "bg-emerald-100 text-emerald-700 ring-2 ring-emerald-500"
                    : "bg-gray-100 text-gray-400",
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
                  idx < currentIdx ? "bg-emerald-500" : "bg-gray-200",
                ].join(" ")}
              />
            )}
          </React.Fragment>
        );
      })}
    </nav>
  );
}
