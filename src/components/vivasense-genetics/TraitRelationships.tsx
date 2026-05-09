/**
 * TraitRelationships
 * ==================
 * Phase 2 / Phase 1 scope: phenotypic correlation only.
 *
 * State machine:
 *
 *   ┌─ datasetContext provided ──► setup ──► correlating ──► results
 *   │                               ▲                           │
 *   │                               └─────── "Adjust" ──────────┘
 *   └─ no context ──► shows "upload first" message
 *
 * Dataset context is provided by DataSourceTabs, which captures it from
 * MultiTraitUpload.onDatasetReady when the user confirms column mapping
 * in the Upload File tab.  No second upload is required.
 */

import React, { useEffect, useState } from "react";
import { UploadDatasetContext } from "@/services/geneticsUploadApi";
import {
  computeCorrelation,
  CorrelationResponse,
  CorrelationStats,
  exportCorrelationWord,
} from "@/services/traitRelationshipsApi";
import { VsSpinner } from "./VsSpinner";
import { CorrelationHeatmap } from "./CorrelationHeatmap";
import { WordExportPreviewModal } from "@/components/export/WordExportPreviewModal";
import { logDebug } from "@/utils/normalizeModuleData";
import type { PreviewSection } from "@/utils/normalizeModuleData";
import {
  getVivaSenseMode,
  ProFeatureError,
  VIVASENSE_MODE_CHANGED_EVENT,
  VivaSenseMode,
} from "@/services/featureMode";
import { ProFeatureModal } from "./ProFeatureModal";
import { deriveAllowedCorrelationModes } from "./domainTerms";

interface TraitRelationshipsProps {
  /**
   * Populated by DataSourceTabs once the user confirms column mapping in the
   * Upload File tab.  Null until then — component shows a prompt.
   */
  datasetContext: UploadDatasetContext | null;
}

type TRStep = "setup" | "correlating" | "results";

// ─────────────────────────────────────────────────────────────────────────────
// Preview builder
// ─────────────────────────────────────────────────────────────────────────────

function buildCorrelationPreview(
  results: CorrelationResponse,
  displayMode: "phenotypic" | "between_genotype" | "genotypic",
  stats: CorrelationStats,
  method: "pearson" | "spearman",
  userObjective: string
): { sections: PreviewSection[]; warnings: string[]; notes: string[] } {
  const sections: PreviewSection[] = [];
  const warnings: string[] = [...results.warnings];

  const modeDesc: Record<string, string> = {
    phenotypic:       "Phenotypic (all observations)",
    between_genotype: "Between-Genotype Association (genotype means)",
    genotypic:        "Genotypic VC (bivariate REML)",
  };

  sections.push({
    title: "Analysis Settings",
    rows: [
      { label: "Method", value: method === "spearman" ? "Spearman ρ" : "Pearson r" },
      { label: "Objective", value: userObjective },
      { label: "Mode shown", value: modeDesc[displayMode] ?? displayMode },
      { label: "Traits", value: results.trait_names.join(", ") },
    ],
  });

  const buildModeRow = (label: string, s: CorrelationStats | null) => ({
    label,
    value: s
      ? `${s.n_observations} obs · matrix ${s.r_matrix.length}×${s.r_matrix[0]?.length ?? 0}`
      : "Not available",
  });

  sections.push({
    title: "Mode Availability",
    rows: [
      buildModeRow("Phenotypic", results.phenotypic),
      buildModeRow("Between-Genotype", results.between_genotype),
      buildModeRow("Genotypic VC", results.genotypic),
    ],
  });

  if (results.genotypic === null && userObjective === "Breeding decision") {
    warnings.push(
      "Genotypic VC unavailable — showing between-genotype association as fallback. " +
      "Ensure sommer is installed on the server and ≥ 3 genotypes per trait pair have complete data."
    );
  }

  if (stats.inference_approximate) {
    warnings.push(
      stats.inference_note ??
        "Genotypic VC p-values and CIs are approximate (Fisher z approximation). Interpret cautiously."
    );
  }

  const notes = [
    "Use SVG or PNG export from the heatmap for publication-quality figures.",
    "For a combined Word report (ANOVA + correlation), run the analysis from the Upload File tab.",
    results.statistical_note,
  ].filter(Boolean) as string[];

  logDebug("TraitRelationships:preview", {
    mode: displayMode,
    n_traits: results.trait_names.length,
    n_warnings: warnings.length,
    genotypic_available: results.genotypic !== null,
  });

  return { sections, warnings, notes };
}

// ─────────────────────────────────────────────────────────────────────────────

export function TraitRelationships({ datasetContext }: TraitRelationshipsProps) {
  const [step, setStep] = useState<TRStep>("setup");
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  // Becomes true after the user first changes the trait selection; gates the
  // validation message so it doesn't show before any interaction.
  const [traitsTouched, setTraitsTouched] = useState(false);
  const [method, setMethod] = useState<"pearson" | "spearman">("pearson");
  const [userObjective, setUserObjective] = useState<"Field understanding" | "Genotype comparison" | "Breeding decision">("Field understanding");
  const [results, setResults] = useState<CorrelationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [displayMode, setDisplayMode] = useState<"phenotypic" | "between_genotype" | "genotypic">("phenotypic");
  const [showPreview, setShowPreview] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [showProModal, setShowProModal] = useState(false);
  const [mode, setMode] = useState<VivaSenseMode>(() => getVivaSenseMode());

  // ── Domain-aware mode governance ───────────────────────────────────────────
  // Compute which correlation modes are scientifically appropriate for the
  // current dataset context. Phenotypic is always allowed; between-genotype
  // and genotypic depend on domain and genotype-column semantics.
  const { showBetweenGenotype, showGenotypic } = deriveAllowedCorrelationModes(
    datasetContext?.research_domain,
    datasetContext?.genotypeColumn ?? ""
  );

  useEffect(() => {
    const syncMode = () => setMode(getVivaSenseMode());
    window.addEventListener(VIVASENSE_MODE_CHANGED_EVENT, syncMode);
    window.addEventListener("storage", syncMode);
    return () => {
      window.removeEventListener(VIVASENSE_MODE_CHANGED_EVENT, syncMode);
      window.removeEventListener("storage", syncMode);
    };
  }, []);

  // When a new (or first) dataset context arrives, reset the trait selection
  // and discard any previous results. Also reset objective/mode to valid defaults.
  useEffect(() => {
    if (datasetContext) {
      setSelectedTraits(datasetContext.availableTraitColumns);
      setTraitsTouched(false);
      setResults(null);
      setError(null);
      setStep("setup");
      setDisplayMode("phenotypic");
      // Reset objective to "Field understanding" — always valid regardless of domain.
      setUserObjective("Field understanding");
    }
  }, [datasetContext]);

  // ── Helpers ────────────────────────────────────────────────────────────────

  const toggleTrait = (t: string) => {
    setTraitsTouched(true);
    setSelectedTraits((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]
    );
  };

  const canRun = datasetContext !== null && selectedTraits.length >= 2;

  const handleRun = async () => {
    if (!datasetContext || !canRun) return;
    setStep("correlating");
    setError(null);

    try {
      console.log("[TraitRelationships] Starting correlation analysis...", {
        traits: selectedTraits,
        method,
        objective: userObjective,
      });

      const data = await computeCorrelation({
        base64_content: datasetContext.base64Content,
        file_type: datasetContext.fileType,
        genotype_column: datasetContext.genotypeColumn,
        rep_column: datasetContext.repColumn,
        environment_column: datasetContext.environmentColumn,
        trait_columns: selectedTraits,
        mode: datasetContext.mode,
        method,
        user_objective: userObjective,
      });

      console.log("DEBUG: Raw correlation response:", data);
      console.log("[TraitRelationships] API response received successfully", {
        trait_names: data.trait_names,
        method: data.method,
        phenotypic_n: data.phenotypic?.n_observations,
        between_genotype_n: data.between_genotype?.n_observations,
        genotypic_available: data.genotypic !== null,
        genotypic_n: data.genotypic?.n_observations,
        interpretation_length: data.interpretation?.length,
        warnings_count: data.warnings?.length,
      });

      // Set results and determine initial display mode
      setResults(data);
      
      // Set initial displayMode based on objective, constrained by allowed modes.
      // Domain governance: between_genotype and genotypic may be hidden for
      // agronomy/general workflows — fall back to phenotypic when disallowed.
      let initialMode: "phenotypic" | "between_genotype" | "genotypic" = "phenotypic";
      if (userObjective === "Field understanding") {
        initialMode = "phenotypic";
      } else if (userObjective === "Genotype comparison") {
        // Only switch if between-genotype mode is allowed for this domain.
        initialMode = showBetweenGenotype ? "between_genotype" : "phenotypic";
      } else {
        // "Breeding decision" → prefer VC-based genotypic; fall back if unavailable
        // or if genotypic mode is not allowed for this domain.
        if (showGenotypic && data.genotypic) {
          initialMode = "genotypic";
        } else if (showBetweenGenotype) {
          initialMode = "between_genotype";
        } else {
          initialMode = "phenotypic";
        }
      }
      setDisplayMode(initialMode);
      
      console.log("DEBUG: State transitioning to results view. activeMode:", initialMode);

      console.log("[TraitRelationships] State updated", {
        step: "results",
        displayMode: initialMode,
      });

      setStep("results");
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Correlation failed";
      console.error("[TraitRelationships] API error:", {
        error: errorMsg,
        errorType: err instanceof Error ? err.constructor.name : typeof err,
      });
      setError(errorMsg);
      setStep("setup");
    }
  };

  const handleAdjust = () => {
    setResults(null);
    setError(null);
    setStep("setup");
    setDisplayMode("phenotypic");
    // Keep selectedTraits and method as-is — user tweaks then re-runs
  };

  const handleDownload = async () => {
    if (!results) return;
    if (mode !== "pro") {
      setShowProModal(true);
      return;
    }
    setDownloading(true);
    try {
      // Client-side docx export flow
      if (typeof exportCorrelationWord === "function") {
        await exportCorrelationWord(results);
      }
    } catch (err) {
      if (err instanceof ProFeatureError) {
        setShowProModal(true);
        return;
      }
      console.error("Export failed", err);
    } finally {
      setDownloading(false);
    }
  };

  // ── No context ─────────────────────────────────────────────────────────────

  if (!datasetContext) {
    return (
      <div className="rounded-xl border-2 border-dashed border-gray-200 bg-gray-50 p-10 text-center">
        <div className="text-4xl mb-3">🔗</div>
        <p className="font-semibold text-gray-700">No dataset loaded yet</p>
        <p className="mt-1 text-sm text-gray-500 max-w-sm mx-auto">
          Switch to the <strong>Upload File</strong> tab, upload your CSV or
          Excel file, and confirm the column mapping. Your dataset will be
          shared here automatically.
        </p>
      </div>
    );
  }

  // ── Setup: trait selection + method picker ─────────────────────────────────

  if (step === "setup") {
    const traits = datasetContext.availableTraitColumns;

    return (
      <div className="space-y-5">
        {/* Dataset banner */}
        <div className="flex items-center gap-3 rounded-lg bg-emerald-50 border border-emerald-200 p-3 text-sm">
          <span className="text-xl">📊</span>
          <div>
            <p className="font-medium text-emerald-800">
              Using uploaded dataset
            </p>
            <p className="text-emerald-600">
              {traits.length} trait{traits.length !== 1 ? "s" : ""} available
              {" · "}
              {datasetContext.mode === "multi"
                ? "Multi-environment"
                : "Single-environment"}
            </p>
          </div>
        </div>

        {/* Error from previous attempt */}
        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            <p className="font-medium">Correlation error</p>
            <p className="mt-0.5">{error}</p>
          </div>
        )}

        {/* Trait selection */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-semibold text-gray-700">
              Traits to Correlate
              {selectedTraits.length >= 2 && (
                <span className="ml-2 font-normal text-emerald-600">
                  {selectedTraits.length} selected
                </span>
              )}
              {selectedTraits.length < 2 && (
                <span className="ml-2 font-normal text-amber-600">
                  select at least 2
                </span>
              )}
            </label>
            <div className="flex gap-2 text-xs">
              <button
                type="button"
                onClick={() => { setTraitsTouched(true); setSelectedTraits(traits); }}
                className="text-emerald-600 hover:underline"
              >
                All
              </button>
              <span className="text-gray-300">|</span>
              <button
                type="button"
                onClick={() => { setTraitsTouched(true); setSelectedTraits([]); }}
                className="text-gray-400 hover:underline"
              >
                None
              </button>
            </div>
          </div>

          {traits.length === 0 ? (
            <p className="text-sm text-amber-600">
              No numeric trait columns found in this dataset. Re-upload with a
              file that contains numeric data columns.
            </p>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {traits.map((t) => (
                <label
                  key={t}
                  className={[
                    "flex items-center gap-2 rounded-lg border px-3 py-2 cursor-pointer transition-colors text-sm",
                    selectedTraits.includes(t)
                      ? "border-emerald-500 bg-emerald-50 text-emerald-800"
                      : "border-gray-200 bg-white text-gray-600 hover:border-emerald-300",
                  ].join(" ")}
                >
                  <input
                    type="checkbox"
                    checked={selectedTraits.includes(t)}
                    onChange={() => toggleTrait(t)}
                    className="rounded border-gray-300 text-emerald-600"
                  />
                  <span className="truncate font-medium">{t}</span>
                </label>
              ))}
            </div>
          )}

          {/* Inline validation — only shown after user has interacted */}
          {traitsTouched && selectedTraits.length < 2 && (
            <p className="mt-2 text-xs text-red-600">
              Select at least 2 traits to continue
            </p>
          )}
        </div>

        {/* Method selector */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Correlation Method
          </label>
          <div className="flex gap-2">
            {(["pearson", "spearman"] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMethod(m)}
                className={[
                  "rounded-lg border px-4 py-2 text-sm font-medium transition-colors",
                  method === m
                    ? "border-emerald-600 bg-emerald-600 text-white"
                    : "border-gray-300 bg-white text-gray-600 hover:border-emerald-400",
                ].join(" ")}
              >
                {m === "pearson" ? "Pearson (r)" : "Spearman (\u03c1)"}
              </button>
            ))}
          </div>
          <p className="mt-1 text-xs text-gray-400">
            Pearson assumes approximate normality. Use Spearman for ordinal or
            non-normal trait distributions.
          </p>
        </div>

        {/* User objective selector */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Analysis Objective
          </label>
          <div className="grid gap-2 sm:grid-cols-1">
            {([
              {
                value: "Field understanding",
                label: "Field Understanding → Phenotypic",
                desc: "Co-variation among all field observations; reflects genetics + environment jointly",
                allowed: true,
              },
              {
                value: "Genotype comparison",
                label: "Genotype Comparison → Between-Genotype",
                desc: "Association among genotype means; not a true genetic parameter",
                allowed: showBetweenGenotype,
              },
              {
                value: "Breeding decision",
                label: "Breeding Decision → Genotypic (VC-based)",
                desc: "Variance-component genotypic correlation via bivariate REML; falls back to between-genotype if unavailable",
                allowed: showGenotypic,
              },
            ] as const).filter((obj) => obj.allowed).map((obj) => (
              <label
                key={obj.value}
                className={[
                  "flex items-start gap-3 rounded-lg border p-3 cursor-pointer transition-colors",
                  userObjective === obj.value
                    ? "border-emerald-500 bg-emerald-50 text-emerald-800"
                    : "border-gray-200 bg-white text-gray-600 hover:border-emerald-300",
                ].join(" ")}
              >
                <input
                  type="radio"
                  name="objective"
                  value={obj.value}
                  checked={userObjective === obj.value}
                  onChange={(e) => setUserObjective(e.target.value as typeof userObjective)}
                  className="mt-0.5 rounded border-gray-300 text-emerald-600"
                />
                <div>
                  <div className="font-medium">{obj.label}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{obj.desc}</div>
                </div>
              </label>
            ))}
          </div>
          <p className="mt-1 text-xs text-gray-400">
            This influences correlation mode selection and interpretation guidance.
          </p>
        </div>

        {/* Run button */}
        <button
          type="button"
          onClick={handleRun}
          disabled={!canRun}
          className="w-full rounded-lg bg-emerald-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {selectedTraits.length < 2
            ? "Select at least 2 traits to continue"
            : `Compute Correlations — ${selectedTraits.length} trait${selectedTraits.length !== 1 ? "s" : ""}`}
        </button>
      </div>
    );
  }

  // ── Correlating: spinner ──────────────────────────────────────────────────

  if (step === "correlating") {
    return (
      <div className="flex flex-col items-center justify-center py-16 gap-4">
        <VsSpinner size="lg" monogram />
        <p className="text-sm text-gray-500">
          Computing {method === "spearman" ? "Spearman" : "Pearson"}{" "}
          correlations for {selectedTraits.length} trait
          {selectedTraits.length !== 1 ? "s" : ""}…
        </p>
        <p className="text-xs text-gray-400">
          Running phenotypic, between-genotype, and genotypic VC analysis…
        </p>
      </div>
    );
  }

  // ── Results ───────────────────────────────────────────────────────────────

  if (step === "results" && results) {
    // Debug: log that we're in results view with current displayMode
    console.log("DEBUG: Results view is ACTIVE.");
    console.log("DEBUG: Selected mode:", displayMode);
    console.log("DEBUG: Rendering temporary marker '3-mode selector mounted'");

    console.log("[TraitRelationships] Rendering results view", {
      displayMode,
      results_available: !!results,
      phenotypic_available: results.phenotypic !== null,
      between_genotype_available: results.between_genotype !== null,
      genotypic_available: results.genotypic !== null,
    });

    // Validate that displayMode has valid data AND is allowed by domain governance.
    // Fall back to phenotypic when the current mode is no longer allowed.
    let resolvedMode = displayMode;
    if (resolvedMode === "between_genotype" && !showBetweenGenotype) {
      resolvedMode = "phenotypic";
    }
    if (resolvedMode === "genotypic" && !showGenotypic) {
      resolvedMode = showBetweenGenotype ? "between_genotype" : "phenotypic";
    }
    // Sync state if we had to fall back (avoids a stale displayMode in re-renders).
    if (resolvedMode !== displayMode) {
      setDisplayMode(resolvedMode);
    }

    // Validate that displayMode has valid data
    let stats: CorrelationStats | null = null;
    if (resolvedMode === "phenotypic") {
      stats = results.phenotypic;
    } else if (resolvedMode === "between_genotype") {
      stats = results.between_genotype;
    } else if (resolvedMode === "genotypic") {
      stats = results.genotypic ?? results.between_genotype;
    }

    if (!stats) {
      console.warn("[TraitRelationships] No stats available for mode:", resolvedMode);
      stats = results.phenotypic ?? results.between_genotype;
    }

    // ── Mode availability logic ─────────────────────────────────────────────────
    const n = results.trait_names.length;

    const hasValidMatrix = (modeData: any) => {
      if (!modeData || !Array.isArray(modeData.r_matrix)) return false;
      return modeData.r_matrix.length === n && modeData.r_matrix[0]?.length === n;
    };

    const hasValidPairwiseResults = (modeData: any) => {
      if (modeData?.pairs && Array.isArray(modeData.pairs) && modeData.pairs.length > 0) return true;
      if (results.interpretation && results.interpretation.length > 0) return true;
      return false;
    };

    const hasValidStats = (modeData: any) => {
      if (!modeData) return false;
      if (typeof modeData.n_observations === "number" && modeData.n_observations > 0) return true;
      if (Array.isArray(modeData.r_matrix) && modeData.r_matrix.length > 0) return true;
      return false;
    };

    const checkModeAvailable = (modeData: any, isGenotypic: boolean) => {
      if (isGenotypic && !modeData) return false;
      if (!isGenotypic && results.interpretation) return true; // Phenotypic/between_genotype NEVER show unavailable if interpretation exists
      return hasValidMatrix(modeData) || hasValidPairwiseResults(modeData) || hasValidStats(modeData);
    };

    const phenoAvail = checkModeAvailable(results.phenotypic, false);
    // Data availability is gated by domain governance: a mode with data but
    // not allowed by domain is treated as unavailable for display purposes.
    const betweenAvail = showBetweenGenotype && checkModeAvailable(results.between_genotype, false);
    const genoAvail = showGenotypic && checkModeAvailable(results.genotypic, true);

    const hasMatrix = hasValidMatrix(stats);
    const hasPairwise = hasValidPairwiseResults(stats);
    const hasStats = hasValidStats(stats);
    
    const canExport = !!results.interpretation && (phenoAvail || betweenAvail || genoAvail);
    if (!canExport) {
      console.log("DEBUG: Export disabled. Interpretation missing or no modes have valid data.");
    }

    const modeLabelMap: Record<"phenotypic" | "between_genotype" | "genotypic", string> = {
      phenotypic:       "Phenotypic (Field-Level)",
      between_genotype: "Between-Genotype Association",
      genotypic:        "Genotypic (Variance-Component)",
    };
    const modeLabel = modeLabelMap[resolvedMode];

    // Debug logs — active mode and matrix dimensions
    const activeMatrix = stats?.r_matrix ?? [];
    console.log("[TraitRelationships] Results view state", {
      activeMode: resolvedMode,
      modeLabel,
      n_traits: results.trait_names.length,
      n_observations: stats?.n_observations,
      matrix_rows: activeMatrix.length,
      matrix_cols: activeMatrix[0]?.length ?? 0,
      genotypic_available: results.genotypic !== null,
      inference_approximate: stats?.inference_approximate,
      domain: datasetContext?.research_domain,
      showBetweenGenotype,
      showGenotypic,
    });

    // Domain-aware mode selector: only show modes appropriate for this workflow.
    const visibleModes = [
      { value: "phenotypic"       as const, label: "Phenotypic correlation",        desc: "field-level",                visible: true,                available: phenoAvail },
      { value: "between_genotype" as const, label: "Between-genotype association",  desc: "from genotype means",       visible: showBetweenGenotype, available: betweenAvail },
      { value: "genotypic"        as const, label: "Genotypic correlation",         desc: "variance-component based",  visible: showGenotypic,       available: genoAvail },
    ].filter((m) => m.visible);

    return (
      <div className="space-y-5">
        {/* Header */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">
              Correlation Analysis
            </h3>
            <p className="text-sm text-gray-500">
              {results.trait_names.length} traits ·{" "}
              {stats?.n_observations ?? 0}{" "}
              {resolvedMode === "phenotypic" ? "observations" : "genotype means"} ·{" "}
              {results.method === "spearman" ? "Spearman" : "Pearson"} · {modeLabel}
            </p>
            {userObjective === "Breeding decision" && !results.genotypic && showGenotypic && (
              <p className="text-xs text-amber-600 mt-0.5">
                ⚠ Genotypic VC not available — showing between-genotype association as fallback
              </p>
            )}
          </div>
        <div className="flex gap-2 shrink-0">
          <button
            type="button"
            onClick={handleAdjust}
            className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm text-gray-600 hover:bg-gray-50"
          >
            Adjust Selection
          </button>
          <button
            type="button"
            onClick={() => setShowPreview(true)}
            disabled={!canExport}
            className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm text-gray-600 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Preview Report
          </button>
          <button
            type="button"
            onClick={handleDownload}
            disabled={!canExport || downloading}
            className="rounded-lg border border-emerald-600 bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {downloading
              ? "Generating…"
              : mode === "pro"
              ? "Download Report (.docx)"
              : "🔒 Pro · Download Report (.docx)"}
          </button>
        </div>
        </div>

        {/* Mode selector — domain-aware: only scientifically appropriate modes shown */}
        <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
          <label className="block text-sm font-semibold text-gray-700 mb-3">
            Display Correlation Mode
          </label>
          <div className={`grid gap-2 ${visibleModes.length === 1 ? "sm:grid-cols-1" : visibleModes.length === 2 ? "sm:grid-cols-2" : "sm:grid-cols-3"}`}>
            {visibleModes.map((m) => (
              <button
                key={m.value}
                type="button"
                disabled={!m.available}
                onClick={() => {
                  console.log("[TraitRelationships] Mode switched to:", m.value);
                  setDisplayMode(m.value);
                }}
                className={[
                  "relative rounded-lg border p-3 text-left text-sm transition-colors",
                  resolvedMode === m.value
                    ? "border-emerald-500 bg-emerald-50 text-emerald-900"
                    : !m.available
                    ? "border-gray-200 bg-gray-100 text-gray-400 cursor-not-allowed opacity-50"
                    : "border-gray-300 bg-white text-gray-700 hover:border-emerald-300",
                ].join(" ")}
              >
                <div className="font-medium">{m.label}</div>
                <div className="text-xs text-gray-500 mt-0.5">
                  {!m.available ? "Not available" : m.desc}
                </div>
              </button>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {showGenotypic
              ? "Switch modes to view different correlation perspectives. Genotypic VC requires ≥3 genotypes per trait pair."
              : showBetweenGenotype
              ? "Phenotypic and between-genotype modes are available for this workflow."
              : "Phenotypic correlation is shown for this agronomy/general workflow."}
          </p>
        </div>

        {/* Data warnings */}
        {results.warnings.length > 0 && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 space-y-0.5">
            <p className="text-xs font-semibold text-amber-700">
              Data warnings
            </p>
            {results.warnings.map((w, idx) => (
              <p
                key={idx}
                className="text-xs text-amber-600 flex items-start gap-1"
              >
                <span className="mt-0.5 shrink-0">⚠</span> {w}
              </p>
            ))}
          </div>
        )}

        {/* Statistical basis notes */}
        <div className="space-y-1">
          <p className="flex items-start gap-1.5 text-xs text-gray-400">
            <span className="shrink-0 mt-px">ℹ</span>
            {results.statistical_note}
          </p>
          <p className="flex items-start gap-1.5 text-xs text-gray-400">
            <span className="shrink-0 mt-px">ℹ</span>
            Phenotypic and between-genotype p-values use standard cor.test(). P-values are
            unadjusted for multiple comparisons; FDR-adjusted values are reported in the
            interpretation text.
          </p>
          {resolvedMode === "genotypic" && results.genotypic?.inference_approximate && (
            <p className="flex items-start gap-1.5 text-xs text-amber-600">
              <span className="shrink-0 mt-px">⚠</span>
              {results.genotypic.inference_note ??
                "Genotypic VC p-values and CIs are approximate (Fisher z on n_genotypes). " +
                "Interpret cautiously — see interpretation text for details."}
            </p>
          )}
          <p className="flex items-start gap-1.5 text-xs text-gray-400">
            <span className="shrink-0 mt-px">ℹ</span>
            Analysis objective: {userObjective}
          </p>
        </div>

        {/* Heatmap — below stats notes, above interpretation */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-gray-700">Correlation Heatmap</h4>
            <span className="text-xs text-gray-400 font-normal">
              {modeLabel} · Red = negative · White = zero · Blue = positive
            </span>
          </div>
          <CorrelationHeatmap data={results} mode={resolvedMode} />
        </div>

        {/* Interpretation */}
        <div className="rounded-lg bg-emerald-50 border border-emerald-100 p-4">
          <p className="text-xs font-semibold text-emerald-700 mb-1">
            Interpretation
          </p>
          <p className="text-sm text-gray-700 leading-relaxed">
            {results.interpretation}
          </p>
        </div>

      {/* Word Export Modal */}
      {showPreview && (
        <WordExportPreviewModal
          moduleName="Correlation Analysis"
          reportTitle="Multi-Trait Correlation Report"
          datasetSummary={`${stats?.n_observations ?? 0} observations · ${results.trait_names.length} traits`}
          sections={buildCorrelationPreview(results, resolvedMode, stats!, results.method, userObjective).sections}
          warnings={buildCorrelationPreview(results, resolvedMode, stats!, results.method, userObjective).warnings}
          notes={buildCorrelationPreview(results, resolvedMode, stats!, results.method, userObjective).notes}
          canExport={canExport}
          onExport={handleDownload}
          onClose={() => setShowPreview(false)}
        />
      )}
      <ProFeatureModal
        open={showProModal}
        onClose={() => setShowProModal(false)}
        onActivated={() => setMode(getVivaSenseMode())}
      />
      </div>
    );
  }

  return null;
}
