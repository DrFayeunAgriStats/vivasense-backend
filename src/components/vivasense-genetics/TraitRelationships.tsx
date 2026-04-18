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
} from "@/services/traitRelationshipsApi";
import { CorrelationHeatmap } from "./CorrelationHeatmap";

interface TraitRelationshipsProps {
  /**
   * Populated by DataSourceTabs once the user confirms column mapping in the
   * Upload File tab.  Null until then — component shows a prompt.
   */
  datasetContext: UploadDatasetContext | null;
}

type TRStep = "setup" | "correlating" | "results";

export function TraitRelationships({ datasetContext }: TraitRelationshipsProps) {
  const [step, setStep] = useState<TRStep>("setup");
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [method, setMethod] = useState<"pearson" | "spearman">("pearson");
  const [userObjective, setUserObjective] = useState<"Field understanding" | "Genotype comparison" | "Breeding decision">("Field understanding");
  const [results, setResults] = useState<CorrelationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // When a new (or first) dataset context arrives, reset the trait selection
  // and discard any previous results.
  useEffect(() => {
    if (datasetContext) {
      setSelectedTraits(datasetContext.availableTraitColumns);
      setResults(null);
      setError(null);
      setStep("setup");
    }
  }, [datasetContext]);

  // ── Helpers ────────────────────────────────────────────────────────────────

  const toggleTrait = (t: string) =>
    setSelectedTraits((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]
    );

  const canRun = datasetContext !== null && selectedTraits.length >= 2;

  const handleRun = async () => {
    if (!datasetContext || !canRun) return;
    setStep("correlating");
    setError(null);

    try {
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
      setResults(data);
      setStep("results");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Correlation failed");
      setStep("setup");
    }
  };

  const handleAdjust = () => {
    setResults(null);
    setError(null);
    setStep("setup");
    // Keep selectedTraits and method as-is — user tweaks then re-runs
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
                onClick={() => setSelectedTraits(traits)}
                className="text-emerald-600 hover:underline"
              >
                All
              </button>
              <span className="text-gray-300">|</span>
              <button
                type="button"
                onClick={() => setSelectedTraits([])}
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
              },
              {
                value: "Genotype comparison",
                label: "Genotype Comparison → Between-Genotype",
                desc: "Association among genotype means; not a true genetic parameter",
              },
              {
                value: "Breeding decision",
                label: "Breeding Decision → Genotypic (VC-based)",
                desc: "Variance-component genotypic correlation via bivariate REML; falls back to between-genotype if unavailable",
              },
            ] as const).map((obj) => (
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
        <div className="h-10 w-10 rounded-full border-4 border-emerald-600 border-t-transparent animate-spin" />
        <p className="text-sm text-gray-500">
          Computing {method === "spearman" ? "Spearman" : "Pearson"}{" "}
          correlations for {selectedTraits.length} trait
          {selectedTraits.length !== 1 ? "s" : ""}…
        </p>
        <p className="text-xs text-gray-400">
          Correlations are computed on per-genotype means
        </p>
      </div>
    );
  }

  // ── Results ───────────────────────────────────────────────────────────────

  if (step === "results" && results) {
    // Map objective → display mode; fall back gracefully when genotypic VC is null
    let displayMode: 'phenotypic' | 'between_genotype' | 'genotypic';
    if (userObjective === "Field understanding") {
      displayMode = "phenotypic";
    } else if (userObjective === "Genotype comparison") {
      displayMode = "between_genotype";
    } else {
      // "Breeding decision" → prefer VC-based genotypic; fall back if unavailable
      displayMode = results.genotypic ? "genotypic" : "between_genotype";
    }

    const modeLabelMap: Record<typeof displayMode, string> = {
      phenotypic:       "Phenotypic (Field-Level)",
      between_genotype: "Between-Genotype Association",
      genotypic:        "Genotypic (Variance-Component)",
    };
    const modeLabel = modeLabelMap[displayMode];
    const stats =
      displayMode === "phenotypic" ? results.phenotypic :
      displayMode === "between_genotype" ? results.between_genotype :
      (results.genotypic ?? results.between_genotype);

    return (
      <div className="space-y-5">
        {/* Header */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-800">
              Three-Mode Correlation Analysis
            </h3>
            <p className="text-sm text-gray-500">
              {results.trait_names.length} traits ·{" "}
              {stats.n_observations}{" "}
              {displayMode === "phenotypic" ? "observations" : "genotype means"} ·{" "}
              {results.method === "spearman" ? "Spearman" : "Pearson"} · {modeLabel}
            </p>
            {userObjective === "Breeding decision" && !results.genotypic && (
              <p className="text-xs text-amber-600 mt-0.5">
                ⚠ Genotypic VC not available — showing between-genotype association as fallback
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={handleAdjust}
            className="shrink-0 rounded-lg border border-gray-300 px-4 py-1.5 text-sm text-gray-600 hover:bg-gray-50"
          >
            Adjust Selection
          </button>
        </div>

        {/* Data warnings */}
        {results.warnings.length > 0 && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 space-y-0.5">
            <p className="text-xs font-semibold text-amber-700">
              Data warnings
            </p>
            {results.warnings.map((w) => (
              <p
                key={w}
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
          {displayMode === "genotypic" && results.genotypic?.inference_approximate && (
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

        {/* Debug logging */}
        {console.log("TraitRelationships: correlation response", {
          phenotypic: results.phenotypic,
          between_genotype: results.between_genotype,
          genotypic: results.genotypic,
          displayMode,
          userObjective,
        })}

        {/* Heatmap — below stats notes, above interpretation */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-gray-700">Correlation Heatmap</h4>
            <span className="text-xs text-gray-400 font-normal">
              {modeLabel} · Red = negative · White = zero · Blue = positive
            </span>
          </div>
          <CorrelationHeatmap data={results} mode={displayMode} />
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
      </div>
    );
  }

  return null;
}
