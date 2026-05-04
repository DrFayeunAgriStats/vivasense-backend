import React, { useEffect, useState } from "react";
import {
  UploadDatasetContext,
  UploadAnalysisResponse,
  analyzeUpload,
} from "@/services/geneticsUploadApi";
import { ResultsDisplay } from "@/components/vivasense-genetics/ResultsDisplay";
import { VsSpinner } from "@/components/vivasense-genetics/VsSpinner";
import {
  DEFAULT_SELECTION_INTENSITY,
  SELECTION_INTENSITIES,
  selectionIntensityDisclosure,
} from "@/components/vivasense-genetics/selectionIntensity";

interface GeneticsWorkspaceModuleProps {
  datasetContext: UploadDatasetContext | null;
}

export function GeneticsWorkspaceModule({ datasetContext }: GeneticsWorkspaceModuleProps) {
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [selectionIntensity, setSelectionIntensity] = useState(DEFAULT_SELECTION_INTENSITY);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<UploadAnalysisResponse | null>(null);

  useEffect(() => {
    if (!datasetContext) return;
    setSelectedTraits(datasetContext.availableTraitColumns);
    setSelectionIntensity(DEFAULT_SELECTION_INTENSITY);
    setError(null);
    setResults(null);
  }, [datasetContext]);

  if (!datasetContext) {
    return (
      <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 p-8 text-center">
        <p className="text-sm font-medium text-gray-600">No dataset loaded</p>
        <p className="mt-1 text-xs text-gray-400">
          Upload a file in the Multi-Trait File tab to compute genetic parameters.
        </p>
      </div>
    );
  }

  const traits = datasetContext.availableTraitColumns;

  const toggleTrait = (trait: string) => {
    setSelectedTraits((prev) =>
      prev.includes(trait) ? prev.filter((t) => t !== trait) : [...prev, trait]
    );
  };

  const run = async () => {
    if (selectedTraits.length === 0) return;
    setRunning(true);
    setError(null);
    setResults(null);
    try {
      const data = await analyzeUpload({
        base64_content: datasetContext.base64Content,
        file_type: datasetContext.fileType,
        genotype_column: datasetContext.genotypeColumn,
        rep_column: datasetContext.repColumn,
        environment_column: datasetContext.environmentColumn ?? null,
        trait_columns: selectedTraits,
        mode: datasetContext.mode,
        random_environment: false,
        selection_intensity: selectionIntensity,
        module: "genetic_parameters",
      });
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Genetics analysis failed");
    } finally {
      setRunning(false);
    }
  };

  const selectAllTraits = () => setSelectedTraits([...traits]);
  const clearAllTraits = () => setSelectedTraits([]);

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2">
        <span className="h-2 w-2 shrink-0 rounded-full bg-emerald-500" />
        <p className="text-xs text-emerald-700">
          Dataset confirmed <span className="font-medium">{datasetContext.file.name}</span>
          {" "}- {traits.length} numeric trait{traits.length !== 1 ? "s" : ""} available
        </p>
      </div>

      <div className="rounded-xl border border-gray-200 bg-white p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h3 className="text-base font-semibold text-gray-800">Genetic Parameters Module Setup</h3>
          <span className="rounded-full bg-violet-50 px-2.5 py-1 text-xs font-medium text-violet-700">
            Pro Module
          </span>
        </div>

        <p className="mb-3 text-sm text-gray-500">
          Run broad-sense heritability, GCV, PCV, and GAM across multiple traits in one pass.
        </p>

        <div className="mb-4">
          <div className="mb-1 flex items-center gap-1.5">
            <label className="block text-sm font-medium text-gray-700">Selection Intensity</label>
            <button
              type="button"
              className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-gray-300 text-xs text-gray-600"
              title="Selection intensity (i) is the standardised selection differential. It determines how aggressively superior genotypes are selected. Default is 10% (i = 1.40) per Falconer & Mackay (1996)."
              aria-label="Selection intensity information"
            >
              i
            </button>
          </div>
          <select
            value={selectionIntensity}
            onChange={(e) => setSelectionIntensity(Number(e.target.value) || DEFAULT_SELECTION_INTENSITY)}
            className="w-full max-w-xs rounded-lg border border-gray-300 px-3 py-2 text-sm"
          >
            {SELECTION_INTENSITIES.map((option) => (
              <option key={option.pct} value={option.value}>
                {option.label} (i = {option.value.toFixed(3)}) - {option.note}
              </option>
            ))}
          </select>
          <p className="mt-1 text-xs text-gray-400">
            {selectionIntensityDisclosure(selectionIntensity)}
          </p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold text-gray-700">
              Select traits to analyse
              {selectedTraits.length > 0 && (
                <span className="ml-2 font-normal text-emerald-600">
                  {selectedTraits.length} selected
                </span>
              )}
            </p>
            <div className="flex gap-2 text-xs">
              <button
                type="button"
                onClick={selectAllTraits}
                className="text-emerald-600 hover:underline"
              >
                Select all
              </button>
              <span className="text-gray-300">|</span>
              <button
                type="button"
                onClick={clearAllTraits}
                className="text-gray-500 hover:underline"
              >
                Clear
              </button>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {traits.map((trait) => {
              const active = selectedTraits.includes(trait);
              return (
                <button
                  key={trait}
                  type="button"
                  onClick={() => toggleTrait(trait)}
                  className={[
                    "rounded-full border px-3 py-1 text-sm font-medium transition-colors",
                    active
                      ? "border-emerald-600 bg-emerald-600 text-white"
                      : "border-gray-300 bg-white text-gray-600 hover:border-emerald-400 hover:text-emerald-700",
                  ].join(" ")}
                >
                  {trait}
                </button>
              );
            })}
          </div>

          {selectedTraits.length === 0 && (
            <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-3 py-2">
              Select at least one trait to compute genetic parameters.
            </p>
          )}
        </div>

        {error && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3">
            <p className="text-sm font-medium text-red-700">Analysis failed</p>
            <p className="mt-0.5 break-all text-xs font-mono text-red-600">{error}</p>
          </div>
        )}

        <div className="mt-4">
          <button
            type="button"
            onClick={run}
            disabled={selectedTraits.length === 0 || running}
            className={[
              "w-full rounded-lg px-4 py-2.5 text-sm font-semibold transition-colors",
              selectedTraits.length > 0 && !running
                ? "bg-emerald-600 text-white hover:bg-emerald-700"
                : "bg-gray-100 text-gray-400 cursor-not-allowed",
            ].join(" ")}
          >
            {running ? (
              <span className="inline-flex items-center gap-2">
                <VsSpinner size="sm" className="border-white" />
                Running...
              </span>
            ) : (
              `Compute Genetic Parameters - ${selectedTraits.length} trait${selectedTraits.length !== 1 ? "s" : ""}`
            )}
          </button>
        </div>
      </div>

      {results && (
        <ResultsDisplay
          results={results}
          onReset={() => setResults(null)}
          domain={datasetContext.research_domain}
        />
      )}
    </div>
  );
}
