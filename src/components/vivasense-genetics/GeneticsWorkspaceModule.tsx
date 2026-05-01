import React, { useState } from "react";
import {
  UploadDatasetContext,
  UploadAnalysisResponse,
  analyzeUpload,
} from "@/services/geneticsUploadApi";
import { ResultsDisplay } from "@/components/vivasense-genetics/ResultsDisplay";
import { VsSpinner } from "@/components/vivasense-genetics/VsSpinner";

interface GeneticsWorkspaceModuleProps {
  datasetContext: UploadDatasetContext | null;
}

export function GeneticsWorkspaceModule({ datasetContext }: GeneticsWorkspaceModuleProps) {
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [selectionIntensity, setSelectionIntensity] = useState(1.4);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<UploadAnalysisResponse | null>(null);

  if (!datasetContext) {
    return (
      <div className="rounded-2xl border border-dashed border-gray-300 bg-white p-10 text-center">
        <p className="text-sm font-medium text-gray-700">Upload and confirm a dataset first to compute genetic parameters.</p>
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

  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-gray-200 bg-white p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h3 className="text-base font-semibold text-gray-800">Genetic Parameters Workspace</h3>
          <span className="rounded-full bg-violet-50 px-2.5 py-1 text-xs font-medium text-violet-700">
            Pro Module
          </span>
        </div>

        <p className="mb-3 text-sm text-gray-600">
          Run broad-sense heritability, GCV, PCV, and GAM across multiple traits in one pass.
        </p>

        <div className="mb-4">
          <label className="mb-1 block text-sm font-medium text-gray-700">Selection intensity (k)</label>
          <input
            type="number"
            min={0}
            step="0.01"
            value={selectionIntensity}
            onChange={(e) => setSelectionIntensity(Number(e.target.value) || 1.4)}
            className="w-full max-w-xs rounded-lg border border-gray-300 px-3 py-2 text-sm"
          />
          <p className="mt-1 text-xs text-gray-500">Default 1.4 (~20% selection). Use 2.06 for 5% selection.</p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-gray-700">Traits</p>
            <div className="flex gap-2 text-xs">
              <button
                type="button"
                onClick={() => setSelectedTraits(traits)}
                className="rounded-full border border-emerald-300 px-2.5 py-1 text-emerald-700"
              >
                Select all
              </button>
              <button
                type="button"
                onClick={() => setSelectedTraits([])}
                className="rounded-full border border-gray-300 px-2.5 py-1 text-gray-600"
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
                    "rounded-full border px-3 py-1.5 text-sm",
                    active
                      ? "border-emerald-600 bg-emerald-600 text-white"
                      : "border-gray-300 text-gray-700 hover:border-emerald-300",
                  ].join(" ")}
                >
                  {trait}
                </button>
              );
            })}
          </div>
        </div>

        {error && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="mt-4">
          <button
            type="button"
            onClick={run}
            disabled={selectedTraits.length === 0 || running}
            className="rounded-lg bg-emerald-600 px-5 py-2 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-50"
          >
            {running ? (
              <span className="inline-flex items-center gap-2">
                <VsSpinner size="sm" className="border-white" />
                Computing parameters…
              </span>
            ) : (
              "Compute Genetic Parameters"
            )}
          </button>
        </div>
      </div>

      {results && <ResultsDisplay results={results} onReset={() => setResults(null)} />}
    </div>
  );
}
