import React, { useEffect, useMemo, useState } from "react";
import {
  UploadDatasetContext,
  UploadAnalysisResponse,
  analyzeUpload,
} from "@/services/geneticsUploadApi";
import { ResultsDisplay } from "@/components/vivasense-genetics/ResultsDisplay";
import { VsSpinner } from "@/components/vivasense-genetics/VsSpinner";

type AnovaDesign = "crd" | "rcbd" | "factorial" | "split_plot_rcbd";

interface AnovaWorkspaceModuleProps {
  datasetContext: UploadDatasetContext | null;
}

export function AnovaWorkspaceModule({ datasetContext }: AnovaWorkspaceModuleProps) {
  const [design, setDesign] = useState<AnovaDesign>("rcbd");
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [treatmentColumn, setTreatmentColumn] = useState<string>(datasetContext?.genotypeColumn ?? "");
  const [repColumn, setRepColumn] = useState<string>(datasetContext?.repColumn ?? "");
  const [factorA, setFactorA] = useState<string>("");
  const [factorB, setFactorB] = useState<string>("");
  const [mainPlot, setMainPlot] = useState<string>("");
  const [subPlot, setSubPlot] = useState<string>("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<UploadAnalysisResponse | null>(null);

  useEffect(() => {
    if (!datasetContext) return;
    setSelectedTraits(datasetContext.availableTraitColumns);
    setTreatmentColumn(datasetContext.genotypeColumn ?? "");
    setRepColumn(datasetContext.repColumn ?? "");
    setFactorA("");
    setFactorB("");
    setMainPlot("");
    setSubPlot("");
    setError(null);
    setResults(null);
  }, [datasetContext]);

  const categoricalColumns = useMemo(() => {
    if (!datasetContext) return [];
    const traits = new Set(datasetContext.availableTraitColumns);
    const candidates = [
      datasetContext.genotypeColumn,
      datasetContext.repColumn,
      datasetContext.environmentColumn,
    ].filter(Boolean) as string[];
    const pool = new Set<string>([...candidates]);
    return Array.from(pool).filter((c) => !traits.has(c));
  }, [datasetContext]);

  if (!datasetContext) {
    return (
      <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 p-8 text-center">
        <p className="text-sm font-medium text-gray-600">No dataset loaded</p>
        <p className="mt-1 text-xs text-gray-400">
          Upload a file in the Multi-Trait File tab to run ANOVA.
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

  const valid =
    selectedTraits.length > 0 &&
    treatmentColumn &&
    (design === "crd" || repColumn) &&
    (design !== "factorial" || (factorA && factorB)) &&
    (design !== "split_plot_rcbd" || (mainPlot && subPlot && repColumn));

  const selectAllTraits = () => setSelectedTraits([...traits]);
  const clearAllTraits = () => setSelectedTraits([]);

  const run = async () => {
    if (!valid) return;
    setRunning(true);
    setError(null);
    setResults(null);
    try {
      const data = await analyzeUpload({
        base64_content: datasetContext.base64Content,
        file_type: datasetContext.fileType,
        genotype_column: treatmentColumn,
        rep_column: design === "crd" ? "" : repColumn,
        environment_column: datasetContext.environmentColumn ?? null,
        trait_columns: selectedTraits,
        mode: datasetContext.mode,
        random_environment: false,
        selection_intensity: 1.4,
        module: "anova",
        design_type: design,
        treatment_column: treatmentColumn,
        factor_a_column: design === "factorial" ? factorA : undefined,
        factor_b_column: design === "factorial" ? factorB : undefined,
        main_plot_column: design === "split_plot_rcbd" ? mainPlot : undefined,
        sub_plot_column: design === "split_plot_rcbd" ? subPlot : undefined,
      });
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "ANOVA failed");
    } finally {
      setRunning(false);
    }
  };

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
          <h3 className="text-base font-semibold text-gray-800">ANOVA Module Setup</h3>
          <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700">
            {design.toUpperCase()} design
          </span>
        </div>

        <p className="mb-3 text-sm text-gray-500">
          Configure design columns once and run ANOVA across multiple response traits.
        </p>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Field label="Design" required>
            <select
              value={design}
              onChange={(e) => setDesign(e.target.value as AnovaDesign)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
            >
              <option value="crd">CRD</option>
              <option value="rcbd">RCBD</option>
              <option value="factorial">Factorial</option>
              <option value="split_plot_rcbd">Split-Plot RCBD</option>
            </select>
          </Field>

          <Field label="Treatment / Genotype" required>
            <ColumnSelect value={treatmentColumn} onChange={setTreatmentColumn} options={categoricalColumns} />
          </Field>

          {design !== "crd" && (
            <Field label="Replication / Block" required>
              <ColumnSelect value={repColumn} onChange={setRepColumn} options={categoricalColumns} />
            </Field>
          )}

          {design === "factorial" && (
            <>
              <Field label="Factor A" required>
                <ColumnSelect value={factorA} onChange={setFactorA} options={categoricalColumns} />
              </Field>
              <Field label="Factor B" required>
                <ColumnSelect value={factorB} onChange={setFactorB} options={categoricalColumns} />
              </Field>
            </>
          )}

          {design === "split_plot_rcbd" && (
            <>
              <Field label="Main plot factor" required>
                <ColumnSelect value={mainPlot} onChange={setMainPlot} options={categoricalColumns} />
              </Field>
              <Field label="Subplot factor" required>
                <ColumnSelect value={subPlot} onChange={setSubPlot} options={categoricalColumns} />
              </Field>
            </>
          )}
        </div>

        <div className="mt-4 space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold text-gray-700">
              Response traits
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
                All
              </button>
              <span className="text-gray-300">|</span>
              <button
                type="button"
                onClick={clearAllTraits}
                className="text-gray-400 hover:underline"
              >
                None
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
                    "rounded-full border px-3 py-1.5 text-sm font-medium transition-colors",
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
              Select at least one trait to run ANOVA.
            </p>
          )}
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
            disabled={!valid || running}
            className={[
              "w-full rounded-lg px-4 py-2.5 text-sm font-semibold transition-colors",
              valid && !running
                ? "bg-emerald-600 text-white hover:bg-emerald-700"
                : "bg-gray-100 text-gray-400 cursor-not-allowed",
            ].join(" ")}
          >
            {running ? (
              <span className="inline-flex items-center gap-2">
                <VsSpinner size="sm" className="border-white" />
                Running ANOVA...
              </span>
            ) : (
              `Run ANOVA - ${selectedTraits.length} trait${selectedTraits.length !== 1 ? "s" : ""}`
            )}
          </button>
        </div>
      </div>

      {results && <ResultsDisplay results={results} onReset={() => setResults(null)} />}
    </div>
  );
}

function Field({
  label,
  required,
  children,
}: {
  label: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-gray-700">
        {label}
        {required && <span className="ml-1 text-red-500">*</span>}
      </label>
      {children}
    </div>
  );
}

function ColumnSelect({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
    >
      <option value="">Select column…</option>
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  );
}
