import React, { useEffect, useMemo, useState } from "react";
import {
  UploadDatasetContext,
  UploadAnalysisResponse,
  analyzeUpload,
} from "@/services/geneticsUploadApi";
import { ResultsDisplay } from "@/components/vivasense-genetics/ResultsDisplay";
import { VsSpinner } from "@/components/vivasense-genetics/VsSpinner";
import {
  detectExperimentalDesign,
  DesignDetectionResult,
  ColumnAnalysis,
} from "@/components/vivasense-genetics/designDetection";
import { DesignRecommendationCard } from "@/components/vivasense-genetics/DesignRecommendationCard";
import {
  getDesignAwareLabels,
  getColumnPlaceholder,
  getFieldHelpText,
} from "@/components/vivasense-genetics/designLabels";
import { DomainKey } from "@/components/vivasense-genetics/domainTerms";

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
  const [designDetection, setDesignDetection] = useState<DesignDetectionResult | null>(null);
  const [showRecommendation, setShowRecommendation] = useState(false);

  // Domain from dataset context
  const domain: DomainKey = datasetContext?.research_domain ?? "general";

  // Get design-aware labels
  const labels = getDesignAwareLabels(design, domain);

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

  useEffect(() => {
    if (!datasetContext) return;
    setSelectedTraits(datasetContext.availableTraitColumns);
    setTreatmentColumn(datasetContext.genotypeColumn ?? "");
    setRepColumn(datasetContext.repColumn ?? "");

    const genoCol = datasetContext.genotypeColumn ?? "";
    const repCol = datasetContext.repColumn ?? "";
    const eligible = categoricalColumns.filter(
      (col) => col !== genoCol && col !== repCol
    );
    setFactorA(eligible[0] ?? "");
    setFactorB(eligible[1] ?? "");

    setMainPlot("");
    setSubPlot("");
    setError(null);
    setResults(null);

    // Auto-detect experimental design
    const columnAnalysis: ColumnAnalysis[] = categoricalColumns.map(col => ({
      name: col,
      uniqueValues: 0, // Would need actual data to compute this precisely
      isNumeric: false,
      sampleValues: [],
    }));
    
    const detection = detectExperimentalDesign(columnAnalysis, categoricalColumns);
    setDesignDetection(detection);
    
    // Show recommendation card if confidence is medium or high
    if (detection.confidence !== "low") {
      setShowRecommendation(true);
      
      // Auto-populate fields if high confidence
      if (detection.confidence === "high") {
        setDesign(detection.suggestedDesign);
        
        // Auto-populate split-plot fields
        if (detection.suggestedDesign === "split_plot_rcbd") {
          if (detection.detectedFactors.blocking?.[0]) {
            setRepColumn(detection.detectedFactors.blocking[0]);
          }
          if (detection.detectedFactors.possibleMainPlot?.[0]) {
            setMainPlot(detection.detectedFactors.possibleMainPlot[0]);
          }
          if (detection.detectedFactors.possibleSubplot?.[0]) {
            setSubPlot(detection.detectedFactors.possibleSubplot[0]);
          }
        }
      }
    }
  }, [datasetContext, categoricalColumns]);

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

      {/* Design Recommendation Card */}
      {showRecommendation && designDetection && (
        <DesignRecommendationCard
          detection={designDetection}
          onAccept={() => {
            setDesign(designDetection.suggestedDesign);
            setShowRecommendation(false);
            
            // Auto-populate fields for split-plot
            if (designDetection.suggestedDesign === "split_plot_rcbd") {
              if (designDetection.detectedFactors.blocking?.[0]) {
                setRepColumn(designDetection.detectedFactors.blocking[0]);
              }
              if (designDetection.detectedFactors.possibleMainPlot?.[0]) {
                setMainPlot(designDetection.detectedFactors.possibleMainPlot[0]);
              }
              if (designDetection.detectedFactors.possibleSubplot?.[0]) {
                setSubPlot(designDetection.detectedFactors.possibleSubplot[0]);
              }
            }
          }}
          onDismiss={() => setShowRecommendation(false)}
        />
      )}

      <div className="rounded-xl border border-gray-200 bg-white p-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h3 className="text-base font-semibold text-gray-800">ANOVA Module Setup</h3>
          <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-medium text-emerald-700">
            {design.toUpperCase()} design
          </span>
        </div>

        <p className="mb-3 text-sm text-gray-500">
          {labels.designDescription}
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

          {/* Treatment/Factor column (design-specific) */}
          {design !== "split_plot_rcbd" && design !== "factorial" && (
            <Field 
              label={labels.treatmentLabel} 
              required
              helpText="Column containing treatment identifiers (e.g., variety names, treatment codes)"
            >
              <ColumnSelect 
                value={treatmentColumn} 
                onChange={setTreatmentColumn} 
                options={categoricalColumns}
                placeholder={getColumnPlaceholder("treatment", design, domain)}
              />
            </Field>
          )}

          {/* Blocking/Replication column */}
          {design !== "crd" && (
            <Field 
              label={labels.blockingLabel} 
              required
              helpText={getFieldHelpText("blocking", design)}
            >
              <ColumnSelect 
                value={repColumn} 
                onChange={setRepColumn} 
                options={categoricalColumns}
                placeholder={getColumnPlaceholder("blocking", design, domain)}
              />
            </Field>
          )}

          {/* Factorial Design Fields */}
          {design === "factorial" && (
            <>
              <Field 
                label={labels.factorALabel} 
                required
                helpText={getFieldHelpText("factorA", design)}
              >
                <ColumnSelect 
                  value={factorA} 
                  onChange={setFactorA} 
                  options={categoricalColumns}
                  placeholder={getColumnPlaceholder("factorA", design, domain)}
                />
              </Field>
              <Field 
                label={labels.factorBLabel} 
                required
                helpText={getFieldHelpText("factorB", design)}
              >
                <ColumnSelect 
                  value={factorB} 
                  onChange={setFactorB} 
                  options={categoricalColumns}
                  placeholder={getColumnPlaceholder("factorB", design, domain)}
                />
              </Field>
            </>
          )}

          {/* Split-Plot Design Fields */}
          {design === "split_plot_rcbd" && (
            <>
              <Field 
                label={labels.mainPlotLabel} 
                required
                helpText={getFieldHelpText("mainPlot", design)}
              >
                <ColumnSelect 
                  value={mainPlot} 
                  onChange={setMainPlot} 
                  options={categoricalColumns}
                  placeholder={getColumnPlaceholder("mainPlot", design, domain)}
                />
              </Field>
              <Field 
                label={labels.subplotLabel} 
                required
                helpText={getFieldHelpText("subplot", design)}
              >
                <ColumnSelect 
                  value={subPlot} 
                  onChange={setSubPlot} 
                  options={categoricalColumns}
                  placeholder={getColumnPlaceholder("subplot", design, domain)}
                />
              </Field>
            </>
          )}
        </div>

        <div className="mt-4 space-y-2">
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold text-gray-700">
              Select response traits to analyse
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
              Select at least one trait to run ANOVA.
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
                Running...
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
  helpText,
  children,
}: {
  label: string;
  required?: boolean;
  helpText?: string | null;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="mb-1 flex items-center gap-1.5 text-sm font-medium text-gray-700">
        {label}
        {required && <span className="text-red-500">*</span>}
        {helpText && (
          <span className="group relative cursor-help">
            <svg
              className="h-4 w-4 text-gray-400 hover:text-gray-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span className="invisible group-hover:visible absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 rounded-lg bg-gray-900 px-2 py-1.5 text-xs text-white shadow-lg z-10">
              {helpText}
            </span>
          </span>
        )}
      </label>
      {children}
    </div>
  );
}

function ColumnSelect({
  value,
  onChange,
  options,
  placeholder = "Select column…",
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  placeholder?: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm"
    >
      <option value="">{placeholder}</option>
      {options.map((option) => (
        <option key={option} value={option}>
          {option}
        </option>
      ))}
    </select>
  );
}
