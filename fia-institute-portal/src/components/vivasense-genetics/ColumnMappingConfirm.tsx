import { inferSelectionPercent, selectionIntensityDisclosure, DEFAULT_SELECTION_INTENSITY, SELECTION_INTENSITY_OPTIONS } from "./selectionIntensity";
import React, { useState } from "react";
import { UploadPreviewResponse } from "@/services/geneticsUploadApi";
import { VsSpinner } from "./VsSpinner";
import {
  DEFAULT_SELECTION_INTENSITY,
  SELECTION_INTENSITY_OPTIONS,
  selectionIntensityDisclosure,
} from "./selectionIntensity";
import {
  DomainKey,
  RESEARCH_DOMAINS,
  detectDomainFromColumns,
} from "./domainTerms";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type ExperimentalDesign = "CRD" | "RCBD" | "Factorial" | "Split-Plot" | "MET";

const ADVANCED_SETTINGS_SESSION_KEY = "vivasense:column-mapping:advanced-settings:open";

interface ColumnMapping {
  genotypeColumn: string;
  repColumn: string;
  environmentColumn: string;
  selectedTraits: string[];
  mode: "single" | "multi";
  randomEnvironment: boolean;
  selectionIntensity: number;
  numericFactorOverrides: string[];
  research_domain: DomainKey;
}

interface ColumnMappingConfirmProps {
  preview: UploadPreviewResponse;
  onConfirm: (mapping: ColumnMapping) => void;
  onBack: () => void;
  loading: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const DESIGN_OPTIONS: { value: ExperimentalDesign; label: string; desc: string }[] = [
  { value: "CRD",        label: "CRD",        desc: "Completely Randomized Design" },
  { value: "RCBD",       label: "RCBD",       desc: "Randomized Complete Block Design" },
  { value: "Factorial",  label: "Factorial",  desc: "Factorial Design" },
  { value: "Split-Plot", label: "Split-Plot", desc: "Split-Plot Design" },
  { value: "MET",        label: "MET",        desc: "Multi-Environment Trial" },
];

function normalizeLabel(value: unknown): string {
  return String(value ?? "").trim().toLowerCase();
}

function getColumnPreviewValues(rows: Record<string, unknown>[], column: string): string[] {
  if (!column) return [];
  return rows
    .map((row) => row[column])
    .map((value) => String(value ?? "").trim())
    .filter((value) => value.length > 0);
}

function isNumericPreview(values: string[]): boolean {
  return values.length > 0 && values.every((value) => /^[-+]?\d+(\.\d+)?$/.test(value));
}

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export function ColumnMappingConfirm({
  preview,
  onConfirm,
  onBack,
  loading,
}: ColumnMappingConfirmProps) {
  const { detected_columns, column_names, data_preview, n_rows, n_columns, warnings } = preview;

  // Numeric columns = server-detected trait columns.
  // Categorical columns = everything else in column_names.
  // Ambiguous columns (neither clearly numeric nor structural) fall through to categorical.
  const numericColumns: string[] = detected_columns.traits;
  const categoricalColumns: string[] = column_names.filter(
    (c: string) => !numericColumns.includes(c)
  );

  // ── Section A ─────────────────────────────────────────────────────────────

  const [design, setDesign] = useState<ExperimentalDesign>("RCBD");
  const [domain, setDomain] = useState<DomainKey>(() =>
    detectDomainFromColumns(column_names)
  );

  const hideTreatmentSelector = design === "Split-Plot" || design === "Factorial";

  // ── Section B ─────────────────────────────────────────────────────────────

  const [selectedTraits, setSelectedTraits] = useState<string[]>(
    detected_columns.traits
  );
  const [treatmentColumn, setTreatmentColumn] = useState(
    detected_columns.genotype?.column ?? ""
  );
  const [replicationColumn, setReplicationColumn] = useState(
    detected_columns.rep?.column ?? ""
  );
  const [mainPlotFactor, setMainPlotFactor] = useState("");
  const [subPlotFactor, setSubPlotFactor] = useState("");
  const [factorA, setFactorA] = useState("");
  const [factorB, setFactorB] = useState("");
  // Factor C is optional, default to empty string
  const [factorC, setFactorC] = useState(""); 

  const [environmentColumn, setEnvironmentColumn] = useState(
    detected_columns.environment?.column ?? ""
  );
  const [numericFactorOverrides, setNumericFactorOverrides] = useState<string[]>([]);
  const [selectionIntensity, setSelectionIntensity] = useState(DEFAULT_SELECTION_INTENSITY);
  const [consentGiven, setConsentGiven] = React.useState(false);

  // TASK: Simplified cleanup logic — clear treatment state when selector is hidden
  React.useEffect(() => {
    if (hideTreatmentSelector) {
      setTreatmentColumn("");
    }
  }, [hideTreatmentSelector]);

  // ── Section C ─────────────────────────────────────────────────────────────

  const [advancedOpen, setAdvancedOpen] = useState(() => {
    if (typeof window === "undefined") {
      return false;
    }
    try {
      return window.sessionStorage.getItem(ADVANCED_SETTINGS_SESSION_KEY) === "open";
    } catch {
      return false;
    }
  });
  const [meanSeparation, setMeanSeparation] = useState(false);
  const [meanSepMethod, setMeanSepMethod] = useState<"tukey" | "lsd">("tukey");
  const [includeInteraction, setIncludeInteraction] = useState(false);
  const [outputFormat, setOutputFormat] = useState<"tables" | "tables_ai">("tables_ai");

  // ── Section D: Validation — touched tracking ──────────────────────────────
  // Validation messages only show for fields the user has interacted with,
  // so the form doesn't open with a wall of red text.
  const [touched, setTouched] = useState<Set<string>>(new Set());
  const touch = (field: string) =>
    setTouched((prev) => new Set([...prev, field]));

  const toggleTrait = (trait: string) => {
    setSelectedTraits((prev) =>
      prev.includes(trait) ? prev.filter((item) => item !== trait) : [...prev, trait]
    );
    touch("selectedTraits");
  };

  const toggleColumnType = (colName: string) => {
    setNumericFactorOverrides((prev) =>
      prev.includes(colName)
        ? prev.filter((c) => c !== colName)
        : [...prev, colName]
    );
  };

  // ── Data quality score ────────────────────────────────────────────────────
  // Heuristic: start at 100, subtract for warnings and data gaps
  const missingWarnings = warnings.filter(
    (w) => /missing|null|empty|blank/i.test(w)
  ).length;
  const qualityScore =
    warnings.length === 0 && n_rows >= 10 && numericColumns.length >= 1
      ? "excellent"
      : missingWarnings > 0 || warnings.length > 2
      ? "poor"
      : warnings.length > 0 || n_rows < 10
      ? "moderate"
      : "good";

  const qualityConfig = {
    excellent: { label: "Ready", bar: "bg-emerald-500", text: "text-emerald-700", border: "border-emerald-200", bg: "bg-emerald-50", width: "w-full" },
    good:      { label: "Ready", bar: "bg-emerald-400", text: "text-emerald-700", border: "border-emerald-200", bg: "bg-emerald-50", width: "w-4/5" },
    moderate:  { label: "Check Required", bar: "bg-amber-400", text: "text-amber-700", border: "border-amber-200", bg: "bg-amber-50", width: "w-3/5" },
    poor:      { label: "Issues Detected", bar: "bg-red-400", text: "text-red-700", border: "border-red-200", bg: "bg-red-50", width: "w-2/5" },
  }[qualityScore];

  const validationItems = [
    !detected_columns.genotype?.column
      ? {
          title: "Treatment or genotype variable",
          severity: "review",
          recommendation: "No structural treatment column was confidently identified. Map this manually before analysis.",
        }
      : {
          title: "Treatment or genotype variable",
          severity: detected_columns.genotype.confidence === "high" ? "ok" : "review",
          recommendation: `Detected ${detected_columns.genotype.column} with ${detected_columns.genotype.confidence} confidence.`,
        },
    !detected_columns.rep?.column && design !== "CRD"
      ? {
          title: "Replication structure",
          severity: "review",
          recommendation: "A replication or block variable is required for this design. Confirm the correct column before running ANOVA.",
        }
      : {
          title: "Replication structure",
          severity: design === "CRD" ? "ok" : detected_columns.rep?.confidence === "high" ? "ok" : "review",
          recommendation: design === "CRD"
            ? "CRD selected, so no replication column is required."
            : `Replication column ${detected_columns.rep?.column ?? ""} detected${detected_columns.rep ? ` with ${detected_columns.rep.confidence} confidence.` : "."}`,
        },
    design === "MET"
      ? {
          title: "Environment balance",
          severity: detected_columns.environment?.column ? (detected_columns.environment.confidence === "high" ? "ok" : "review") : "attention",
          recommendation: detected_columns.environment?.column
            ? `Environment column ${detected_columns.environment.column} detected. Confirm this before G×E interpretation.`
            : "MET selected, but no environment column has been confirmed yet.",
        }
      : null,
    warnings.length > 0
      ? {
          title: "Preview warnings",
          severity: warnings.length > 2 ? "attention" : "review",
          recommendation: warnings[0],
        }
      : {
          title: "Preview warnings",
          severity: "ok",
          recommendation: "No preview warnings were raised by the upload validator.",
        },
  ].filter(Boolean) as Array<{ title: string; severity: "ok" | "review" | "attention"; recommendation: string }>;

  // ── Derived visibility ────────────────────────────────────────────────────
  const showReplication = design !== "CRD";
  const showSplitPlot   = design === "Split-Plot";
  const showEnvironment = design === "MET";
  const showInteraction = design === "Factorial" || design === "MET";

  const sampleValidationItems = React.useMemo(() => {
    const items: Array<{
      title: string;
      severity: "ok" | "review" | "attention";
      explanation: string;
      action: string;
      proceed: string;
    }> = [];

    const previewRows = data_preview as Record<string, unknown>[];
    const treatmentValues = getColumnPreviewValues(previewRows, treatmentColumn);
    const replicationValues = getColumnPreviewValues(previewRows, replicationColumn);
    const environmentValues = getColumnPreviewValues(previewRows, environmentColumn);
    const emptyColumns = column_names.filter((column) => getColumnPreviewValues(previewRows, column).length === 0);
    const traitIssues = selectedTraits.filter((trait) => {
      const values = getColumnPreviewValues(previewRows, trait);
      return values.length > 0 && !isNumericPreview(values);
    });

    if (showReplication) {
      const missingRepCount = previewRows.length - replicationValues.length;
      if (missingRepCount > 0) {
        items.push({
          title: "Missing replication values",
          severity: "attention",
          explanation: `${missingRepCount} preview row${missingRepCount !== 1 ? "s appear" : " appears"} to lack a replication entry.`,
          action: "Check the replication or block column for blanks before analysis.",
          proceed: "Analysis should be reviewed before proceeding.",
        });
      }
    }

    if (treatmentValues.length > 0) {
      const treatmentCounts = treatmentValues.reduce<Record<string, number>>((acc, value) => {
        acc[value] = (acc[value] ?? 0) + 1;
        return acc;
      }, {});
      const counts = Object.values(treatmentCounts);
      if (counts.length > 1 && Math.max(...counts) - Math.min(...counts) > 1) {
        items.push({
          title: "Unbalanced treatment counts",
          severity: "review",
          explanation: "Preview rows suggest unequal treatment frequencies.",
          action: "Confirm whether this imbalance is expected or due to omitted observations.",
          proceed: "Analysis can proceed, but interpretation should consider imbalance.",
        });
      }

      const normalizedGroups = treatmentValues.reduce<Record<string, Set<string>>>((acc, value) => {
        const normalized = normalizeLabel(value);
        if (!acc[normalized]) acc[normalized] = new Set<string>();
        acc[normalized].add(value);
        return acc;
      }, {});
      const inconsistentTreatmentLabels = Object.values(normalizedGroups).filter((set) => set.size > 1).length;
      if (inconsistentTreatmentLabels > 0) {
        items.push({
          title: "Inconsistent treatment labels",
          severity: "review",
          explanation: "Preview rows suggest the same treatment may appear with inconsistent spelling or spacing.",
          action: "Standardize labels such as genotype or treatment names before running final analyses.",
          proceed: "Analysis can proceed, but label inconsistency can fragment groups.",
        });
      }
    }

    if (showEnvironment) {
      if (environmentValues.length === 0) {
        items.push({
          title: "Environment structure",
          severity: "attention",
          explanation: "No non-empty environment labels were found in preview rows for the selected MET structure.",
          action: "Confirm the correct environment column before G×E analysis.",
          proceed: "Analysis should not proceed until environment structure is confirmed.",
        });
      } else {
        const normalizedGroups = environmentValues.reduce<Record<string, Set<string>>>((acc, value) => {
          const normalized = normalizeLabel(value);
          if (!acc[normalized]) acc[normalized] = new Set<string>();
          acc[normalized].add(value);
          return acc;
        }, {});
        const inconsistentEnvironmentLabels = Object.values(normalizedGroups).filter((set) => set.size > 1).length;
        if (inconsistentEnvironmentLabels > 0) {
          items.push({
            title: "Inconsistent environment labels",
            severity: "review",
            explanation: "Environment names in preview rows appear to vary in formatting.",
            action: "Normalize environment labels to avoid splitting the same site or season into separate groups.",
            proceed: "Analysis can proceed, but G×E interpretation may be distorted.",
          });
        }
        if (new Set(environmentValues.map(normalizeLabel)).size < 2) {
          items.push({
            title: "Suspicious MET structure",
            severity: "review",
            explanation: "The selected MET structure currently shows fewer than two distinct environments in preview rows.",
            action: "Confirm that the uploaded dataset truly contains multiple environments and that the environment column is mapped correctly.",
            proceed: "Analysis can proceed only after confirming the intended MET structure.",
          });
        }
      }
    }

    if (showReplication && new Set(replicationValues.map(normalizeLabel)).size < 2) {
      items.push({
        title: "Insufficient replication",
        severity: "review",
        explanation: "Preview rows show fewer than two distinct replication levels.",
        action: "Verify the block or replication column and confirm the design selection.",
        proceed: "Analysis can proceed cautiously, but inferential strength may be limited.",
      });
    }

    if (traitIssues.length > 0) {
      items.push({
        title: "Non-numeric response values",
        severity: "attention",
        explanation: `Selected response variable${traitIssues.length !== 1 ? "s" : ""} ${traitIssues.join(", ")} include non-numeric values in preview rows.`,
        action: "Review data entry or formatting for the affected response variables.",
        proceed: "Analysis should be reviewed before proceeding.",
      });
    }

    if (numericFactorOverrides.length > 0) {
      items.push({
        title: "Numeric-coded factor handling",
        severity: "ok",
        explanation: `Numeric-coded treatment factor${numericFactorOverrides.length !== 1 ? "s" : ""} marked: ${numericFactorOverrides.join(", ")}.`,
        action: "These columns will be treated as categorical design factors rather than numeric response variables.",
        proceed: "Analysis can proceed safely if this reflects the intended experimental structure.",
      });
    }

    if (design === "Split-Plot") {
      if (!mainPlotFactor || !subPlotFactor || mainPlotFactor === subPlotFactor) {
        items.push({
          title: "Split-plot hierarchy",
          severity: "attention",
          explanation: "Main-plot and sub-plot factors are incomplete or duplicated.",
          action: "Assign distinct main-plot and sub-plot variables before running the split-plot model.",
          proceed: "Analysis should not proceed until the split-plot hierarchy is complete.",
        });
      }
    }

    if (emptyColumns.length > 0) {
      items.push({
        title: "Empty columns in preview",
        severity: "review",
        explanation: `Preview rows suggest empty columns such as ${emptyColumns.slice(0, 3).join(", ")}${emptyColumns.length > 3 ? " and others" : ""}.`,
        action: "Remove unused columns or confirm they are intentionally blank beyond the preview sample.",
        proceed: "Analysis can proceed, but unnecessary columns may complicate mapping.",
      });
    }

    if (items.length === 0) {
      items.push({
        title: "Scientific validation",
        severity: "ok",
        explanation: "No additional structural concerns were identified in the preview sample.",
        action: "Proceed to analysis after confirming mapped variables.",
        proceed: "Analysis can proceed safely.",
      });
    }

    return items;
  }, [
    column_names,
    data_preview,
    design,
    environmentColumn,
    numericFactorOverrides,
    replicationColumn,
    selectedTraits,
    showEnvironment,
    showReplication,
    treatmentColumn,
    mainPlotFactor,
    subPlotFactor,
  ]);

  // ── Handlers ─────────────────────────────────────────────────────────────

  // Design change: clear conditional fields; retain selected traits and treatment.
  const handleDesignChange = (d: ExperimentalDesign) => {
    setDesign(d);
    setReplicationColumn("");
    setMainPlotFactor("");
    setSubPlotFactor("");
    setFactorA("");
    setFactorB("");
    setFactorC("");
    setEnvironmentColumn("");
    // Remove touched state for cleared fields so their errors reset.
    setTouched((prev) => {
      const next = new Set(prev);
      next.delete("replicationColumn");
      next.delete("mainPlotFactor");
      next.delete("subPlotFactor");
      next.delete("environmentColumn");
      return next;
    });
  };

  // Readiness check — all required fields filled.
  const canSubmit =
    selectedTraits.length > 0 &&
    (hideTreatmentSelector || treatmentColumn !== "") &&
    (!showReplication || replicationColumn !== "") &&
    (!showSplitPlot   || (mainPlotFactor !== "" && subPlotFactor !== "" && mainPlotFactor !== subPlotFactor)) &&
    (design !== "Factorial" || (
      factorA !== "" && factorB !== "" && factorA !== factorB &&
      (factorC === "" || (factorC !== factorA && factorC !== factorB))
    )) &&
    (!showEnvironment || environmentColumn !== "");

  // ── Validation messages (Section D) ──────────────────────────────────────
  // Each message is null when the field is untouched or valid.
  const errors = {
    selectedTraits:
      touched.has("selectedTraits") && selectedTraits.length === 0
        ? "Select at least one trait to continue"
        : null,
    treatmentColumn:
      !hideTreatmentSelector && touched.has("treatmentColumn") && treatmentColumn === ""
        ? "Select a treatment or genotype variable"
        : null,
    replicationColumn:
      showReplication && touched.has("replicationColumn") && replicationColumn === ""
        ? "Select a replication column for RCBD"
        : null,
    mainPlotFactor:
      showSplitPlot && touched.has("mainPlotFactor") && mainPlotFactor === ""
        ? "Select main plot factor for Split-Plot design"
        : null,
    subPlotFactor:
      showSplitPlot && touched.has("subPlotFactor") && subPlotFactor === ""
        ? "Select sub-plot factor for Split-Plot design"
        : null,
    environmentColumn:
      showEnvironment && touched.has("environmentColumn") && environmentColumn === ""
        ? "Select environment column for MET"
        : null,
  };

  const handleSubmit = () => {
    if (!canSubmit || loading) return;
    // Always allow genotypeColumn to be optional (empty string) for all designs
    onConfirm({
      genotypeColumn: treatmentColumn,
      repColumn:         showReplication ? replicationColumn : "",
      environmentColumn: showEnvironment ? environmentColumn : "",
      selectedTraits,
      mode:              design === "MET" ? "multi" : "single",
      randomEnvironment: false,
      selectionIntensity,
      numericFactorOverrides,
      research_domain:   domain,
    });
  };

  const toggleAdvancedOpen = () => {
    setAdvancedOpen((current) => {
      const next = !current;
      if (typeof window !== "undefined") {
        try {
          window.sessionStorage.setItem(
            ADVANCED_SETTINGS_SESSION_KEY,
            next ? "open" : "closed"
          );
        } catch {
          // Ignore storage failures and keep the in-memory toggle working.
        }
      }
      return next;
    });
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">

      <div className={`rounded-2xl border ${qualityConfig.border} ${qualityConfig.bg} p-4 space-y-3`}>
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span className="text-2xl">📊</span>
            <div>
              <p className={`font-semibold text-sm ${qualityConfig.text}`}>
                Dataset Quality — <span className="font-bold">{qualityConfig.label}</span>
              </p>
              <p className="text-xs text-gray-500 mt-0.5">
                {n_rows} rows · {n_columns} columns
              </p>
            </div>
          </div>
          <div className="text-right text-xs text-gray-500 shrink-0">
            <span className={`inline-block font-semibold ${qualityConfig.text}`}>
              {numericColumns.length} numeric
            </span>
            {" · "}
            {categoricalColumns.length} categorical
          </div>
        </div>

        {/* Readiness bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-400">
            <span>Data readiness</span>
            <span className={qualityConfig.text}>{qualityConfig.label}</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-gray-200/70 overflow-hidden">
            <div className={`h-full rounded-full transition-all ${qualityConfig.bar} ${qualityConfig.width}`} />
          </div>
        </div>

        {/* Column type legend */}
        {(numericColumns.length > 0 || categoricalColumns.length > 0) && (
          <div className="flex flex-wrap gap-1.5">
            {numericColumns.slice(0, 6).map((c) => (
              <span key={c} className="inline-block rounded-full bg-emerald-100 text-emerald-700 text-xs px-2 py-0.5">
                {c}
              </span>
            ))}
            {numericColumns.length > 6 && (
              <span className="inline-block rounded-full bg-emerald-100 text-emerald-600 text-xs px-2 py-0.5">
                +{numericColumns.length - 6} more
              </span>
            )}
            {categoricalColumns.slice(0, 4).map((c) => (
              <span key={c} className="inline-block rounded-full bg-gray-100 text-gray-500 text-xs px-2 py-0.5">
                {c}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="rounded-2xl border border-gray-200 bg-white p-4 space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-gray-500">Scientific validation</p>
            <p className="mt-1 text-sm text-gray-700 leading-relaxed">
              Review these checks before analysis. The goal is to confirm that variable mapping and dataset structure are methodologically defensible.
            </p>
          </div>
          <span className={[
            "inline-flex rounded-full px-3 py-1 text-xs font-semibold",
            qualityScore === "poor"
              ? "bg-red-100 text-red-700"
              : qualityScore === "moderate"
              ? "bg-amber-100 text-amber-700"
              : "bg-emerald-100 text-emerald-700",
          ].join(" ")}>
            {qualityScore === "poor" ? "Review carefully" : qualityScore === "moderate" ? "Review before proceeding" : "Proceed after confirmation"}
          </span>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          {[...validationItems, ...sampleValidationItems].map((item) => (
            <div key={`${item.title}-${"recommendation" in item ? item.recommendation : item.explanation}`} className="rounded-xl border border-gray-200 bg-gray-50 px-4 py-3">
              <div className="flex items-center gap-2">
                <span className={[
                  "rounded-full px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide",
                  item.severity === "ok"
                    ? "bg-emerald-100 text-emerald-700"
                    : item.severity === "review"
                    ? "bg-amber-100 text-amber-700"
                    : "bg-red-100 text-red-700",
                ].join(" ")}>{item.severity}</span>
                <p className="text-sm font-semibold text-gray-800">{item.title}</p>
              </div>
              <p className="mt-2 text-sm text-gray-600 leading-relaxed">{"explanation" in item ? item.explanation : item.recommendation}</p>
              {"action" in item && (
                <p className="mt-2 text-xs text-gray-700"><span className="font-semibold">Recommended action:</span> {item.action}</p>
              )}
              {"proceed" in item && (
                <p className="mt-1 text-xs text-gray-500"><span className="font-semibold">Proceeding note:</span> {item.proceed}</p>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Server-side warnings */}
      {warnings.length > 0 && (
        <div className="space-y-1">
          {warnings.map((w) => (
            <p key={w} className="text-sm text-amber-700 flex items-start gap-1.5">
              <span className="mt-0.5">⚠</span> {w}
            </p>
          ))}
        </div>
      )}

      {/* ── Research Domain selector ────────────────────────────────────────── */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Research Domain
          <span className="ml-2 text-xs font-normal text-gray-400">
            (auto-detected · you can override)
          </span>
        </label>
        <div className="grid grid-cols-3 gap-2">
          {RESEARCH_DOMAINS.map((rd) => (
            <button
              key={rd.value}
              type="button"
              onClick={() => setDomain(rd.value)}
              className={[
                "rounded-xl border px-3 py-2.5 text-left transition-colors",
                domain === rd.value
                  ? "border-emerald-600 bg-emerald-50 ring-1 ring-emerald-500"
                  : "border-gray-200 bg-white hover:border-emerald-300 hover:bg-emerald-50/50",
              ].join(" ")}
            >
              <span className="text-lg">{rd.icon}</span>
              <p className="mt-1 text-xs font-semibold text-gray-800 leading-tight">{rd.label}</p>
              <p className="mt-0.5 text-[10px] text-gray-400 leading-snug">{rd.desc}</p>
            </button>
          ))}
        </div>
      </div>

      {/* ── Section A: Experimental Design ─────────────────────────────────── */}

      <FormField
        label="Experimental Design"
        helper="Select the design used in your experiment."
        required
      >
        <select
          value={design}
          onChange={(e) => handleDesignChange(e.target.value as ExperimentalDesign)}
          className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
        >
          {DESIGN_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label} — {opt.desc}
            </option>
          ))}
        </select>
      </FormField>

      <FormField
        label="Selection Intensity"
        helper="Controls how stringent selection is for GA and GAM calculations."
        required
      >
          {domain === "plant_breeding" && (
            <div className="space-y-2">
              <div className="flex items-center gap-1.5">
                <select
                  value={selectionIntensity}
                  onChange={(e) => setSelectionIntensity(Number(e.target.value) || DEFAULT_SELECTION_INTENSITY)}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
                >
                  {SELECTION_INTENSITY_OPTIONS.map((option) => (
                    <option key={option.pct} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-gray-300 text-xs text-gray-600"
                  title="Selection intensity (i) is the standardised selection differential. It determines how aggressively superior genotypes are selected. Default is 10% (i = 1.40) per Falconer & Mackay (1996)."
                  aria-label="Selection intensity information"
                >
                  i
                </button>
              </div>
              <p className="text-xs text-gray-500">{selectionIntensityDisclosure(selectionIntensity)}</p>
            </div>
          )}
      </FormField>

      {/* ── Section B: Core Variables ───────────────────────────────────────── */}

      <div className="space-y-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-gray-400">Variables</p>

        {/* B1 — Response Variables */}
        <FormField
          label="Response Variables (Traits)"
          helper="Select one or more numeric traits for batch analysis"
          required
          error={errors.selectedTraits}
        >
          <div className="space-y-3 rounded-lg border border-gray-200 p-3">
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  setSelectedTraits(numericColumns);
                  touch("selectedTraits");
                }}
                className="rounded-full border border-emerald-300 px-3 py-1 text-xs font-medium text-emerald-700 hover:bg-emerald-50"
              >
                Select all
              </button>
              <button
                type="button"
                onClick={() => {
                  setSelectedTraits([]);
                  touch("selectedTraits");
                }}
                className="rounded-full border border-gray-300 px-3 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50"
              >
                Clear
              </button>
              <span className="text-xs text-gray-500">
                {selectedTraits.length} of {numericColumns.length} selected
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {numericColumns.map((col) => {
                const active = selectedTraits.includes(col);
                const isOverridden = numericFactorOverrides.includes(col);
                return (
                  <div key={col}>
                    <button
                      type="button"
                      onClick={() => toggleTrait(col)}
                      className={[
                        "rounded-full border px-3 py-1.5 text-sm transition-colors",
                        active
                          ? "border-emerald-600 bg-emerald-600 text-white"
                          : "border-gray-300 text-gray-700 hover:border-emerald-300 hover:bg-emerald-50",
                      ].join(" ")}
                    >
                      {col}
                    </button>
                    <div style={{ marginTop: 4 }}>
                      <button
                        onClick={() => toggleColumnType(col)}
                        className="text-xs px-2 py-1 rounded-full border transition-all"
                        style={{
                          background: isOverridden ? "#0F6E56" : "#F3F4F6",
                          color: isOverridden ? "white" : "#374151",
                          borderColor: isOverridden ? "#0F6E56" : "#D1D5DB",
                          cursor: "pointer",
                        }}
                        type="button"
                      >
                        {isOverridden ? "✓ Treatment Factor" : "Numeric Trait"}
                      </button>
                      {isOverridden && (
                        <p style={{ fontSize: 11, color: "#0F6E56", marginTop: 2 }}>
                          Will be used as a categorical treatment factor in the analysis model.
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
            <p style={{ fontSize: 11, color: "#6B7280", marginTop: 8, fontStyle: "italic" }}>
              💡 If your column contains numeric treatment levels (e.g. 0, 50, 100 kg N/ha
              or storage days 1, 3, 6, 9), toggle it to Treatment Factor.
            </p>
          </div>
        </FormField>

        {/* B2 — Treatment / Genotype */}
        {!hideTreatmentSelector && <FormField
          label="Treatment / Genotype"
          helper="Select the genotype or treatment variable"
          required
          error={errors.treatmentColumn}
        >
          <ColumnSelect
            value={treatmentColumn}
            onChange={(v) => { setTreatmentColumn(v); touch("treatmentColumn"); }}
            options={categoricalColumns}
            placeholder="— Select categorical column —"
          />
        </FormField>}

        {/* B3 — Replication / Block (hidden for CRD) */}
        {showReplication && (
          <FormField
            label="Replication / Block"
            helper="Select the replication or block column"
            required
            error={errors.replicationColumn}
          >
            <ColumnSelect
              value={replicationColumn}
              onChange={(v) => { setReplicationColumn(v); touch("replicationColumn"); }}
              options={categoricalColumns}
              placeholder="— Select categorical column —"
            />
          </FormField>
        )}

        {/* B4a — Main Plot Factor (Split-Plot only) */}
        {showSplitPlot && (
          <FormField
            label="Main Plot Factor"
            required
            error={errors.mainPlotFactor}
          >
            <ColumnSelect
              value={mainPlotFactor}
              onChange={(v) => { setMainPlotFactor(v); touch("mainPlotFactor"); }}
              options={categoricalColumns}
              placeholder="— Select categorical column —"
            />
          </FormField>
        )}

        {/* B4b — Sub-Plot Factor (Split-Plot only) */}
        {showSplitPlot && (
          <FormField
            label="Sub-Plot Factor"
            required
            error={errors.subPlotFactor}
          >
            <ColumnSelect
              value={subPlotFactor}
              onChange={(v) => { setSubPlotFactor(v); touch("subPlotFactor"); }}
              options={categoricalColumns}
              placeholder="— Select categorical column —"
            />
          </FormField>
        )}

        {/* Factorial Factors */}
        {design === "Factorial" && (
          <div className="grid gap-4 sm:grid-cols-3">
            <FormField label="Factor A" required>
              <ColumnSelect value={factorA} onChange={setFactorA} options={categoricalColumns} />
            </FormField>
            <FormField label="Factor B" required>
              <ColumnSelect value={factorB} onChange={setFactorB} options={categoricalColumns} />
            </FormField>
            <FormField 
              label="Factor C (Optional)"
              // Factor C is optional, so no 'required' prop
              // No error prop needed as it's optional
            >
              {/* Add "None" option to Factor C selector */}
              <ColumnSelect value={factorC} onChange={setFactorC} options={["", ...categoricalColumns]} placeholder="None" />
              <ColumnSelect value={factorC} onChange={setFactorC} options={categoricalColumns} />
            </FormField>
          </div>
        )}

        {/* B5 — Environment / Location (MET only) */}
        {showEnvironment && (
          <FormField
            label="Environment / Location"
            helper="Required for G×E analysis"
            required
            error={errors.environmentColumn}
          >
            <ColumnSelect
              value={environmentColumn}
              onChange={(v) => { setEnvironmentColumn(v); touch("environmentColumn"); }}
              options={categoricalColumns}
              placeholder="— Select categorical column —"
            />
          </FormField>
        )}
      </div>

      {/* ── Section C: Advanced Settings (collapsible) ──────────────────────── */}

      <div className="rounded-lg border border-gray-200">
        <button
          type="button"
          onClick={toggleAdvancedOpen}
          aria-expanded={advancedOpen}
          className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
        >
          Advanced Settings
          <span className="text-gray-400 text-xs">{advancedOpen ? "▲" : "▼"}</span>
        </button>

        {advancedOpen && (
          <div className="border-t border-gray-200 px-4 py-4 space-y-4">

            {/* Mean separation */}
            <label className="flex items-start gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={meanSeparation}
                onChange={(e) => setMeanSeparation(e.target.checked)}
                className="mt-0.5 rounded border-gray-300 text-emerald-600"
              />
              <div>
                <span className="text-sm text-gray-700">Perform Mean Separation</span>
                {meanSeparation && (
                  <div className="mt-2 flex gap-2">
                    {(["tukey", "lsd"] as const).map((m) => (
                      <button
                        key={m}
                        type="button"
                        onClick={() => setMeanSepMethod(m)}
                        className={[
                          "rounded border px-3 py-1 text-xs font-medium transition-colors",
                          meanSepMethod === m
                            ? "border-emerald-600 bg-emerald-600 text-white"
                            : "border-gray-300 text-gray-600 hover:border-emerald-400",
                        ].join(" ")}
                      >
                        {m === "tukey" ? "Tukey HSD" : "LSD"}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </label>

            {/* Interaction effects (Factorial / MET only) */}
            {showInteraction && (
              <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={includeInteraction}
                  onChange={(e) => setIncludeInteraction(e.target.checked)}
                  className="rounded border-gray-300 text-emerald-600"
                />
                Include Interaction Effects
              </label>
            )}

            {/* Output format */}
            <div>
              <p className="text-sm font-medium text-gray-700 mb-2">Output Format</p>
              {([
                { value: "tables",    label: "Tables only" },
                { value: "tables_ai", label: "Tables + AI Interpretation" },
              ] as const).map((opt) => (
                <label
                  key={opt.value}
                  className="flex items-center gap-2 text-sm text-gray-700 mb-1.5 cursor-pointer"
                >
                  <input
                    type="radio"
                    name="output_format"
                    value={opt.value}
                    checked={outputFormat === opt.value}
                    onChange={() => setOutputFormat(opt.value)}
                    className="border-gray-300 text-emerald-600"
                  />
                  {opt.label}
                  {opt.value === "tables_ai" && (
                    <span className="text-xs text-gray-400">(default)</span>
                  )}
                </label>
              ))}
            </div>

          </div>
        )}
      </div>

      {/* Data preview (collapsible) */}
      <details className="group">
        <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700 select-none">
          Preview first rows ▸
        </summary>
        <div className="mt-2 overflow-x-auto rounded-lg border border-gray-200">
          <table className="min-w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                {column_names.map((col) => (
                  <th
                    key={col}
                    className="px-3 py-2 text-left font-semibold text-gray-600 whitespace-nowrap"
                  >
                    {col}
                    {numericColumns.includes(col) && (
                      <span className="ml-1 text-emerald-500 font-normal">#</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data_preview.map((row, i) => (
                <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                  {column_names.map((col) => (
                    <td key={col} className="px-3 py-1.5 text-gray-700 whitespace-nowrap">
                      {row[col] == null ? (
                        <span className="text-gray-300">—</span>
                      ) : (
                        String(row[col])
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-1 text-xs text-gray-400">
          # = numeric column available for analysis
        </p>
      </details>

      {/* ── Section E: Analysis Summary (shown only when all fields are filled) ── */}

      {canSubmit && (
        <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-4 space-y-3">
          <p className="text-sm font-semibold text-emerald-800">Analysis Configuration Summary</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2 text-sm">
            <SummaryRow
              label="Design"
              value={DESIGN_OPTIONS.find((o) => o.value === design)?.desc ?? design}
            />
            <SummaryRow label="Research domain" value={RESEARCH_DOMAINS.find((item) => item.value === domain)?.label ?? domain} />
            <SummaryRow
              label="Response variables"
              value={selectedTraits.length === 1 ? selectedTraits[0] : `${selectedTraits.length} selected`}
            />
            <SummaryRow
              label="Treatment factor"
              value={treatmentColumn || "—"}
            />
            <SummaryRow
              label="Replications"
              value={design === "CRD" ? "Not required" : replicationColumn || "—"}
            />
            {showEnvironment && <SummaryRow label="Environment factor" value={environmentColumn || "—"} />}
            <SummaryRow label="Observations" value={n_rows.toLocaleString()} />
            <SummaryRow label="Selection intensity" value={`i = ${selectionIntensity.toFixed(3)}`} />
          </div>
          <p className="text-xs text-emerald-800/80 leading-relaxed">
            Review this configuration before execution. VivaSense will use these mapped variables to build the statistical model and downstream interpretation.
          </p>
        </div>
      )}

      {/* ── Actions ─────────────────────────────────────────────────────────── */}

      {canSubmit && (
        <p className="flex items-center gap-1.5 text-sm font-medium text-emerald-600">
          <span aria-hidden="true">✓</span> Ready to run analysis
        </p>
      )}

      {/* ── Data Privacy Notice ─────────────────────────────────────────── */}
      <div style={{
        marginTop: 8,
        padding: "10px 14px",
        background: "#F9FAFB",
        borderRadius: 8,
        fontSize: 12,
        color: "#6B7280",
        lineHeight: 1.6,
        border: "0.5px solid #E5E7EB"
      }}>
        <div style={{
          display: "inline-flex", alignItems: "center",
          gap: 4, fontSize: 11, fontWeight: 500,
          color: "#1D9E75", marginBottom: 6
        }}>
          <span>🔒</span>
          <span>NDPA 2023 Compliant — Zero Data Retention</span>
        </div>
        <p>
          <span style={{ fontWeight: 500, color: "#111827" }}>Your data is private.</span>{" "}
          Files are processed in-session only and are not stored on our servers
          after your analysis is complete. We do not share your data with third
          parties or use it to train AI models.{" "}
          <a href="/privacy" style={{ color: "#1D9E75" }}>Privacy Policy →</a>
        </p>
        <label style={{
          display: "flex", alignItems: "flex-start",
          gap: 8, marginTop: 8, cursor: "pointer", fontSize: 12
        }}>
          <input
            type="checkbox"
            checked={consentGiven}
            onChange={(e) => setConsentGiven(e.target.checked)}
            style={{ marginTop: 2 }}
          />
          <span>
            I understand that my data will be processed in-session only and agree
            to the{" "}
            <a href="/privacy" style={{ color: "#1D9E75" }}>
              data processing terms
            </a>.
          </span>
        </label>
      </div>

      <div className="flex gap-3">
        <button
          type="button"
          onClick={onBack}
          disabled={loading}
          className="rounded-lg border border-gray-300 px-5 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 disabled:opacity-50"
        >
          ← Back
        </button>
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!canSubmit || loading || !consentGiven}
          title={!canSubmit ? "Complete required fields to run analysis." : undefined}
          className="flex-1 rounded-lg bg-emerald-600 px-5 py-2 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <VsSpinner size="sm" className="border-white" />
              Running analysis…
            </span>
          ) : (
            "Run Analysis"
          )}
        </button>
      </div>

    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

interface FormFieldProps {
  label: string;
  helper?: string;
  required?: boolean;
  /** Validation message — shown in red below the field when non-null. */
  error?: string | null;
  children: React.ReactNode;
}

function FormField({ label, helper, required, error, children }: FormFieldProps) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>
      {helper && <p className="text-xs text-gray-400 mb-1.5">{helper}</p>}
      {children}
      {error && (
        <p className="mt-1 text-xs text-red-600">{error}</p>
      )}
    </div>
  );
}

interface ColumnSelectProps {
  value: string;
  onChange: (v: string) => void;
  options: string[];
  placeholder?: string;
}

function ColumnSelect({ value, onChange, options, placeholder }: ColumnSelectProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
    >
      <option value="">{placeholder ?? "— Select column —"}</option>
      {options.map((col) => (
        <option key={col} value={col}>
          {col}
        </option>
      ))}
    </select>
  );
}

interface SummaryRowProps {
  label: string;
  value: string;
}

function SummaryRow({ label, value }: SummaryRowProps) {
  return (
    <>
      <span className="text-gray-500">{label}</span>
      <span className="font-medium text-gray-800 truncate">{value}</span>
    </>
  );
}
