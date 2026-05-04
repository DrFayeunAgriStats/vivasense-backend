import React, { useState } from "react";
import { UploadPreviewResponse } from "@/services/geneticsUploadApi";
import { VsSpinner } from "./VsSpinner";
import {
  DEFAULT_SELECTION_INTENSITY,
  SELECTION_INTENSITIES,
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
  const [environmentColumn, setEnvironmentColumn] = useState(
    detected_columns.environment?.column ?? ""
  );
  const [selectionIntensity, setSelectionIntensity] = useState(DEFAULT_SELECTION_INTENSITY);

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

  // ── Derived visibility ────────────────────────────────────────────────────
  const showReplication = design !== "CRD";
  const showSplitPlot   = design === "Split-Plot";
  const showEnvironment = design === "MET";
  const showInteraction = design === "Factorial" || design === "MET";

  // ── Handlers ─────────────────────────────────────────────────────────────

  // Design change: clear conditional fields; retain selected traits and treatment.
  const handleDesignChange = (d: ExperimentalDesign) => {
    setDesign(d);
    setReplicationColumn("");
    setMainPlotFactor("");
    setSubPlotFactor("");
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
    treatmentColumn !== "" &&
    (!showReplication || replicationColumn !== "") &&
    (!showSplitPlot   || (mainPlotFactor !== "" && subPlotFactor !== "")) &&
    (!showEnvironment || environmentColumn !== "");

  // ── Validation messages (Section D) ──────────────────────────────────────
  // Each message is null when the field is untouched or valid.
  const errors = {
    selectedTraits:
      touched.has("selectedTraits") && selectedTraits.length === 0
        ? "Select at least one trait to continue"
        : null,
    treatmentColumn:
      touched.has("treatmentColumn") && treatmentColumn === ""
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
    onConfirm({
      genotypeColumn:    treatmentColumn,
      repColumn:         showReplication ? replicationColumn : "",
      environmentColumn: showEnvironment ? environmentColumn : "",
      selectedTraits,
      mode:              design === "MET" ? "multi" : "single",
      randomEnvironment: false,
      selectionIntensity,
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

      {/* ── Data quality card ─────────────────────────────────────────────── */}
      <div className={`rounded-xl border ${qualityConfig.border} ${qualityConfig.bg} p-4 space-y-3`}>
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
        <div className="space-y-2">
          <div className="flex items-center gap-1.5">
            <select
              value={selectionIntensity}
              onChange={(e) => setSelectionIntensity(Number(e.target.value) || DEFAULT_SELECTION_INTENSITY)}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
            >
              {SELECTION_INTENSITIES.map((option) => (
                <option key={option.pct} value={option.value}>
                  {option.label} (i = {option.value.toFixed(3)}) - {option.note}
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
              {numericColumns.map((trait) => {
                const active = selectedTraits.includes(trait);
                return (
                  <button
                    key={trait}
                    type="button"
                    onClick={() => toggleTrait(trait)}
                    className={[
                      "rounded-full border px-3 py-1.5 text-sm transition-colors",
                      active
                        ? "border-emerald-600 bg-emerald-600 text-white"
                        : "border-gray-300 text-gray-700 hover:border-emerald-300 hover:bg-emerald-50",
                    ].join(" ")}
                  >
                    {trait}
                  </button>
                );
              })}
            </div>
          </div>
        </FormField>

        {/* B2 — Treatment / Genotype */}
        <FormField
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
        </FormField>

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
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 space-y-2">
          <p className="text-sm font-semibold text-emerald-800">Analysis Summary</p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            <SummaryRow
              label="Design"
              value={DESIGN_OPTIONS.find((o) => o.value === design)?.desc ?? design}
            />
            <SummaryRow
              label="Traits"
              value={selectedTraits.length === 1 ? selectedTraits[0] : `${selectedTraits.length} selected`}
            />
            <SummaryRow
              label="Treatments"
              value="—"
            />
            <SummaryRow
              label="Replications"
              value={design === "CRD" ? "—" : "—"}
            />
            <SummaryRow label="Observations" value={n_rows.toLocaleString()} />
            <SummaryRow label="Selection intensity" value={`i = ${selectionIntensity.toFixed(3)}`} />
          </div>
        </div>
      )}

      {/* ── Actions ─────────────────────────────────────────────────────────── */}

      {canSubmit && (
        <p className="flex items-center gap-1.5 text-sm font-medium text-emerald-600">
          <span aria-hidden="true">✓</span> Ready to run analysis
        </p>
      )}

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
          disabled={!canSubmit || loading}
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
