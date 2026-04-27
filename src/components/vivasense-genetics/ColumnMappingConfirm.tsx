import React, { useState } from "react";
import { UploadPreviewResponse } from "@/services/geneticsUploadApi";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

type ExperimentalDesign = "CRD" | "RCBD" | "Factorial" | "Split-Plot" | "MET";

interface ColumnMapping {
  genotypeColumn: string;
  repColumn: string;
  environmentColumn: string;
  selectedTraits: string[];
  mode: "single" | "multi";
  randomEnvironment: boolean;
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

  // ── Section B ─────────────────────────────────────────────────────────────

  const [responseVariable, setResponseVariable] = useState(
    detected_columns.traits[0] ?? ""
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

  // ── Section C ─────────────────────────────────────────────────────────────

  const [advancedOpen, setAdvancedOpen] = useState(false);
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

  // ── Derived visibility ────────────────────────────────────────────────────

  const showReplication = design !== "CRD";
  const showSplitPlot   = design === "Split-Plot";
  const showEnvironment = design === "MET";
  const showInteraction = design === "Factorial" || design === "MET";

  // ── Handlers ─────────────────────────────────────────────────────────────

  // Design change: clear conditional fields; retain Response Variable and Treatment.
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
    responseVariable !== "" &&
    treatmentColumn !== "" &&
    (!showReplication || replicationColumn !== "") &&
    (!showSplitPlot   || (mainPlotFactor !== "" && subPlotFactor !== "")) &&
    (!showEnvironment || environmentColumn !== "");

  // ── Validation messages (Section D) ──────────────────────────────────────
  // Each message is null when the field is untouched or valid.
  const errors = {
    responseVariable:
      touched.has("responseVariable") && responseVariable === ""
        ? "Select a response variable to continue"
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
      selectedTraits:    [responseVariable],
      mode:              design === "MET" ? "multi" : "single",
      randomEnvironment: false,
    });
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="space-y-6">

      {/* Dataset summary */}
      <div className="flex items-center gap-3 rounded-lg bg-emerald-50 border border-emerald-200 p-3">
        <span className="text-2xl">📊</span>
        <div>
          <p className="font-medium text-emerald-800">
            {n_rows} rows · {n_columns} columns detected
          </p>
          <p className="text-sm text-emerald-600">
            {numericColumns.length} numeric · {categoricalColumns.length} categorical
          </p>
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

      {/* ── Section B: Core Variables ───────────────────────────────────────── */}

      <div className="space-y-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-gray-400">Variables</p>

        {/* B1 — Response Variable */}
        <FormField
          label="Response Variable (Trait)"
          helper="Select the trait to analyse (e.g., Yield, Plant Height)"
          required
          error={errors.responseVariable}
        >
          <ColumnSelect
            value={responseVariable}
            onChange={(v) => { setResponseVariable(v); touch("responseVariable"); }}
            options={numericColumns}
            placeholder="— Select numeric column —"
          />
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
          onClick={() => setAdvancedOpen((o) => !o)}
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
          # = numeric column available for Response Variable
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
            <SummaryRow label="Trait" value={responseVariable} />
            <SummaryRow
              label="Treatments"
              value="—"
            />
            <SummaryRow
              label="Replications"
              value={design === "CRD" ? "—" : "—"}
            />
            <SummaryRow label="Observations" value={n_rows.toLocaleString()} />
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
              <span className="h-4 w-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
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
