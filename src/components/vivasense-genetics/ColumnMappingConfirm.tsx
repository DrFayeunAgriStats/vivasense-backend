import React, { useState } from "react";
import { UploadPreviewResponse } from "@/services/geneticsUploadApi";

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

export function ColumnMappingConfirm({
  preview,
  onConfirm,
  onBack,
  loading,
}: ColumnMappingConfirmProps) {
  const { detected_columns, column_names, data_preview, n_rows, warnings } = preview;

  const [genotypeColumn, setGenotypeColumn] = useState(
    detected_columns.genotype?.column ?? ""
  );
  const [repColumn, setRepColumn] = useState(
    detected_columns.rep?.column ?? ""
  );
  const [environmentColumn, setEnvironmentColumn] = useState(
    detected_columns.environment?.column ?? ""
  );
  const [selectedTraits, setSelectedTraits] = useState<string[]>(
    detected_columns.traits
  );
  const [mode, setMode] = useState<"single" | "multi">(
    preview.mode_suggestion
  );
  const [randomEnvironment, setRandomEnvironment] = useState(false);

  const toggleTrait = (trait: string) => {
    setSelectedTraits((prev) =>
      prev.includes(trait) ? prev.filter((t) => t !== trait) : [...prev, trait]
    );
  };

  const canSubmit =
    genotypeColumn &&
    repColumn &&
    selectedTraits.length > 0 &&
    (mode === "single" || environmentColumn);

  const handleSubmit = () => {
    if (!canSubmit) return;
    onConfirm({ genotypeColumn, repColumn, environmentColumn, selectedTraits, mode, randomEnvironment });
  };

  return (
    <div className="space-y-6">
      {/* File summary */}
      <div className="flex items-center gap-3 rounded-lg bg-emerald-50 border border-emerald-200 p-3">
        <span className="text-2xl">📊</span>
        <div>
          <p className="font-medium text-emerald-800">
            {n_rows} rows · {preview.n_columns} columns detected
          </p>
          <p className="text-sm text-emerald-600">
            Confirm column assignments below before running analysis
          </p>
        </div>
      </div>

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="space-y-1">
          {warnings.map((w) => (
            <p key={w} className="text-sm text-amber-700 flex items-start gap-1.5">
              <span className="mt-0.5">⚠</span> {w}
            </p>
          ))}
        </div>
      )}

      {/* Mode toggle */}
      <div>
        <label className="block text-sm font-semibold text-gray-700 mb-2">
          Analysis Mode
        </label>
        <div className="flex gap-2">
          {(["single", "multi"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={[
                "flex-1 rounded-lg border px-4 py-2 text-sm font-medium transition-colors",
                mode === m
                  ? "border-emerald-600 bg-emerald-600 text-white"
                  : "border-gray-300 bg-white text-gray-600 hover:border-emerald-400",
              ].join(" ")}
            >
              {m === "single" ? "Single Environment" : "Multi-Environment"}
            </button>
          ))}
        </div>
      </div>

      {/* Required columns */}
      <div className="grid gap-4 sm:grid-cols-2">
        <ColumnSelect
          label="Genotype Column"
          value={genotypeColumn}
          onChange={setGenotypeColumn}
          options={column_names}
          confidence={detected_columns.genotype?.confidence}
          required
        />
        <ColumnSelect
          label="Replication Column"
          value={repColumn}
          onChange={setRepColumn}
          options={column_names}
          confidence={detected_columns.rep?.confidence}
          required
        />
        {mode === "multi" && (
          <ColumnSelect
            label="Environment Column"
            value={environmentColumn}
            onChange={setEnvironmentColumn}
            options={column_names}
            confidence={detected_columns.environment?.confidence}
            required
          />
        )}
      </div>

      {/* Random environment (multi-env only) */}
      {mode === "multi" && (
        <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
          <input
            type="checkbox"
            checked={randomEnvironment}
            onChange={(e) => setRandomEnvironment(e.target.checked)}
            className="rounded border-gray-300 text-emerald-600"
          />
          Treat environment as random effect (advanced)
        </label>
      )}

      {/* Trait selection */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-semibold text-gray-700">
            Traits to Analyze
            {selectedTraits.length > 0 && (
              <span className="ml-2 text-emerald-600 font-normal">
                {selectedTraits.length} selected
              </span>
            )}
          </label>
          <div className="flex gap-2 text-xs">
            <button
              type="button"
              onClick={() => setSelectedTraits(detected_columns.traits)}
              className="text-emerald-600 hover:underline"
            >
              Select all
            </button>
            <span className="text-gray-300">|</span>
            <button
              type="button"
              onClick={() => setSelectedTraits([])}
              className="text-gray-400 hover:underline"
            >
              Clear
            </button>
          </div>
        </div>
        {detected_columns.traits.length === 0 ? (
          <p className="text-sm text-amber-600">
            No numeric trait columns found. Check that your file has numeric data columns.
          </p>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {detected_columns.traits.map((trait) => (
              <label
                key={trait}
                className={[
                  "flex items-center gap-2 rounded-lg border px-3 py-2 cursor-pointer transition-colors text-sm",
                  selectedTraits.includes(trait)
                    ? "border-emerald-500 bg-emerald-50 text-emerald-800"
                    : "border-gray-200 bg-white text-gray-600 hover:border-emerald-300",
                ].join(" ")}
              >
                <input
                  type="checkbox"
                  checked={selectedTraits.includes(trait)}
                  onChange={() => toggleTrait(trait)}
                  className="rounded border-gray-300 text-emerald-600"
                />
                <span className="truncate font-medium">{trait}</span>
              </label>
            ))}
          </div>
        )}
      </div>

      {/* Data preview */}
      <details className="group">
        <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700 select-none">
          Preview first 5 rows ▸
        </summary>
        <div className="mt-2 overflow-x-auto rounded-lg border border-gray-200">
          <table className="min-w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                {column_names.map((col) => (
                  <th key={col} className="px-3 py-2 text-left font-semibold text-gray-600 whitespace-nowrap">
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data_preview.map((row, i) => (
                <tr key={i} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                  {column_names.map((col) => (
                    <td key={col} className="px-3 py-1.5 text-gray-700 whitespace-nowrap">
                      {row[col] == null ? <span className="text-gray-300">—</span> : String(row[col])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      {/* Actions */}
      <div className="flex gap-3 pt-2">
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
          className="flex-1 rounded-lg bg-emerald-600 px-5 py-2 text-sm font-semibold text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="h-4 w-4 rounded-full border-2 border-white border-t-transparent animate-spin" />
              Analyzing {selectedTraits.length} trait{selectedTraits.length !== 1 ? "s" : ""}…
            </span>
          ) : (
            `Analyze ${selectedTraits.length} Trait${selectedTraits.length !== 1 ? "s" : ""}`
          )}
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

interface ColumnSelectProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  confidence?: string;
  required?: boolean;
}

function ColumnSelect({ label, value, onChange, options, confidence, required }: ColumnSelectProps) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
        {confidence && (
          <span
            className={[
              "ml-2 text-xs px-1.5 py-0.5 rounded-full",
              confidence === "high"
                ? "bg-emerald-100 text-emerald-700"
                : confidence === "medium"
                ? "bg-yellow-100 text-yellow-700"
                : "bg-gray-100 text-gray-500",
            ].join(" ")}
          >
            {confidence} confidence
          </span>
        )}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none"
      >
        <option value="">— Select column —</option>
        {options.map((col) => (
          <option key={col} value={col}>
            {col}
          </option>
        ))}
      </select>
    </div>
  );
}
