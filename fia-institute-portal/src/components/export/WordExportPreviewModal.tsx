import React, { useState } from "react";
import type { PreviewSection } from "@/utils/normalizeModuleData";

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

export interface WordExportPreviewProps {
  moduleName: string;
  reportTitle: string;
  datasetSummary: string;
  sections: PreviewSection[];
  warnings: string[];
  notes: string[];
  /** Set false to disable the Download button (no valid data). */
  canExport: boolean;
  /** Called when user clicks Download — caller owns the actual fetch/download. */
  onExport: () => Promise<void>;
  onClose: () => void;
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

export function WordExportPreviewModal({
  moduleName,
  reportTitle,
  datasetSummary,
  sections,
  warnings,
  notes,
  canExport,
  onExport,
  onClose,
}: WordExportPreviewProps) {
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);

  const handleExport = async () => {
    if (!canExport || exporting) return;
    setExporting(true);
    setExportError(null);
    try {
      await onExport();
      onClose();
    } catch (err) {
      setExportError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setExporting(false);
    }
  };

  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="relative flex w-full max-w-2xl max-h-[90vh] flex-col rounded-2xl bg-white shadow-2xl">
        {/* ── Header ─────────────────────────────────────────────────────── */}
        <div className="flex items-start justify-between border-b border-gray-200 px-6 py-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-emerald-600">
              {moduleName}
            </p>
            <h2 className="mt-0.5 text-lg font-semibold text-gray-800">{reportTitle}</h2>
            <p className="mt-0.5 text-sm text-gray-500">{datasetSummary}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="ml-4 rounded-lg p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-colors"
            aria-label="Close preview"
          >
            <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>

        {/* ── Body ───────────────────────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">

          {/* Warnings */}
          {warnings.length > 0 && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 space-y-1">
              <p className="text-xs font-semibold text-amber-700">Warnings</p>
              {warnings.map((w, i) => (
                <p key={i} className="text-xs text-amber-600 flex items-start gap-1.5">
                  <span className="mt-0.5 shrink-0">⚠</span> {w}
                </p>
              ))}
            </div>
          )}

          {/* No-data guard */}
          {!canExport && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3">
              <p className="text-xs font-semibold text-red-700">No exportable data</p>
              <p className="text-xs text-red-600 mt-0.5">
                The report cannot be generated because required analysis data is missing or
                malformed. Run the analysis and verify results before exporting.
              </p>
            </div>
          )}

          {/* Sections */}
          {sections.map((section, si) => (
            <div key={si} className="space-y-2">
              <h3 className="text-sm font-semibold text-gray-700 border-b border-gray-100 pb-1">
                {section.title}
              </h3>
              <div className="rounded-lg border border-gray-200 overflow-hidden">
                <table className="min-w-full text-xs">
                  <tbody>
                    {section.rows.map((row, ri) => (
                      <tr key={ri} className={ri % 2 === 0 ? "bg-white" : "bg-gray-50/60"}>
                        <td className="px-3 py-2 font-medium text-gray-600 w-1/2">
                          {row.label}
                        </td>
                        <td className="px-3 py-2 text-gray-800 font-mono">{row.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {section.note && (
                <p className="text-xs text-gray-400 italic">{section.note}</p>
              )}
            </div>
          ))}

          {/* Notes */}
          {notes.length > 0 && (
            <div className="rounded-lg border border-blue-100 bg-blue-50 px-4 py-3 space-y-1">
              <p className="text-xs font-semibold text-blue-700">Report notes</p>
              {notes.map((n, i) => (
                <p key={i} className="text-xs text-blue-600">{n}</p>
              ))}
            </div>
          )}

          {/* Export error */}
          {exportError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-2">
              <p className="text-xs font-semibold text-red-700">Export failed</p>
              <p className="text-xs text-red-600 mt-0.5">{exportError}</p>
            </div>
          )}
        </div>

        {/* ── Footer ─────────────────────────────────────────────────────── */}
        <div className="flex items-center justify-end gap-3 border-t border-gray-200 px-6 py-4">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-gray-300 px-4 py-1.5 text-sm text-gray-600 hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleExport}
            disabled={!canExport || exporting}
            className="rounded-lg bg-emerald-600 px-5 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {exporting ? (
              <>
                <svg className="h-3.5 w-3.5 animate-spin" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Generating…
              </>
            ) : (
              "Download Report (.docx)"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
