import React, { useCallback, useState } from "react";
import { UploadPreviewResponse } from "@/services/geneticsUploadApi";
import { VsSpinner } from "./VsSpinner";

interface FileUploadProps {
  onPreviewReady: (file: File, preview: UploadPreviewResponse) => void;
  onPreviewStart: () => void;
  previewFn: (file: File) => Promise<UploadPreviewResponse>;
}

export function FileUpload({ onPreviewReady, onPreviewStart, previewFn }: FileUploadProps) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const ACCEPTED = [".csv", ".xlsx", ".xls"];

  const handleFile = useCallback(
    async (file: File) => {
      const ext = "." + file.name.split(".").pop()?.toLowerCase();
      if (!ACCEPTED.includes(ext)) {
        setError(`Unsupported file type "${ext}". Please upload a .csv or .xlsx file.`);
        return;
      }

      setError(null);
      setSelectedFile(file);
      setLoading(true);
      onPreviewStart();

      try {
        const preview = await previewFn(file);
        onPreviewReady(file, preview);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Preview failed");
      } finally {
        setLoading(false);
      }
    },
    [onPreviewReady, onPreviewStart, previewFn]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  return (
    <div className="w-full">
      <div className="mb-4 rounded-2xl border border-emerald-100 bg-gradient-to-r from-emerald-50 via-white to-teal-50 p-4">
        <p className="text-sm font-semibold text-emerald-900">Upload Dataset</p>
        <p className="mt-1 text-xs text-emerald-800/80">
          Start with a CSV or Excel file. VivaSense will preview your columns and guide mapping before analysis.
        </p>
      </div>

      <label
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={[
          "group relative flex flex-col items-center justify-center gap-3 overflow-hidden rounded-2xl border-2 border-dashed p-10 cursor-pointer transition-all duration-200",
          dragging
            ? "border-emerald-500 bg-emerald-50 shadow-lg shadow-emerald-200/60"
            : "border-gray-300 bg-gradient-to-b from-white to-gray-50 hover:border-emerald-400 hover:shadow-md hover:shadow-emerald-100",
        ].join(" ")}
      >
        <div className="pointer-events-none absolute -right-10 -top-10 h-36 w-36 rounded-full bg-emerald-100/60 blur-2xl transition-opacity group-hover:opacity-100 opacity-70" />

        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={onInputChange}
          disabled={loading}
        />

        {loading ? (
          <>
            <VsSpinner size="lg" monogram />
            <p className="text-sm text-gray-500">Reading file…</p>
          </>
        ) : (
          <>
            <UploadIcon />
            <div className="text-center">
              <p className="font-semibold text-gray-800">
                Drag &amp; drop your file here, or{" "}
                <span className="text-emerald-600 underline">browse</span>
              </p>
              <p className="mt-1 text-sm text-gray-500">
                Supports .csv, .xlsx — max 10 MB
              </p>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-gray-600">
              <span className="rounded-full border border-emerald-200 bg-white px-2.5 py-1">CSV</span>
              <span className="rounded-full border border-emerald-200 bg-white px-2.5 py-1">XLSX</span>
              <span className="rounded-full border border-gray-200 bg-white px-2.5 py-1">Auto column detection</span>
            </div>

            {selectedFile && (
              <div className="flex items-center gap-2 rounded-lg bg-emerald-100 px-3 py-1.5 text-sm text-emerald-700">
                <FileIcon />
                <span>{selectedFile.name}</span>
                <span className="text-emerald-500">
                  ({(selectedFile.size / 1024).toFixed(0)} KB)
                </span>
              </div>
            )}
          </>
        )}
      </label>

      {error && (
        <p className="mt-2 flex items-center gap-1 rounded-lg border border-red-100 bg-red-50 px-3 py-2 text-sm text-red-700">
          <span>⚠</span> {error}
        </p>
      )}
    </div>
  );
}

function UploadIcon() {
  return (
    <svg className="h-10 w-10 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
    </svg>
  );
}

function FileIcon() {
  return (
    <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  );
}
