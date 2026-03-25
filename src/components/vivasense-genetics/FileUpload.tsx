import React, { useCallback, useState } from "react";
import { UploadPreviewResponse } from "@/services/geneticsUploadApi";

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
      <label
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={[
          "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 cursor-pointer transition-colors",
          dragging
            ? "border-emerald-500 bg-emerald-50"
            : "border-gray-300 bg-gray-50 hover:border-emerald-400 hover:bg-emerald-50/50",
        ].join(" ")}
      >
        <input
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={onInputChange}
          disabled={loading}
        />

        {loading ? (
          <>
            <div className="h-10 w-10 rounded-full border-4 border-emerald-600 border-t-transparent animate-spin" />
            <p className="text-sm text-gray-500">Reading file…</p>
          </>
        ) : (
          <>
            <UploadIcon />
            <div className="text-center">
              <p className="font-medium text-gray-700">
                Drag &amp; drop your file here, or{" "}
                <span className="text-emerald-600 underline">browse</span>
              </p>
              <p className="mt-1 text-sm text-gray-400">
                Supports .csv, .xlsx — max 10 MB
              </p>
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
        <p className="mt-2 text-sm text-red-600 flex items-center gap-1">
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
