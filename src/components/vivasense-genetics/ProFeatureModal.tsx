import React, { useMemo, useState } from "react";
import {
  activateProWithCode,
  BOOK_DATA_CLINIC_URL,
} from "@/services/featureMode";

interface ProFeatureModalProps {
  open: boolean;
  onClose: () => void;
  onActivated?: () => void;
}

export function ProFeatureModal({
  open,
  onClose,
  onActivated,
}: ProFeatureModalProps) {
  const [accessCode, setAccessCode] = useState("");
  const [error, setError] = useState<string | null>(null);

  const canSubmit = useMemo(() => accessCode.trim().length > 0, [accessCode]);

  if (!open) return null;

  const handleActivate = () => {
    const ok = activateProWithCode(accessCode);
    if (!ok) {
      setError("Invalid access code. Please try again.");
      return;
    }
    setAccessCode("");
    setError(null);
    onActivated?.();
    onClose();
  };

  const handleBookClinic = () => {
    window.open(BOOK_DATA_CLINIC_URL, "_blank", "noopener,noreferrer");
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="w-full max-w-md rounded-2xl border border-gray-200 bg-white p-5 shadow-2xl">
        <div className="mb-3 flex items-center gap-2">
          <span className="text-lg">🔒</span>
          <h3 className="text-lg font-semibold" style={{ color: "#1B5E20" }}>
            VivaSense Pro Feature
          </h3>
        </div>

        <p className="text-sm text-gray-700 leading-relaxed">
          This analysis is part of VivaSense Pro. Upgrade to access advanced
          modules like G×E, PCA, Genetic Parameters, and Path Analysis.
        </p>

        <div className="mt-4">
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-gray-600">
            Enter Pro Access Code
          </label>
          <input
            type="text"
            value={accessCode}
            onChange={(e) => setAccessCode(e.target.value)}
            className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none ring-0 focus:border-green-700"
            placeholder="VIVASENSE-PILOT-2026"
          />
          {error && <p className="mt-1 text-xs text-red-600">{error}</p>}
        </div>

        <p className="mt-3 text-xs text-gray-500">
          Let us help you interpret your results or handle full analysis professionally.
        </p>

        <div className="mt-4 flex flex-wrap items-center justify-end gap-2">
          <button
            type="button"
            onClick={handleActivate}
            disabled={!canSubmit}
            className="rounded-lg px-3 py-1.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
            style={{ backgroundColor: "#1B5E20" }}
          >
            Enter Pro Access Code
          </button>
          <button
            type="button"
            onClick={handleBookClinic}
            className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
          >
            Book Data Clinic
          </button>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
