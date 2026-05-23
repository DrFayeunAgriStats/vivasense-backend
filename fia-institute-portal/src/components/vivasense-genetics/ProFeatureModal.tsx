import React, { useMemo, useState } from "react";
import {
  activateProWithCode,
  BOOK_DATA_CLINIC_URL,
} from "@/services/featureMode";

interface ProFeatureModalProps {
  open: boolean;
  onClose: () => void;
  onActivated?: () => void;
  /** Optional: name of the feature that triggered this modal */
  featureName?: string;
}

const FREE_FEATURES = [
  "Field layout generator (CRD / RCBD)",
  "Single-trait ANOVA",
  "Basic descriptive statistics",
  "Column mapping & data preview",
];

const PRO_FEATURES = [
  "Multi-trait batch analysis",
  "G×E interaction & stability analysis",
  "PCA & cluster analysis",
  "Genetic parameters (H², GCV, GAM)",
  "Path coefficient & correlation analysis",
  "Academic AI interpretation (per trait)",
  "Word report download (.docx)",
  "Trait relationships heatmap",
];

export function ProFeatureModal({
  open,
  onClose,
  onActivated,
  featureName,
}: ProFeatureModalProps) {
  const [accessCode, setAccessCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"compare" | "code">("compare");

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
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="w-full max-w-2xl rounded-2xl border border-gray-200 bg-white shadow-2xl overflow-hidden">

        {/* ── Header ── */}
        <div className="flex items-center justify-between px-6 py-4 bg-gradient-to-r from-emerald-800 to-emerald-600">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🌿</span>
            <div>
              <h3 className="text-base font-bold text-white leading-tight">VivaSense Pro</h3>
              {featureName ? (
                <p className="text-xs text-emerald-100 mt-0.5">
                  Requires Pro: <span className="font-semibold">{featureName}</span>
                </p>
              ) : (
                <p className="text-xs text-emerald-100 mt-0.5">Unlock advanced analysis modules</p>
              )}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg p-1.5 text-emerald-200 hover:bg-emerald-700 transition-colors"
          >
            ✕
          </button>
        </div>

        {/* ── Tab bar ── */}
        <div className="flex border-b border-gray-100 bg-white">
          {(["compare", "code"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={[
                "flex-1 py-3 text-sm font-medium transition-colors",
                tab === t
                  ? "border-b-2 border-emerald-600 text-emerald-700"
                  : "text-gray-500 hover:text-gray-700",
              ].join(" ")}
            >
              {t === "compare" ? "Free vs Pro" : "Enter Access Code"}
            </button>
          ))}
        </div>

        {/* ── Body ── */}
        <div className="p-6">
          {tab === "compare" && (
            <div className="grid sm:grid-cols-2 gap-4">
              {/* Free column */}
              <div className="rounded-xl border border-gray-200 bg-gray-50 p-4">
                <span className="inline-block rounded-full bg-gray-200 px-2.5 py-0.5 text-xs font-semibold text-gray-600 mb-3">
                  Free
                </span>
                <ul className="space-y-2">
                  {FREE_FEATURES.map((f) => (
                    <li key={f} className="flex items-start gap-2 text-sm text-gray-600">
                      <span className="mt-0.5 shrink-0 text-gray-400">✓</span>
                      {f}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Pro column */}
              <div className="rounded-xl border border-emerald-300 bg-gradient-to-b from-emerald-50 to-white p-4 relative overflow-hidden">
                <div className="absolute -right-4 -top-4 h-20 w-20 rounded-full bg-emerald-200/50 blur-xl pointer-events-none" />
                <div className="flex items-center gap-2 mb-3">
                  <span className="inline-block rounded-full bg-emerald-700 px-2.5 py-0.5 text-xs font-semibold text-white">Pro</span>
                  <span className="text-xs text-emerald-600">Everything in Free, plus:</span>
                </div>
                <ul className="space-y-2">
                  {PRO_FEATURES.map((f) => (
                    <li key={f} className="flex items-start gap-2 text-sm font-medium text-emerald-800">
                      <span className="mt-0.5 shrink-0 text-emerald-500">✓</span>
                      {f}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {tab === "code" && (
            <div className="max-w-sm mx-auto space-y-4">
              <p className="text-sm text-gray-600 leading-relaxed">
                Enter your VivaSense Pro access code to unlock all features immediately.
              </p>
              <div>
                <label className="mb-1.5 block text-xs font-semibold uppercase tracking-wide text-gray-500">
                  Pro Access Code
                </label>
                <input
                  type="text"
                  value={accessCode}
                  onChange={(e) => { setAccessCode(e.target.value); setError(null); }}
                  onKeyDown={(e) => e.key === "Enter" && canSubmit && handleActivate()}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-emerald-600 focus:ring-1 focus:ring-emerald-200 transition-shadow"
                  placeholder="VIVASENSE-PILOT-2026"
                  autoFocus
                />
                {error && <p className="mt-1.5 text-xs text-red-600">{error}</p>}
              </div>
              <button
                type="button"
                onClick={handleActivate}
                disabled={!canSubmit}
                className="w-full rounded-lg bg-emerald-700 px-4 py-2.5 text-sm font-semibold text-white hover:bg-emerald-800 disabled:cursor-not-allowed disabled:opacity-50 transition-colors"
              >
                Activate Pro
              </button>
            </div>
          )}
        </div>

        {/* ── Footer ── */}
        <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-4 border-t border-gray-100 bg-gray-50">
          <p className="text-xs text-gray-500">
            Need hands-on help? Our Data Clinic team can run the analysis for you.
          </p>
          <div className="flex gap-2 shrink-0">
            <button
              type="button"
              onClick={handleBookClinic}
              className="rounded-lg border border-emerald-600 px-3 py-1.5 text-xs font-medium text-emerald-700 hover:bg-emerald-50 transition-colors"
            >
              Book Data Clinic
            </button>
            {tab === "compare" && (
              <button
                type="button"
                onClick={() => setTab("code")}
                className="rounded-lg bg-emerald-700 px-3 py-1.5 text-xs font-semibold text-white hover:bg-emerald-800 transition-colors"
              >
                I have a code →
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
