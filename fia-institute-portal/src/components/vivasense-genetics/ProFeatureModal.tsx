import React from "react";
import { BOOK_DATA_CLINIC_URL } from "@/services/featureMode";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

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
  const PRO_INTEREST_URL = "https://wa.me/2349022158026?text=Hi%20Dr.%20Fayeun%2C%20I'm%20interested%20in%20VivaSense%20Pro.%20Please%20send%20payment%20details.";

  if (!open) return null;

  const recordInterest = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session?.user) {
        toast.error("Please sign in first.");
        return;
      }

      await supabase.from("profiles").update({
        pro_interest: true,
        pro_interest_date: new Date().toISOString(),
      }).eq("id", session.user.id);

      toast.success("Thanks! We will contact you within 24 hours.");
      onActivated?.();
      onClose();
      window.open(PRO_INTEREST_URL, "_blank", "noopener,noreferrer");
    } catch (err) {
      console.error("Failed to record Pro interest:", err);
      toast.error("Could not save your interest. Please try again.");
    }
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

        <div className="border-b border-gray-100 bg-white px-6 py-4 text-center">
          <p className="text-sm text-gray-600 leading-relaxed">
            Get access to advanced analysis features. {featureName ? <span className="font-semibold">{featureName}</span> : null}
          </p>
        </div>

        {/* ── Body ── */}
        <div className="p-6">
          <div className="text-center">
            <div className="text-5xl mb-3">⭐</div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Unlock VivaSense Pro</h2>
            <p className="text-gray-600 text-sm">Get access to advanced features</p>
          </div>

          <div className="bg-green-50 rounded-xl p-5 my-6">
            <p className="font-semibold text-green-900 mb-3">✨ Pro includes:</p>
            <ul className="space-y-2 text-sm text-green-800">
              {PRO_FEATURES.map((feature) => (
                <li key={feature}>✓ {feature}</li>
              ))}
              <li>✓ Priority email support</li>
            </ul>
          </div>

          <div className="text-center mb-6">
            <p className="text-3xl font-bold text-gray-900">
              ₦5,000<span className="text-base font-normal text-gray-500">/month</span>
            </p>
            <p className="text-sm text-gray-500 mt-1">Special rate for African researchers</p>
          </div>

          <a
            href={PRO_INTEREST_URL}
            target="_blank"
            rel="noopener noreferrer"
            onClick={() => void recordInterest()}
            className="block w-full bg-green-700 text-white text-center py-3 rounded-xl font-semibold hover:bg-green-800"
          >
            💬 Get Pro via WhatsApp
          </a>

          <button
            type="button"
            onClick={() => void recordInterest()}
            className="w-full mt-3 border border-gray-300 text-gray-700 py-3 rounded-xl font-medium hover:bg-gray-50"
          >
            I'm Interested — Contact Me Later
          </button>
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
          </div>
        </div>
      </div>
    </div>
  );
}
