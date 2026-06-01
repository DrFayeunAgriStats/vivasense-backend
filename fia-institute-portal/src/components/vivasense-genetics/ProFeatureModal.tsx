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
  "Single-trait analysis",
  "Basic statistical outputs",
  "Standard interpretation support",
  "Community resources",
];

const PRO_FEATURES = [
  "Multi-trait batch analysis",
  "G×E interaction & stability analysis",
  "PCA & cluster analysis",
  "Genetic parameters (H², GCV, GAM)",
  "Academic AI interpretation",
  "Publication-ready Word reports",
  "Priority support",
];

const PRO_INTEREST_URL =
  "https://wa.me/2349022158026?text=Hi%20Dr.%20Fayeun%2C%20I'm%20interested%20in%20VivaSense%20Pro.%20Please%20send%20payment%20details.";

export function ProFeatureModal({
  open,
  onClose,
  onActivated,
  featureName,
}: ProFeatureModalProps) {
  if (!open) return null;

  const handleActivatePro = async () => {
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
      onClose();
      window.open(PRO_INTEREST_URL, "_blank", "noopener,noreferrer");
    } catch (err) {
      console.error("Failed to record Pro interest:", err);
    }
  };

  const handleWaitingList = async () => {
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
      toast.success(
        "Thank you for your interest in VivaSense Pro. We have added you to the Pro waiting list and will contact you about upcoming premium features and researcher benefits."
      );
      onActivated?.();
      onClose();
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
      <div className="w-full max-w-2xl rounded-2xl border border-gray-200 bg-white shadow-2xl overflow-hidden max-h-[90vh] flex flex-col">

        {/* ── Header ── */}
        <div className="flex items-center justify-between px-6 py-4 bg-gradient-to-r from-emerald-800 to-emerald-600 shrink-0">
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

        {/* ── Social Proof ── */}
        <div className="border-b border-gray-100 bg-white px-6 py-3 text-center shrink-0">
          <p className="text-xs text-gray-500 leading-relaxed">
            Trusted by MSc students, PhD researchers, lecturers, and agricultural professionals across Africa.
          </p>
        </div>

        {/* ── Scrollable Body ── */}
        <div className="overflow-y-auto flex-1">
          <div className="p-6 space-y-6">

            {/* Free vs Pro comparison */}
            <div className="grid sm:grid-cols-2 gap-4">
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

            {/* Pricing */}
            <div className="text-center">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">VivaSense Pro</p>
              <p className="text-3xl font-bold text-gray-900">
                ₦5,000<span className="text-base font-normal text-gray-500">/month</span>
              </p>
              <p className="text-sm text-gray-500 mt-1">Special rate for African researchers</p>
            </div>

            {/* CTAs */}
            <div className="space-y-3">
              <button
                type="button"
                onClick={() => void handleActivatePro()}
                className="block w-full bg-emerald-700 text-white text-center py-3 rounded-xl font-semibold hover:bg-emerald-800 transition-colors"
              >
                🚀 Activate VivaSense Pro
              </button>
              <p className="text-center text-xs text-gray-500">
                Speak with the FIA team to activate Pro access.
              </p>
              <button
                type="button"
                onClick={() => void handleWaitingList()}
                className="w-full border border-gray-300 text-gray-700 py-3 rounded-xl font-medium hover:bg-gray-50 transition-colors"
              >
                🌱 Join the Pro Waiting List
              </button>
            </div>

          </div>
        </div>

        {/* ── Footer ── */}
        <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-4 border-t border-gray-100 bg-gray-50 shrink-0">
          <div className="min-w-0">
            <p className="text-xs font-semibold text-gray-700">Need help with your data?</p>
            <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">
              Our FIA Data Clinic team can assist with analysis, interpretation, and reporting.
            </p>
          </div>
          <button
            type="button"
            onClick={handleBookClinic}
            className="rounded-lg border border-emerald-600 px-3 py-1.5 text-xs font-medium text-emerald-700 hover:bg-emerald-50 transition-colors shrink-0"
          >
            Book Data Clinic
          </button>
        </div>
      </div>
    </div>
  );
}
