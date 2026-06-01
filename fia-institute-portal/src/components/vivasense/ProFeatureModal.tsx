import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Lock } from "lucide-react";
import { type ProGuardInfo } from "@/lib/vivasenseGating";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { BOOK_DATA_CLINIC_URL } from "@/services/featureMode";

interface ProFeatureModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  guard: ProGuardInfo | null;
}

const PRO_INTEREST_URL = "https://wa.me/2349022158026?text=Hi%20Dr.%20Fayeun%2C%20I'm%20interested%20in%20VivaSense%20Pro.%20Please%20send%20payment%20details.";

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

export function ProFeatureModal({ open, onOpenChange, guard }: ProFeatureModalProps) {
  if (!guard) return null;

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
      onOpenChange(false);
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
      onOpenChange(false);
    } catch (err) {
      console.error("Failed to record Pro interest:", err);
      toast.error("Could not save your interest. Please try again.");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full" style={{ backgroundColor: "#1B5E2015" }}>
            <Lock className="h-6 w-6" style={{ color: "#1B5E20" }} />
          </div>
          <DialogTitle className="text-center font-serif text-xl">
            {guard.title}
          </DialogTitle>
          <DialogDescription className="text-center pt-1">
            {guard.description}
          </DialogDescription>
          <p className="text-center text-xs text-muted-foreground pt-2 leading-relaxed">
            Trusted by MSc students, PhD researchers, lecturers, and agricultural professionals across Africa.
          </p>
        </DialogHeader>

        {/* Free vs Pro comparison */}
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="rounded-xl border border-gray-200 bg-gray-50 p-3">
            <span className="inline-block rounded-full bg-gray-200 px-2.5 py-0.5 text-xs font-semibold text-gray-600 mb-2">
              Free
            </span>
            <ul className="space-y-1.5">
              {FREE_FEATURES.map((f) => (
                <li key={f} className="flex items-start gap-1.5 text-gray-600">
                  <span className="shrink-0 text-gray-400">✓</span>
                  {f}
                </li>
              ))}
            </ul>
          </div>
          <div className="rounded-xl border border-emerald-300 bg-gradient-to-b from-emerald-50 to-white p-3">
            <div className="flex items-center gap-1.5 mb-2">
              <span className="inline-block rounded-full bg-emerald-700 px-2.5 py-0.5 text-xs font-semibold text-white">
                Pro
              </span>
            </div>
            <ul className="space-y-1.5">
              {PRO_FEATURES.map((f) => (
                <li key={f} className="flex items-start gap-1.5 font-medium text-emerald-800">
                  <span className="shrink-0 text-emerald-500">✓</span>
                  {f}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Pricing */}
        <div className="text-center py-1">
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">VivaSense Pro</p>
          <p className="text-3xl font-bold text-gray-900">
            ₦5,000<span className="text-base font-normal text-gray-500">/month</span>
          </p>
          <p className="text-sm text-gray-500 mt-1">Special rate for African researchers</p>
        </div>

        {/* CTAs */}
        <div className="space-y-2">
          <button
            type="button"
            onClick={() => void handleActivatePro()}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-700 px-4 py-3 text-sm font-semibold text-white hover:bg-emerald-800 transition-colors"
          >
            🚀 Activate VivaSense Pro
          </button>
          <p className="text-center text-xs text-muted-foreground">
            Speak with the FIA team to activate Pro access.
          </p>
          <button
            type="button"
            onClick={() => void handleWaitingList()}
            className="w-full rounded-xl border border-gray-300 px-4 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
          >
            🌱 Join the Pro Waiting List
          </button>
        </div>

        {/* Data Clinic upsell */}
        <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm">
          <p className="font-semibold text-emerald-900 mb-1">Need help with your data?</p>
          <p className="text-xs text-emerald-800 mb-3 leading-relaxed">
            Our FIA Data Clinic team can assist with analysis, interpretation, and reporting.
          </p>
          <button
            type="button"
            onClick={() => window.open(BOOK_DATA_CLINIC_URL, "_blank", "noopener,noreferrer")}
            className="rounded-lg border border-emerald-600 px-3 py-1.5 text-xs font-medium text-emerald-700 hover:bg-emerald-100 transition-colors"
          >
            Book Data Clinic
          </button>
        </div>

        <DialogFooter className="sm:justify-center">
          <Button variant="ghost" onClick={() => onOpenChange(false)} className="text-xs">
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
