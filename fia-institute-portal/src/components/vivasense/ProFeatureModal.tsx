import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Sparkles, Lock, CalendarCheck } from "lucide-react";
import { type ProGuardInfo } from "@/lib/vivasenseGating";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

interface ProFeatureModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  guard: ProGuardInfo | null;
}

const PRO_INTEREST_URL = "https://wa.me/2349022158026?text=Hi%20Dr.%20Fayeun%2C%20I'm%20interested%20in%20VivaSense%20Pro.%20Please%20send%20payment%20details.";

export function ProFeatureModal({ open, onOpenChange, guard }: ProFeatureModalProps) {
  if (!guard) return null;

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
      onOpenChange(false);
      window.open(PRO_INTEREST_URL, "_blank", "noopener,noreferrer");
    } catch (err) {
      console.error("Failed to record Pro interest:", err);
      toast.error("Could not save your interest. Please try again.");
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-full" style={{ backgroundColor: "#1B5E2015" }}>
            <Lock className="h-6 w-6" style={{ color: "#1B5E20" }} />
          </div>
          <DialogTitle className="text-center font-serif text-xl">
            {guard.title}
          </DialogTitle>
          <DialogDescription className="text-center pt-2">
            Get access to advanced analysis features. {guard.description}
          </DialogDescription>
        </DialogHeader>

        <div className="rounded-lg border p-4 text-sm text-muted-foreground" style={{ borderColor: "#1B5E2030", backgroundColor: "#1B5E2008" }}>
          <p className="flex items-start gap-2">
            <Sparkles className="mt-0.5 h-4 w-4 flex-shrink-0" style={{ color: "#1B5E20" }} />
            <span>
              VivaSense <strong className="text-foreground">Pro</strong> unlocks combined / multi-environment ANOVA, genetic parameters, PCA, path analysis, clustering, selection indices, and Word exports.
            </span>
          </p>
        </div>

        <div className="rounded-xl bg-green-50 p-5 text-sm text-green-800">
          <p className="font-semibold text-green-900 mb-3">✨ Pro includes:</p>
          <ul className="space-y-2">
            <li>✓ Download publication-ready Word reports</li>
            <li>✓ AI-powered academic interpretation</li>
            <li>✓ Genetic parameters (H², GCV, PCV, GAM)</li>
            <li>✓ Multi-trait batch analysis</li>
            <li>✓ G×E interaction & stability analysis</li>
            <li>✓ Trait relationships & heatmaps</li>
            <li>✓ Priority email support</li>
          </ul>
        </div>

        <div className="text-center">
          <p className="text-3xl font-bold text-gray-900">
            ₦5,000<span className="text-base font-normal text-gray-500">/month</span>
          </p>
          <p className="text-sm text-gray-500 mt-1">Special rate for African researchers</p>
        </div>

        <DialogFooter className="sm:justify-center gap-2 flex-col sm:flex-row">
          <Button variant="outline" className="gap-1.5" asChild>
            <a href={PRO_INTEREST_URL} target="_blank" rel="noopener noreferrer" onClick={() => void recordInterest()}>
              <Sparkles className="h-4 w-4" />
              Get Pro via WhatsApp
            </a>
          </Button>
          <Button variant="outline" onClick={() => void recordInterest()} className="gap-1.5">
            <CalendarCheck className="h-4 w-4" />
            I'm Interested — Contact Me Later
          </Button>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
