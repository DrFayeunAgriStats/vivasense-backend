import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Sparkles, Lock, KeyRound, CalendarCheck } from "lucide-react";
import { useState } from "react";
import { Input } from "@/components/ui/input";
import { setVivaSenseMode, type ProGuardInfo } from "@/lib/vivasenseGating";

interface ProFeatureModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  guard: ProGuardInfo | null;
}

const PRO_CODES = ["FIA-PRO-2026", "VIVASENSE-PRO"];

export function ProFeatureModal({ open, onOpenChange, guard }: ProFeatureModalProps) {
  const [showCodeInput, setShowCodeInput] = useState(false);
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);

  if (!guard) return null;

  const submitCode = () => {
    if (PRO_CODES.includes(code.trim().toUpperCase())) {
      setVivaSenseMode("pro");
      setError(null);
      setCode("");
      setShowCodeInput(false);
      onOpenChange(false);
    } else {
      setError("Invalid Pro code. Contact FIA for access.");
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
            Upgrade to access advanced analysis features. {guard.description}
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

        {showCodeInput && (
          <div className="space-y-2">
            <Input
              placeholder="Enter Pro access code"
              value={code}
              onChange={(e) => { setCode(e.target.value); setError(null); }}
              onKeyDown={(e) => e.key === "Enter" && submitCode()}
              autoFocus
            />
            {error && <p className="text-xs text-destructive">{error}</p>}
            <Button onClick={submitCode} className="w-full" style={{ backgroundColor: "#1B5E20" }}>
              Activate Pro
            </Button>
          </div>
        )}

        <DialogFooter className="sm:justify-center gap-2 flex-col sm:flex-row">
          <Button variant="outline" onClick={() => setShowCodeInput((s) => !s)} className="gap-1.5">
            <KeyRound className="h-4 w-4" />
            Enter Pro Code
          </Button>
          <Button variant="outline" asChild className="gap-1.5">
            <a href="mailto:hello@fieldtoinsightacademy.com.ng?subject=VivaSense%20Data%20Clinic%20Booking">
              <CalendarCheck className="h-4 w-4" />
              Book Data Clinic
            </a>
          </Button>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            Close
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
