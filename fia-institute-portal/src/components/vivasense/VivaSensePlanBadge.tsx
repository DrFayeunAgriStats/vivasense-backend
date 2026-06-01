import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { getVivaSenseMode, subscribeVivaSenseMode, type VivaSenseMode } from "@/lib/vivasenseGating";
import { useAuth } from "@/contexts/AuthContext";

export function VivaSensePlanBadge() {
  const [mode, setMode] = useState<VivaSenseMode>(() => getVivaSenseMode());
  const { profile } = useAuth();

  useEffect(() => {
    setMode(getVivaSenseMode());
    return subscribeVivaSenseMode(setMode);
  }, []);

  const normalizedPlan = (profile?.plan || "").toLowerCase();
  const isInstitutional = normalizedPlan === "institutional";
  const isPro = isInstitutional || normalizedPlan === "pro" || mode === "pro";
  const planLabel = isInstitutional ? "Institutional Plan" : isPro ? "Pro Plan" : "Free Plan";
  const planHint = isInstitutional
    ? "Institution account access"
    : isPro
    ? "Advanced modules unlocked"
    : "Basic analysis access";

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground hidden sm:inline">
        {planHint}
      </span>
      <Badge
        variant={isPro ? "default" : "secondary"}
        className={
          isPro
            ? "text-white border-transparent hover:opacity-90"
            : "bg-muted text-foreground border-border"
        }
        style={isPro ? { backgroundColor: "#1B5E20" } : undefined}
      >
        {planLabel}
      </Badge>
    </div>
  );
}
