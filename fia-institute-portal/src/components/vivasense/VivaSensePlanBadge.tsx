import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { getVivaSenseMode, subscribeVivaSenseMode, type VivaSenseMode } from "@/lib/vivasenseGating";

export function VivaSensePlanBadge() {
  const [mode, setMode] = useState<VivaSenseMode>(() => getVivaSenseMode());

  useEffect(() => {
    setMode(getVivaSenseMode());
    return subscribeVivaSenseMode(setMode);
  }, []);

  const isPro = mode === "pro";

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground hidden sm:inline">
        {isPro ? "Advanced modules unlocked" : "Basic analysis access"}
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
        {isPro ? "Pro Plan" : "Free Plan"}
      </Badge>
    </div>
  );
}
