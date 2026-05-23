import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FlaskConical } from "lucide-react";

interface Props {
  metaEntries: Array<[string, unknown]>;
  nTreatments?: number | null;
  nReps?: number | null;
  nObservations?: number | null;
}

/** Human-readable labels for common meta keys */
const LABELS: Record<string, string> = {
  design: "Experimental Design",
  response: "Trait Analysed",
  treatment: "Treatment Factor",
  formula: "Statistical Model",
  grand_mean: "Grand Mean",
  cv_percent: "CV (%)",
  n_treatments: "Treatments",
  n_reps: "Replications",
  n_observations: "Total Observations",
};

function prettyLabel(key: string): string {
  return LABELS[key] || key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Format design names to standard abbreviations */
function formatDesign(val: string): string {
  const s = val.toLowerCase().trim();
  if (s.includes("rcbd") || s.includes("randomized complete block")) return "RCBD";
  if (s.includes("crd") || s.includes("completely randomized")) return "CRD";
  if (s.includes("split") && s.includes("plot")) return "Split-Plot";
  if (s.includes("factorial")) return "Factorial";
  if (s.includes("lattice")) return "Lattice";
  return val;
}

export function ExperimentSummary({ metaEntries, nTreatments, nReps, nObservations }: Props) {
  // Build enriched entries: inject computed counts if not already present
  const enriched = [...metaEntries];
  const existingKeys = new Set(metaEntries.map(([k]) => k));
  if (nTreatments != null && !existingKeys.has("n_treatments")) enriched.push(["n_treatments", nTreatments]);
  if (nReps != null && !existingKeys.has("n_reps")) enriched.push(["n_reps", nReps]);
  if (nObservations != null && !existingKeys.has("n_observations")) enriched.push(["n_observations", nObservations]);

  if (enriched.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-3">
          <FlaskConical className="w-6 h-6 text-primary" />
          Experiment Summary
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-4">
          {enriched.map(([key, value]) => {
            const label = prettyLabel(String(key));
            let displayValue = String(value);
            if (String(key) === "design") displayValue = formatDesign(displayValue);

            return (
              <div key={String(key)} className="flex flex-col border-l-2 border-primary/20 pl-3">
                <dt className="text-xs text-muted-foreground font-medium uppercase tracking-wider">
                  {label}
                </dt>
                <dd className="text-foreground font-semibold text-base mt-0.5">{displayValue}</dd>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
