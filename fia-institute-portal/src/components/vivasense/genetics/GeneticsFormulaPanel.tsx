import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { ChevronDown, BookOpen } from "lucide-react";

const FORMULAS = [
  { label: "Genetic Variance", formula: "σ²g = (MSg − MSe) / r", note: "MSg = Mean Square for Genotype, MSe = Mean Square Error, r = replications" },
  { label: "Environmental Variance", formula: "σ²e = MSe", note: "Mean Square Error from ANOVA" },
  { label: "G×E Interaction Variance", formula: "σ²ge = (MSge − MSe) / r", note: "Only for multilocational trials" },
  { label: "Phenotypic Variance", formula: "σ²p = σ²g + σ²e", note: "Total observed variance" },
  { label: "Broad-sense Heritability", formula: "H² = σ²g / σ²p", note: "Proportion of phenotypic variance due to genetics" },
  { label: "Genetic Advance", formula: "GA = k × √(σ²p) × H²", note: "k = selection differential (2.06 at 5%)" },
  { label: "GA as % of Mean", formula: "GA% = (GA / X̄) × 100", note: "Relative genetic advance" },
  { label: "GCV", formula: "GCV = (√σ²g / X̄) × 100", note: "Genotypic Coefficient of Variation" },
  { label: "PCV", formula: "PCV = (√σ²p / X̄) × 100", note: "Phenotypic Coefficient of Variation" },
];

export function GeneticsFormulaPanel() {
  const [open, setOpen] = useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <Card className="border-primary/20">
        <CardHeader className="pb-2">
          <CollapsibleTrigger className="w-full flex items-center justify-between">
            <CardTitle className="flex items-center gap-3 text-lg">
              <BookOpen className="w-5 h-5 text-primary" />
              Show Formulas
            </CardTitle>
            <ChevronDown className={`w-5 h-5 text-muted-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
          </CollapsibleTrigger>
        </CardHeader>
        <CollapsibleContent>
          <CardContent>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {FORMULAS.map((f) => (
                <div key={f.label} className="rounded-lg border border-border bg-muted/30 p-4 text-center space-y-2">
                  <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">{f.label}</p>
                  <p className="font-mono text-base text-foreground font-bold">{f.formula}</p>
                  <p className="text-xs text-muted-foreground">{f.note}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
