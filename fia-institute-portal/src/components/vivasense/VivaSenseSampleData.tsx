import { Download, FileText, FileSpreadsheet } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { BetaBadge } from "./BetaBadge";

const SAMPLE_FILES = [
  {
    file: "/sample-data/01_oneway_CRD.csv",
    design: "One-way ANOVA (CRD)",
    crop: "Cowpea genotype trial — 5 genotypes, 3 reps",
    tags: ["CRD"],
    beta: false,
  },
  {
    file: "/sample-data/05_oneway_RCBD.csv",
    design: "One-way ANOVA (RCBD)",
    crop: "Maize variety trial — 5 varieties, 4 blocks",
    tags: ["RCBD"],
    beta: false,
  },
  {
    file: "/sample-data/02_twoway_CRD_factorial.csv",
    design: "Two-way Factorial (CRD)",
    crop: "Maize nitrogen × variety — 3 N levels, 2 varieties",
    tags: ["Factorial", "CRD"],
    beta: false,
  },
  {
    file: "/sample-data/03_factorial_RCBD.csv",
    design: "Factorial ANOVA (RCBD)",
    crop: "Sorghum variety × fertilizer — 3 varieties, 2 fertilizers, 3 blocks",
    tags: ["Factorial", "RCBD"],
    beta: false,
  },
  {
    file: "/sample-data/04_splitplot.csv",
    design: "Split-plot ANOVA",
    crop: "Cassava irrigation × variety — 2 irrigations, 3 varieties, 3 blocks",
    tags: ["Split-Plot"],
    beta: true,
  },
  {
    file: "/sample_data/vivasense_sample_MET.csv",
    design: "Multi-Environment Trial (MET)",
    crop: "5 genotypes × 4 environments × 3 replications · 4 traits · GGE/AMMI ready",
    tags: ["MET"],
    beta: false,
  },
  {
    file: "/sample-data/07_oneway_RCBD_fertiliser.csv",
    design: "One-way ANOVA (RCBD) — Fertiliser Trial",
    crop: "Maize nitrogen rate trial — 5 N levels, 3 reps · Grain yield, stover, plant height",
    tags: ["RCBD"],
    beta: false,
  },
  {
    file: "/sample-data/08_oneway_RCBD_spacing.csv",
    design: "One-way ANOVA (RCBD) — Plant Spacing Trial",
    crop: "Crop spacing trial — 5 spacing treatments, 3 reps · Yield, LAI, days to flowering",
    tags: ["RCBD"],
    beta: false,
  },
];

export const VivaSenseSampleData = () => {
  return (
    <section className="py-20 bg-muted/30">
      <div className="container mx-auto px-4 max-w-6xl">
        <h2 className="text-3xl md:text-4xl font-bold text-center mb-3 text-foreground font-serif">
          Download Sample Data
        </h2>
        <p className="text-center text-muted-foreground mb-10 text-base md:text-lg">
          Ready-to-run datasets covering CRD, RCBD, Factorial, Split-Plot, and Multi-Environment Trial designs.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 mb-8">
          {SAMPLE_FILES.map((sf) => (
            <Card key={sf.file} className="border-border/60 hover:shadow-md hover:-translate-y-0.5 transition-all duration-300">
              <CardContent className="p-5 flex flex-col gap-3 h-full">
                <div className="flex items-start gap-3">
                  <FileSpreadsheet className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2 flex-wrap">
                      <p className="font-semibold text-sm text-foreground leading-tight">
                        {sf.design}
                      </p>
                      {sf.beta && <BetaBadge />}
                    </div>
                    <div className="flex items-center gap-1.5 flex-wrap mt-2">
                      {sf.tags.map((tag) => (
                        <Badge
                          key={`${sf.file}-${tag}`}
                          variant="outline"
                          className="text-[10px] font-medium tracking-wide"
                        >
                          {tag}
                        </Badge>
                      ))}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2 leading-relaxed">
                      {sf.crop}
                    </p>
                  </div>
                </div>
                <a href={sf.file} download className="mt-auto">
                  <Button variant="outline" size="sm" className="w-full gap-1.5">
                    <Download className="h-3.5 w-3.5" />
                    Download CSV
                  </Button>
                </a>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="flex justify-center">
          <a href="/sample-data/README_VivaSense_DataFormat.txt" download>
            <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground hover:text-foreground">
              <FileText className="h-4 w-4" />
              Download Format Guide (README)
            </Button>
          </a>
        </div>
      </div>
    </section>
  );
};
