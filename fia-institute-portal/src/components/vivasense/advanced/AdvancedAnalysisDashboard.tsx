import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, TrendingUp, Sigma, Compass, Network, GitBranch, Award, BarChart3 } from "lucide-react";
import type { DatasetContext } from "@/types/geneticsUpload";
import { StabilityPanel } from "./StabilityPanel";
import { BlupPanel } from "./BlupPanel";
import { PcaPanel } from "./PcaPanel";
import { ClusterPanel } from "./ClusterPanel";
import { NonparametricPanel } from "./NonparametricPanel";
import { ManovaPanel } from "./ManovaPanel";
import { PathAnalysisPanel } from "./PathAnalysisPanel";
import { SelectionIndexPanel } from "./SelectionIndexPanel";

export type AdvancedModuleId =
  | "stability" | "blup" | "pca" | "cluster"
  | "nonparametric" | "manova" | "path-analysis" | "selection-index";

interface Props {
  datasetContext: DatasetContext | null;
  initialModule?: AdvancedModuleId | null;
  /** When true, renders without the back-to-modules navigation (used inside parent tabs). */
  embedded?: boolean;
}

const MODULES: {
  id: AdvancedModuleId;
  title: string;
  description: string;
  icon: React.ElementType;
  badge: string;
}[] = [
  {
    id: "stability",
    title: "Stability Analysis",
    description: "GGE biplot and AMMI framing with stability diagnostics for adaptable genotypes across environments.",
    icon: TrendingUp,
    badge: "G×E",
  },
  {
    id: "blup",
    title: "BLUP Predictions",
    description: "Best Linear Unbiased Predictions of breeding values with reliability and standard errors.",
    icon: Sigma,
    badge: "Mixed model",
  },
  {
    id: "pca",
    title: "Principal Component Analysis",
    description: "Reduce dimensionality and visualize multivariate genotype patterns through scree and biplots.",
    icon: Compass,
    badge: "Multivariate",
  },
  {
    id: "cluster",
    title: "Cluster Analysis",
    description: "Hierarchical grouping of genotypes by trait similarity with silhouette and radar profiles.",
    icon: Network,
    badge: "Grouping",
  },
  {
    id: "nonparametric",
    title: "Non-Parametric Tests",
    description: "Robust statistics (Kruskal–Wallis, Friedman, Dunn) for non-normal or ordinal data.",
    icon: BarChart3,
    badge: "Robust",
  },
  {
    id: "manova",
    title: "MANOVA",
    description: "Multivariate testing across correlated traits with univariate follow-up and effect sizes.",
    icon: Network,
    badge: "Multivariate",
  },
  {
    id: "path-analysis",
    title: "Path Analysis",
    description: "Decompose trait correlations into direct and indirect effects with a clear path diagram.",
    icon: GitBranch,
    badge: "Causal",
  },
  {
    id: "selection-index",
    title: "Selection Index",
    description: "Multi-trait genotype ranking using economic weights and selection intensity.",
    icon: Award,
    badge: "Breeding",
  },
];

export function AdvancedAnalysisDashboard({ datasetContext, initialModule = null, embedded = false }: Props) {
  const [active, setActive] = useState<AdvancedModuleId | null>(initialModule);

  if (active) {
    const Panel = active === "stability" ? StabilityPanel
      : active === "blup" ? BlupPanel
      : active === "pca" ? PcaPanel
      : active === "cluster" ? ClusterPanel
      : active === "nonparametric" ? NonparametricPanel
      : active === "manova" ? ManovaPanel
      : active === "path-analysis" ? PathAnalysisPanel
      : SelectionIndexPanel;
    return (
      <div className="space-y-4">
        <Button
          variant="ghost"
          size="sm"
          className="gap-1.5 text-muted-foreground"
          onClick={() => setActive(null)}
        >
          <ArrowLeft className="h-4 w-4" />
          Back to advanced modules
        </Button>
        <Panel datasetContext={datasetContext} />
      </div>
    );
  }

  return (
    <div className={embedded ? "" : "space-y-6"}>
      {!embedded && (
        <div className="max-w-3xl">
          <h2 className="font-serif text-2xl font-bold text-foreground mb-2">Advanced Analysis</h2>
          <p className="text-muted-foreground">
            Multivariate and mixed-model methods for breeding decisions: GGE/AMMI stability framing, BLUPs, PCA, and clustering.
          </p>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {MODULES.map((m) => {
          const Icon = m.icon;
          return (
            <Card
              key={m.id}
              className="cursor-pointer transition-all duration-200 hover:shadow-lg hover:border-primary/40 hover:-translate-y-0.5 group"
              onClick={() => setActive(m.id)}
            >
              <CardContent className="p-5 space-y-3">
                <div className="flex items-start justify-between">
                  <div className="rounded-lg bg-primary/5 p-2.5 group-hover:bg-primary/10 transition-colors">
                    <Icon className="h-6 w-6 text-primary" />
                  </div>
                  <Badge variant="secondary" className="text-[10px] uppercase">{m.badge}</Badge>
                </div>
                <div>
                  <h3 className="font-semibold text-foreground text-base mb-1">{m.title}</h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">{m.description}</p>
                </div>
                <Button size="sm" variant="outline" className="w-full" onClick={(e) => { e.stopPropagation(); setActive(m.id); }}>
                  Run Analysis
                </Button>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
