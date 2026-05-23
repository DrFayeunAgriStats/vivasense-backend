import { useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ArrowLeft, Sparkles } from "lucide-react";
import { Link } from "react-router-dom";
import { DatasetUpload } from "@/components/vivasense/genetics-params/DatasetUpload";
import { AdvancedAnalysisDashboard } from "@/components/vivasense/advanced/AdvancedAnalysisDashboard";
import type { DatasetContext } from "@/types/geneticsUpload";

export default function VivaSenseAdvanced() {
  const [datasetContext, setDatasetContext] = useState<DatasetContext | null>(null);

  return (
    <Layout>
      <div className="min-h-screen bg-gradient-to-b from-primary/5 via-background to-background">
        <div className="container-wide py-12">
          <div className="mb-6">
            <Button variant="ghost" size="sm" asChild className="gap-1.5 text-muted-foreground">
              <Link to="/vivasense">
                <ArrowLeft className="h-4 w-4" />
                Back to VivaSense
              </Link>
            </Button>
          </div>

          <div className="max-w-3xl mb-10">
            <div className="flex items-center gap-3 mb-3">
              <Sparkles className="h-8 w-8 text-primary" />
              <h1 className="font-serif text-3xl lg:text-4xl font-bold text-foreground">
                Advanced Analysis
              </h1>
              <Badge variant="secondary" className="text-xs">Beta</Badge>
            </div>
            <p className="text-muted-foreground leading-relaxed">
              Stability, BLUP, PCA and Cluster analysis for breeding decisions. Upload once, then choose
              any module — interpretation is presented before the raw statistics.
            </p>
          </div>

          <div className="max-w-5xl mb-8">
            <DatasetUpload onDatasetReady={setDatasetContext} datasetContext={datasetContext} />
          </div>

          <div className="max-w-6xl">
            <AdvancedAnalysisDashboard datasetContext={datasetContext} />
          </div>
        </div>
      </div>
    </Layout>
  );
}
