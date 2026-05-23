import { Layout } from "@/components/layout/Layout";
import { BarChart3 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { DataSourceTabs } from "@/components/vivasense-genetics/DataSourceTabs";
import { MultiTraitUpload } from "@/components/vivasense-genetics/MultiTraitUpload";
import { TraitRelationships } from "@/components/vivasense-genetics/TraitRelationships";
import { DescriptiveStatsModule } from "@/components/vivasense-genetics/DescriptiveStatsModule";
import { AnovaWorkspaceModule } from "@/components/vivasense-genetics/AnovaWorkspaceModule";
import { GeneticsWorkspaceModule } from "@/components/vivasense-genetics/GeneticsWorkspaceModule";
import { AdvancedAnalysisDashboard } from "@/components/vivasense/advanced/AdvancedAnalysisDashboard";
import { FieldDataCollection } from "@/components/vivasense/FieldDataCollection";
import { InstallVivaSenseButton } from "@/components/vivasense/InstallVivaSenseButton";
import type { UploadDatasetContext } from "@/services/geneticsUploadApi";
import type { DatasetContext } from "@/types/geneticsUpload";

function toLegacyDatasetContext(ctx: UploadDatasetContext | null): DatasetContext | null {
  if (!ctx) return null;
  return {
    file: ctx.file,
    base64Content: ctx.base64Content,
    fileType: ctx.fileType,
    genotypeColumn: ctx.genotypeColumn,
    repColumn: ctx.repColumn,
    environmentColumn: ctx.environmentColumn ?? null,
    availableTraitColumns: ctx.availableTraitColumns,
    mode: ctx.mode,
    datasetToken: ctx.datasetToken ?? null,
  };
}

export default function VivaSenseGenetics() {
  return (
    <Layout>
      <div className="min-h-screen bg-gradient-to-b from-primary/5 via-background to-background">
        <div className="mx-auto w-full max-w-[1680px] px-4 py-10 sm:px-6 lg:px-8">

          <div className="mb-6">
            <Button variant="ghost" size="sm" asChild className="gap-1.5 text-muted-foreground">
              <Link to="/vivasense">
                <ArrowLeft className="h-4 w-4" />
                Back to VivaSense
              </Link>
            </Button>
          </div>

          <div className="mb-8 rounded-3xl border border-primary/10 bg-white/90 px-6 py-6 shadow-sm lg:px-8">
            <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
              <div className="max-w-4xl">
                <div className="flex flex-wrap items-center gap-3 mb-3">
                  <BarChart3 className="h-8 w-8 text-primary" />
                  <h1 className="font-serif text-3xl lg:text-4xl font-bold text-foreground">
                    VivaSense Statistical Workspace
                  </h1>
                  <Badge variant="secondary" className="text-xs">Beta</Badge>
                </div>
                <p className="text-muted-foreground leading-relaxed text-base lg:text-[17px] max-w-3xl">
                  Methodology-aware agricultural research environment for upload, design detection, variable mapping,
                  validation, analysis, interpretation, and publication-ready reporting.
                </p>
                <p className="mt-3 text-sm text-muted-foreground max-w-3xl">
                  Supports ANOVA, descriptive statistics, genetic parameters, trait relationships, GGE biplot,
                  AMMI, PCA, and related analytical workflows from one integrated workspace.
                </p>
              </div>
              <div className="flex shrink-0 items-center">
                <InstallVivaSenseButton />
              </div>
            </div>
          </div>

          <DataSourceTabs
            uploadContent={<MultiTraitUpload />}
            fieldDataContent={<FieldDataCollection />}
            anovaContent={(ctx: UploadDatasetContext | null) => (
              <AnovaWorkspaceModule datasetContext={ctx} />
            )}
            geneticsContent={(ctx: UploadDatasetContext | null) => (
              <GeneticsWorkspaceModule datasetContext={ctx} />
            )}
            traitRelationshipsContent={(ctx: UploadDatasetContext | null) => (
              <TraitRelationships datasetContext={ctx} />
            )}
            descriptiveStatsContent={(ctx: UploadDatasetContext | null) => (
              <DescriptiveStatsModule datasetContext={ctx} />
            )}
            advancedContent={(ctx: UploadDatasetContext | null) => (
              <AdvancedAnalysisDashboard datasetContext={toLegacyDatasetContext(ctx)} embedded />
            )}
          />

        </div>
      </div>
    </Layout>
  );
}


