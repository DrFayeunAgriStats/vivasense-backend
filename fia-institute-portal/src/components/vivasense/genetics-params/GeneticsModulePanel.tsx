import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2, Play, Dna, Download, CheckCircle2, AlertTriangle, FileSpreadsheet } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { toast as sonnerToast } from "sonner";
import { analyzeUpload, downloadReport } from "@/lib/geneticsUploadApi";
import { AcademicResultsPanel } from "./AcademicResultsPanel";
import type { DatasetContext, AnalyzeUploadResponse } from "@/types/geneticsUpload";

const MODULE = "genetic_parameters" as const;

interface Props {
  datasetContext: DatasetContext | null;
}

function deriveReliability(h2: number | undefined): string {
  if (h2 == null) return "Unknown";
  if (h2 >= 0.6) return "High";
  if (h2 >= 0.3) return "Moderate";
  return "Low";
}

function deriveGamClass(gam: number | undefined): string {
  if (gam == null) return "Unknown";
  if (gam >= 20) return "High";
  if (gam >= 10) return "Moderate";
  return "Low";
}

function derivePcvGcvGap(pcv: number | undefined, gcv: number | undefined): string {
  if (pcv == null || gcv == null) return "Unknown";
  const gap = pcv - gcv;
  if (gap <= 5) return "Small";
  if (gap <= 15) return "Moderate";
  return "Large";
}

export function GeneticsModulePanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [selectionIntensity, setSelectionIntensity] = useState(1.4);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalyzeUploadResponse | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  if (!datasetContext) {
    return (
      <Card className="border-dashed">
        <CardContent className="py-16 text-center space-y-3">
          <Dna className="h-10 w-10 mx-auto text-muted-foreground/50" />
          <p className="text-muted-foreground font-medium">Upload a dataset first to compute genetic parameters.</p>
        </CardContent>
      </Card>
    );
  }

  const toggleTrait = (t: string) => {
    setSelectedTraits((prev) => prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]);
  };

  const handleAnalyze = async () => {
    if (selectedTraits.length === 0) return;
    setIsAnalyzing(true);
    setResults(null);
    try {
      console.log("[MODULE]", MODULE);
      const res = await analyzeUpload({
        base64_content: datasetContext.base64Content,
        file_type: datasetContext.fileType,
        genotype_column: datasetContext.genotypeColumn,
        rep_column: datasetContext.repColumn,
        environment_column: datasetContext.environmentColumn,
        trait_columns: selectedTraits,
        mode: datasetContext.mode,
        random_environment: false,
        selection_intensity: selectionIntensity,
        module: MODULE,
      });
      setResults(res);
      toast({ title: "Genetic parameters computed", description: `${res.summary_table.length} trait(s) analyzed.` });
    } catch (err: any) {
      toast({ title: "Analysis failed", description: err.message, variant: "destructive" });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDownload = async () => {
    if (!results) return;
    setIsDownloading(true);
    try {
      const payload = {
        analysis_type: MODULE,
        dataset_summary: results.dataset_summary,
        summary_table: results.summary_table,
        trait_results: Object.fromEntries(
          Object.entries(results.trait_results)
            .filter(([, tr]) => tr.status === "success" && tr.analysis_result)
            .map(([trait, tr]) => [trait, {
              variance_components: tr.analysis_result.result.variance_components,
              genetic_parameters: tr.analysis_result.result.genetic_parameters,
              heritability: tr.analysis_result.result.heritability,
              grand_mean: tr.analysis_result.result.grand_mean,
              n_genotypes: tr.analysis_result.result.n_genotypes,
              n_reps: tr.analysis_result.result.n_reps,
              interpretation: tr.analysis_result.interpretation,
            }])
        ),
        failed_traits: results.failed_traits,
      };

      const blob = await downloadReport(MODULE, payload);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `VivaSense_GeneticParameters_${new Date().toISOString().slice(0, 10)}.docx`;
      a.click();
      URL.revokeObjectURL(url);
      sonnerToast.success("Genetic Parameters report downloaded");
    } catch {
      sonnerToast.error("Download failed");
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-md border border-primary/20 bg-primary/5 p-3 flex items-center gap-2 text-sm">
        <FileSpreadsheet className="h-4 w-4 text-primary shrink-0" />
        <span>Using: <span className="font-medium">{datasetContext.file.name}</span></span>
        <Badge variant="outline" className="ml-auto text-xs">{datasetContext.availableTraitColumns.length} traits · {datasetContext.mode} mode</Badge>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Dna className="h-5 w-5 text-emerald-600" />
            Genetic Parameters
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm font-medium">Select traits for analysis</p>
            <div className="flex flex-wrap gap-3">
              {datasetContext.availableTraitColumns.map((t) => (
                <label key={t} className="flex items-center gap-2 text-sm cursor-pointer">
                  <Checkbox checked={selectedTraits.includes(t)} onCheckedChange={() => toggleTrait(t)} />
                  {t}
                </label>
              ))}
            </div>
          </div>

          <div className="max-w-xs space-y-1.5">
            <Label className="text-sm font-medium">Selection Intensity (k)</Label>
            <Input
              type="number"
              step="0.01"
              min={0}
              value={selectionIntensity}
              onChange={(e) => setSelectionIntensity(Number(e.target.value) || 1.4)}
            />
            <p className="text-xs text-muted-foreground">Default 1.4 (~20% selection). Use 2.06 for 5%.</p>
          </div>

          <Button onClick={handleAnalyze} disabled={isAnalyzing || selectedTraits.length === 0} className="gap-2">
            {isAnalyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Compute Genetic Parameters
          </Button>
        </CardContent>
      </Card>

      {results && (
        <div className="space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-emerald-600" />
                Genetic Parameter Results
              </CardTitle>
              <Button onClick={handleDownload} disabled={isDownloading} size="sm" className="gap-2 bg-primary hover:bg-primary/90 text-primary-foreground">
                {isDownloading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                {isDownloading ? "Downloading..." : "Download Genetic Parameters Report"}
              </Button>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2 text-sm">
                <Badge variant="secondary">{results.dataset_summary.n_genotypes} genotypes</Badge>
                <Badge variant="secondary">{results.dataset_summary.n_reps} reps</Badge>
                <Badge variant="outline">{results.dataset_summary.mode} mode</Badge>
              </div>
            </CardContent>
          </Card>

          {results.failed_traits.length > 0 && (
            <Card className="border-destructive/30">
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-sm text-destructive font-medium mb-1">
                  <AlertTriangle className="h-4 w-4" /> Failed Traits
                </div>
                <ul className="list-disc pl-5 text-sm text-muted-foreground">
                  {results.failed_traits.map((t) => <li key={t}>{t}</li>)}
                </ul>
              </CardContent>
            </Card>
          )}

          {Object.entries(results.trait_results).map(([trait, tr]) => {
            if (tr.status !== "success" || !tr.analysis_result) return null;
            const r = tr.analysis_result.result;
            const vc = r.variance_components;
            const gp = r.genetic_parameters;
            const h2 = r.heritability.h2_broad_sense;
            const summaryRow = results.summary_table.find((s) => s.trait === trait);

            return (
              <div key={trait} className="space-y-1">
                <h3 className="text-base font-semibold text-foreground px-1">{trait}</h3>
                <AcademicResultsPanel
                  moduleLabel="Genetic Parameters"
                  selectionReliability={deriveReliability(h2)}
                  insightSummary={`Selection for ${trait} appears ${deriveReliability(h2).toLowerCase()}ly reliable under the observed conditions.`}
                  insightBasis={`Based on heritability (H² = ${h2?.toFixed(3) ?? "—"}) and environmental influence`}
                  interpretation={tr.analysis_result.interpretation}
                  recommendation={
                    h2 >= 0.6
                      ? `High heritability (${h2.toFixed(3)}) with GAM of ${gp.GAM_percent?.toFixed(1)}% suggests simple selection methods can be effective for ${trait}.`
                      : h2 >= 0.3
                      ? `Moderate heritability for ${trait} — consider progeny testing or replicated trials to improve selection accuracy.`
                      : `Low heritability for ${trait} — environmental influence is dominant. Heterosis breeding or environmental management may be more effective.`
                  }
                  classifications={[
                    {
                      label: "Heritability",
                      value: summaryRow?.heritability_class ? summaryRow.heritability_class.charAt(0).toUpperCase() + summaryRow.heritability_class.slice(1) : deriveReliability(h2),
                      tooltip: `Broad-sense heritability (H²) = ${h2?.toFixed(3) ?? "—"}. Indicates the proportion of phenotypic variance due to genetic factors.`,
                    },
                    {
                      label: "GAM",
                      value: deriveGamClass(gp.GAM_percent),
                      tooltip: `Genetic Advance as % of Mean = ${gp.GAM_percent?.toFixed(2) ?? "—"}%. Predicts expected genetic gain under selection.`,
                    },
                    {
                      label: "PCV–GCV Gap",
                      value: derivePcvGcvGap(gp.PCV, gp.GCV),
                      tooltip: `PCV = ${gp.PCV?.toFixed(2) ?? "—"}%, GCV = ${gp.GCV?.toFixed(2) ?? "—"}%. A small gap indicates low environmental influence on trait variation.`,
                    },
                  ]}
                  statisticalNotes={[
                    { text: `Basis: ${r.heritability.interpretation_basis} | k = ${gp.selection_intensity}` },
                  ]}
                  anovaTable={r.anova_table}
                  meanSeparation={r.mean_separation}
                  descriptiveStats={[
                    { label: "Grand Mean", value: r.grand_mean?.toFixed(4) ?? "—" },
                    { label: "σ²g (Genotypic)", value: vc.sigma2_genotype?.toFixed(4) ?? "—" },
                    { label: "σ²e (Error)", value: vc.sigma2_error?.toFixed(4) ?? "—" },
                    { label: "σ²p (Phenotypic)", value: vc.sigma2_phenotypic?.toFixed(4) ?? "—" },
                    { label: "H² (Broad-sense)", value: h2?.toFixed(3) ?? "—" },
                    { label: "GCV (%)", value: gp.GCV?.toFixed(2) ?? "—" },
                    { label: "PCV (%)", value: gp.PCV?.toFixed(2) ?? "—" },
                    { label: "GAM (%)", value: gp.GAM_percent?.toFixed(2) ?? "—" },
                  ]}
                />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
