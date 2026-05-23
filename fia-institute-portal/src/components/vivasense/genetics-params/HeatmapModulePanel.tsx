import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Loader2, Grid3X3, FileSpreadsheet, Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { toast as sonnerToast } from "sonner";
import { computeCorrelation } from "@/lib/geneticsUploadApi";
import { CorrelationHeatmap } from "./CorrelationHeatmap";
import { AcademicResultsPanel } from "./AcademicResultsPanel";
import type { DatasetContext, CorrelationResponse } from "@/types/geneticsUpload";

const MODULE = "heatmap" as const;

interface Props {
  datasetContext: DatasetContext | null;
}

export function HeatmapModulePanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [isComputing, setIsComputing] = useState(false);
  const [result, setResult] = useState<CorrelationResponse | null>(null);

  if (!datasetContext) {
    return (
      <Card className="border-dashed">
        <CardContent className="py-16 text-center space-y-3">
          <Grid3X3 className="h-10 w-10 mx-auto text-muted-foreground/50" />
          <p className="text-muted-foreground font-medium">Upload a dataset first to generate a heatmap.</p>
        </CardContent>
      </Card>
    );
  }

  const toggleTrait = (t: string) => {
    setSelectedTraits((prev) => prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]);
  };

  const handleGenerate = async () => {
    if (selectedTraits.length < 2) {
      toast({ title: "Select at least 2 traits", variant: "destructive" });
      return;
    }
    setIsComputing(true);
    setResult(null);
    try {
      console.log("[MODULE]", MODULE);
      const res = await computeCorrelation({
        base64_content: datasetContext.base64Content,
        file_type: datasetContext.fileType,
        genotype_column: datasetContext.genotypeColumn,
        rep_column: datasetContext.repColumn,
        environment_column: datasetContext.environmentColumn,
        trait_columns: selectedTraits,
      });
      setResult(res);
      toast({ title: "Heatmap generated" });
    } catch (err: any) {
      toast({ title: "Heatmap generation failed", description: err.message, variant: "destructive" });
    } finally {
      setIsComputing(false);
    }
  };

  const handleDownload = () => {
    if (!result) return;
    console.log("[MODULE]", MODULE);
    console.log("[REQUEST] download-heatmap (client-side export)");
    const lines: string[] = [];
    lines.push("VivaSense Heatmap Data");
    lines.push(`Date: ${new Date().toLocaleString()}`);
    lines.push("");
    lines.push("Correlation Matrix:");
    lines.push(["", ...result.trait_names].join("\t"));
    result.r_matrix.forEach((row, i) => {
      lines.push([result.trait_names[i], ...row.map(v => v.toFixed(4))].join("\t"));
    });
    if (result.interpretation) {
      lines.push("");
      lines.push("Interpretation:");
      lines.push(result.interpretation);
    }
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `VivaSense_Heatmap_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    sonnerToast.success("Heatmap data downloaded");
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
            <Grid3X3 className="h-5 w-5 text-primary" />
            Heatmap & Visualization
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm font-medium">Select traits for heatmap (minimum 2)</p>
            <div className="flex flex-wrap gap-3">
              {datasetContext.availableTraitColumns.map((t) => (
                <label key={t} className="flex items-center gap-2 text-sm cursor-pointer">
                  <Checkbox checked={selectedTraits.includes(t)} onCheckedChange={() => toggleTrait(t)} />
                  {t}
                </label>
              ))}
            </div>
          </div>
          <Button onClick={handleGenerate} disabled={isComputing || selectedTraits.length < 2} className="gap-2">
            {isComputing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Grid3X3 className="h-4 w-4" />}
            Generate Heatmap
          </Button>
        </CardContent>
      </Card>

      {result && (
        <div className="space-y-6">
          <div className="flex justify-end">
            <Button onClick={handleDownload} size="sm" variant="outline" className="gap-2">
              <Download className="h-4 w-4" />
              Download Heatmap Report
            </Button>
          </div>

          {result.interpretation && (
            <AcademicResultsPanel
              moduleLabel="Heatmap"
              interpretation={result.interpretation}
              statisticalNotes={[
                { text: "Correlations were computed using genotype-level means; significance based on number of genotypes." },
              ]}
            />
          )}

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Correlation Heatmap</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <CorrelationHeatmap
                traits={result.trait_names}
                rMatrix={result.r_matrix}
                pMatrix={result.p_matrix}
              />
              {result.n_observations != null && (
                <p className="text-xs text-muted-foreground mt-2">Based on {result.n_observations} observations</p>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
