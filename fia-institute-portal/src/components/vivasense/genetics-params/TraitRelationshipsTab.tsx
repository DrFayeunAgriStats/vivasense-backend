import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Loader2, TrendingUp, AlertTriangle, Info, FileSpreadsheet } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { computeCorrelation } from "@/lib/geneticsUploadApi";
import { CorrelationHeatmap } from "./CorrelationHeatmap";
import type { DatasetContext, CorrelationResponse } from "@/types/geneticsUpload";

interface Props {
  datasetContext: DatasetContext | null;
}

export function TraitRelationshipsTab({ datasetContext }: Props) {
  const { toast } = useToast();
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [isComputing, setIsComputing] = useState(false);
  const [result, setResult] = useState<CorrelationResponse | null>(null);

  if (!datasetContext) {
    return (
      <Card className="border-dashed">
        <CardContent className="py-16 text-center space-y-3">
          <FileSpreadsheet className="h-10 w-10 mx-auto text-muted-foreground/50" />
          <p className="text-muted-foreground font-medium">
            Upload and analyze a dataset first to use Trait Relationships.
          </p>
          <p className="text-sm text-muted-foreground">
            Switch to the <span className="font-medium">Upload File</span> tab, upload your dataset, and confirm the column mapping.
          </p>
        </CardContent>
      </Card>
    );
  }

  const toggleTrait = (t: string) => {
    setSelectedTraits((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]
    );
  };

  const handleCompute = async () => {
    if (selectedTraits.length < 2) {
      toast({ title: "Select at least 2 traits", variant: "destructive" });
      return;
    }
    setIsComputing(true);
    setResult(null);
    try {
      console.log("[TraitRelationshipsTab] datasetContext check:", {
        hasBase64: !!datasetContext.base64Content,
        base64Length: datasetContext.base64Content?.length,
        fileType: datasetContext.fileType,
        genotypeCol: datasetContext.genotypeColumn,
        repCol: datasetContext.repColumn,
        envCol: datasetContext.environmentColumn,
        selectedTraits,
      });
      const res = await computeCorrelation({
        base64_content: datasetContext.base64Content,
        file_type: datasetContext.fileType,
        genotype_column: datasetContext.genotypeColumn,
        rep_column: datasetContext.repColumn,
        environment_column: datasetContext.environmentColumn,
        trait_columns: selectedTraits,
      });
      setResult(res);
      toast({ title: "Correlation computed" });
    } catch (err: any) {
      const errMsg = typeof err === "object" && err !== null
        ? (err.message || JSON.stringify(err))
        : String(err);
      console.error("[CORRELATION CATCH]", errMsg);
      toast({ title: "Correlation failed", description: errMsg, variant: "destructive" });
    } finally {
      setIsComputing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Dataset banner */}
      <div className="rounded-md border border-primary/20 bg-primary/5 p-3 flex items-center gap-2 text-sm">
        <FileSpreadsheet className="h-4 w-4 text-primary shrink-0" />
        <span>
          Using uploaded dataset: <span className="font-medium">{datasetContext.file.name}</span>
          <span className="text-muted-foreground ml-2">
            ({datasetContext.availableTraitColumns.length} traits available)
          </span>
        </span>
      </div>

      {/* Trait selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-primary" />
            Phenotypic Correlation Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm font-medium">Select traits for correlation (minimum 2)</p>
            <div className="flex flex-wrap gap-3">
              {datasetContext.availableTraitColumns.map((t) => (
                <label key={t} className="flex items-center gap-2 text-sm cursor-pointer">
                  <Checkbox
                    checked={selectedTraits.includes(t)}
                    onCheckedChange={() => toggleTrait(t)}
                  />
                  {t}
                </label>
              ))}
            </div>
          </div>
          <Button
            onClick={handleCompute}
            disabled={isComputing || selectedTraits.length < 2}
            className="gap-2"
          >
            {isComputing ? <Loader2 className="h-4 w-4 animate-spin" /> : <TrendingUp className="h-4 w-4" />}
            Compute Correlation
          </Button>
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Warnings */}
          {result.warnings.length > 0 && (
            <div className="rounded-md border border-yellow-300 bg-yellow-50 p-3 text-sm text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-200 dark:border-yellow-700">
              <div className="flex items-center gap-2 mb-1 font-medium">
                <AlertTriangle className="h-4 w-4" /> Warnings
              </div>
              <ul className="list-disc pl-5 space-y-0.5">
                {result.warnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            </div>
          )}

          {/* Statistical note */}
          {result.statistical_note && (
            <div className="rounded-md border border-border bg-muted/30 p-3 text-sm flex gap-2">
              <Info className="h-4 w-4 text-muted-foreground mt-0.5 shrink-0" />
              <div>
                <p className="text-muted-foreground">{result.statistical_note}</p>
                <p className="text-xs text-muted-foreground mt-1 italic">
                  P-values are unadjusted for multiple comparisons.
                </p>
              </div>
            </div>
          )}

          {/* Heatmap */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Correlation Matrix</CardTitle>
            </CardHeader>
            <CardContent>
              <CorrelationHeatmap
                traits={result.trait_names}
                rMatrix={result.r_matrix}
                pMatrix={result.p_matrix}
              />
              {result.n_observations != null && (
                <p className="text-xs text-muted-foreground mt-2">
                  Based on {result.n_observations} observations
                </p>
              )}
            </CardContent>
          </Card>

          {/* Interpretation */}
          {result.interpretation && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Interpretation</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {result.interpretation}
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
