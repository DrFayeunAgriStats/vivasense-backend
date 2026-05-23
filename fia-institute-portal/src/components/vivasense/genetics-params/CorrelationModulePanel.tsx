import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Loader2, TrendingUp, AlertTriangle, FileSpreadsheet, GitCompareArrows, Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { toast as sonnerToast } from "sonner";
import { computeCorrelation } from "@/lib/geneticsUploadApi";
import { AcademicResultsPanel } from "./AcademicResultsPanel";
import type { DatasetContext, CorrelationResponse } from "@/types/geneticsUpload";

const MODULE = "correlation" as const;

interface Props {
  datasetContext: DatasetContext | null;
}

export function CorrelationModulePanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [isComputing, setIsComputing] = useState(false);
  const [result, setResult] = useState<CorrelationResponse | null>(null);

  if (!datasetContext) {
    return (
      <Card className="border-dashed">
        <CardContent className="py-16 text-center space-y-3">
          <GitCompareArrows className="h-10 w-10 mx-auto text-muted-foreground/50" />
          <p className="text-muted-foreground font-medium">Upload a dataset first to compute correlations.</p>
        </CardContent>
      </Card>
    );
  }

  const toggleTrait = (t: string) => {
    setSelectedTraits((prev) => prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]);
  };

  const handleCompute = async () => {
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
      toast({ title: "Correlation computed" });
    } catch (err: any) {
      toast({ title: "Correlation failed", description: err.message, variant: "destructive" });
    } finally {
      setIsComputing(false);
    }
  };

  const handleDownload = () => {
    if (!result) return;
    console.log("[MODULE]", MODULE);
    console.log("[REQUEST] download-correlation (client-side export)");
    const lines: string[] = [];
    lines.push("VivaSense Correlation Analysis Report");
    lines.push(`Date: ${new Date().toLocaleString()}`);
    lines.push("");
    lines.push("Correlation Matrix (r-values):");
    lines.push(["", ...result.trait_names].join("\t"));
    result.r_matrix.forEach((row, i) => {
      lines.push([result.trait_names[i], ...row.map(v => v.toFixed(4))].join("\t"));
    });
    lines.push("");
    if (result.p_matrix.length > 0) {
      lines.push("P-value Matrix:");
      lines.push(["", ...result.trait_names].join("\t"));
      result.p_matrix.forEach((row, i) => {
        lines.push([result.trait_names[i], ...row.map(v => v != null ? (v < 0.001 ? "<0.001" : v.toFixed(4)) : "—")].join("\t"));
      });
      lines.push("");
    }
    if (result.interpretation) {
      lines.push("Interpretation:");
      lines.push(result.interpretation);
    }

    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `VivaSense_Correlation_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    sonnerToast.success("Correlation report downloaded");
  };

  const deriveInsight = () => {
    if (!result) return undefined;
    const pairs: { t1: string; t2: string; r: number }[] = [];
    result.trait_names.forEach((t1, i) => {
      result.trait_names.slice(i + 1).forEach((t2, jOff) => {
        const j = i + 1 + jOff;
        pairs.push({ t1, t2, r: result.r_matrix[i][j] });
      });
    });
    const strongest = pairs.reduce((a, b) => Math.abs(a.r) > Math.abs(b.r) ? a : b, pairs[0]);
    if (!strongest) return undefined;
    const dir = strongest.r > 0 ? "positive" : "negative";
    return `Strongest relationship: ${strongest.t1} vs ${strongest.t2} (r = ${strongest.r.toFixed(3)}, ${dir})`;
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
            <GitCompareArrows className="h-5 w-5 text-primary" />
            Trait Relationships
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <p className="text-sm font-medium">Select traits for correlation (minimum 2)</p>
            <div className="flex flex-wrap gap-3">
              {datasetContext.availableTraitColumns.map((t) => (
                <label key={t} className="flex items-center gap-2 text-sm cursor-pointer">
                  <Checkbox checked={selectedTraits.includes(t)} onCheckedChange={() => toggleTrait(t)} />
                  {t}
                </label>
              ))}
            </div>
          </div>
          <Button onClick={handleCompute} disabled={isComputing || selectedTraits.length < 2} className="gap-2">
            {isComputing ? <Loader2 className="h-4 w-4 animate-spin" /> : <TrendingUp className="h-4 w-4" />}
            Compute Correlation
          </Button>
        </CardContent>
      </Card>

      {result && (
        <div className="space-y-6">
          <div className="flex justify-end">
            <Button onClick={handleDownload} size="sm" variant="outline" className="gap-2">
              <Download className="h-4 w-4" />
              Download Correlation Report
            </Button>
          </div>

          <AcademicResultsPanel
            moduleLabel="Correlation"
            insightSummary={deriveInsight()}
            insightBasis={result.n_observations != null ? `Based on ${result.n_observations} observations` : undefined}
            interpretation={result.interpretation}
            statisticalNotes={[
              ...(result.statistical_note ? [{ text: result.statistical_note }] : []),
              { text: "Correlations were computed using genotype-level means; significance based on number of genotypes." },
              { text: "P-values are unadjusted for multiple comparisons." },
            ]}
          />

          {result.warnings.length > 0 && (
            <div className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm dark:bg-amber-900/20 dark:border-amber-700">
              <div className="flex items-center gap-2 mb-1 font-medium text-amber-800 dark:text-amber-200"><AlertTriangle className="h-4 w-4" /> Warnings</div>
              <ul className="list-disc pl-5 space-y-0.5 text-amber-800 dark:text-amber-200">{result.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
            </div>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Correlation Matrix</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="border-collapse text-sm w-full">
                  <thead>
                    <tr>
                      <th className="border border-border px-3 py-2 bg-muted font-medium text-left">Trait Pair</th>
                      <th className="border border-border px-3 py-2 bg-muted font-medium text-right">r-value</th>
                      <th className="border border-border px-3 py-2 bg-muted font-medium text-right">p-value</th>
                      <th className="border border-border px-3 py-2 bg-muted font-medium text-center">Sig.</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.trait_names.map((t1, i) =>
                      result.trait_names.slice(i + 1).map((t2, jOff) => {
                        const j = i + 1 + jOff;
                        const r = result.r_matrix[i]?.[j] ?? 0;
                        const p = result.p_matrix[i]?.[j] ?? null;
                        const stars = p != null ? (p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "ns") : "—";
                        return (
                          <tr key={`${i}-${j}`} className="even:bg-muted/30">
                            <td className="border border-border px-3 py-2 font-medium">{t1} vs {t2}</td>
                            <td className={`border border-border px-3 py-2 text-right font-mono ${Math.abs(r) >= 0.7 ? "font-bold text-emerald-700 dark:text-emerald-400" : ""}`}>
                              {r.toFixed(3)}
                            </td>
                            <td className="border border-border px-3 py-2 text-right font-mono">
                              {p != null ? (p < 0.001 ? "<0.001" : p.toFixed(3)) : "—"}
                            </td>
                            <td className="border border-border px-3 py-2 text-center font-mono">{stars}</td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted-foreground mt-1 italic">
                *** p &lt; 0.001, ** p &lt; 0.01, * p &lt; 0.05, ns = not significant
              </p>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
