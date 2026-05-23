import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Loader2, TrendingUp, AlertTriangle, FileSpreadsheet, GitCompareArrows, Download, Grid3X3, LineChart, Dna, Info, ArrowRight, Sparkles } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useToast } from "@/hooks/use-toast";
import { toast as sonnerToast } from "sonner";
import { computeCorrelation } from "@/lib/geneticsUploadApi";
import { AcademicResultsPanel } from "./AcademicResultsPanel";
import { CorrelationHeatmap } from "./CorrelationHeatmap";
import { RegressionAnalysisTab } from "./RegressionAnalysisTab";
import type { DatasetContext, CorrelationResponse, CorrelationModeKey, CorrelationModeBlock } from "@/types/geneticsUpload";

const MODE_ORDER: CorrelationModeKey[] = ["phenotypic", "between_genotype", "genotypic"];
const MODE_META: Record<CorrelationModeKey, { label: string; description: string }> = {
  phenotypic: {
    label: "Phenotypic correlation",
    description: "Field-level — uses all observations (raw plot data).",
  },
  between_genotype: {
    label: "Between-genotype association",
    description: "Computed from genotype means across replications/environments.",
  },
  genotypic: {
    label: "Genotypic correlation",
    description: "Variance-component based (true genetic correlation).",
  },
};

const MODULE = "relationship" as const;

interface Props {
  datasetContext: DatasetContext | null;
}

export function RelationshipModulePanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [isComputing, setIsComputing] = useState(false);
  const [result, setResult] = useState<CorrelationResponse | null>(null);
  const [selectedMode, setSelectedMode] = useState<CorrelationModeKey>("phenotypic");

  // Detect mode-block shape: scalar pairwise (new backend) vs matrix (legacy)
  // NOTE: We trust DATA presence over the `available` flag — if the backend
  // sent scalar r/rg/p_value or a matrix, the mode IS available regardless
  // of any explicit `available: false` flag.
  const isBlockAvailable = (b: CorrelationModeBlock | undefined | null): boolean => {
    if (!b) return false;
    const hasScalar =
      typeof b.r === "number" || typeof b.rg === "number" || typeof b.p_value === "number";
    const hasMatrix = Array.isArray(b.r_matrix) && b.r_matrix.length > 0;
    if (hasScalar || hasMatrix) return true;
    if (b.available === false) return false;
    return false;
  };

  const modeBlocks = useMemo(() => {
    if (!result) return null;
    const legacyBlock: CorrelationModeBlock = {
      r_matrix: result.r_matrix ?? [],
      p_matrix: result.p_matrix ?? [],
      n_observations: result.n_observations ?? null,
      available: Array.isArray(result.r_matrix) && result.r_matrix.length > 0,
    };
    const blocks: Record<CorrelationModeKey, CorrelationModeBlock> = {
      phenotypic: result.phenotypic ?? legacyBlock,
      between_genotype: result.between_genotype ?? { available: false },
      genotypic: result.genotypic ?? { available: false },
    };
    MODE_ORDER.forEach((k) => {
      blocks[k] = { ...blocks[k], available: isBlockAvailable(blocks[k]) };
    });
    return blocks;
  }, [result]);

  useEffect(() => {
    if (!modeBlocks) return;
    if (!modeBlocks[selectedMode]?.available) {
      const firstAvail = MODE_ORDER.find((k) => modeBlocks[k]?.available);
      if (firstAvail) setSelectedMode(firstAvail);
    }
  }, [modeBlocks, selectedMode]);

  useEffect(() => {
    if (!result) return;
    console.log("FULL CORRELATION RESPONSE:", result);
    console.log("[CORRELATION SELECTED MODE]", selectedMode);
    console.log("[CORRELATION MODE BLOCKS]", modeBlocks);
    console.log("[3-MODE MARKER MOUNTED]", true);
  }, [result, selectedMode, modeBlocks]);

  const activeBlock: CorrelationModeBlock | null = modeBlocks ? modeBlocks[selectedMode] : null;
  const isScalarShape = !!activeBlock && (typeof activeBlock.r === "number" || typeof activeBlock.rg === "number");
  const hasMatrix = !!activeBlock && Array.isArray(activeBlock.r_matrix) && activeBlock.r_matrix.length > 0;
  const genotypicFallback = !!modeBlocks && !modeBlocks.genotypic.available && modeBlocks.between_genotype.available;

  if (!datasetContext) {
    return (
      <Card className="border-dashed">
        <CardContent className="py-16 text-center space-y-3">
          <GitCompareArrows className="h-10 w-10 mx-auto text-muted-foreground/50" />
          <p className="text-muted-foreground font-medium">Upload a dataset first to analyze trait relationships.</p>
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
      console.log("[MODULE]", MODULE, "[SUBMODULE] correlation");
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
    console.log("[MODULE]", MODULE, "[REQUEST] download-correlation (client-side export)");
    const lines: string[] = [];
    lines.push("VivaSense Relationship Analysis Report");
    lines.push(`Date: ${new Date().toLocaleString()}`);
    lines.push(`Traits: ${result.trait_names.join(", ")}`);
    if (result.method) lines.push(`Method: ${result.method}`);
    lines.push("");

    if (modeBlocks) {
      MODE_ORDER.forEach((k) => {
        const b = modeBlocks[k];
        if (!b?.available) return;
        lines.push(`── ${MODE_META[k].label} ──`);
        const isGeno = k === "genotypic";
        const rVal = isGeno ? b.rg : b.r;
        if (typeof rVal === "number") lines.push(`${isGeno ? "rg" : "r"} = ${rVal.toFixed(4)}`);
        if (typeof b.p_value === "number") lines.push(`p-value = ${b.p_value < 0.001 ? "<0.001" : b.p_value.toFixed(4)}`);
        if (typeof b.df === "number") lines.push(`df = ${b.df}`);
        if (typeof b.critical_r === "number") lines.push(`Critical r (α=0.05) = ${b.critical_r.toFixed(4)}`);
        if (typeof b.ci_lower === "number" && typeof b.ci_upper === "number") {
          lines.push(`95% CI = ${b.ci_lower.toFixed(4)} to ${b.ci_upper.toFixed(4)}`);
        }
        if (Array.isArray(b.r_matrix) && b.r_matrix.length > 0) {
          lines.push("Correlation Matrix:");
          lines.push(["", ...result.trait_names].join("\t"));
          b.r_matrix.forEach((row, i) => {
            lines.push([result.trait_names[i], ...row.map((v) => (typeof v === "number" ? v.toFixed(4) : "—"))].join("\t"));
          });
        }
        lines.push("");
      });
    }

    if (result.interpretation) {
      lines.push("Interpretation:");
      lines.push(result.interpretation);
    }
    if (result.statistical_note) {
      lines.push("");
      lines.push("Statistical Note:");
      lines.push(result.statistical_note);
    }
    const blob = new Blob([lines.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `VivaSense_Relationship_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    sonnerToast.success("Relationship report downloaded");
  };

  const deriveInsight = () => {
    if (!result || !activeBlock || !activeBlock.available) return undefined;
    const r = activeBlock.r_matrix;
    if (!Array.isArray(r) || r.length === 0) return undefined;
    const pairs: { t1: string; t2: string; r: number }[] = [];
    result.trait_names.forEach((t1, i) => {
      result.trait_names.slice(i + 1).forEach((t2, jOff) => {
        const j = i + 1 + jOff;
        const val = r[i]?.[j];
        if (typeof val === "number" && !Number.isNaN(val)) pairs.push({ t1, t2, r: val });
      });
    });
    if (pairs.length === 0) return undefined;
    const strongest = pairs.reduce((a, b) => Math.abs(a.r) > Math.abs(b.r) ? a : b, pairs[0]);
    const dir = strongest.r > 0 ? "positive" : "negative";
    return `Strongest relationship: ${strongest.t1} vs ${strongest.t2} (r = ${strongest.r.toFixed(3)}, ${dir})`;
  };

  return (
    <div className="space-y-6">
      <div className="rounded-md border border-primary/20 bg-primary/5 p-3 flex items-center gap-2 text-sm">
        <FileSpreadsheet className="h-4 w-4 text-primary shrink-0" />
        <span>Using: <span className="font-medium">{datasetContext.file.name}</span></span>
        <Badge variant="outline" className="ml-auto text-xs">{datasetContext.availableTraitColumns.length} variables · {datasetContext.mode} mode</Badge>
      </div>

      <TooltipProvider delayDuration={150}>
      <Tabs defaultValue="correlation" className="w-full">
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <Tooltip>
            <TooltipTrigger asChild>
              <TabsTrigger value="correlation" className="gap-2">
                <GitCompareArrows className="h-4 w-4" />
                Correlation
              </TabsTrigger>
            </TooltipTrigger>
            <TooltipContent>Measure how strongly two variables move together.</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <TabsTrigger value="regression" className="gap-2">
                <LineChart className="h-4 w-4" />
                Regression Analysis
              </TabsTrigger>
            </TooltipTrigger>
            <TooltipContent>Used to explore relationships between variables.</TooltipContent>
          </Tooltip>
        </TabsList>

        {/* ============== CORRELATION ============== */}
        <TabsContent value="correlation" className="space-y-6 mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <GitCompareArrows className="h-5 w-5 text-primary" />
                Correlation Analysis
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Select at least two variables to compute pairwise correlations.
              </p>
              <div className="space-y-2">
                <p className="text-sm font-medium">Variables</p>
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
              <div className="rounded-md border border-border bg-muted/30 p-3 text-xs text-muted-foreground space-y-1">
                <p className="font-medium text-foreground">Results will include:</p>
                <ul className="list-disc pl-5 space-y-0.5">
                  <li>Correlation matrix</li>
                  <li>Significance levels</li>
                  <li>Heatmap visualization</li>
                  <li>Interpretation of relationships</li>
                </ul>
              </div>
            </CardContent>
          </Card>

          {result && (
            <div className="space-y-6" data-testid="correlation-results-view">
              {/* Deployment marker */}
              <div className="flex items-center justify-between gap-3 rounded-md border border-emerald-300 bg-emerald-50 px-3 py-2 text-xs font-medium text-emerald-800 dark:bg-emerald-900/20 dark:border-emerald-700 dark:text-emerald-200">
                <span className="flex items-center gap-1.5">
                  <Sparkles className="h-3.5 w-3.5" />
                  3-mode correlation UI active
                </span>
                <Button onClick={handleDownload} size="sm" variant="outline" className="gap-2 h-7">
                  <Download className="h-3.5 w-3.5" />
                  Download Report
                </Button>
              </div>

              {/* Debug: raw response inspector — collapsible */}
              <details className="rounded-md border border-border bg-muted/30 p-3 text-xs">
                <summary className="cursor-pointer font-medium text-muted-foreground">
                  Debug · Raw backend response (click to expand)
                </summary>
                <pre className="mt-2 overflow-x-auto text-[11px] leading-snug">
{JSON.stringify(result, null, 2)}
                </pre>
              </details>

              {/* 3-mode selector */}
              <Card>
                <CardHeader className="pb-3">
                  <CardTitle className="text-base flex items-center gap-2">
                    <GitCompareArrows className="h-4 w-4 text-primary" />
                    Correlation Mode
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <RadioGroup
                    value={selectedMode}
                    onValueChange={(v) => setSelectedMode(v as CorrelationModeKey)}
                    className="grid grid-cols-1 md:grid-cols-3 gap-3"
                  >
                    {MODE_ORDER.map((key) => {
                      const meta = MODE_META[key];
                      const block = modeBlocks?.[key];
                      const available = !!block?.available;
                      const isActive = selectedMode === key;
                      return (
                        <label
                          key={key}
                          htmlFor={`mode-${key}`}
                          className={[
                            "flex items-start gap-2 rounded-md border p-3 transition-colors",
                            available ? "cursor-pointer hover:bg-muted/40" : "opacity-60 cursor-not-allowed",
                            isActive && available ? "border-primary bg-primary/5" : "border-border",
                          ].join(" ")}
                        >
                          <RadioGroupItem id={`mode-${key}`} value={key} disabled={!available} className="mt-0.5" />
                          <div className="space-y-1">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium">{meta.label}</span>
                              {!available && <Badge variant="outline" className="text-[10px]">Unavailable</Badge>}
                            </div>
                            <p className="text-xs text-muted-foreground leading-snug">{meta.description}</p>
                          </div>
                        </label>
                      );
                    })}
                  </RadioGroup>

                  {genotypicFallback && (
                    <div className="mt-3 rounded-md border border-amber-300 bg-amber-50 p-3 text-xs text-amber-800 dark:bg-amber-900/20 dark:border-amber-700 dark:text-amber-200 flex gap-2">
                      <Info className="h-3.5 w-3.5 mt-0.5 shrink-0" />
                      <span>
                        Genotypic correlation (variance-component based) is unavailable for this dataset.
                        Use <span className="font-medium">Between-genotype association</span> as an approximate substitute.
                      </span>
                    </div>
                  )}
                </CardContent>
              </Card>

              <AcademicResultsPanel
                moduleLabel={`Correlation — ${MODE_META[selectedMode].label}`}
                insightSummary={deriveInsight()}
                insightBasis={activeBlock?.n_observations != null ? `Based on ${activeBlock.n_observations} observations` : undefined}
                interpretation={result.interpretation}
                statisticalNotes={[
                  ...(activeBlock?.note ? [{ text: activeBlock.note }] : []),
                  ...(result.statistical_note ? [{ text: result.statistical_note }] : []),
                  { text: "Correlations were computed using available observations; significance based on sample size." },
                  { text: "P-values are unadjusted for multiple comparisons." },
                ]}
              />

              {result.warnings.length > 0 && (
                <div className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm dark:bg-amber-900/20 dark:border-amber-700">
                  <div className="flex items-center gap-2 mb-1 font-medium text-amber-800 dark:text-amber-200"><AlertTriangle className="h-4 w-4" /> Warnings</div>
                  <ul className="list-disc pl-5 space-y-0.5 text-amber-800 dark:text-amber-200">{result.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
                </div>
              )}

              {/* Scalar pairwise shape (new backend) */}
              {isScalarShape && activeBlock ? (
                <>
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">
                        {MODE_META[selectedMode].label}
                        {result.trait_names.length === 2 && (
                          <span className="ml-2 text-sm text-muted-foreground font-normal">
                            · {result.trait_names[0]} × {result.trait_names[1]}
                          </span>
                        )}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ScalarModeTable block={activeBlock} mode={selectedMode} />
                    </CardContent>
                  </Card>

                  {/* Critical r comparison table (all 3 modes) */}
                  {modeBlocks && (
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-base">Critical r Value Comparison (α = 0.05)</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="overflow-x-auto">
                          <table className="border-collapse text-sm w-full">
                            <thead>
                              <tr>
                                <th className="border border-border px-3 py-2 bg-muted text-left font-medium">Mode</th>
                                <th className="border border-border px-3 py-2 bg-muted text-right font-medium">n</th>
                                <th className="border border-border px-3 py-2 bg-muted text-right font-medium">Critical r</th>
                                <th className="border border-border px-3 py-2 bg-muted text-right font-medium">Your r / rg</th>
                                <th className="border border-border px-3 py-2 bg-muted text-center font-medium">Result</th>
                              </tr>
                            </thead>
                            <tbody>
                              {MODE_ORDER.map((k) => {
                                const b = modeBlocks[k];
                                if (!b?.available) return null;
                                const isGeno = k === "genotypic";
                                const obs = isGeno
                                  ? (typeof b.df === "number" ? b.df + 2 : null)
                                  : k === "between_genotype"
                                    ? (b.n_genotypes ?? (typeof b.df === "number" ? b.df + 2 : null))
                                    : (typeof b.df === "number" ? b.df + 2 : b.n_observations ?? null);
                                const rVal = isGeno ? b.rg : b.r;
                                const exceeds =
                                  typeof rVal === "number" && typeof b.critical_r === "number"
                                    ? Math.abs(rVal) > b.critical_r
                                    : null;
                                return (
                                  <tr key={k} className="even:bg-muted/30">
                                    <td className="border border-border px-3 py-2 font-medium">{MODE_META[k].label}</td>
                                    <td className="border border-border px-3 py-2 text-right font-mono">{obs ?? "—"}</td>
                                    <td className="border border-border px-3 py-2 text-right font-mono">
                                      {typeof b.critical_r === "number" ? b.critical_r.toFixed(4) : "—"}
                                    </td>
                                    <td className="border border-border px-3 py-2 text-right font-mono">
                                      {typeof rVal === "number" ? rVal.toFixed(4) : "—"}
                                    </td>
                                    <td className="border border-border px-3 py-2 text-center">
                                      {exceeds == null ? (
                                        <span className="text-muted-foreground">—</span>
                                      ) : exceeds ? (
                                        <Badge className="bg-emerald-100 text-emerald-800 hover:bg-emerald-100 dark:bg-emerald-900/40 dark:text-emerald-200">EXCEEDS</Badge>
                                      ) : (
                                        <Badge variant="outline" className="text-muted-foreground">BELOW</Badge>
                                      )}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </>
              ) : hasMatrix && activeBlock ? (
                <>
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base">Correlation Matrix · {MODE_META[selectedMode].label}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-x-auto">
                        <table className="border-collapse text-sm w-full">
                          <thead>
                            <tr>
                              <th className="border border-border px-3 py-2 bg-muted font-medium text-left">Variable Pair</th>
                              <th className="border border-border px-3 py-2 bg-muted font-medium text-right">r-value</th>
                              <th className="border border-border px-3 py-2 bg-muted font-medium text-right">p-value</th>
                              <th className="border border-border px-3 py-2 bg-muted font-medium text-center">Sig.</th>
                            </tr>
                          </thead>
                          <tbody>
                            {result.trait_names.map((t1, i) =>
                              result.trait_names.slice(i + 1).map((t2, jOff) => {
                                const j = i + 1 + jOff;
                                const r = activeBlock.r_matrix?.[i]?.[j] ?? 0;
                                const p = activeBlock.p_matrix?.[i]?.[j] ?? null;
                                const stars = p != null ? (p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "ns") : "—";
                                return (
                                  <tr key={`${i}-${j}`} className="even:bg-muted/30">
                                    <td className="border border-border px-3 py-2 font-medium">{t1} vs {t2}</td>
                                    <td className={`border border-border px-3 py-2 text-right font-mono ${Math.abs(r) >= 0.7 ? "font-bold text-emerald-700 dark:text-emerald-400" : ""}`}>
                                      {typeof r === "number" ? r.toFixed(3) : "—"}
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

                  <Card>
                    <CardHeader>
                      <CardTitle className="text-base flex items-center gap-2">
                        <Grid3X3 className="h-4 w-4 text-primary" />
                        Correlation Heatmap · {MODE_META[selectedMode].label}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <CorrelationHeatmap
                        traits={result.trait_names}
                        rMatrix={activeBlock.r_matrix ?? []}
                        pMatrix={activeBlock.p_matrix ?? []}
                      />
                      {activeBlock.n_observations != null && (
                        <p className="text-xs text-muted-foreground">Based on {activeBlock.n_observations} observations</p>
                      )}
                    </CardContent>
                  </Card>
                </>
              ) : (
                <Card className="border-dashed">
                  <CardContent className="py-10 text-center space-y-2">
                    <Info className="h-8 w-8 mx-auto text-muted-foreground/50" />
                    <p className="text-sm text-muted-foreground">
                      No data available for <span className="font-medium">{MODE_META[selectedMode].label}</span>.
                    </p>
                    <p className="text-xs text-muted-foreground">Try selecting another mode above.</p>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </TabsContent>

        {/* ============== REGRESSION ANALYSIS ============== */}
        <TabsContent value="regression" className="mt-6">
          <RegressionAnalysisTab datasetContext={datasetContext} />
        </TabsContent>
      </Tabs>
      </TooltipProvider>
    </div>
  );
}

function ScalarModeTable({ block, mode }: { block: CorrelationModeBlock; mode: CorrelationModeKey }) {
  const isGeno = mode === "genotypic";
  const rValue = isGeno ? block.rg : block.r;
  const rLabel = isGeno ? "rg" : "r";
  const p = block.p_value;
  const df = block.df;
  const n = isGeno
    ? (typeof df === "number" ? df + 2 : null)
    : mode === "between_genotype"
      ? (block.n_genotypes ?? (typeof df === "number" ? df + 2 : null))
      : (typeof df === "number" ? df + 2 : block.n_observations ?? null);
  const sig = typeof p === "number" ? p < 0.05 : null;
  const sigLabel = sig == null ? "—" : sig ? (isGeno ? "Significant (approx.)" : "Significant") : "Not significant";
  const fmt = (v: number | undefined | null, d = 4) =>
    typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—";
  const fmtP = (v: number | undefined | null) =>
    typeof v === "number" && Number.isFinite(v) ? (v < 0.001 ? "<0.001" : v.toFixed(4)) : "—";

  const rows: Array<[string, string]> = [
    ["n", n != null ? String(n) : "—"],
    ["df", typeof df === "number" ? String(df) : "—"],
    [rLabel, fmt(rValue)],
    ["p-value", fmtP(p) + (isGeno ? " (approximate)" : "")],
    ["Critical r (α = 0.05)", fmt(block.critical_r)],
  ];
  if (typeof block.ci_lower === "number" && typeof block.ci_upper === "number") {
    rows.push(["95% CI", `${fmt(block.ci_lower)} to ${fmt(block.ci_upper)}`]);
  }
  rows.push(["Significance", sigLabel]);

  return (
    <div className="space-y-3">
      <div className="overflow-x-auto">
        <table className="border-collapse text-sm w-full">
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={k} className="even:bg-muted/30">
                <td className="border border-border px-3 py-2 font-medium w-1/3">{k}</td>
                <td className="border border-border px-3 py-2 font-mono">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {isGeno && (
        <p className="text-xs text-muted-foreground italic">
          Note: Approximate inference using Fisher z-transformation.
        </p>
      )}
    </div>
  );
}
