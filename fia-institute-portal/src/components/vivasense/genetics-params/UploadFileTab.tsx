import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Upload, Loader2, Play, AlertTriangle, CheckCircle2, FileSpreadsheet, Download } from "lucide-react";
import { toast as sonnerToast } from "sonner";
import { useToast } from "@/hooks/use-toast";
import { uploadPreview, analyzeUpload, fileToBase64 } from "@/lib/geneticsUploadApi";
import type { DatasetContext, UploadPreviewResponse, AnalyzeUploadResponse } from "@/types/geneticsUpload";
import { extractRows, fmtNum, formatP } from "./GeneticsResultsDashboard";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown } from "lucide-react";

interface Props {
  onDatasetReady: (ctx: DatasetContext) => void;
  datasetContext: DatasetContext | null;
  activeModule?: "anova" | "genetics" | "correlation" | "heatmap";
}

export function UploadFileTab({ onDatasetReady, datasetContext, activeModule = "genetics" }: Props) {
  const { toast } = useToast();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<UploadPreviewResponse | null>(null);
  const [isPreviewing, setIsPreviewing] = useState(false);

  // Mapping state
  const [genotypeCol, setGenotypeCol] = useState("");
  const [repCol, setRepCol] = useState("");
  const [envCol, setEnvCol] = useState("");
  const [mode, setMode] = useState<"single" | "multi">("single");
  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [selectionIntensity, setSelectionIntensity] = useState(1.4);

  // Analysis state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalyzeUploadResponse | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setPreview(null);
      setResults(null);
    }
  };

  const handlePreview = async () => {
    if (!file) return;
    setIsPreviewing(true);
    setPreview(null);
    try {
      const res = await uploadPreview(file);
      setPreview(res);
      // Auto-fill mapping
      setGenotypeCol(res.detected_columns.genotype?.column ?? "");
      setRepCol(res.detected_columns.rep?.column ?? "");
      setEnvCol(res.detected_columns.environment?.column ?? "");
      setMode(res.mode_suggestion);
      setSelectedTraits(res.detected_columns.traits ?? []);
    } catch (err: any) {
      toast({ title: "Preview failed", description: err.message, variant: "destructive" });
    } finally {
      setIsPreviewing(false);
    }
  };

  const toggleTrait = (t: string) => {
    setSelectedTraits((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]
    );
  };

  const handleAnalyze = async () => {
    if (!file || !preview || selectedTraits.length === 0) return;
    setIsAnalyzing(true);
    setResults(null);
    try {
      const base64 = await fileToBase64(file);
      const ext = file.name.split(".").pop()?.toLowerCase() ?? "csv";
      const fileType = ext as "csv" | "xlsx" | "xls";

      const ctx: DatasetContext = {
        file,
        base64Content: base64,
        fileType,
        genotypeColumn: genotypeCol,
        repColumn: repCol,
        environmentColumn: envCol || null,
        availableTraitColumns: preview.detected_columns.traits,
        mode,
      };
      onDatasetReady(ctx);

      const res = await analyzeUpload({
        base64_content: base64,
        file_type: fileType,
        genotype_column: genotypeCol,
        rep_column: repCol,
        environment_column: envCol || null,
        trait_columns: selectedTraits,
        mode,
        random_environment: false,
        selection_intensity: selectionIntensity,
        module: "genetic_parameters",
      });
      setResults(res);
      toast({ title: "Analysis complete", description: `${res.summary_table.length} trait(s) analyzed.` });
    } catch (err: any) {
      toast({ title: "Analysis failed", description: err.message, variant: "destructive" });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* File upload */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Upload className="h-5 w-5 text-primary" />
            Upload Dataset
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <Input
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={handleFileChange}
              className="flex-1"
            />
            <Button onClick={handlePreview} disabled={!file || isPreviewing} className="gap-2">
              {isPreviewing ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileSpreadsheet className="h-4 w-4" />}
              Preview
            </Button>
          </div>
          {file && (
            <p className="text-xs text-muted-foreground">
              Selected: <span className="font-medium">{file.name}</span>
            </p>
          )}
        </CardContent>
      </Card>

      {/* Preview + mapping */}
      {preview && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Column Mapping & Trait Selection</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            {/* Dataset info */}
            <div className="flex flex-wrap gap-2 text-sm">
              <Badge variant="secondary">{preview.n_rows} rows</Badge>
              <Badge variant="secondary">{preview.n_columns} columns</Badge>
              <Badge variant="outline">Suggested: {preview.mode_suggestion}</Badge>
            </div>

            {preview.warnings.length > 0 && (
              <div className="rounded-md border border-yellow-300 bg-yellow-50 p-3 text-sm text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-200 dark:border-yellow-700">
                <div className="flex items-center gap-2 mb-1 font-medium">
                  <AlertTriangle className="h-4 w-4" /> Warnings
                </div>
                <ul className="list-disc pl-5 space-y-0.5">
                  {preview.warnings.map((w, i) => <li key={i}>{w}</li>)}
                </ul>
              </div>
            )}

            {/* Column selectors */}
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="space-y-1.5">
                <Label className="text-sm font-medium">Genotype Column</Label>
                <Select value={genotypeCol} onValueChange={setGenotypeCol}>
                  <SelectTrigger><SelectValue placeholder="Select…" /></SelectTrigger>
                  <SelectContent>
                    {preview.column_names.map((c) => (
                      <SelectItem key={c} value={c}>{c}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {preview.detected_columns.genotype && (
                  <ConfidenceBadge confidence={preview.detected_columns.genotype.confidence} />
                )}
              </div>
              <div className="space-y-1.5">
                <Label className="text-sm font-medium">Rep Column</Label>
                <Select value={repCol} onValueChange={setRepCol}>
                  <SelectTrigger><SelectValue placeholder="Select…" /></SelectTrigger>
                  <SelectContent>
                    {preview.column_names.map((c) => (
                      <SelectItem key={c} value={c}>{c}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {preview.detected_columns.rep && (
                  <ConfidenceBadge confidence={preview.detected_columns.rep.confidence} />
                )}
              </div>
              <div className="space-y-1.5">
                <Label className="text-sm font-medium">Environment Column</Label>
                <Select value={envCol || "__none__"} onValueChange={(v) => setEnvCol(v === "__none__" ? "" : v)}>
                  <SelectTrigger><SelectValue placeholder="None" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">None</SelectItem>
                    {preview.column_names.map((c) => (
                      <SelectItem key={c} value={c}>{c}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-sm font-medium">Mode</Label>
                <Select value={mode} onValueChange={(v) => setMode(v as "single" | "multi")}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="single">Single Environment</SelectItem>
                    <SelectItem value="multi">Multi-Environment</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Selection intensity — only for genetics module */}
            {activeModule !== "anova" && (
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
            )}

            {/* Trait selection */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Select Traits for Analysis</Label>
              <div className="flex flex-wrap gap-3">
                {preview.detected_columns.traits.map((t) => (
                  <label key={t} className="flex items-center gap-2 text-sm cursor-pointer">
                    <Checkbox
                      checked={selectedTraits.includes(t)}
                      onCheckedChange={() => toggleTrait(t)}
                    />
                    {t}
                  </label>
                ))}
              </div>
              {preview.detected_columns.traits.length === 0 && (
                <p className="text-sm text-muted-foreground">No numeric trait columns detected.</p>
              )}
            </div>

            {/* Analyze */}
            <Button
              onClick={handleAnalyze}
              disabled={isAnalyzing || selectedTraits.length === 0 || !genotypeCol || !repCol}
              className="gap-2"
            >
              {isAnalyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {activeModule === "anova" ? "Run ANOVA" : "Run Genetic Parameter Analysis"}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Data preview table */}
      {preview && preview.data_preview.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Data Preview (first rows)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr>
                    {preview.column_names.map((c) => (
                      <th key={c} className="border border-border px-2 py-1.5 bg-muted font-medium text-left">{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.data_preview.slice(0, 8).map((row, i) => (
                    <tr key={i} className="even:bg-muted/30">
                      {preview.column_names.map((c) => (
                        <td key={c} className="border border-border px-2 py-1">{String(row[c] ?? "")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {results && <UploadAnalysisResults results={results} />}
    </div>
  );
}

function ConfidenceBadge({ confidence }: { confidence: string }) {
  const variant = confidence === "high" ? "default" : confidence === "medium" ? "secondary" : "outline";
  return <Badge variant={variant} className="text-[10px]">{confidence} confidence</Badge>;
}

// ── Results display ─────────────────────────────────────────────────────────

function UploadAnalysisResults({ results }: { results: AnalyzeUploadResponse }) {
  const [isDownloading, setIsDownloading] = useState(false);

  const handleDownloadWord = async () => {
    setIsDownloading(true);
    try {
      const payload = {
        analysis_type: "multi_trait_genetic_parameters",
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

      const res = await fetch("https://vivasense-genetics-docker.onrender.com/genetics/download-results", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errText = await res.text();
        let msg = errText;
        try { const parsed = JSON.parse(errText); msg = parsed.detail || parsed.error || errText; } catch {}
        throw new Error(msg);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const date = new Date().toISOString().slice(0, 10);
      a.download = `VivaSense_Genetics_MultiTrait_${date}.docx`;
      a.click();
      URL.revokeObjectURL(url);
      sonnerToast.success("Downloaded successfully");
    } catch (err: any) {
      console.error("[DOWNLOAD ERROR]", err);
      sonnerToast.error("Download failed. Try again.");
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Dataset summary */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-emerald-600" />
            Analysis Results
          </CardTitle>
          <Button
            onClick={handleDownloadWord}
            disabled={isDownloading}
            size="sm"
            className="gap-2 bg-emerald-600 hover:bg-emerald-700 text-white"
          >
            {isDownloading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            {isDownloading ? "Downloading..." : "📥 Download as Word"}
          </Button>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2 text-sm">
            <Badge variant="secondary">{results.dataset_summary.n_genotypes} genotypes</Badge>
            <Badge variant="secondary">{results.dataset_summary.n_reps} reps</Badge>
            {results.dataset_summary.n_environments && (
              <Badge variant="secondary">{results.dataset_summary.n_environments} environments</Badge>
            )}
            <Badge variant="outline">{results.dataset_summary.mode} mode</Badge>
          </div>

          {/* Summary table */}
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="bg-muted">
                  {["Trait", "Grand Mean", "H²", "GCV (%)", "PCV (%)", "GAM (%)", "Class", "Status"].map((h) => (
                    <th key={h} className="border border-border px-3 py-2 text-left font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {results.summary_table.map((row) => (
                  <tr key={row.trait} className="even:bg-muted/30">
                    <td className="border border-border px-3 py-1.5 font-medium">{row.trait}</td>
                    <td className="border border-border px-3 py-1.5">{row.grand_mean?.toFixed(2) ?? "—"}</td>
                    <td className="border border-border px-3 py-1.5">{row.h2?.toFixed(3) ?? "—"}</td>
                    <td className="border border-border px-3 py-1.5">{row.gcv?.toFixed(2) ?? "—"}</td>
                    <td className="border border-border px-3 py-1.5">{row.pcv?.toFixed(2) ?? "—"}</td>
                    <td className="border border-border px-3 py-1.5">{row.gam_percent?.toFixed(2) ?? "—"}</td>
                    <td className="border border-border px-3 py-1.5">
                      <HeritabilityBadge cls={row.heritability_class} />
                    </td>
                    <td className="border border-border px-3 py-1.5">
                      {row.status === "success"
                        ? <Badge variant="default" className="text-[10px] bg-emerald-600">OK</Badge>
                        : <Badge variant="destructive" className="text-[10px]">Failed</Badge>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Failed traits */}
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

      {/* Per-trait detail cards */}
      {Object.entries(results.trait_results).map(([trait, tr]) => {
        if (tr.status !== "success" || !tr.analysis_result) return null;
        const r = tr.analysis_result.result;
        const vc = r.variance_components;
        return (
          <Card key={trait}>
            <CardHeader>
              <CardTitle className="text-base">{trait} — Detailed Results</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Warnings */}
              {tr.data_warnings.length > 0 && (
                <div className="rounded-md border border-yellow-300 bg-yellow-50 p-3 text-sm text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-200 dark:border-yellow-700">
                  <ul className="list-disc pl-4">{tr.data_warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
                </div>
              )}

              <div className="grid gap-4 sm:grid-cols-2">
                {/* Variance components */}
                <div className="space-y-1 text-sm">
                  <p className="font-medium text-foreground mb-1">Variance Components</p>
                  <Row label="σ²g (Genotypic)" value={vc.sigma2_genotype} />
                  <Row label="σ²e (Error)" value={vc.sigma2_error} />
                  {vc.sigma2_ge != null && <Row label="σ²ge (G×E)" value={vc.sigma2_ge} />}
                  <Row label="σ²p (Phenotypic)" value={vc.sigma2_phenotypic} />
                </div>

                {/* Genetic parameters */}
                <div className="space-y-1 text-sm">
                  <p className="font-medium text-foreground mb-1">Genetic Parameters</p>
                  <Row label="H² (Broad-sense)" value={r.heritability.h2_broad_sense} decimals={3} />
                  <Row label="GCV (%)" value={r.genetic_parameters.GCV} />
                  <Row label="PCV (%)" value={r.genetic_parameters.PCV} />
                  <Row label="GAM (%)" value={r.genetic_parameters.GAM_percent} />
                  <p className="text-xs text-muted-foreground pt-1">
                    Basis: {r.heritability.interpretation_basis} | k = {r.genetic_parameters.selection_intensity}
                  </p>
                </div>
              </div>

              {/* ANOVA Table */}
              {r.anova_table && (() => {
                const anovaRows = extractRows(r.anova_table);
                if (anovaRows.length === 0) return null;
                return (
                  <Collapsible>
                    <CollapsibleTrigger className="w-full flex items-center justify-between rounded-md bg-muted/50 px-3 py-2 text-sm font-medium text-foreground hover:bg-muted/80">
                      ANOVA Table
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="overflow-x-auto mt-2">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Source</TableHead>
                              <TableHead className="text-right">DF</TableHead>
                              <TableHead className="text-right">SS</TableHead>
                              <TableHead className="text-right">MS</TableHead>
                              <TableHead className="text-right">F</TableHead>
                              <TableHead className="text-right">p-value</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {anovaRows.map((row, i) => {
                              const pVal = row.p_value ?? row.pvalue;
                              const pNum = pVal != null ? Number(pVal) : NaN;
                              const isSig = !isNaN(pNum) && pNum < 0.05;
                              return (
                                <TableRow key={i}>
                                  <TableCell className="font-medium">{String(row.source ?? "")}</TableCell>
                                  <TableCell className="text-right font-mono">{String(row.df ?? "—")}</TableCell>
                                  <TableCell className="text-right font-mono">{fmtNum(row.ss)}</TableCell>
                                  <TableCell className="text-right font-mono">{fmtNum(row.ms)}</TableCell>
                                  <TableCell className="text-right font-mono">{fmtNum(row.f_value)}</TableCell>
                                  <TableCell className={`text-right font-mono ${isSig ? "text-green-600 font-semibold" : ""}`}>
                                    {pVal != null ? formatP(pVal) : "—"}
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1 italic">* p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001</p>
                    </CollapsibleContent>
                  </Collapsible>
                );
              })()}

              {/* Mean Separation */}
              {r.mean_separation && (() => {
                const msRows = extractRows(r.mean_separation);
                if (msRows.length === 0) return null;
                return (
                  <Collapsible defaultOpen>
                    <CollapsibleTrigger className="w-full flex items-center justify-between rounded-md bg-muted/50 px-3 py-2 text-sm font-medium text-foreground hover:bg-muted/80">
                      Mean Separation (Tukey HSD)
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <div className="overflow-x-auto mt-2">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Genotype</TableHead>
                              <TableHead className="text-right">Mean ± SE</TableHead>
                              <TableHead className="text-center">Group</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {msRows.map((row, i) => {
                              const grp = String(row.group ?? "—").trim().charAt(0).toLowerCase();
                              return (
                                <TableRow key={i} className={groupRowColor(grp)}>
                                  <TableCell className="font-medium">{String(row.genotype ?? "")}</TableCell>
                                  <TableCell className="text-right font-mono">
                                    {fmtNum(row.mean)} ± {fmtNum(row.se)}
                                  </TableCell>
                                  <TableCell className="text-center font-bold text-lg">
                                    {String(row.group ?? "—")}
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1 italic">
                        Means with the same letter are not significantly different (Tukey HSD, α = 0.05)
                      </p>
                    </CollapsibleContent>
                  </Collapsible>
                );
              })()}

              {/* Interpretation */}
              {tr.analysis_result.interpretation && (
                <div className="rounded-md bg-muted/50 p-4 text-sm leading-relaxed">
                  <p className="font-medium text-foreground mb-1">Interpretation</p>
                  <p className="text-muted-foreground">{tr.analysis_result.interpretation}</p>
                </div>
              )}
            </CardContent>
          </Card>
        );
      })}

      {/* Correlation Matrix */}
      <CorrelationMatrixSection results={results} />
    </div>
  );
}

function Row({ label, value, decimals = 2 }: { label: string; value: number | null | undefined; decimals?: number }) {
  return (
    <div className="flex justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono">{
        typeof value === "number" && !isNaN(value)
          ? value.toFixed(decimals)
          : "N/A"
      }</span>
    </div>
  );
}

function HeritabilityBadge({ cls }: { cls: string }) {
  const colors: Record<string, string> = {
    high: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300",
    moderate: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
    low: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  };
  return (
    <span className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-medium ${colors[cls] ?? colors.moderate}`}>
      {cls}
    </span>
  );
}

/** Color-code mean separation rows by Tukey group letter */
function groupRowColor(group: string): string {
  const map: Record<string, string> = {
    a: "bg-emerald-50 dark:bg-emerald-900/20",
    b: "bg-orange-50 dark:bg-orange-900/20",
    c: "bg-red-50 dark:bg-red-900/20",
    d: "bg-gray-100 dark:bg-gray-800/30",
    e: "bg-gray-200 dark:bg-gray-700/30",
  };
  return map[group] ?? "";
}

/** Correlation matrix display for multi-trait analysis */
function CorrelationMatrixSection({ results }: { results: AnalyzeUploadResponse }) {
  // Try to find correlation_matrix in the response (backend may include it)
  const corrData = (results as any).correlation_matrix;
  if (!corrData) return null;

  const traits: string[] = corrData.trait_names ?? corrData.traits ?? [];
  const rMatrix: number[][] = corrData.r_matrix ?? corrData.correlations ?? [];
  const pMatrix: (number | null)[][] = corrData.p_matrix ?? [];

  if (traits.length === 0 || rMatrix.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg flex items-center gap-2">
          <FileSpreadsheet className="h-5 w-5 text-primary" />
          Correlation Matrix
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="overflow-x-auto">
          <table className="border-collapse text-sm w-full">
            <thead>
              <tr>
                <th className="border border-border px-3 py-2 bg-muted font-medium text-left">Trait Pair</th>
                <th className="border border-border px-3 py-2 bg-muted font-medium text-right">r-value</th>
                <th className="border border-border px-3 py-2 bg-muted font-medium text-right">p-value</th>
                <th className="border border-border px-3 py-2 bg-muted font-medium text-center">Significance</th>
              </tr>
            </thead>
            <tbody>
              {traits.map((t1, i) =>
                traits.slice(i + 1).map((t2, jOff) => {
                  const j = i + 1 + jOff;
                  const r = rMatrix[i]?.[j] ?? 0;
                  const p = pMatrix[i]?.[j] ?? null;
                  const stars = p != null ? getSigStars(p) : "—";
                  const strength = getCorrelationStrength(r);
                  return (
                    <tr key={`${i}-${j}`} className="even:bg-muted/30">
                      <td className="border border-border px-3 py-2 font-medium">
                        {t1} vs {t2}
                      </td>
                      <td className={`border border-border px-3 py-2 text-right font-mono ${
                        Math.abs(r) >= 0.7 ? "font-bold text-emerald-700 dark:text-emerald-400" : ""
                      }`}>
                        {r.toFixed(3)}
                      </td>
                      <td className="border border-border px-3 py-2 text-right font-mono">
                        {p != null ? (p < 0.001 ? "<0.001" : p.toFixed(3)) : "—"}
                      </td>
                      <td className="border border-border px-3 py-2 text-center font-mono">
                        {stars}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
        {corrData.interpretation && (
          <p className="text-sm text-muted-foreground leading-relaxed">{corrData.interpretation}</p>
        )}
        <p className="text-xs text-muted-foreground italic">
          Significance: *** p &lt; 0.001, ** p &lt; 0.01, * p &lt; 0.05, ns = not significant.
          Strong positive correlations indicate traits that improve together under selection.
        </p>
      </CardContent>
    </Card>
  );
}

function getSigStars(p: number): string {
  if (p < 0.001) return "***";
  if (p < 0.01) return "**";
  if (p < 0.05) return "*";
  return "ns";
}

function getCorrelationStrength(r: number): string {
  const abs = Math.abs(r);
  if (abs >= 0.7) return "strong";
  if (abs >= 0.4) return "moderate";
  if (abs >= 0.1) return "weak";
  return "negligible";
}
