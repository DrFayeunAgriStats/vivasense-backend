import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Loader2, Play, BarChart3, Download, CheckCircle2, AlertTriangle,
  FileSpreadsheet, Info,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { toast as sonnerToast } from "sonner";
import { analyzeUpload, downloadReport } from "@/lib/geneticsUploadApi";
import { AcademicResultsPanel } from "./AcademicResultsPanel";
import type {
  DatasetContext, AnalyzeUploadResponse, AnovaDesignType,
} from "@/types/geneticsUpload";

const MODULE = "anova" as const;

interface Props {
  datasetContext: DatasetContext | null;
}

interface DesignMeta {
  id: AnovaDesignType;
  label: string;
  hint: string;
}

const DESIGNS: DesignMeta[] = [
  { id: "crd",              label: "CRD",              hint: "Completely Randomized Design — single treatment factor, no blocking." },
  { id: "rcbd",             label: "RCBD",             hint: "Randomized Complete Block Design — one treatment factor with replication blocks." },
  { id: "factorial",        label: "Factorial",        hint: "Two crossed treatment factors (A × B). Optionally add replication blocks." },
  { id: "split_plot_rcbd",  label: "Split-Plot RCBD",  hint: "Restricted randomisation: main-plot factor inside replication blocks, subplot factor inside each main plot." },
];

export function AnovaModulePanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const [design, setDesign] = useState<AnovaDesignType>("rcbd");

  // Mappings (per-design)
  const [treatmentCol, setTreatmentCol] = useState<string>("");
  const [repColumn, setRepColumn] = useState<string>("");
  const [factorA, setFactorA] = useState<string>("");
  const [factorB, setFactorB] = useState<string>("");
  const [factorC, setFactorC] = useState<string>("");
  const [includeRepFactorial, setIncludeRepFactorial] = useState(false);
  const [mainPlot, setMainPlot] = useState<string>("");
  const [subPlot, setSubPlot] = useState<string>("");

  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalyzeUploadResponse | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);

  // All available columns for selectors — computed safely even when no dataset (returns []).
  const allColumns = useMemo(() => {
    if (!datasetContext) return [];
    const traits = new Set(datasetContext.availableTraitColumns);
    // Broad discovery: use all columns from the dataset pool
    const all = (datasetContext as any).columns ?? (datasetContext as any).availableColumns ?? [];
    if (all.length > 0) {
      return all.filter((c: string) => !traits.has(c));
    }
    // Fallback to detected structural columns
    const candidates = [
      datasetContext.genotypeColumn,
      datasetContext.repColumn,
      datasetContext.environmentColumn,
    ].filter(Boolean) as string[];
    return Array.from(new Set(candidates));
  }, [datasetContext]);

  if (!datasetContext) {
    return (
      <Card className="border-dashed">
        <CardContent className="py-16 text-center space-y-3">
          <BarChart3 className="h-10 w-10 mx-auto text-muted-foreground/50" />
          <p className="text-muted-foreground font-medium">Upload a dataset first to run ANOVA analysis.</p>
        </CardContent>
      </Card>
    );
  }

  const factorAColumns = allColumns.filter(
    (col) => col !== treatmentCol && col !== repColumn
  );

  const factorBColumns = allColumns.filter(
    (col) => col !== treatmentCol && col !== repColumn && col !== factorA
  );

  const factorCColumns = allColumns.filter(
    (col) => col !== treatmentCol && col !== repColumn && col !== factorA && col !== factorB
  );

  const toggleTrait = (t: string) =>
    setSelectedTraits((prev) => prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]);

  // ── Validation ─────────────────────────────────────────────────────────
  const validation = (() => {
    if (selectedTraits.length === 0) return "Select at least one response variable.";
    if (design === "crd") {
      if (!treatmentCol) return "Select a Treatment column.";
    }
    if (design === "rcbd") {
      if (!repColumn) return "Select a Replication / Block column.";
      if (!treatmentCol) return "Select a Treatment column.";
      if (repColumn === treatmentCol) return "Replication and Treatment must be different columns.";
    }
    if (design === "factorial") {
      if (!factorA || !factorB) return "Select Factor A and Factor B columns.";
      if (factorA === factorB) return "Factor A and Factor B must be different columns.";
      if (factorC && factorC !== "None" && (factorC === factorA || factorC === factorB)) {
        return "Factor C must be different from Factors A and B.";
      }
      if (includeRepFactorial && !repColumn) return "Replication column required when blocking is enabled.";
    }
    if (design === "split_plot_rcbd") {
      if (!repColumn) return "Select a Replication / Block column.";
      if (!mainPlot) return "Select a Main-Plot factor column.";
      if (!subPlot) return "Select a Subplot factor column.";
      const cols = [repColumn, mainPlot, subPlot];
      if (new Set(cols).size !== cols.length) return "Replication, Main-plot and Subplot must be different columns.";
    }
    return null;
  })();

  const isSplitPlot = design === "split_plot_rcbd";
  const isFactorialFamily = design === "factorial";

  // ── Run analysis ──────────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (validation) return;
    setIsAnalyzing(true);
    setResults(null);
    try {
      const effectiveDesign: AnovaDesignType =
        design === "factorial" && includeRepFactorial ? "factorial_rcbd" : design;

      console.log("[MODULE]", MODULE, "[DESIGN]", effectiveDesign);

      const res = await analyzeUpload({
        base64_content: datasetContext.base64Content,
        file_type: datasetContext.fileType,
        // legacy fields kept for backend back-compat
        genotype_column: treatmentCol || datasetContext.genotypeColumn,
        rep_column: repColumn || datasetContext.repColumn,
        environment_column: datasetContext.environmentColumn,
        trait_columns: selectedTraits,
        mode: datasetContext.mode,
        random_environment: false,
        selection_intensity: 1.4,
        module: MODULE,
        // design-aware fields
        design_type: effectiveDesign,
        treatment_column: design === "crd" || design === "rcbd" ? treatmentCol : undefined,
        factor_a_column: isFactorialFamily ? factorA : undefined,
        factor_b_column: isFactorialFamily ? factorB : undefined,
        factor_c_column: isFactorialFamily && factorC && factorC !== "None" ? factorC : undefined,
        main_plot_column: isSplitPlot ? mainPlot : undefined,
        sub_plot_column: isSplitPlot ? subPlot : undefined,
      });
      setResults(res);
      toast({ title: "ANOVA complete", description: `${res.summary_table.length} response variable(s) analyzed.` });
    } catch (err: any) {
      toast({ title: "ANOVA failed", description: err.message, variant: "destructive" });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ── Download report ───────────────────────────────────────────────────
  const handleDownload = async () => {
    if (!results) return;
    setIsDownloading(true);
    try {
      const payload = {
        analysis_type: MODULE,
        design_type: design === "factorial" && includeRepFactorial ? "factorial_rcbd" : design,
        dataset_summary: results.dataset_summary,
        summary_table: results.summary_table,
        trait_results: Object.fromEntries(
          Object.entries(results.trait_results)
            .filter(([, tr]) => tr.status === "success" && tr.analysis_result)
            .map(([trait, tr]) => [trait, {
              anova_table: tr.analysis_result.result.anova_table,
              mean_separation: isSplitPlot ? undefined : tr.analysis_result.result.mean_separation,
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
      a.download = `VivaSense_ANOVA_${design}_${new Date().toISOString().slice(0, 10)}.docx`;
      a.click();
      URL.revokeObjectURL(url);
      sonnerToast.success("ANOVA report downloaded");
    } catch {
      sonnerToast.error("Download failed");
    } finally {
      setIsDownloading(false);
    }
  };

  // ── Field selector helper ─────────────────────────────────────────────
  const ColumnSelect = ({
    label, value, onChange, placeholder = "Select column…", options = allColumns,
  }: {
    label: string;
    value: string;
    onChange: (v: string) => void;
    placeholder?: string;
    options?: string[];
  }) => (
    <div className="space-y-1.5">
      <Label className="text-sm font-medium">{label}</Label>
      <Select value={value || undefined} onValueChange={onChange}>
        <SelectTrigger><SelectValue placeholder={placeholder} /></SelectTrigger>
        <SelectContent>
          {options.map((c) => (
            <SelectItem key={c} value={c}>{c}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Dataset banner */}
      <div className="rounded-md border border-primary/20 bg-primary/5 p-3 flex items-center gap-2 text-sm">
        <FileSpreadsheet className="h-4 w-4 text-primary shrink-0" />
        <span>Using: <span className="font-medium">{datasetContext.file.name}</span></span>
        <Badge variant="outline" className="ml-auto text-xs">
          {datasetContext.availableTraitColumns.length} response variable(s) · {datasetContext.mode} mode
        </Badge>
      </div>

      {/* Design selector */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-primary" />
            ANOVA Analysis
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Select your experimental design to ensure correct analysis and interpretation.
          </p>
        </CardHeader>
        <CardContent className="space-y-5">
          <Tabs value={design} onValueChange={(v) => setDesign(v as AnovaDesignType)}>
            <TabsList className="grid w-full grid-cols-2 sm:grid-cols-4">
              {DESIGNS.map((d) => (
                <TabsTrigger key={d.id} value={d.id} className="text-xs sm:text-sm">{d.label}</TabsTrigger>
              ))}
            </TabsList>
          </Tabs>

          <div className="rounded-md border bg-muted/30 p-3 flex gap-2 text-xs text-muted-foreground">
            <Info className="h-4 w-4 shrink-0 mt-0.5" />
            <span>{DESIGNS.find((d) => d.id === design)?.hint}</span>
          </div>

          {/* Field mapping per design */}
          <div className="grid gap-4 sm:grid-cols-2">
            {design === "crd" && (
              <ColumnSelect label="Treatment Column" value={treatmentCol} onChange={setTreatmentCol} />
            )}

            {design === "rcbd" && (
              <>
                <ColumnSelect label="Replication / Block Column" value={repColumn} onChange={setRepColumn} />
                <ColumnSelect label="Treatment Column" value={treatmentCol} onChange={setTreatmentCol} />
              </>
            )}

            {design === "factorial" && (
              <>
                <ColumnSelect label="Factor A Column" value={factorA} onChange={setFactorA} options={factorAColumns} />
                <ColumnSelect label="Factor B Column" value={factorB} onChange={setFactorB} options={factorBColumns} />
                <ColumnSelect
                  label="Factor C Column (optional)"
                  value={factorC}
                  onChange={setFactorC}
                  options={["None", ...factorCColumns]}
                  placeholder="None"
                />
                <div className="sm:col-span-2 flex items-center gap-2 text-sm">
                  <Checkbox
                    checked={includeRepFactorial}
                    onCheckedChange={(v) => setIncludeRepFactorial(Boolean(v))}
                    id="include-rep"
                  />
                  <Label htmlFor="include-rep" className="cursor-pointer">Include Replication (Block Design)</Label>
                </div>
                {includeRepFactorial && (
                  <ColumnSelect label="Replication / Block Column" value={repColumn} onChange={setRepColumn} />
                )}
              </>
            )}

            {design === "split_plot_rcbd" && (
              <>
                <ColumnSelect label="Replication / Block Column" value={repColumn} onChange={setRepColumn} />
                <ColumnSelect label="Main-Plot Factor Column" value={mainPlot} onChange={setMainPlot} />
                <ColumnSelect label="Subplot Factor Column" value={subPlot} onChange={setSubPlot} />
              </>
            )}
          </div>

          {isSplitPlot && (
            <div className="rounded-md border border-amber-300 bg-amber-50 dark:bg-amber-950/20 dark:border-amber-800 p-3 text-xs text-amber-900 dark:text-amber-200 space-y-1">
              <p className="font-semibold flex items-center gap-1.5"><Info className="h-3.5 w-3.5" /> Split-Plot RCBD</p>
              <p>Main-plot effects are tested against whole-plot error. Subplot effects and interactions are tested against subplot error.</p>
            </div>
          )}

          {/* Response variable selection */}
          <div className="space-y-2">
            <Label className="text-sm font-medium">Response Variable(s)</Label>
            <div className="flex flex-wrap gap-3">
              {datasetContext.availableTraitColumns.map((t) => (
                <label key={t} className="flex items-center gap-2 text-sm cursor-pointer">
                  <Checkbox checked={selectedTraits.includes(t)} onCheckedChange={() => toggleTrait(t)} />
                  {t}
                </label>
              ))}
            </div>
          </div>

          {validation && (
            <div className="rounded-md border border-destructive/30 bg-destructive/5 p-2.5 text-xs text-destructive flex items-center gap-2">
              <AlertTriangle className="h-3.5 w-3.5" /> {validation}
            </div>
          )}

          <Button onClick={handleAnalyze} disabled={isAnalyzing || !!validation} className="gap-2">
            {isAnalyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run Analysis
          </Button>
        </CardContent>
      </Card>

      {/* Results */}
      {results && (
        <div className="space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-emerald-600" />
                ANOVA Results — {DESIGNS.find((d) => d.id === design)?.label}
              </CardTitle>
              <Button onClick={handleDownload} disabled={isDownloading} size="sm" className="gap-2 bg-primary hover:bg-primary/90 text-primary-foreground">
                {isDownloading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                {isDownloading ? "Downloading..." : "Download ANOVA Report"}
              </Button>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2 text-sm">
                <Badge variant="secondary">{results.dataset_summary.n_genotypes} treatment level(s)</Badge>
                <Badge variant="secondary">{results.dataset_summary.n_reps} replication(s)</Badge>
                <Badge variant="outline">{results.dataset_summary.mode} mode</Badge>
              </div>
            </CardContent>
          </Card>

          {isSplitPlot && (
            <Card className="border-amber-200 dark:border-amber-800 bg-amber-50/50 dark:bg-amber-950/10">
              <CardContent className="py-4 px-5 text-sm text-amber-900 dark:text-amber-200">
                <p className="font-semibold mb-1">Error strata</p>
                <p>Main-plot effects are evaluated using whole-plot variability. Subplot effects and interactions are evaluated using subplot variability.</p>
              </CardContent>
            </Card>
          )}

          {results.failed_traits.length > 0 && (
            <Card className="border-destructive/30">
              <CardContent className="p-4">
                <div className="flex items-center gap-2 text-sm text-destructive font-medium mb-1">
                  <AlertTriangle className="h-4 w-4" /> Failed Response Variables
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

            return (
              <div key={trait} className="space-y-1">
                <h3 className="text-base font-semibold text-foreground px-1">{trait}</h3>
                <AcademicResultsPanel
                  moduleLabel="ANOVA"
                  domainNeutral
                  insightSummary={`Grand mean: ${r.grand_mean?.toFixed(2) ?? "—"} | ${r.n_genotypes} treatment level(s) × ${r.n_reps} replication(s)`}
                  interpretation={tr.analysis_result.interpretation}
                  statisticalNotes={
                    tr.data_warnings.length > 0
                      ? tr.data_warnings.map((w) => ({ text: w }))
                      : undefined
                  }
                  anovaTable={r.anova_table}
                  // Split-plot: hide mean separation per spec
                  meanSeparation={isSplitPlot ? undefined : r.mean_separation}
                  descriptiveStats={[
                    { label: "Grand Mean", value: r.grand_mean?.toFixed(4) ?? "—" },
                    { label: "Treatment Levels", value: String(r.n_genotypes) },
                    { label: "Replications", value: String(r.n_reps) },
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
