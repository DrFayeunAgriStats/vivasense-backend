import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Loader2, BarChart3, AlertTriangle, Download, Info, CheckCircle2, FileSpreadsheet,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import {
  computeDescriptiveStats,
  exportDescriptiveStatsWord,
} from "@/lib/descriptiveStatsApi";
import type {
  DatasetContext,
} from "@/types/geneticsUpload";
import type {
  DescriptiveStatsResponse,
  DescriptiveStatsRow,
  DescriptiveStatsRequest,
} from "@/types/descriptiveStats";

interface Props {
  datasetContext: DatasetContext | null;
}

const fmt = (v: number | null | undefined, digits = 2): string => {
  if (typeof v === "number" && !isNaN(v)) {
    return v.toFixed(digits);
  }
  if (v === null || v === undefined) return "Not available";
  return String(v);
};

const fmtInt = (v: number | null | undefined): string => {
  if (v === null || v === undefined) return "Not available";
  return String(Math.trunc(v));
};

function precisionBadgeClass(p?: string | null): string {
  if (!p) return "bg-muted text-muted-foreground border-border";
  const s = p.toLowerCase();
  if (s.includes("excellent") || s.includes("good")) {
    return "bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-900/30 dark:text-emerald-200 dark:border-emerald-800";
  }
  if (s.includes("moderate") || s.includes("acceptable")) {
    return "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-900/30 dark:text-amber-200 dark:border-amber-800";
  }
  if (s.includes("poor") || s.includes("high") || s.includes("caution")) {
    return "bg-red-100 text-red-800 border-red-300 dark:bg-red-900/30 dark:text-red-200 dark:border-red-800";
  }
  return "bg-muted text-muted-foreground border-border";
}

export function DescriptiveStatsPanel({ datasetContext }: Props) {
  const { toast } = useToast();

  const traitOptions = datasetContext?.availableTraitColumns ?? [];
  const columnOptions = useMemo(() => {
    // Best-effort list of available columns for genotype/rep selectors
    if (!datasetContext) return [];
    const set = new Set<string>(traitOptions);
    if (datasetContext.genotypeColumn) set.add(datasetContext.genotypeColumn);
    if (datasetContext.repColumn) set.add(datasetContext.repColumn);
    if (datasetContext.environmentColumn) set.add(datasetContext.environmentColumn);
    return Array.from(set);
  }, [datasetContext, traitOptions]);

  const [selectedTraits, setSelectedTraits] = useState<string[]>([]);
  const [genotypeCol, setGenotypeCol] = useState<string>(datasetContext?.genotypeColumn ?? "__none__");
  const [repCol, setRepCol] = useState<string>(datasetContext?.repColumn ?? "__none__");
  const [expectedRep, setExpectedRep] = useState<string>("");

  const [isRunning, setIsRunning] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DescriptiveStatsResponse | null>(null);
  const [lastRequest, setLastRequest] = useState<DescriptiveStatsRequest | null>(null);

  const toggleTrait = (t: string) =>
    setSelectedTraits((prev) =>
      prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]
    );

  const datasetToken = datasetContext?.datasetToken ?? null;
  const isUploadPending = !!datasetContext && !datasetToken;

  const handleRun = async () => {
    if (!datasetContext) {
      toast({ title: "Upload a dataset first", variant: "destructive" });
      return;
    }
    // Guard: backend requires dataset_token for /analysis/descriptive-stats.
    // Prevent firing the request before the upload-preview completes.
    if (!datasetToken) {
      const msg = "Dataset is still being prepared. Please wait for the upload to finish, then try again.";
      setError(msg);
      toast({ title: "Dataset not ready", description: msg, variant: "destructive" });
      return;
    }
    if (selectedTraits.length === 0) {
      setError("Select at least one numeric trait to compute descriptive statistics.");
      return;
    }
    setError(null);
    setIsRunning(true);
    setResult(null);
    try {
      const req: DescriptiveStatsRequest = {
        dataset_token: datasetToken,
        trait_columns: selectedTraits,
        genotype_column: genotypeCol && genotypeCol !== "__none__" ? genotypeCol : null,
        rep_column: repCol && repCol !== "__none__" ? repCol : null,
        expected_replication: expectedRep ? Number(expectedRep) : null,
      };
      console.log("[DESCRIPTIVE-STATS] request:", req);
      const res = await computeDescriptiveStats(req);
      console.log("[DESCRIPTIVE-STATS] full response:", res);
      setResult(res);
      setLastRequest(req);
      toast({ title: "Descriptive statistics ready", description: `${res.summary_table?.length ?? 0} traits summarized.` });
    } catch (e: any) {
      const msg = e?.message || "Failed to compute descriptive statistics.";
      setError(msg);
      toast({ title: "Analysis failed", description: msg, variant: "destructive" });
    } finally {
      setIsRunning(false);
    }
  };

  const handleExport = async () => {
    if (!result || !lastRequest || !datasetToken) {
      const msg = "Please confirm dataset and run descriptive statistics before exporting.";
      setError(msg);
      toast({ title: "Cannot export yet", description: msg, variant: "destructive" });
      return;
    }
    if (selectedTraits.length === 0) {
      toast({
        title: "No traits selected",
        description: "Please confirm dataset and run descriptive statistics before exporting.",
        variant: "destructive",
      });
      return;
    }
    setIsExporting(true);
    try {
      // Backend expects a FLAT payload — fields at root, NOT nested under `response`.
      const r: any = result;
      const payload = {
        dataset_token: datasetToken,

        overview:        r?.overview        ?? null,
        summary_table:   r?.summary_table   ?? [],
        reliable_traits: r?.reliable_traits ?? [],
        caution_traits:  r?.caution_traits  ?? [],
        global_flags:    r?.global_flags    ?? [],
        recommendation:  r?.recommendation  ?? "",

        trait_columns: lastRequest.trait_columns ?? [],
        genotype_column: lastRequest.genotype_column ?? null,
        rep_column: lastRequest.rep_column ?? null,
        expected_replication: lastRequest.expected_replication ?? null,
      };
      const blob = await exportDescriptiveStatsWord(payload);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const tokenSlug = datasetToken.replace(/[^A-Za-z0-9_-]/g, "").slice(0, 12);
      a.download = `descriptive_statistics_${tokenSlug || new Date().toISOString().slice(0, 10)}.docx`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      toast({ title: "Report downloaded", description: a.download });
    } catch (e: any) {
      const msg = e?.message || "Word export endpoint is unavailable.";
      toast({ title: "Export failed", description: msg, variant: "destructive" });
    } finally {
      setIsExporting(false);
    }
  };

  // ─── Empty state ───────────────────────────────────────────────────────────
  if (!datasetContext) {
    return (
      <Card>
        <CardContent className="py-10 text-center">
          <FileSpreadsheet className="h-10 w-10 mx-auto text-muted-foreground mb-3" />
          <p className="text-sm text-muted-foreground">
            Upload and confirm a dataset above to enable Descriptive Statistics.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-1">
          <BarChart3 className="h-6 w-6 text-primary" />
          <h2 className="font-serif text-2xl font-semibold text-foreground">
            Descriptive Statistics
          </h2>
        </div>
        <p className="text-sm text-muted-foreground">
          Summarize trait distributions, data quality, and variability before ANOVA.
        </p>
      </div>

      {/* Selection panel */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Trait & Column Selection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Trait multi-select */}
          <div>
            <Label className="text-sm font-medium mb-2 block">
              Numeric traits <span className="text-muted-foreground">({selectedTraits.length} selected)</span>
            </Label>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 max-h-56 overflow-y-auto rounded-md border border-border p-3 bg-muted/20">
              {traitOptions.length === 0 ? (
                <p className="text-xs text-muted-foreground col-span-full">
                  No numeric traits detected in dataset.
                </p>
              ) : (
                traitOptions.map((t) => (
                  <label
                    key={t}
                    className="flex items-center gap-2 text-sm cursor-pointer hover:bg-background rounded px-2 py-1"
                  >
                    <Checkbox
                      checked={selectedTraits.includes(t)}
                      onCheckedChange={() => toggleTrait(t)}
                    />
                    <span className="truncate">{t}</span>
                  </label>
                ))
              )}
            </div>
          </div>

          {/* Optional columns */}
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="space-y-1.5">
              <Label className="text-sm">Genotype column (optional)</Label>
              <Select value={genotypeCol} onValueChange={setGenotypeCol}>
                <SelectTrigger><SelectValue placeholder="None" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">None</SelectItem>
                  {columnOptions.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-sm">Replication column (optional)</Label>
              <Select value={repCol} onValueChange={setRepCol}>
                <SelectTrigger><SelectValue placeholder="None" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">None</SelectItem>
                  {columnOptions.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <Label className="text-sm">Expected replication (optional)</Label>
              <Input
                type="number"
                min={1}
                placeholder="e.g. 3"
                value={expectedRep}
                onChange={(e) => setExpectedRep(e.target.value)}
              />
            </div>
          </div>

          {error && (
            <div className="rounded-md border border-red-300 bg-red-50 text-red-800 px-3 py-2 text-sm flex items-start gap-2 dark:bg-red-900/20 dark:text-red-200 dark:border-red-800">
              <AlertTriangle className="h-4 w-4 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          {isUploadPending && (
            <div className="rounded-md border border-amber-300 bg-amber-50 text-amber-800 px-3 py-2 text-sm flex items-start gap-2 dark:bg-amber-900/20 dark:text-amber-200 dark:border-amber-800">
              <Loader2 className="h-4 w-4 mt-0.5 animate-spin" />
              <span>Preparing dataset on the server… please wait before running analysis.</span>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3">
            <Button
              onClick={handleRun}
              disabled={isRunning || isUploadPending || !datasetToken || selectedTraits.length === 0}
              className="gap-2"
            >
              {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <BarChart3 className="h-4 w-4" />}
              Run Descriptive Statistics
            </Button>
            {result && (
              <Button onClick={handleExport} disabled={isExporting} variant="outline" className="gap-2">
                {isExporting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                Export Descriptive Report
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {isRunning && (
        <Card>
          <CardContent className="py-12 text-center">
            <Loader2 className="h-8 w-8 mx-auto animate-spin text-primary mb-3" />
            <p className="text-sm text-muted-foreground">Computing descriptive statistics…</p>
          </CardContent>
        </Card>
      )}

      {!isRunning && !result && !error && (
        <Card>
          <CardContent className="py-10 text-center">
            <Info className="h-8 w-8 mx-auto text-muted-foreground mb-3" />
            <p className="text-sm text-muted-foreground">
              Select traits and click <span className="font-medium">Run Descriptive Statistics</span> to view results.
            </p>
          </CardContent>
        </Card>
      )}

      {result && <DescriptiveResults result={result} />}
    </div>
  );
}

// ─── Results sub-component ──────────────────────────────────────────────────
function DescriptiveResults({ result }: { result: DescriptiveStatsResponse }) {
  const rows: DescriptiveStatsRow[] = Array.isArray(result.summary_table) ? result.summary_table : [];

  // Derived data-quality lists (tolerant — falls back to table-derived if backend omits)
  const traitsWithMissing = rows.filter((r) => (r.missing_count ?? 0) > 0);
  const traitsWithZeros = rows.filter((r) => (r.zero_count ?? 0) > 0);
  const traitsHighCV = rows.filter((r) => typeof r.cv_percent === "number" && (r.cv_percent as number) > 20);

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-base flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
            Results
          </CardTitle>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="secondary">{rows.length} traits</Badge>
            {result.global_flags?.length ? (
              <Badge variant="outline" className="border-amber-300 text-amber-700 dark:text-amber-300">
                {result.global_flags.length} global flag{result.global_flags.length !== 1 ? "s" : ""}
              </Badge>
            ) : null}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Section A: Overview */}
        {(() => {
          const ov = result.overview;
          if (!ov) return null;
          if (typeof ov === "string") {
            return <p className="text-sm text-muted-foreground mb-4 leading-relaxed">{ov}</p>;
          }
          const nTraits = ov.n_traits ?? rows.length;
          const nObs = ov.n_observations ?? null;
          return (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-5">
              <div className="rounded-md border border-border bg-muted/20 px-4 py-3">
                <div className="text-xs uppercase tracking-wide text-muted-foreground">Traits</div>
                <div className="text-lg font-semibold text-foreground tabular-nums">{fmtInt(nTraits)}</div>
              </div>
              <div className="rounded-md border border-border bg-muted/20 px-4 py-3">
                <div className="text-xs uppercase tracking-wide text-muted-foreground">Observations</div>
                <div className="text-lg font-semibold text-foreground tabular-nums">{fmtInt(nObs)}</div>
              </div>
              {result.reliable_traits && (
                <div className="rounded-md border border-emerald-200 dark:border-emerald-800 bg-emerald-50/40 dark:bg-emerald-900/10 px-4 py-3">
                  <div className="text-xs uppercase tracking-wide text-emerald-700 dark:text-emerald-300">Reliable</div>
                  <div className="text-lg font-semibold text-emerald-800 dark:text-emerald-200 tabular-nums">
                    {result.reliable_traits.length}
                  </div>
                </div>
              )}
            </div>
          );
        })()}

        <Tabs defaultValue="summary" className="w-full">
          <TabsList className="flex flex-wrap h-auto">
            <TabsTrigger value="summary">Summary Table</TabsTrigger>
            <TabsTrigger value="interpretation">Interpretation</TabsTrigger>
            <TabsTrigger value="quality">Data Quality</TabsTrigger>
            <TabsTrigger value="recommendation">Recommendation</TabsTrigger>
          </TabsList>

          {/* Tab A: Summary Table */}
          <TabsContent value="summary" className="mt-4">
            <div className="overflow-x-auto rounded-md border border-border">
              <table className="w-full text-sm border-collapse">
                <thead className="bg-muted/50">
                  <tr className="text-left">
                    {["Trait", "n", "Mean", "Min", "Max", "SD", "CV (%)", "Median", "Missing", "Zero", "Precision"].map((h) => (
                      <th key={h} className="px-3 py-2 font-medium text-foreground border-b border-border whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={r.trait + i} className="even:bg-muted/20">
                      <td className="px-3 py-2 font-medium text-foreground whitespace-nowrap">{r.trait}</td>
                      <td className="px-3 py-2 tabular-nums">{fmtInt(r.n)}</td>
                      <td className="px-3 py-2 tabular-nums">{fmt(r.mean)}</td>
                      <td className="px-3 py-2 tabular-nums">{fmt(r.minimum)}</td>
                      <td className="px-3 py-2 tabular-nums">{fmt(r.maximum)}</td>
                      <td className="px-3 py-2 tabular-nums">{fmt(r.sd)}</td>
                      <td className="px-3 py-2 tabular-nums">{fmt(r.cv_percent)}</td>
                      <td className="px-3 py-2 tabular-nums">{fmt(r.median)}</td>
                      <td className="px-3 py-2 tabular-nums">{fmtInt(r.missing_count)}</td>
                      <td className="px-3 py-2 tabular-nums">{fmtInt(r.zero_count)}</td>
                      <td className="px-3 py-2">
                        {r.precision_class ? (
                          <Badge variant="outline" className={precisionBadgeClass(r.precision_class)}>
                            {r.precision_class}
                          </Badge>
                        ) : (
                          <span className="text-xs text-muted-foreground">Not available</span>
                        )}
                      </td>
                    </tr>
                  ))}
                  {rows.length === 0 && (
                    <tr><td colSpan={11} className="px-3 py-6 text-center text-muted-foreground">No rows returned.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </TabsContent>

          {/* Tab B: Interpretation */}
          <TabsContent value="interpretation" className="mt-4 space-y-3">
            {rows.length === 0 && (
              <p className="text-sm text-muted-foreground">No interpretation available.</p>
            )}
            {rows.map((r, i) => (
              <Card key={r.trait + i} className="border-border">
                <CardContent className="py-4">
                  <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
                    <h3 className="font-semibold text-foreground">{r.trait}</h3>
                    {r.precision_class && (
                      <Badge variant="outline" className={precisionBadgeClass(r.precision_class)}>
                        {r.precision_class}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line">
                    {r.interpretation || "No interpretation provided for this trait."}
                  </p>
                  {Array.isArray(r.flags) && r.flags.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mt-3">
                      {r.flags.map((f, j) => (
                        <Badge key={j} variant="outline" className="border-amber-300 text-amber-700 dark:text-amber-300 text-xs">
                          ⚠ {f}
                        </Badge>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </TabsContent>

          {/* Tab C: Data Quality */}
          <TabsContent value="quality" className="mt-4 space-y-4">
            <QualityList
              title="Global flags"
              items={result.global_flags ?? []}
              emptyText="No global data-quality flags raised."
              tone="amber"
            />
            <QualityTraitList title="Traits with missing values" items={traitsWithMissing.map((r) => `${r.trait} — ${r.missing_count} missing`)} />
            <QualityTraitList title="Traits with zero values" items={traitsWithZeros.map((r) => `${r.trait} — ${r.zero_count} zeros`)} />
            <QualityTraitList
              title="Traits with high CV (> 20%)"
              items={traitsHighCV.map((r) => `${r.trait} — CV ${fmt(r.cv_percent)}%`)}
            />
            <QualityList
              title="Caution traits"
              items={result.caution_traits ?? []}
              emptyText="No traits flagged for caution."
              tone="red"
            />
          </TabsContent>

          {/* Tab D: Recommendation */}
          <TabsContent value="recommendation" className="mt-4 space-y-4">
            <Card className="border-emerald-200 dark:border-emerald-800 bg-emerald-50/40 dark:bg-emerald-900/10">
              <CardContent className="py-4">
                <h3 className="font-semibold text-emerald-800 dark:text-emerald-200 mb-2">Reliable traits</h3>
                {result.reliable_traits && result.reliable_traits.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {result.reliable_traits.map((t) => (
                      <Badge key={t} variant="outline" className="border-emerald-300 text-emerald-800 dark:text-emerald-200">
                        {t}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">Not available.</p>
                )}
              </CardContent>
            </Card>

            <Card className="border-amber-200 dark:border-amber-800 bg-amber-50/40 dark:bg-amber-900/10">
              <CardContent className="py-4">
                <h3 className="font-semibold text-amber-800 dark:text-amber-200 mb-2">Caution traits</h3>
                {result.caution_traits && result.caution_traits.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {result.caution_traits.map((t) => (
                      <Badge key={t} variant="outline" className="border-amber-300 text-amber-800 dark:text-amber-200">
                        {t}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">No traits flagged for caution.</p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardContent className="py-4">
                <h3 className="font-semibold text-foreground mb-2">Recommendation</h3>
                <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line">
                  {result.recommendation || "No recommendation provided by backend."}
                </p>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

function QualityList({
  title, items, emptyText, tone = "amber",
}: { title: string; items: string[]; emptyText: string; tone?: "amber" | "red" }) {
  const toneClass =
    tone === "red"
      ? "border-red-200 bg-red-50/50 dark:bg-red-900/10 dark:border-red-800"
      : "border-amber-200 bg-amber-50/50 dark:bg-amber-900/10 dark:border-amber-800";
  return (
    <Card className={toneClass}>
      <CardContent className="py-4">
        <h3 className="font-semibold text-foreground mb-2">{title}</h3>
        {items.length === 0 ? (
          <p className="text-sm text-muted-foreground">{emptyText}</p>
        ) : (
          <ul className="text-sm text-muted-foreground list-disc pl-5 space-y-0.5">
            {items.map((it, i) => <li key={i}>{it}</li>)}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function QualityTraitList({ title, items }: { title: string; items: string[] }) {
  return (
    <Card>
      <CardContent className="py-4">
        <h3 className="font-semibold text-foreground mb-2">{title}</h3>
        {items.length === 0 ? (
          <p className="text-sm text-muted-foreground">None.</p>
        ) : (
          <ul className="text-sm text-muted-foreground list-disc pl-5 space-y-0.5">
            {items.map((it, i) => <li key={i}>{it}</li>)}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}
