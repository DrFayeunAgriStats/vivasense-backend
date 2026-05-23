import { useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  BarChart, Bar, ScatterChart, Scatter, CartesianGrid, XAxis, YAxis, ZAxis, Tooltip, Legend,
  ReferenceLine, ResponsiveContainer,
} from "recharts";
import { Loader2, Play, AlertTriangle, Network } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { runManova, buildManovaPayload } from "@/lib/advancedAnalysisApi";
import type { DatasetContext } from "@/types/geneticsUpload";
import type { ManovaResponse, ManovaTestStat } from "@/types/advancedAnalysis";
import {
  fmt, SummaryCard, InterpretationPanel, ExportToolbar, downloadCsv, exportChartPng, DatasetTokenWarning,
} from "./shared";

interface Props { datasetContext: DatasetContext | null; }

const TEST_STATS: ManovaTestStat[] = ["Wilks", "Pillai", "Hotelling-Lawley", "Roy"];

function allColumns(ctx: DatasetContext | null): string[] {
  if (!ctx) return [];
  const cols = new Set<string>();
  if (ctx.genotypeColumn) cols.add(ctx.genotypeColumn);
  if (ctx.repColumn) cols.add(ctx.repColumn);
  if (ctx.environmentColumn) cols.add(ctx.environmentColumn);
  ctx.availableTraitColumns.forEach((t) => cols.add(t));
  return Array.from(cols);
}

const SCATTER_COLORS = ["#0A7F5A", "#2563EB", "#D97706", "#9333EA", "#DC2626", "#0891B2", "#65A30D", "#DB2777"];

export function ManovaPanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const allTraits = datasetContext?.availableTraitColumns ?? [];
  const cols = allColumns(datasetContext);
  const [traitCols, setTraitCols] = useState<string[]>(allTraits.slice(0, Math.max(2, allTraits.length)));
  const [factorCol, setFactorCol] = useState<string>(datasetContext?.genotypeColumn ?? "");
  const [covariates, setCovariates] = useState<string[]>([]);
  const [testStat, setTestStat] = useState<ManovaTestStat>("Wilks");
  const [alpha, setAlpha] = useState<number>(0.05);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ManovaResponse | null>(null);
  const etaRef = useRef<HTMLDivElement>(null);
  const scatterRef = useRef<HTMLDivElement>(null);
  const datasetToken = datasetContext?.datasetToken ?? null;

  const toggleTrait = (t: string) => setTraitCols((p) => p.includes(t) ? p.filter((x) => x !== t) : [...p, t]);
  const toggleCov = (t: string) => setCovariates((p) => p.includes(t) ? p.filter((x) => x !== t) : [...p, t]);

  const handleRun = async () => {
    if (!datasetToken) return toast({ title: "Please upload a dataset first." });
    if (traitCols.length < 2) return toast({ title: "Select at least two numeric traits." });
    if (!factorCol) return toast({ title: "Select a factor column." });
    if (alpha <= 0 || alpha >= 1) return toast({ title: "Alpha must be between 0 and 1." });
    setIsRunning(true); setError(null); setResult(null);
    try {
      const res = await runManova(buildManovaPayload({
        datasetToken,
        traitColumns: traitCols,
        factorColumn: factorCol,
        covariates,
        testStatistic: testStat,
        alpha,
      }));
      // eslint-disable-next-line no-console
      console.log("[manova response]", res);
      if (res.status !== "success") throw new Error("MANOVA failed on the server.");
      setResult(res);
      toast({ title: "MANOVA complete" });
    } catch (e) {
      const raw = (e as Error).message ?? "Unexpected error";
      const msg = /singular|constant|missing|rank/i.test(raw)
        ? "This analysis could not be completed. Check trait type, missing values, or constant columns."
        : raw;
      setError(msg);
      toast({ title: "Analysis failed", description: msg, variant: "destructive" });
    } finally { setIsRunning(false); }
  };

  const etaData = useMemo(() => {
    const rows = result?.univariate_results ?? [];
    return rows
      .filter((r) => typeof r.eta_squared === "number")
      .map((r) => ({ trait: r.trait, eta_squared: r.eta_squared as number }));
  }, [result]);

  const scatterByGroup = useMemo(() => {
    const coords = result?.coordinates ?? [];
    if (coords.length === 0) return [];
    const groups = new Map<string, { name: string; points: { x: number; y: number; label: string }[] }>();
    coords.forEach((c) => {
      const key = c.group ?? "all";
      if (!groups.has(key)) groups.set(key, { name: key, points: [] });
      groups.get(key)!.points.push({ x: c.x, y: c.y, label: c.label });
    });
    return Array.from(groups.values());
  }, [result]);

  if (!datasetToken) return <DatasetTokenWarning />;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="font-serif text-xl flex items-center gap-2">
            <Network className="h-5 w-5 text-primary" /> MANOVA
            <Badge variant="secondary" className="text-[10px] uppercase">Multivariate</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <p className="text-sm text-muted-foreground">
            Multivariate analysis of variance across correlated traits using Wilks, Pillai, Hotelling-Lawley or Roy statistics.
          </p>
          <div>
            <Label className="text-xs uppercase tracking-wide block mb-2">Trait columns (≥ 2)</Label>
            <div className="flex flex-wrap gap-2">
              {allTraits.map((t) => {
                const on = traitCols.includes(t);
                return (
                  <button key={t} type="button" onClick={() => toggleTrait(t)}
                    className={`px-3 py-1.5 rounded-md border text-sm transition-colors ${on ? "bg-primary/10 border-primary" : "bg-background border-border text-muted-foreground hover:border-primary/40"}`}>
                    {t}
                  </button>
                );
              })}
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <Label className="text-xs uppercase tracking-wide">Factor column</Label>
              <Select value={factorCol} onValueChange={setFactorCol}>
                <SelectTrigger><SelectValue placeholder="Select factor" /></SelectTrigger>
                <SelectContent>
                  {cols.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs uppercase tracking-wide">Test statistic</Label>
              <Select value={testStat} onValueChange={(v) => setTestStat(v as ManovaTestStat)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {TEST_STATS.map((s) => <SelectItem key={s} value={s}>{s}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs uppercase tracking-wide">Alpha</Label>
              <Input type="number" step="0.01" min="0.001" max="0.999" value={alpha} onChange={(e) => setAlpha(Number(e.target.value))} />
            </div>
          </div>
          <div>
            <Label className="text-xs uppercase tracking-wide block mb-2">Covariates (optional)</Label>
            <div className="flex flex-wrap gap-2">
              {allTraits.filter((t) => !traitCols.includes(t)).map((t) => {
                const on = covariates.includes(t);
                return (
                  <button key={t} type="button" onClick={() => toggleCov(t)}
                    className={`px-3 py-1.5 rounded-md border text-sm transition-colors ${on ? "bg-primary/10 border-primary" : "bg-background border-border text-muted-foreground hover:border-primary/40"}`}>
                    {t}
                  </button>
                );
              })}
              {allTraits.filter((t) => !traitCols.includes(t)).length === 0 && (
                <p className="text-xs text-muted-foreground">No traits available as covariates (deselect a response trait first).</p>
              )}
            </div>
          </div>
          <Button onClick={handleRun} disabled={isRunning || traitCols.length < 2 || !factorCol} className="gap-2">
            {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run MANOVA
          </Button>
        </CardContent>
      </Card>

      {error && (
        <Card className="border-destructive/40 bg-destructive/5">
          <CardContent className="p-4 flex items-start gap-3 text-sm">
            <AlertTriangle className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
            <div className="text-destructive">{error}</div>
          </CardContent>
        </Card>
      )}

      {isRunning && <Skeleton className="h-72" />}

      {result && (
        <>
          {result.interpretation && <InterpretationPanel text={result.interpretation} />}

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <SummaryCard label="Test statistic" value={result.test_statistic ?? testStat} />
            <SummaryCard label="Value" value={fmt(result.statistic_value, 4)} />
            <SummaryCard label="F statistic" value={fmt(result.f_statistic, 3)} />
            <SummaryCard
              label="p-value"
              value={result.p_value == null ? "—" : (result.p_value < 0.001 ? "< 0.001" : result.p_value.toFixed(4))}
              accent={result.significant ? "emerald" : "muted"}
              hint={result.significant ? "Significant" : "Not significant"}
            />
          </div>

          {Array.isArray(result.manova_table) && result.manova_table.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">MANOVA results</CardTitle>
                <ExportToolbar onCsv={() => downloadCsv("manova_results.csv", result.manova_table as unknown as Record<string, unknown>[])} />
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Effect</TableHead>
                      <TableHead>Statistic</TableHead>
                      <TableHead className="text-right">Value</TableHead>
                      <TableHead className="text-right">F</TableHead>
                      <TableHead className="text-right">df₁</TableHead>
                      <TableHead className="text-right">df₂</TableHead>
                      <TableHead className="text-right">p-value</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.manova_table.map((r, i) => (
                      <TableRow key={i} className={r.significant ? "bg-primary/5" : ""}>
                        <TableCell className="font-medium">{r.effect}</TableCell>
                        <TableCell>{r.test_statistic ?? "—"}</TableCell>
                        <TableCell className="text-right tabular-nums">{fmt(r.value, 4)}</TableCell>
                        <TableCell className="text-right tabular-nums">{fmt(r.f_statistic, 3)}</TableCell>
                        <TableCell className="text-right tabular-nums">{r.num_df ?? "—"}</TableCell>
                        <TableCell className="text-right tabular-nums">{r.den_df ?? "—"}</TableCell>
                        <TableCell className="text-right tabular-nums">{r.p_value == null ? "—" : (r.p_value < 0.001 ? "< 0.001" : r.p_value.toFixed(4))}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="font-serif text-lg">Univariate follow-up ANOVA</CardTitle>
              {Array.isArray(result.univariate_results) && result.univariate_results.length > 0 && (
                <ExportToolbar onCsv={() => downloadCsv("manova_univariate.csv", result.univariate_results as unknown as Record<string, unknown>[])} />
              )}
            </CardHeader>
            <CardContent>
              {Array.isArray(result.univariate_results) && result.univariate_results.length > 0 ? (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Trait</TableHead>
                        <TableHead className="text-right">F</TableHead>
                        <TableHead className="text-right">p-value</TableHead>
                        <TableHead className="text-right">η²</TableHead>
                        <TableHead>Significant</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {result.univariate_results.map((r) => (
                        <TableRow key={r.trait}>
                          <TableCell className="font-medium">{r.trait}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(r.f_statistic, 3)}</TableCell>
                          <TableCell className="text-right tabular-nums">{r.p_value == null ? "—" : (r.p_value < 0.001 ? "< 0.001" : r.p_value.toFixed(4))}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(r.eta_squared, 3)}</TableCell>
                          <TableCell>{r.significant ? <Badge className="bg-emerald-600 hover:bg-emerald-600">Yes</Badge> : <Badge variant="secondary">No</Badge>}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Univariate follow-up results were not returned for this run.</p>
              )}
            </CardContent>
          </Card>

          {etaData.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">Effect size (η²) by trait</CardTitle>
                <ExportToolbar onPng={() => exportChartPng(etaRef.current, "manova_effect_size.png")} />
              </CardHeader>
              <CardContent>
                <div ref={etaRef} className="w-full h-[300px]">
                  <ResponsiveContainer>
                    <BarChart data={etaData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="trait" />
                      <YAxis label={{ value: "η²", angle: -90, position: "insideLeft" }} />
                      <Tooltip formatter={(v: number) => v.toFixed(3)} />
                      <Bar dataKey="eta_squared" fill="#0A7F5A" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {scatterByGroup.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">Discriminant projection</CardTitle>
                <ExportToolbar onPng={() => exportChartPng(scatterRef.current, "manova_projection.png")} />
              </CardHeader>
              <CardContent>
                <div ref={scatterRef} className="w-full h-[420px]">
                  <ResponsiveContainer>
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="x" name="Dim 1" />
                      <YAxis type="number" dataKey="y" name="Dim 2" />
                      <ZAxis range={[60, 60]} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                      <ReferenceLine x={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
                      <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
                      <Legend />
                      {scatterByGroup.map((g, i) => (
                        <Scatter key={g.name} name={g.name} data={g.points} fill={SCATTER_COLORS[i % SCATTER_COLORS.length]} />
                      ))}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {result.assumptions_note && (
            <Card className="border-amber-300 bg-amber-50/60 dark:bg-amber-900/20">
              <CardContent className="p-4 text-sm text-amber-900 dark:text-amber-200">
                <strong>Assumptions:</strong> {result.assumptions_note}
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
