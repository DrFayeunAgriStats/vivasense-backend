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
  BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { Loader2, Play, AlertTriangle, Sigma } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { runNonparametric, buildNonparametricPayload } from "@/lib/advancedAnalysisApi";
import type { DatasetContext } from "@/types/geneticsUpload";
import type { NonparametricResponse, NonparametricTestType } from "@/types/advancedAnalysis";
import {
  fmt, SummaryCard, InterpretationPanel, ExportToolbar, downloadCsv, exportChartPng, DatasetTokenWarning,
} from "./shared";

interface Props { datasetContext: DatasetContext | null; }

const TEST_OPTIONS: { id: NonparametricTestType; label: string; hint: string }[] = [
  { id: "kruskal-wallis", label: "Kruskal–Wallis", hint: "≥3 independent groups" },
  { id: "friedman", label: "Friedman", hint: "Repeated measures (requires block)" },
  { id: "dunn", label: "Dunn's test", hint: "Post-hoc pairwise" },
];

function allColumns(ctx: DatasetContext | null): string[] {
  if (!ctx) return [];
  const cols = new Set<string>();
  if (ctx.genotypeColumn) cols.add(ctx.genotypeColumn);
  if (ctx.repColumn) cols.add(ctx.repColumn);
  if (ctx.environmentColumn) cols.add(ctx.environmentColumn);
  ctx.availableTraitColumns.forEach((t) => cols.add(t));
  return Array.from(cols);
}

export function NonparametricPanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const traits = datasetContext?.availableTraitColumns ?? [];
  const cols = allColumns(datasetContext);
  const [trait, setTrait] = useState<string>(traits[0] ?? "");
  const [groupCol, setGroupCol] = useState<string>(datasetContext?.genotypeColumn ?? "");
  const [testType, setTestType] = useState<NonparametricTestType>("kruskal-wallis");
  const [blockCol, setBlockCol] = useState<string>(datasetContext?.repColumn ?? "");
  const [alpha, setAlpha] = useState<number>(0.05);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<NonparametricResponse | null>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  const datasetToken = datasetContext?.datasetToken ?? null;

  const handleRun = async () => {
    if (!datasetToken) return toast({ title: "Please upload a dataset first." });
    if (!trait) return toast({ title: "Select a trait." });
    if (!groupCol) return toast({ title: "Select a group column." });
    if (testType === "friedman" && !blockCol) {
      return toast({ title: "Block column required", description: "Friedman test needs a block/subject column." });
    }
    if (alpha <= 0 || alpha >= 1) {
      return toast({ title: "Alpha must be between 0 and 1." });
    }
    setIsRunning(true); setError(null); setResult(null);
    try {
      const res = await runNonparametric(buildNonparametricPayload({
        datasetToken,
        traitColumn: trait,
        groupColumn: groupCol,
        testType,
        blockColumn: testType === "friedman" ? blockCol : undefined,
        alpha,
      }));
      // eslint-disable-next-line no-console
      console.log("[nonparametric response]", res);
      if (res.status !== "success") {
        throw new Error("Non-parametric analysis failed on the server.");
      }
      setResult(res);
      toast({ title: "Analysis complete" });
    } catch (e) {
      const raw = (e as Error).message ?? "Unexpected error";
      const msg = /constant|missing|nan|insufficient/i.test(raw)
        ? "This analysis could not be completed. Check trait type, missing values, or constant columns."
        : raw;
      setError(msg);
      toast({ title: "Analysis failed", description: msg, variant: "destructive" });
    } finally { setIsRunning(false); }
  };

  const groupChartData = useMemo(() => {
    const rows = result?.group_summary ?? [];
    return rows.map((r) => ({ group: r.group, value: r.median ?? r.mean_rank ?? 0 }));
  }, [result]);

  const valueLabel = useMemo(() => {
    const rows = result?.group_summary ?? [];
    return rows.some((r) => r.median != null) ? "Median" : "Mean rank";
  }, [result]);

  if (!datasetToken) return <DatasetTokenWarning />;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="font-serif text-xl flex items-center gap-2">
            <Sigma className="h-5 w-5 text-primary" /> Non-Parametric Tests
            <Badge variant="secondary" className="text-[10px] uppercase">Robust</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <p className="text-sm text-muted-foreground">
            Robust tests (Kruskal–Wallis, Friedman, Dunn) for non-normal or ordinal data.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label className="text-xs uppercase tracking-wide">Trait</Label>
              <Select value={trait} onValueChange={setTrait}>
                <SelectTrigger><SelectValue placeholder="Select numeric trait" /></SelectTrigger>
                <SelectContent>
                  {traits.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs uppercase tracking-wide">Group column</Label>
              <Select value={groupCol} onValueChange={setGroupCol}>
                <SelectTrigger><SelectValue placeholder="Select group" /></SelectTrigger>
                <SelectContent>
                  {cols.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs uppercase tracking-wide">Test type</Label>
              <Select value={testType} onValueChange={(v) => setTestType(v as NonparametricTestType)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {TEST_OPTIONS.map((t) => (
                    <SelectItem key={t.id} value={t.id}>{t.label} — {t.hint}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            {testType === "friedman" && (
              <div>
                <Label className="text-xs uppercase tracking-wide">Block / subject column</Label>
                <Select value={blockCol} onValueChange={setBlockCol}>
                  <SelectTrigger><SelectValue placeholder="Required for Friedman" /></SelectTrigger>
                  <SelectContent>
                    {cols.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
            )}
            <div>
              <Label className="text-xs uppercase tracking-wide">Alpha</Label>
              <Input
                type="number" step="0.01" min="0.001" max="0.999"
                value={alpha} onChange={(e) => setAlpha(Number(e.target.value))}
              />
            </div>
          </div>

          <Button onClick={handleRun} disabled={isRunning || !trait || !groupCol} className="gap-2">
            {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run Analysis
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
            <SummaryCard label="Test" value={result.test_type ?? testType} />
            <SummaryCard label="Statistic" value={fmt(result.statistic, 3)} />
            <SummaryCard
              label="p-value"
              value={result.p_value == null ? "—" : (result.p_value < 0.001 ? "< 0.001" : result.p_value.toFixed(4))}
              accent={result.significant ? "emerald" : "muted"}
            />
            <SummaryCard
              label="Significance"
              value={result.significant ? "Significant" : "Not significant"}
              accent={result.significant ? "emerald" : "muted"}
              hint={`α = ${result.alpha ?? alpha}`}
            />
          </div>

          {Array.isArray(result.group_summary) && result.group_summary.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">{valueLabel} by group</CardTitle>
                <ExportToolbar
                  onCsv={() => downloadCsv("nonparametric_groups.csv", result.group_summary as unknown as Record<string, unknown>[])}
                  onPng={() => exportChartPng(chartRef.current, "nonparametric_groups.png")}
                />
              </CardHeader>
              <CardContent>
                <div ref={chartRef} className="w-full h-[320px]">
                  <ResponsiveContainer>
                    <BarChart data={groupChartData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="group" />
                      <YAxis label={{ value: valueLabel, angle: -90, position: "insideLeft" }} />
                      <Tooltip />
                      <Bar dataKey="value" fill="#0A7F5A">
                        {groupChartData.map((_, i) => <Cell key={i} fill="#0A7F5A" />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-4 overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Group</TableHead>
                        <TableHead className="text-right">n</TableHead>
                        <TableHead className="text-right">Median</TableHead>
                        <TableHead className="text-right">Mean rank</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {result.group_summary.map((r) => (
                        <TableRow key={r.group}>
                          <TableCell className="font-medium">{r.group}</TableCell>
                          <TableCell className="text-right tabular-nums">{r.n ?? "—"}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(r.median, 3)}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(r.mean_rank, 2)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="font-serif text-lg">Pairwise post-hoc comparisons</CardTitle>
              {Array.isArray(result.posthoc_results) && result.posthoc_results.length > 0 && (
                <ExportToolbar
                  onCsv={() => downloadCsv("nonparametric_posthoc.csv", result.posthoc_results as unknown as Record<string, unknown>[])}
                />
              )}
            </CardHeader>
            <CardContent>
              {Array.isArray(result.posthoc_results) && result.posthoc_results.length > 0 ? (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Group A</TableHead>
                        <TableHead>Group B</TableHead>
                        <TableHead className="text-right">Statistic</TableHead>
                        <TableHead className="text-right">p-value</TableHead>
                        <TableHead className="text-right">p-adjusted</TableHead>
                        <TableHead>Significant</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {result.posthoc_results.map((r, i) => (
                        <TableRow key={i}>
                          <TableCell>{r.group_a}</TableCell>
                          <TableCell>{r.group_b}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(r.statistic, 3)}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(r.p_value, 4)}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(r.p_adjusted, 4)}</TableCell>
                          <TableCell>
                            {r.significant ? <Badge className="bg-emerald-600 hover:bg-emerald-600">Yes</Badge> : <Badge variant="secondary">No</Badge>}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              ) : (
                <p className="text-sm text-muted-foreground">Post-hoc results were not returned or not required.</p>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
