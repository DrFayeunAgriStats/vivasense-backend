import { useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Collapsible, CollapsibleContent, CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, ErrorBar, Cell, Legend,
  PieChart, Pie,
} from "recharts";
import { Loader2, Play, AlertTriangle, Settings2, Sigma } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { runBlup, buildBlupPayload } from "@/lib/advancedAnalysisApi";
import type { DatasetContext } from "@/types/geneticsUpload";
import type { BlupResponse } from "@/types/advancedAnalysis";
import {
  fmt, SummaryCard, InterpretationPanel, ExportToolbar, downloadCsv, exportChartPng, DatasetTokenWarning,
} from "./shared";

interface Props {
  datasetContext: DatasetContext | null;
}

export function BlupPanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const traits = datasetContext?.availableTraitColumns ?? [];
  const [trait, setTrait] = useState<string>(traits[0] ?? "");
  const [fixedEffects, setFixedEffects] = useState<string[]>([]);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BlupResponse | null>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  const datasetToken = datasetContext?.datasetToken ?? null;

  const fixedOptions = useMemo(() => {
    if (!datasetContext) return [] as string[];
    const opts: string[] = [];
    if (datasetContext.environmentColumn) opts.push(datasetContext.environmentColumn);
    if (datasetContext.repColumn) opts.push(datasetContext.repColumn);
    return opts;
  }, [datasetContext]);

  const handleRun = async () => {
    if (!datasetToken || !trait) {
      toast({ title: "Missing input", description: "Dataset and trait are required." });
      return;
    }
    setIsRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await runBlup(buildBlupPayload({
        datasetToken,
        trait,
        randomEffects: ["genotype"],
        fixedEffects: fixedEffects,
      }));
      if (res.status !== "success") throw new Error("BLUP analysis failed on the server.");
      setResult(res);
      toast({ title: "BLUPs computed", description: `${res.genotype_blups.length} genotypes (${res.model_type})` });
    } catch (e) {
      const msg = (e as Error).message ?? "Unexpected error";
      setError(msg);
      toast({ title: "Analysis failed", description: msg, variant: "destructive" });
    } finally {
      setIsRunning(false);
    }
  };

  // Bar data: BLUP with SE error bars, color by reliability
  const barData = useMemo(() => {
    if (!result) return [];
    const sorted = [...result.genotype_blups].sort((a, b) => b.blup - a.blup);
    return sorted.map((g) => ({
      genotype: g.genotype,
      blup: g.blup,
      se: g.se,
      reliability: g.reliability,
    }));
  }, [result]);

  const varianceData = useMemo(() => {
    if (!result) return [];
    const v = result.variance_components;
    const arr = [
      { name: "Genotype", value: v.genotype, fill: "#0A7F5A" },
      { name: "Residual", value: v.residual, fill: "hsl(var(--muted-foreground))" },
    ];
    if (typeof v.environment === "number") {
      arr.push({ name: "Environment", value: v.environment, fill: "#2563EB" });
    }
    return arr;
  }, [result]);

  if (!datasetToken) return <DatasetTokenWarning />;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="font-serif text-xl flex items-center gap-2">
            <Sigma className="h-5 w-5 text-primary" />
            BLUP Predictions
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Best Linear Unbiased Predictions of genotypic values, with reliability and standard errors —
            the breeding-value standard for selection decisions.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <div className="md:col-span-2">
              <Label className="text-xs uppercase tracking-wide">Trait</Label>
              <Select value={trait} onValueChange={setTrait}>
                <SelectTrigger><SelectValue placeholder="Select a trait" /></SelectTrigger>
                <SelectContent>
                  {traits.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={handleRun} disabled={isRunning || !trait} className="gap-2">
              {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              Run Analysis
            </Button>
          </div>

          {fixedOptions.length > 0 && (
            <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="gap-1.5 -ml-2 text-muted-foreground">
                  <Settings2 className="h-4 w-4" />
                  Advanced options
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="pt-3">
                <Label className="text-xs uppercase tracking-wide block mb-2">Fixed effects</Label>
                <div className="flex flex-wrap gap-3">
                  {fixedOptions.map((opt) => {
                    const checked = fixedEffects.includes(opt);
                    return (
                      <label key={opt} className="flex items-center gap-2 text-sm cursor-pointer">
                        <Checkbox
                          checked={checked}
                          onCheckedChange={(v) => {
                            setFixedEffects((prev) =>
                              v ? [...prev, opt] : prev.filter((x) => x !== opt)
                            );
                          }}
                        />
                        {opt}
                      </label>
                    );
                  })}
                </div>
              </CollapsibleContent>
            </Collapsible>
          )}
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

      {isRunning && (
        <div className="space-y-4">
          <Skeleton className="h-24" />
          <Skeleton className="h-72" />
        </div>
      )}

      {result && (
        <>
          <InterpretationPanel text={result.interpretation} />

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <SummaryCard label="Model" value={result.model_type === "multi-environment" ? "Multi-env" : "Single-env"} />
            <SummaryCard label="Trait" value={result.trait} accent="emerald" />
            <SummaryCard
              label="Best genotypes"
              value={result.best_genotypes.length}
              hint={result.best_genotypes.slice(0, 3).join(", ") || "—"}
              accent="emerald"
            />
          </div>

          {/* Variance components */}
          <Card>
            <CardHeader>
              <CardTitle className="font-serif text-lg">Variance components</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="w-full h-[260px]">
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={varianceData} dataKey="value" nameKey="name" outerRadius={90} label />
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* BLUP bar chart */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="font-serif text-lg">BLUP values (sorted)</CardTitle>
              <ExportToolbar
                onCsv={() => downloadCsv(`blup_${result.trait}.csv`, result.genotype_blups as unknown as Record<string, unknown>[])}
                onPng={() => exportChartPng(chartRef.current, `blup_${result.trait}.png`)}
                onCopy={() => navigator.clipboard.writeText(result.interpretation)}
              />
            </CardHeader>
            <CardContent>
              <div ref={chartRef} className="w-full h-[420px]">
                <ResponsiveContainer>
                  <BarChart data={barData} margin={{ top: 10, right: 30, bottom: 80, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="genotype" angle={-45} textAnchor="end" interval={0} height={80} tick={{ fontSize: 11 }} />
                    <YAxis label={{ value: result.trait + " (BLUP)", angle: -90, position: "insideLeft" }} />
                    <Tooltip />
                    <Bar dataKey="blup" name="BLUP">
                      {barData.map((d, i) => (
                        <Cell key={i} fill={reliabilityColor(d.reliability)} />
                      ))}
                      <ErrorBar dataKey="se" width={4} stroke="hsl(var(--muted-foreground))" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-3 flex items-center gap-4 text-xs text-muted-foreground">
                <span>Color = reliability:</span>
                <span className="inline-flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ background: "#94A3B8" }} />low</span>
                <span className="inline-flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ background: "#3B82F6" }} />moderate</span>
                <span className="inline-flex items-center gap-1.5"><span className="h-3 w-3 rounded-sm" style={{ background: "#0A7F5A" }} />high</span>
              </div>
            </CardContent>
          </Card>

          {/* Table */}
          <Card>
            <CardHeader>
              <CardTitle className="font-serif text-lg">Per-genotype BLUPs</CardTitle>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Genotype</TableHead>
                    <TableHead className="text-right">BLUP</TableHead>
                    <TableHead className="text-right">SE</TableHead>
                    <TableHead>Reliability</TableHead>
                    <TableHead className="text-right">Rank</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {result.genotype_blups.map((g) => (
                    <TableRow key={g.genotype}>
                      <TableCell className="font-medium">
                        {g.genotype}
                        {result.best_genotypes.includes(g.genotype) && (
                          <Badge variant="outline" className="ml-2 bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-900/30 dark:text-emerald-200">
                            Best
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{fmt(g.blup, 3)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fmt(g.se, 3)}</TableCell>
                      <TableCell className="min-w-[160px]">
                        <div className="flex items-center gap-2">
                          <Progress value={Math.max(0, Math.min(100, g.reliability * 100))} className="h-2" />
                          <span className="text-xs tabular-nums text-muted-foreground w-12 text-right">{(g.reliability * 100).toFixed(0)}%</span>
                        </div>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{g.rank}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function reliabilityColor(r: number): string {
  if (r >= 0.7) return "#0A7F5A";
  if (r >= 0.4) return "#3B82F6";
  return "#94A3B8";
}
