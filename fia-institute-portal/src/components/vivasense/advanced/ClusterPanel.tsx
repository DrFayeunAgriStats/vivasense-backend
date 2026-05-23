import { useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Collapsible, CollapsibleTrigger, CollapsibleContent,
} from "@/components/ui/collapsible";
import {
  BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  RadarChart, Radar, PolarAngleAxis, PolarGrid, PolarRadiusAxis, Legend,
} from "recharts";
import { Loader2, Play, AlertTriangle, Network, ChevronDown } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { runCluster, buildClusterPayload } from "@/lib/advancedAnalysisApi";
import type { DatasetContext } from "@/types/geneticsUpload";
import type { ClusterResponse } from "@/types/advancedAnalysis";
import {
  fmt, SummaryCard, InterpretationPanel, ExportToolbar, downloadCsv, exportChartPng, DatasetTokenWarning,
} from "./shared";

interface Props { datasetContext: DatasetContext | null; }

const LINKAGE = ["ward", "average", "complete", "single"] as const;
type Linkage = typeof LINKAGE[number];

const CLUSTER_PALETTE = ["#0A7F5A", "#2563EB", "#D97706", "#9333EA", "#DC2626", "#0891B2", "#65A30D", "#DB2777"];

export function ClusterPanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const allTraits = datasetContext?.availableTraitColumns ?? [];
  const [traits, setTraits] = useState<string[]>(allTraits);
  const [linkage, setLinkage] = useState<Linkage>("ward");
  const [k, setK] = useState<string>("");
  const [standardize, setStandardize] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClusterResponse | null>(null);
  const silRef = useRef<HTMLDivElement>(null);
  const radarRef = useRef<HTMLDivElement>(null);
  const datasetToken = datasetContext?.datasetToken ?? null;

  const handleRun = async () => {
    if (!datasetToken) return toast({ title: "Dataset required" });
    if (traits.length < 2) return toast({ title: "Pick at least 2 traits" });
    setIsRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await runCluster(buildClusterPayload({
        datasetToken,
        traits,
        method: linkage,
        k: k ? Number(k) : null,
        scale: standardize,
      }));
      if (res.status !== "success") throw new Error("Cluster analysis failed.");
      setResult(res);
      toast({ title: "Cluster analysis complete", description: `k=${res.optimal_k}, ${res.method}` });
    } catch (e) {
      const msg = (e as Error).message ?? "Unexpected error";
      setError(msg);
      toast({ title: "Analysis failed", description: msg, variant: "destructive" });
    } finally { setIsRunning(false); }
  };

  const toggleTrait = (t: string) => {
    setTraits((prev) => prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]);
  };

  const avgSil = useMemo(() => {
    if (!result || !result.silhouette_scores?.length) return null;
    return result.silhouette_scores.reduce((a, b) => a + b, 0) / result.silhouette_scores.length;
  }, [result]);

  const silData = useMemo(() => {
    if (!result) return [];
    return result.cluster_assignments
      .map((g) => ({
        genotype: g.genotype,
        score: g.silhouette_score ?? 0,
        cluster: g.cluster_id,
      }))
      .sort((a, b) => a.cluster - b.cluster || b.score - a.score);
  }, [result]);

  const radarData = useMemo(() => {
    if (!result) return [];
    // rows = traits, columns = clusters
    const traitKeys = Array.from(
      new Set(result.cluster_summary.flatMap((c) => Object.keys(c.mean_per_trait)))
    );
    // normalize per trait so radar is comparable
    const minMax: Record<string, { min: number; max: number }> = {};
    traitKeys.forEach((t) => {
      const vals = result.cluster_summary.map((c) => c.mean_per_trait[t] ?? 0);
      minMax[t] = { min: Math.min(...vals), max: Math.max(...vals) };
    });
    return traitKeys.map((t) => {
      const row: Record<string, number | string> = { trait: t };
      result.cluster_summary.forEach((c) => {
        const v = c.mean_per_trait[t] ?? 0;
        const { min, max } = minMax[t];
        row[`C${c.cluster_id}`] = max === min ? 50 : ((v - min) / (max - min)) * 100;
      });
      return row;
    });
  }, [result]);

  const membersByCluster = useMemo(() => {
    if (!result) return new Map<number, string[]>();
    const map = new Map<number, string[]>();
    for (const a of result.cluster_assignments) {
      if (!map.has(a.cluster_id)) map.set(a.cluster_id, []);
      map.get(a.cluster_id)!.push(a.genotype);
    }
    return map;
  }, [result]);

  if (!datasetToken) return <DatasetTokenWarning />;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="font-serif text-xl flex items-center gap-2">
            <Network className="h-5 w-5 text-primary" /> Cluster Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Group genotypes with similar multivariate trait profiles using hierarchical clustering.
          </p>
          <div>
            <Label className="text-xs uppercase tracking-wide block mb-2">Traits</Label>
            <div className="flex flex-wrap gap-2">
              {allTraits.map((t) => {
                const checked = traits.includes(t);
                return (
                  <label key={t} className={`px-3 py-1.5 rounded-md border text-sm cursor-pointer transition-colors ${checked ? "bg-primary/10 border-primary text-foreground" : "bg-background border-border text-muted-foreground hover:border-primary/40"}`}>
                    <input type="checkbox" className="sr-only" checked={checked} onChange={() => toggleTrait(t)} />
                    {t}
                  </label>
                );
              })}
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
            <div>
              <Label className="text-xs uppercase tracking-wide">Linkage</Label>
              <Select value={linkage} onValueChange={(v) => setLinkage(v as Linkage)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {LINKAGE.map((l) => <SelectItem key={l} value={l}>{l}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs uppercase tracking-wide">k (optional)</Label>
              <Input type="number" min={2} value={k} onChange={(e) => setK(e.target.value)} placeholder="auto" />
            </div>
            <label className="flex items-center gap-2 text-sm">
              <Checkbox checked={standardize} onCheckedChange={(v) => setStandardize(!!v)} />
              Standardize traits
            </label>
            <Button onClick={handleRun} disabled={isRunning || traits.length < 2} className="gap-2">
              {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              Run Analysis
            </Button>
          </div>
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
          <InterpretationPanel text={result.interpretation} />

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <SummaryCard label="Optimal k" value={result.optimal_k} accent="emerald" />
            <SummaryCard label="Method" value={result.method} />
            <SummaryCard label="Genotypes" value={result.n_genotypes} />
            <SummaryCard
              label="Avg silhouette"
              value={avgSil !== null ? fmt(avgSil, 3) : "—"}
              accent={avgSil !== null && avgSil >= 0.5 ? "emerald" : avgSil !== null && avgSil >= 0.25 ? "amber" : "muted"}
            />
          </div>

          {/* Cluster table */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="font-serif text-lg">Cluster summary</CardTitle>
              <ExportToolbar
                onCsv={() => downloadCsv("cluster_assignments.csv", result.cluster_assignments as unknown as Record<string, unknown>[])}
                onCopy={() => navigator.clipboard.writeText(result.interpretation)}
              />
            </CardHeader>
            <CardContent className="space-y-3">
              {result.cluster_summary.map((c) => {
                const members = membersByCluster.get(c.cluster_id) ?? [];
                const color = CLUSTER_PALETTE[(c.cluster_id - 1) % CLUSTER_PALETTE.length];
                return (
                  <Collapsible key={c.cluster_id} className="border rounded-lg">
                    <CollapsibleTrigger className="w-full flex items-center justify-between p-3 hover:bg-muted/40 transition-colors">
                      <div className="flex items-center gap-3">
                        <span className="h-3 w-3 rounded-sm" style={{ background: color }} />
                        <span className="font-medium">Cluster {c.cluster_id}</span>
                        <Badge variant="secondary">{c.size} genotypes</Badge>
                      </div>
                      <ChevronDown className="h-4 w-4 text-muted-foreground" />
                    </CollapsibleTrigger>
                    <CollapsibleContent className="p-3 pt-0 space-y-3">
                      <div className="text-xs text-muted-foreground">
                        <span className="font-medium text-foreground">Members: </span>
                        {members.join(", ")}
                      </div>
                      <div className="overflow-x-auto">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Trait</TableHead>
                              <TableHead className="text-right">Mean</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {Object.entries(c.mean_per_trait).map(([t, v]) => (
                              <TableRow key={t}>
                                <TableCell>{t}</TableCell>
                                <TableCell className="text-right tabular-nums">{fmt(v, 3)}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                );
              })}
            </CardContent>
          </Card>

          {/* Silhouette */}
          {silData.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">Silhouette per genotype</CardTitle>
                <ExportToolbar onPng={() => exportChartPng(silRef.current, "silhouette.png")} />
              </CardHeader>
              <CardContent>
                <div ref={silRef} className="w-full h-[420px]">
                  <ResponsiveContainer>
                    <BarChart data={silData} layout="vertical" margin={{ left: 80, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" domain={[-1, 1]} />
                      <YAxis type="category" dataKey="genotype" width={120} tick={{ fontSize: 10 }} />
                      <Tooltip />
                      <Bar dataKey="score">
                        {silData.map((d, i) => (
                          <Cell key={i} fill={CLUSTER_PALETTE[(d.cluster - 1) % CLUSTER_PALETTE.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Radar profiles */}
          {radarData.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">Cluster trait profiles</CardTitle>
                <ExportToolbar onPng={() => exportChartPng(radarRef.current, "cluster_radar.png")} />
              </CardHeader>
              <CardContent>
                <div ref={radarRef} className="w-full h-[420px]">
                  <ResponsiveContainer>
                    <RadarChart data={radarData} outerRadius="70%">
                      <PolarGrid />
                      <PolarAngleAxis dataKey="trait" tick={{ fontSize: 11 }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} />
                      {result.cluster_summary.map((c, i) => (
                        <Radar
                          key={c.cluster_id}
                          name={`Cluster ${c.cluster_id}`}
                          dataKey={`C${c.cluster_id}`}
                          stroke={CLUSTER_PALETTE[i % CLUSTER_PALETTE.length]}
                          fill={CLUSTER_PALETTE[i % CLUSTER_PALETTE.length]}
                          fillOpacity={0.2}
                        />
                      ))}
                      <Legend />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Trait values normalized to 0–100 across clusters for visual comparison.</p>
              </CardContent>
            </Card>
          )}

          {!result.dendrogram_data && (
            <Card className="bg-muted/40">
              <CardContent className="p-4 text-xs text-muted-foreground">
                Dendrogram visualization not available for this response. Use the cluster summary and silhouette plot above.
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
