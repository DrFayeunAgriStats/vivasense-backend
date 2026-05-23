import { useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  BarChart, Bar, ComposedChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, ReferenceLine, Legend, LabelList,
} from "recharts";
import { Loader2, Play, AlertTriangle, Compass } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { runPca } from "@/lib/advancedAnalysisApi";
import type { DatasetContext } from "@/types/geneticsUpload";
import type { PcaResponse } from "@/types/advancedAnalysis";
import {
  fmt, SummaryCard, InterpretationPanel, ExportToolbar, downloadCsv, exportChartPng, DatasetTokenWarning,
} from "./shared";

interface Props { datasetContext: DatasetContext | null; }

export function PcaPanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const allTraits = datasetContext?.availableTraitColumns ?? [];
  const [traits, setTraits] = useState<string[]>(allTraits);
  const [standardize, setStandardize] = useState(true);
  const [nComponents, setNComponents] = useState<number>(0); // 0 = let backend decide
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PcaResponse | null>(null);
  const [pcA, setPcA] = useState(0); // index 0 = PC1
  const [pcB, setPcB] = useState(1); // index 1 = PC2
  const screeRef = useRef<HTMLDivElement>(null);
  const biplotRef = useRef<HTMLDivElement>(null);

  const datasetToken = datasetContext?.datasetToken ?? null;

  const handleRun = async () => {
    if (!datasetToken) return toast({ title: "Dataset required" });
    if (traits.length < 2) return toast({ title: "Pick at least 2 traits", description: "PCA needs ≥2 variables." });
    setIsRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await runPca({
        dataset_token: datasetToken,
        trait_columns: traits,
        scale: standardize,
        n_components: nComponents > 0 ? nComponents : null,
      });
      if (res.status !== "success") throw new Error("PCA failed on the server.");
      setResult(res);
      setPcA(0);
      setPcB(Math.min(1, res.variance_explained.length - 1));
      toast({ title: "PCA complete", description: `${res.n_traits} traits × ${res.n_genotypes} genotypes` });
    } catch (e) {
      const msg = (e as Error).message ?? "Unexpected error";
      setError(msg);
      toast({ title: "Analysis failed", description: msg, variant: "destructive" });
    } finally { setIsRunning(false); }
  };

  // Detect whether backend returns proportions (0..1) or percentages (0..100)
  const varScale = useMemo(() => {
    if (!result) return 1;
    const maxV = Math.max(...result.variance_explained, ...(result.cumulative_variance ?? [0]));
    return maxV > 1 ? 1 : 100;
  }, [result]);
  const toPct = (v: number) => v * varScale;

  const screeData = useMemo(() => {
    if (!result) return [];
    return result.variance_explained.map((v, i) => ({
      pc: `PC${i + 1}`,
      variance: toPct(v),
      cumulative: toPct(result.cumulative_variance[i] ?? 0),
    }));
  }, [result, varScale]);

  const biplot = useMemo(() => {
    if (!result) return null;
    const src = result.biplot_data ?? { loadings: result.loadings, scores: result.scores };
    const nPCs = result.variance_explained.length;
    const a = Math.min(pcA, nPCs - 1);
    const b = Math.min(pcB, nPCs - 1);
    const scores = src.scores.map((s) => ({
      genotype: s.genotype,
      x: s.scores[a] ?? 0,
      y: s.scores[b] ?? 0,
    }));
    // scale loadings to score range for visual overlay
    const maxScore = Math.max(1e-9, ...scores.flatMap((p) => [Math.abs(p.x), Math.abs(p.y)]));
    const loadingEntries = Object.entries(src.loadings);
    const maxLoad = Math.max(1e-9, ...loadingEntries.flatMap(([, l]) => [Math.abs(l[a] ?? 0), Math.abs(l[b] ?? 0)]));
    const scale = (maxScore / maxLoad) * 0.8;
    const loadings = loadingEntries.map(([trait, l]) => ({
      trait,
      x: (l[a] ?? 0) * scale,
      y: (l[b] ?? 0) * scale,
    }));
    return { scores, loadings, a, b };
  }, [result, pcA, pcB]);

  const loadingsRows = useMemo(() => {
    if (!result) return [];
    return Object.entries(result.loadings).map(([trait, vals]) => ({
      trait, ...Object.fromEntries(vals.map((v, i) => [`PC${i + 1}`, v])),
    }));
  }, [result]);

  const toggleTrait = (t: string) => {
    setTraits((prev) => prev.includes(t) ? prev.filter((x) => x !== t) : [...prev, t]);
  };

  if (!datasetToken) return <DatasetTokenWarning />;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="font-serif text-xl flex items-center gap-2">
            <Compass className="h-5 w-5 text-primary" /> Principal Component Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Reduce trait dimensionality and reveal multivariate patterns of genotype variation.
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
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
            <label className="flex items-center gap-2 text-sm">
              <Checkbox checked={standardize} onCheckedChange={(v) => setStandardize(!!v)} />
              Standardize traits (z-score)
            </label>
            <div>
              <Label className="text-xs uppercase tracking-wide">Number of PCs (0 = auto)</Label>
              <div className="flex items-center gap-3">
                <Slider value={[nComponents]} max={Math.max(2, allTraits.length)} step={1} onValueChange={(v) => setNComponents(v[0])} className="flex-1" />
                <span className="text-sm tabular-nums w-8 text-right">{nComponents || "auto"}</span>
              </div>
            </div>
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
            <SummaryCard label="Traits" value={result.n_traits} />
            <SummaryCard label="Genotypes" value={result.n_genotypes} />
            <SummaryCard label="PC1 variance" value={`${toPct(result.variance_explained[0]).toFixed(1)}%`} accent="emerald" />
            <SummaryCard label="PC1+PC2" value={`${toPct(result.cumulative_variance[1] ?? result.variance_explained[0]).toFixed(1)}%`} accent="emerald" />
          </div>

          {/* Scree plot */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="font-serif text-lg">Scree plot</CardTitle>
              <ExportToolbar onPng={() => exportChartPng(screeRef.current, "pca_scree.png")} />
            </CardHeader>
            <CardContent>
              <div ref={screeRef} className="w-full h-[300px]">
                <ResponsiveContainer>
                  <ComposedChart data={screeData} margin={{ top: 10, right: 30, bottom: 10, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="pc" />
                    <YAxis yAxisId="l" label={{ value: "% variance", angle: -90, position: "insideLeft" }} />
                    <YAxis yAxisId="r" orientation="right" domain={[0, 100]} label={{ value: "Cumulative %", angle: 90, position: "insideRight" }} />
                    <Tooltip />
                    <Legend />
                    <Bar yAxisId="l" dataKey="variance" name="Variance %" fill="#0A7F5A" />
                    <Line yAxisId="r" dataKey="cumulative" name="Cumulative %" stroke="hsl(var(--primary))" strokeWidth={2} dot />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Biplot */}
          {biplot && result.variance_explained.length >= 2 && (
            <Card>
              <CardHeader className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                <CardTitle className="font-serif text-lg">Biplot (PC{biplot.a + 1} vs PC{biplot.b + 1})</CardTitle>
                <div className="flex flex-wrap items-center gap-2">
                  <Select value={String(pcA)} onValueChange={(v) => setPcA(Number(v))}>
                    <SelectTrigger className="w-28"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {result.variance_explained.map((_, i) => <SelectItem key={i} value={String(i)}>PC{i + 1}</SelectItem>)}
                    </SelectContent>
                  </Select>
                  <span className="text-xs text-muted-foreground">vs</span>
                  <Select value={String(pcB)} onValueChange={(v) => setPcB(Number(v))}>
                    <SelectTrigger className="w-28"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {result.variance_explained.map((_, i) => <SelectItem key={i} value={String(i)}>PC{i + 1}</SelectItem>)}
                    </SelectContent>
                  </Select>
                  <ExportToolbar onPng={() => exportChartPng(biplotRef.current, "pca_biplot.png")} />
                </div>
              </CardHeader>
              <CardContent>
                <div ref={biplotRef} className="w-full h-[460px]">
                  <ResponsiveContainer>
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" dataKey="x" name={`PC${biplot.a + 1}`} label={{ value: `PC${biplot.a + 1} (${toPct(result.variance_explained[biplot.a]).toFixed(1)}%)`, position: "insideBottom", offset: -5 }} />
                      <YAxis type="number" dataKey="y" name={`PC${biplot.b + 1}`} label={{ value: `PC${biplot.b + 1} (${toPct(result.variance_explained[biplot.b]).toFixed(1)}%)`, angle: -90, position: "insideLeft" }} />
                      <ZAxis range={[60, 60]} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v: number) => v.toFixed(3)} />
                      <ReferenceLine x={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
                      <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
                      <Scatter name="Genotypes" data={biplot.scores} fill="#0A7F5A">
                        <LabelList dataKey="genotype" position="top" style={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }} />
                      </Scatter>
                      <Scatter name="Trait loadings" data={biplot.loadings} fill="#DC2626" shape="triangle">
                        <LabelList dataKey="trait" position="right" style={{ fontSize: 11, fill: "#DC2626", fontWeight: 600 }} />
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                <p className="text-xs text-muted-foreground mt-2">Triangles are trait loading vectors (scaled for overlay); circles are genotype scores.</p>
              </CardContent>
            </Card>
          )}

          {/* Loadings table */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="font-serif text-lg">Trait loadings</CardTitle>
              <div className="flex flex-wrap gap-2">
                <ExportToolbar onCsv={() => downloadCsv("pca_loadings.csv", loadingsRows as Record<string, unknown>[])} csvLabel="Download loadings" />
                <ExportToolbar onCsv={() => downloadCsv("pca_scores.csv", result.scores.map(s => ({ genotype: s.genotype, ...Object.fromEntries(s.scores.map((v, i) => [`PC${i + 1}`, v])) })) as Record<string, unknown>[])} csvLabel="Download scores" />
              </div>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Trait</TableHead>
                    {result.variance_explained.map((_, i) => <TableHead key={i} className="text-right">PC{i + 1}</TableHead>)}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {Object.entries(result.loadings).map(([trait, vals]) => (
                    <TableRow key={trait}>
                      <TableCell className="font-medium">{trait}</TableCell>
                      {vals.map((v, i) => (
                        <TableCell key={i} className="text-right tabular-nums" style={{ background: loadingHeat(v) }}>{fmt(v, 3)}</TableCell>
                      ))}
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

function loadingHeat(v: number): string {
  // map -1..1 → red..white..green at low alpha
  const a = Math.min(1, Math.abs(v));
  if (v >= 0) return `rgba(10, 127, 90, ${a * 0.22})`;
  return `rgba(220, 38, 38, ${a * 0.22})`;
}
