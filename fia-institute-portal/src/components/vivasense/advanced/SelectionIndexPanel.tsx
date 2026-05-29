import { useEffect, useMemo, useRef, useState } from "react";
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
  BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarAngleAxis, PolarGrid, PolarRadiusAxis,
} from "recharts";
import { Loader2, Play, AlertTriangle, Award } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { runSelectionIndex, buildSelectionIndexPayload } from "@/lib/advancedAnalysisApi";
import type { DatasetContext } from "@/types/geneticsUpload";
import type { SelectionIndexResponse } from "@/types/advancedAnalysis";
import {
  fmt, SummaryCard, InterpretationPanel, ExportToolbar, downloadCsv, exportChartPng, DatasetTokenWarning,
} from "./shared";

interface Props { datasetContext: DatasetContext | null; }

const INTENSITY_PRESETS = [
  { label: "Top 10% (i = 1.755)", value: 1.755 },
  { label: "Top 5% (i = 2.063)", value: 2.063 },
  { label: "Top 1% (i = 2.665)", value: 2.665 },
];

export function SelectionIndexPanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const allTraits = datasetContext?.availableTraitColumns ?? [];
  const [traitCols, setTraitCols] = useState<string[]>(allTraits);
  const [weights, setWeights] = useState<Record<string, string>>({});
  const [heritability, setHeritability] = useState<Record<string, string>>({});
  const [intensityPreset, setIntensityPreset] = useState<string>("1.755");
  const [customIntensity, setCustomIntensity] = useState<string>("");
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SelectionIndexResponse | null>(null);
  const weightsChartRef = useRef<HTMLDivElement>(null);
  const radarRef = useRef<HTMLDivElement>(null);
  const datasetToken = datasetContext?.datasetToken ?? null;

  useEffect(() => {
    setWeights((prev) => {
      const next: Record<string, string> = {};
      traitCols.forEach((t) => { next[t] = prev[t] ?? "1"; });
      return next;
    });
  }, [traitCols]);

  const toggleTrait = (t: string) => {
    setTraitCols((p) => p.includes(t) ? p.filter((x) => x !== t) : [...p, t]);
  };

  const intensity = useMemo(() => {
    if (intensityPreset === "custom") {
      const n = Number(customIntensity);
      return Number.isFinite(n) ? n : NaN;
    }
    return Number(intensityPreset);
  }, [intensityPreset, customIntensity]);

  const validateWeights = (): { ok: boolean; parsed: Record<string, number>; errMsg?: string } => {
    const parsed: Record<string, number> = {};
    for (const t of traitCols) {
      const raw = weights[t];
      const n = Number(raw);
      if (raw === "" || raw == null || !Number.isFinite(n)) {
        return { ok: false, parsed, errMsg: `Economic weight for "${t}" is required and must be numeric.` };
      }
      parsed[t] = n;
    }
    return { ok: true, parsed };
  };

  const handleRun = async () => {
    if (!datasetToken) return toast({ title: "Please upload a dataset first." });
    if (traitCols.length < 2) return toast({ title: "Select at least two numeric traits." });
    const v = validateWeights();
    if (!v.ok) {
      toast({ title: "Economic weights are required for all selected traits.", description: v.errMsg, variant: "destructive" });
      return;
    }
    if (!Number.isFinite(intensity) || intensity <= 0) {
      return toast({ title: "Selection intensity must be a positive number." });
    }
    // Optional heritability
    let h2: Record<string, number> | undefined;
    const h2Entries = Object.entries(heritability).filter(([, val]) => val !== "" && val != null);
    if (h2Entries.length > 0) {
      h2 = {};
      for (const [t, raw] of h2Entries) {
        const n = Number(raw);
        if (!Number.isFinite(n) || n < 0 || n > 1) {
          return toast({ title: `Heritability for "${t}" must be between 0 and 1.` });
        }
        h2[t] = n;
      }
    }

    setIsRunning(true); setError(null); setResult(null);
    try {
      const res = await runSelectionIndex(buildSelectionIndexPayload({
        datasetToken,
        traitColumns: traitCols,
        economicWeights: v.parsed,
        geneticParameters: h2,
        selectionIntensity: intensity,
      }));
      // eslint-disable-next-line no-console
      console.log("[selection-index response]", res);
      if (res.status !== "success") throw new Error("Selection index failed on the server.");
      setResult(res);
      toast({ title: "Selection index complete" });
    } catch (e) {
      const raw = (e as Error).message ?? "Unexpected error";
      const msg = /singular|constant|missing|rank/i.test(raw)
        ? "This analysis could not be completed. Check trait type, missing values, or constant columns."
        : raw;
      setError(msg);
      toast({ title: "Analysis failed", description: msg, variant: "destructive" });
    } finally { setIsRunning(false); }
  };

  const indexWeightsData = useMemo(() => {
    return (result?.index_weights ?? []).map((w) => ({ trait: w.trait, weight: w.weight }));
  }, [result]);

  const expectedGainData = useMemo(() => {
    return (result?.expected_gain ?? []).map((g) => ({ trait: g.trait, gain: g.expected_gain }));
  }, [result]);

  // Defensive: derive selected genotypes if missing
  const selectedSet = useMemo(() => {
    if (Array.isArray(result?.selected_genotypes)) return new Set(result!.selected_genotypes);
    if (Array.isArray(result?.rankings)) {
      const sel = result!.rankings.filter((r) => r.selected).map((r) => r.genotype);
      return new Set(sel);
    }
    return new Set<string>();
  }, [result]);

  if (!datasetToken) return <DatasetTokenWarning />;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="font-serif text-xl flex items-center gap-2">
            <Award className="h-5 w-5 text-primary" /> Selection Index
            <Badge variant="secondary" className="text-[10px] uppercase">Multi-trait</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <p className="text-sm text-muted-foreground">
            Smith–Hazel style multi-trait genotype ranking with economic weights and optional genetic parameters.
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

          {traitCols.length > 0 && (
            <div>
              <Label className="text-xs uppercase tracking-wide block mb-2">Economic weights & h² (optional)</Label>
              <div className="overflow-x-auto rounded border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Trait</TableHead>
                      <TableHead className="w-40">Economic weight *</TableHead>
                      <TableHead className="w-40">Heritability h² (0–1, optional)</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {traitCols.map((t) => (
                      <TableRow key={t}>
                        <TableCell className="font-medium">{t}</TableCell>
                        <TableCell>
                          <Input
                            type="number" step="0.01"
                            value={weights[t] ?? ""}
                            onChange={(e) => setWeights({ ...weights, [t]: e.target.value })}
                            placeholder="e.g. 1.0"
                          />
                        </TableCell>
                        <TableCell>
                          <Input
                            type="number" step="0.01" min="0" max="1"
                            value={heritability[t] ?? ""}
                            onChange={(e) => setHeritability({ ...heritability, [t]: e.target.value })}
                            placeholder="optional"
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label className="text-xs uppercase tracking-wide">Selection intensity</Label>
              <Select value={intensityPreset} onValueChange={setIntensityPreset}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {INTENSITY_PRESETS.map((p) => (
                    <SelectItem key={p.value} value={String(p.value)}>{p.label}</SelectItem>
                  ))}
                  <SelectItem value="custom">Custom value…</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {intensityPreset === "custom" && (
              <div>
                <Label className="text-xs uppercase tracking-wide">Custom intensity</Label>
                <Input type="number" step="0.001" min="0" value={customIntensity} onChange={(e) => setCustomIntensity(e.target.value)} placeholder="e.g. 1.4" />
              </div>
            )}
          </div>

          <Button onClick={handleRun} disabled={isRunning || traitCols.length < 2} className="gap-2">
            {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run Selection Index
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
            <SummaryCard label="Selection accuracy" value={fmt(result.selection_accuracy, 3)} accent="emerald" />
            <SummaryCard label="Total merit" value={fmt(result.total_merit, 3)} />
            <SummaryCard label="# Selected" value={result.n_selected ?? selectedSet.size ?? "—"} />
            <SummaryCard label="Top entry" value={result.top_genotype ?? (result.rankings?.[0]?.genotype ?? "—")} />
          </div>

          {indexWeightsData.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">Index weights</CardTitle>
                <ExportToolbar
                  onCsv={() => downloadCsv("selection_index_weights.csv", result.index_weights as unknown as Record<string, unknown>[])}
                  onPng={() => exportChartPng(weightsChartRef.current, "selection_index_weights.png")}
                />
              </CardHeader>
              <CardContent>
                <div ref={weightsChartRef} className="w-full h-[300px]">
                  <ResponsiveContainer>
                    <BarChart data={indexWeightsData} margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="trait" />
                      <YAxis label={{ value: "Weight", angle: -90, position: "insideLeft" }} />
                      <Tooltip formatter={(v: number) => v.toFixed(3)} />
                      <Bar dataKey="weight" fill="#0A7F5A" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {Array.isArray(result.rankings) && result.rankings.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">Genotype rankings</CardTitle>
                <ExportToolbar onCsv={() => downloadCsv("selection_rankings.csv", result.rankings as unknown as Record<string, unknown>[])} />
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-16">Rank</TableHead>
                      <TableHead>Genotype</TableHead>
                      <TableHead className="text-right">Index value</TableHead>
                      <TableHead>Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.rankings.map((r) => {
                      const isSel = r.selected ?? selectedSet.has(r.genotype);
                      return (
                        <TableRow key={r.genotype} className={isSel ? "bg-primary/5" : ""}>
                          <TableCell className="tabular-nums">{r.rank}</TableCell>
                          <TableCell className="font-medium">{r.genotype}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(r.index_value, 3)}</TableCell>
                          <TableCell>
                            {isSel ? <Badge className="bg-emerald-600 hover:bg-emerald-600">Selected</Badge> : <Badge variant="secondary">—</Badge>}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          {expectedGainData.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">Expected genetic gain</CardTitle>
                <ExportToolbar
                  onCsv={() => downloadCsv("selection_expected_gain.csv", result.expected_gain as unknown as Record<string, unknown>[])}
                  onPng={() => exportChartPng(radarRef.current, "selection_expected_gain.png")}
                />
              </CardHeader>
              <CardContent>
                <div ref={radarRef} className="w-full h-[360px]">
                  <ResponsiveContainer>
                    <RadarChart data={expectedGainData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="trait" />
                      <PolarRadiusAxis />
                      <Radar name="Expected gain" dataKey="gain" stroke="#0A7F5A" fill="#0A7F5A" fillOpacity={0.3} />
                      <Tooltip formatter={(v: number) => v.toFixed(3)} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {Array.isArray(result.relative_efficiency) && result.relative_efficiency.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">Relative efficiency</CardTitle>
                <ExportToolbar onCsv={() => downloadCsv("selection_relative_efficiency.csv", result.relative_efficiency as unknown as Record<string, unknown>[])} />
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Trait</TableHead>
                      <TableHead className="text-right">Efficiency</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.relative_efficiency.map((r) => (
                      <TableRow key={r.trait}>
                        <TableCell className="font-medium">{r.trait}</TableCell>
                        <TableCell className="text-right tabular-nums">{fmt(r.efficiency, 3)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          {selectedSet.size > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="font-serif text-lg">Selected genotypes</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-2">
                  {Array.from(selectedSet).map((g) => (
                    <Badge key={g} className="bg-emerald-600 hover:bg-emerald-600">{g}</Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
