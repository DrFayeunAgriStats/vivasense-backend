import { forwardRef, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Loader2, Play, AlertTriangle, GitBranch } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { runPathAnalysis, buildPathAnalysisPayload } from "@/lib/advancedAnalysisApi";
import type { DatasetContext } from "@/types/geneticsUpload";
import type { PathAnalysisResponse, PathAnalysisMethod, PathDiagramEdge } from "@/types/advancedAnalysis";
import {
  fmt, SummaryCard, InterpretationPanel, ExportToolbar, downloadCsv, DatasetTokenWarning,
} from "./shared";

interface Props { datasetContext: DatasetContext | null; }

// ── Defensive normalization helpers ─────────────────────────────────────────
const asArray = <T,>(v: unknown): T[] => (Array.isArray(v) ? (v as T[]) : []);
const asObj = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
const getEdgeSource = (edge: unknown): string => {
  const e = asObj(edge);
  return String(e.source ?? e.from ?? "");
};
const getEdgeTarget = (edge: unknown): string => {
  const e = asObj(edge);
  return String(e.target ?? e.to ?? "");
};
const asNumber = (v: unknown, fallback = 0): number => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};

export function PathAnalysisPanel({ datasetContext }: Props) {
  const { toast } = useToast();
  const allTraits = datasetContext?.availableTraitColumns ?? [];
  const [outcome, setOutcome] = useState<string>(allTraits[0] ?? "");
  const [predictors, setPredictors] = useState<string[]>(allTraits.slice(1));
  const [method, setMethod] = useState<PathAnalysisMethod>("correlation");
  const [standardize, setStandardize] = useState<boolean>(true);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PathAnalysisResponse | null>(null);
  const diagramRef = useRef<SVGSVGElement>(null);
  const datasetToken = datasetContext?.datasetToken ?? null;

  const togglePredictor = (t: string) => {
    if (t === outcome) return;
    setPredictors((p) => p.includes(t) ? p.filter((x) => x !== t) : [...p, t]);
  };

  const handleRun = async () => {
    if (!datasetToken) return toast({ title: "Please upload a dataset first." });
    if (!outcome) return toast({ title: "Select an outcome trait." });
    if (predictors.length < 2) return toast({ title: "Select at least two predictor traits." });
    if (predictors.includes(outcome)) {
      return toast({ title: "Outcome cannot also be a predictor." });
    }
    setIsRunning(true); setError(null); setResult(null);
    try {
      const res = await runPathAnalysis(buildPathAnalysisPayload({
        datasetToken,
        outcomeTrait: outcome,
        predictorTraits: predictors,
        method, standardize,
      }));
      // eslint-disable-next-line no-console
      console.log("[PATH RESPONSE]", res);
      if (res.status !== "success") throw new Error("Path analysis failed on the server.");
      setResult(res);
      toast({ title: "Path analysis complete" });
    } catch (e) {
      const raw = (e as Error).message ?? "Unexpected error";
      const msg = /singular|constant|missing|rank/i.test(raw)
        ? "This analysis could not be completed. Check trait type, missing values, or constant columns."
        : raw;
      setError(msg);
      toast({ title: "Analysis failed", description: msg, variant: "destructive" });
    } finally { setIsRunning(false); }
  };

  // ── Defensive normalization helpers ──────────────────────────────────────
  // Backend schema:
  //   { status, outcome_trait, predictor_traits, n_observations,
  //     path_coefficients, correlation_decomposition, r_squared,
  //     residual_effect, indirect_effects, interpretation, path_diagram_data }
  const r = (result ?? {}) as Partial<PathAnalysisResponse> & Record<string, unknown>;

  const pathCoefficients = useMemo(
    () => asArray<NonNullable<PathAnalysisResponse["path_coefficients"]>[number]>(r.path_coefficients),
    [r.path_coefficients]
  );
  const decomposition = useMemo(() => {
    const primary = asArray<Record<string, unknown>>(r.correlation_decomposition);
    if (primary.length > 0) return primary;
    return asArray<Record<string, unknown>>((r as Record<string, unknown>).decomposition); // legacy
  }, [r.correlation_decomposition, (r as Record<string, unknown>).decomposition]);
  const responsePredictors = useMemo(
    () => asArray<string>(r.predictor_traits),
    [r.predictor_traits]
  );
  const responseOutcome =
    (r.outcome_trait as string | undefined) ??
    (r.outcome as string | undefined) ??
    outcome;

  const diagramRaw = asObj(r.path_diagram_data);
  const explicitNodes = asArray<Record<string, unknown>>(diagramRaw.nodes);
  const explicitEdges = asArray<Record<string, unknown>>(diagramRaw.edges);

  const normalizedEdges = useMemo<PathDiagramEdge[]>(() => {
    if (explicitEdges.length > 0) {
      return explicitEdges.map((e, i) => ({
        ...(e as Record<string, unknown>),
        source: getEdgeSource(e),
        target: getEdgeTarget(e),
        weight: asNumber(e.weight ?? e.direct_effect, 0),
        significant: Boolean(e.significant),
      })) as PathDiagramEdge[];
    }

    return pathCoefficients.map((p, i) => {
      const row = asObj(p);
      return {
        source: String(row.predictor ?? `predictor-${i + 1}`),
        target: responseOutcome,
        weight: asNumber(row.direct_effect, 0),
        significant: Boolean(row.significant),
      } as PathDiagramEdge;
    });
  }, [explicitEdges, pathCoefficients, responseOutcome]);

  const diagram = useMemo(() => {
    if (!result || normalizedEdges.length === 0) return null;
    return { edges: normalizedEdges, nodes: explicitNodes, outcome: responseOutcome };
  }, [result, responseOutcome, normalizedEdges, explicitNodes]);

  const diagramPredictors = useMemo(
    () => asArray<PathDiagramEdge>(diagram?.edges).map((edge, i) => {
      const edgeObj = asObj(edge);
      return {
        name: getEdgeSource(edge) || `Predictor ${i + 1}`,
        weight: asNumber(edgeObj.weight, 0),
        significant: Boolean(edgeObj.significant),
      };
    }),
    [diagram]
  );

  const indirectEffects = asObj(r.indirect_effects) as Record<string, unknown>;

  const indirectMatrix = useMemo(() => {
    const rowKeys = Object.keys(indirectEffects);
    if (rowKeys.length === 0) return null;
    const colKeys = Array.from(
      new Set(rowKeys.flatMap((k) => Object.keys(asObj(indirectEffects[k]))))
    );
    if (colKeys.length === 0) return null;
    return { rowKeys, colKeys, ie: indirectEffects };
  }, [indirectEffects]);

  if (result) {
    // eslint-disable-next-line no-console
    console.log("DEBUG PATH", {
      hasResponse: !!result,
      status: result?.status,
      pathCoefficientsType: typeof r.path_coefficients,
      corrDecompType: typeof r.correlation_decomposition,
      predictorTraitsType: typeof r.predictor_traits,
      indirectEffectsType: typeof r.indirect_effects,
      diagramType: typeof r.path_diagram_data,
      nodesType: typeof (asObj(r.path_diagram_data) as { nodes?: unknown }).nodes,
      edgesType: typeof (asObj(r.path_diagram_data) as { edges?: unknown }).edges,
    });
  }

  if (!datasetToken) return <DatasetTokenWarning />;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="font-serif text-xl flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-primary" /> Path Analysis
            <Badge variant="secondary" className="text-[10px] uppercase">Direct + Indirect</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          <p className="text-sm text-muted-foreground">
            Decompose the correlation of each predictor with the outcome into direct and indirect effects.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label className="text-xs uppercase tracking-wide">Outcome trait</Label>
              <Select value={outcome} onValueChange={(v) => { setOutcome(v); setPredictors((p) => p.filter((x) => x !== v)); }}>
                <SelectTrigger><SelectValue placeholder="Select outcome" /></SelectTrigger>
                <SelectContent>
                  {allTraits.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label className="text-xs uppercase tracking-wide">Method</Label>
              <Select value={method} onValueChange={(v) => setMethod(v as PathAnalysisMethod)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="correlation">Correlation matrix</SelectItem>
                  <SelectItem value="covariance">Covariance matrix</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <div>
            <Label className="text-xs uppercase tracking-wide block mb-2">Predictor traits (≥ 2)</Label>
            <div className="flex flex-wrap gap-2">
              {allTraits.filter((t) => t !== outcome).map((t) => {
                const on = predictors.includes(t);
                return (
                  <button key={t} type="button" onClick={() => togglePredictor(t)}
                    className={`px-3 py-1.5 rounded-md border text-sm transition-colors ${on ? "bg-primary/10 border-primary" : "bg-background border-border text-muted-foreground hover:border-primary/40"}`}>
                    {t}
                  </button>
                );
              })}
            </div>
          </div>
          <label className="flex items-center gap-2 text-sm">
            <Checkbox checked={standardize} onCheckedChange={(v) => setStandardize(!!v)} />
            Standardize variables (recommended)
          </label>
          <Button onClick={handleRun} disabled={isRunning || !outcome || predictors.length < 2} className="gap-2">
            {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            Run Path Analysis
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
            <SummaryCard label="Outcome" value={responseOutcome} />
            <SummaryCard label="R²" value={fmt(result.r_squared, 3)} accent="emerald" />
            <SummaryCard label="Residual effect" value={fmt(result.residual_effect, 3)} />
            <SummaryCard
              label="# predictors"
              value={
                responsePredictors.length > 0
                  ? responsePredictors.length
                  : (result.n_predictors ?? predictors.length)
              }
            />
          </div>

          {/* Path diagram */}
          {diagram && diagramPredictors.length > 0 ? (
            <Card>
              <CardHeader>
                <CardTitle className="font-serif text-lg">Path diagram</CardTitle>
              </CardHeader>
              <CardContent>
                <PathDiagram
                  ref={diagramRef}
                  predictors={diagramPredictors}
                  outcome={diagram.outcome}
                  residual={result.residual_effect}
                />
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="p-4 text-sm text-muted-foreground">
                Path diagram data is not available.
              </CardContent>
            </Card>
          )}

          {/* Path coefficients */}
          {pathCoefficients.length > 0 ? (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">Path coefficients</CardTitle>
                <ExportToolbar onCsv={() => downloadCsv("path_coefficients.csv", pathCoefficients as unknown as Record<string, unknown>[])} />
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Predictor</TableHead>
                      <TableHead className="text-right">Direct effect</TableHead>
                      <TableHead className="text-right">SE</TableHead>
                      <TableHead className="text-right">t</TableHead>
                      <TableHead className="text-right">p-value</TableHead>
                      <TableHead>Significant</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {pathCoefficients.map((r) => (
                      <TableRow key={r.predictor}>
                        <TableCell className="font-medium">{r.predictor}</TableCell>
                        <TableCell className="text-right tabular-nums">{fmt(r.direct_effect, 3)}</TableCell>
                        <TableCell className="text-right tabular-nums">{fmt(r.std_error, 3)}</TableCell>
                        <TableCell className="text-right tabular-nums">{fmt(r.t_statistic, 2)}</TableCell>
                        <TableCell className="text-right tabular-nums">{r.p_value == null ? "—" : (r.p_value < 0.001 ? "< 0.001" : r.p_value.toFixed(4))}</TableCell>
                        <TableCell>{r.significant ? <Badge className="bg-emerald-600 hover:bg-emerald-600">Yes</Badge> : <Badge variant="secondary">No</Badge>}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="p-4 text-sm text-muted-foreground">
                Path coefficients are not available for this response.
              </CardContent>
            </Card>
          )}

          {/* Decomposition */}
          {decomposition.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="font-serif text-lg">Correlation decomposition</CardTitle>
                <ExportToolbar onCsv={() => downloadCsv("path_decomposition.csv", decomposition as unknown as Record<string, unknown>[])} />
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Predictor</TableHead>
                      <TableHead className="text-right">Total r</TableHead>
                      <TableHead className="text-right">Direct</TableHead>
                      <TableHead className="text-right">Indirect</TableHead>
                      <TableHead className="text-right">% Direct</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {decomposition.map((row, i) => {
                      const d = row as Record<string, unknown>;
                      const predictor = String(d.predictor ?? `row-${i}`);
                      return (
                        <TableRow key={predictor}>
                          <TableCell className="font-medium">{predictor}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(d.total_correlation as number | undefined, 3)}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(d.direct_effect as number | undefined, 3)}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(d.indirect_effect as number | undefined, 3)}</TableCell>
                          <TableCell className="text-right tabular-nums">{fmt(d.percent_direct as number | undefined, 1)}</TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}

          {/* Indirect effects heatmap */}
          {indirectMatrix ? (
            <Card>
              <CardHeader>
                <CardTitle className="font-serif text-lg">Indirect effects via</CardTitle>
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Predictor</TableHead>
                      {indirectMatrix.colKeys.map((c) => <TableHead key={c} className="text-right">via {c}</TableHead>)}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {indirectMatrix.rowKeys.map((rk) => (
                      <TableRow key={rk}>
                        <TableCell className="font-medium">{rk}</TableCell>
                        {indirectMatrix.colKeys.map((ck) => {
                          const rowObj = asObj(indirectMatrix.ie[rk]);
                          const raw = rowObj[ck];
                          const v = typeof raw === "number" ? raw : (raw == null ? undefined : Number(raw));
                          return (
                            <TableCell key={ck} className="text-right tabular-nums" style={{ background: heatColor(v) }}>
                              {fmt(v, 3)}
                            </TableCell>
                          );
                        })}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="p-4 text-sm text-muted-foreground">
                No indirect effects available.
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function heatColor(v: number | undefined): string {
  if (v == null || Number.isNaN(v)) return "transparent";
  const a = Math.min(1, Math.abs(v));
  return v >= 0 ? `rgba(10,127,90,${a * 0.22})` : `rgba(220,38,38,${a * 0.22})`;
}

// ── SVG path diagram ───────────────────────────────────────────────────────

interface PathDiagramProps {
  predictors: { name: string; weight: number; significant?: boolean }[];
  outcome: string;
  residual?: number;
}

const PathDiagram = forwardRef<SVGSVGElement, PathDiagramProps>(function PathDiagram(
  { predictors, outcome, residual },
  ref
) {
  const safePredictors = asArray<PathDiagramProps["predictors"][number]>(predictors).map((p, i) => ({
    name: String(p?.name ?? `Predictor ${i + 1}`),
    weight: asNumber(p?.weight, 0),
    significant: p?.significant,
  }));
  const W = 760;
  const padding = 40;
  const rowH = 60;
  const H = Math.max(280, safePredictors.length * rowH + padding * 2);
  const leftX = 130;
  const rightX = W - 160;
  const cy = H / 2;
  const maxAbs = Math.max(0.0001, ...safePredictors.map((p) => Math.abs(p.weight)));
  return (
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
      <defs>
        <marker id="arrow-pos" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="hsl(var(--primary))" />
        </marker>
        <marker id="arrow-neg" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="hsl(var(--destructive))" />
        </marker>
        <marker id="arrow-residual" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="hsl(var(--muted-foreground))" />
        </marker>
      </defs>
      {safePredictors.map((p, i) => {
        const y = padding + i * ((H - padding * 2) / Math.max(1, safePredictors.length - 1 || 1));
        const stroke = p.weight >= 0 ? "hsl(var(--primary))" : "hsl(var(--destructive))";
        const widthLine = 1 + (Math.abs(p.weight) / maxAbs) * 4;
        const dash = p.significant === false ? "5 4" : undefined;
        return (
          <g key={p.name}>
            <rect x={20} y={y - 18} width={leftX - 30} height={36} rx={6} fill="hsl(var(--muted))" stroke="hsl(var(--border))" />
            <text x={(20 + leftX - 10) / 2} y={y + 4} textAnchor="middle" fontSize="12" fill="hsl(var(--foreground))">
              {truncate(p.name, 18)}
            </text>
            <line
              x1={leftX} y1={y}
              x2={rightX} y2={cy}
              stroke={stroke} strokeWidth={widthLine}
              markerEnd={p.weight >= 0 ? "url(#arrow-pos)" : "url(#arrow-neg)"}
              strokeDasharray={dash}
            />
            <text
              x={(leftX + rightX) / 2}
              y={(y + cy) / 2 - 4}
              fontSize="11" fill={stroke}
              textAnchor="middle"
            >{p.weight.toFixed(2)}</text>
          </g>
        );
      })}
      <rect x={rightX} y={cy - 22} width={140} height={44} rx={8} fill="hsl(var(--primary) / 0.1)" stroke="hsl(var(--primary))" />
      <text x={rightX + 70} y={cy + 5} textAnchor="middle" fontSize="13" fontWeight="600" fill="hsl(var(--primary))">
        {truncate(outcome, 18)}
      </text>
      {typeof residual === "number" && (
        <g>
          <line x1={rightX + 70} y1={20} x2={rightX + 70} y2={cy - 22}
            stroke="hsl(var(--muted-foreground))" strokeWidth={1.5} strokeDasharray="6 4" markerEnd="url(#arrow-residual)" />
          <text x={rightX + 76} y={36} fontSize="11" fill="hsl(var(--muted-foreground))">residual = {residual.toFixed(2)}</text>
        </g>
      )}
    </svg>
  );
});

function truncate(s: unknown, n: number) {
  const text = String(s ?? "");
  return text.length > n ? text.slice(0, n - 1) + "…" : text;
}
