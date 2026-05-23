import { useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Legend, Customized,
} from "recharts";
import { Crown, Maximize2, Star } from "lucide-react";
import { fmt, InterpretationPanel, ExportToolbar, downloadCsv, exportChartPng } from "../shared";
import type { GGEResults } from "@/types/advancedAnalysis";
import { convexHull } from "./geometry";
import {
  computeAEC,
  computeIdealGenotype,
  drawAbscissaAxis,
  drawStabilityAxis,
  projectGenotypesToAEC,
} from "./ggeMeanStability";

export function GgeTab({
  gge,
  trait,
  biplotType,
}: {
  gge: GGEResults;
  trait: string;
  biplotType: "which-won-where" | "mean-stability" | "discriminativeness";
}) {
  const biplotRef = useRef<HTMLDivElement>(null);
  const [chartZoom, setChartZoom] = useState(1);

  const variance = useMemo(
    () => (Array.isArray(gge?.variance_explained) ? gge.variance_explained : []),
    [gge?.variance_explained]
  );
  const varScale = useMemo(() => {
    const cum = typeof gge?.cumulative_variance === "number" ? gge.cumulative_variance : 0;
    if (variance.length === 0 && cum === 0) return 100;
    const maxV = Math.max(...(variance.length ? variance : [0]), cum);
    return maxV > 1 ? 1 : 100;
  }, [variance, gge?.cumulative_variance]);
  const toPct = (v: number) => (typeof v === "number" ? v * varScale : 0);
  const pc1Var = toPct(variance[0] ?? 0);
  const pc2Var = toPct(variance[1] ?? 0);
  const cumVar = toPct(gge?.cumulative_variance ?? 0);

  const rawGenos = Array.isArray(gge?.biplot_data?.genotypes) ? gge.biplot_data.genotypes.filter(Boolean) : [];
  const rawEnvs = Array.isArray(gge?.biplot_data?.environments) ? gge.biplot_data.environments.filter(Boolean) : [];
  const biplotGenos = rawGenos.map((g) => ({ ...g, kind: "genotype" as const }));
  const biplotEnvs = rawEnvs.map((e) => ({ ...e, kind: "environment" as const, code: e.environment }));
  const hasBiplot = biplotGenos.length > 0 || biplotEnvs.length > 0;

  const meanStabilityModel = useMemo(() => {
    if (biplotType !== "mean-stability" || !biplotGenos.length || !biplotEnvs.length) return null;

    const aec = computeAEC(
      biplotEnvs.map((e) => ({ environment: e.environment, mean: e.mean, pc1: e.pc1, pc2: e.pc2 })),
      biplotGenos.map((g) => ({ mean: g.mean, pc1: g.pc1, pc2: g.pc2 }))
    );
    const projections = projectGenotypesToAEC(
      biplotGenos.map((g) => ({ genotype: g.genotype, mean: g.mean, pc1: g.pc1, pc2: g.pc2 })),
      aec
    );
    const ideal = computeIdealGenotype(projections);
    const environmentProjections = biplotEnvs.map((e) => ({
      environment: e.environment,
      mean: e.mean,
      pc1: e.pc1,
      pc2: e.pc2,
      abscissaScore: e.pc1 * aec.abscissaUnit.x + e.pc2 * aec.abscissaUnit.y,
      ordinateScore: e.pc1 * aec.ordinateUnit.x + e.pc2 * aec.ordinateUnit.y,
    }));

    const maxAbs = Math.max(
      ...biplotGenos.flatMap((g) => [Math.abs(g.pc1), Math.abs(g.pc2)]),
      ...biplotEnvs.flatMap((e) => [Math.abs(e.pc1), Math.abs(e.pc2)]),
      Math.abs(ideal.idealPoint.x),
      Math.abs(ideal.idealPoint.y),
      1
    );

    return {
      aec,
      projections,
      environmentProjections,
      ideal,
      aecAbscissa: drawAbscissaAxis(aec, maxAbs),
      stabilityAxis: drawStabilityAxis(aec, maxAbs),
    };
  }, [biplotType, biplotEnvs, biplotGenos]);

  const chartGenos = useMemo(() => {
    if (biplotType === "mean-stability" && meanStabilityModel) {
      const byGeno = new Map(meanStabilityModel.projections.map((p) => [p.genotype, p]));
      return biplotGenos.map((g) => {
        const p = byGeno.get(g.genotype);
        return {
          ...g,
          xCoord: p?.abscissaScore ?? g.pc1,
          yCoord: p?.ordinateScore ?? g.pc2,
          aecAbscissa: p?.abscissaScore ?? null,
          aecOrdinate: p?.ordinateScore ?? null,
        };
      });
    }
    return biplotGenos.map((g) => ({ ...g, xCoord: g.pc1, yCoord: g.pc2 }));
  }, [biplotType, meanStabilityModel, biplotGenos]);

  const chartEnvs = useMemo(() => {
    if (biplotType === "mean-stability" && meanStabilityModel) {
      const byEnv = new Map(meanStabilityModel.environmentProjections.map((e) => [e.environment, e]));
      return biplotEnvs.map((e) => {
        const p = byEnv.get(e.environment);
        return {
          ...e,
          xCoord: p?.abscissaScore ?? e.pc1,
          yCoord: p?.ordinateScore ?? e.pc2,
        };
      });
    }
    return biplotEnvs.map((e) => ({ ...e, xCoord: e.pc1, yCoord: e.pc2 }));
  }, [biplotType, meanStabilityModel, biplotEnvs]);

  const xAxisLabel =
    biplotType === "mean-stability"
      ? "AEC abscissa (higher mean ->)"
      : `PC1 (${pc1Var.toFixed(1)}%)`;
  const yAxisLabel =
    biplotType === "mean-stability"
      ? "AEC ordinate (instability)"
      : `PC2 (${pc2Var.toFixed(1)}%)`;

  return (
    <div className="space-y-6">
      {gge?.interpretation && <InterpretationPanel text={gge.interpretation} />}

      {/* Variance summary */}
      <Card>
        <CardHeader>
          <CardTitle className="font-serif text-lg">Variance summary</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>PC1</span><span className="tabular-nums font-medium">{pc1Var.toFixed(1)}%</span>
            </div>
            <Progress value={pc1Var} />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1">
              <span>PC2</span><span className="tabular-nums font-medium">{pc2Var.toFixed(1)}%</span>
            </div>
            <Progress value={pc2Var} />
          </div>
          <p className="text-xs text-muted-foreground">
            Cumulative (PC1 + PC2): <span className="font-medium text-foreground">{cumVar.toFixed(1)}%</span>
            {cumVar < 50 && <span className="ml-2 text-amber-600">— below 50%, interpret with caution.</span>}
          </p>
        </CardContent>
      </Card>

      {/* Biplot */}
      {hasBiplot ? (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2">
          <CardTitle className="font-serif text-lg">
            GGE biplot — {biplotType === "which-won-where" ? "Which-Won-Where" : biplotType === "mean-stability" ? "Mean vs Stability" : "Discriminativeness"}
          </CardTitle>
          <div className="flex flex-wrap items-center gap-2">
            <ExportToolbar
              onCsv={() => {
                const rows = meanStabilityModel
                  ? biplotGenos.map((g) => {
                      const proj = meanStabilityModel.projections.find((p) => p.genotype === g.genotype);
                      const rank = meanStabilityModel.ideal.rankings.find((r) => r.genotype === g.genotype);
                      return {
                        genotype: g.genotype,
                        mean: g.mean,
                        pc1: g.pc1,
                        pc2: g.pc2,
                        aec_abscissa_score: proj?.abscissaScore ?? null,
                        aec_ordinate_score: proj?.ordinateScore ?? null,
                        stability_distance: proj?.stabilityDistance ?? null,
                        distance_from_ideal: rank?.distance_from_ideal ?? null,
                        ideal_rank: rank?.rank ?? null,
                      };
                    })
                  : (biplotGenos as unknown as Record<string, unknown>[]);
                downloadCsv(`gge_biplot_genotypes_${trait}.csv`, rows as unknown as Record<string, unknown>[]);
              }}
              csvLabel="Genotype scores CSV"
            />
            <ExportToolbar
              onCsv={() =>
                downloadCsv(`gge_biplot_environments_${trait}.csv`, biplotEnvs as unknown as Record<string, unknown>[])
              }
              csvLabel="Environment scores CSV"
            />
            <ExportToolbar onPng={() => exportChartPng(biplotRef.current, `gge_biplot_${trait}.png`)} />
            <Dialog>
              <DialogTrigger asChild>
                <Button type="button" variant="outline" size="sm" className="gap-2" onClick={() => setChartZoom(1.25)}>
                  <Maximize2 className="h-4 w-4" />
                  Enlarge
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-6xl w-[95vw] p-4 sm:p-6">
                <div className="sr-only">
                  <DialogTitle>Expanded GGE biplot</DialogTitle>
                  <DialogDescription>Zoomable workspace view for publication-oriented inspection of GGE output.</DialogDescription>
                </div>
                <div className="flex items-center justify-between gap-3 pb-3 border-b border-border">
                  <div>
                    <p className="text-sm font-semibold text-foreground">Expanded GGE view</p>
                    <p className="text-xs text-muted-foreground">Inspect labels, axes, and genotype-environment geometry at a larger scale.</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button type="button" variant="outline" size="sm" onClick={() => setChartZoom((prev) => Math.max(1, Number((prev - 0.15).toFixed(2))))}>−</Button>
                    <span className="min-w-[48px] text-center text-xs text-muted-foreground">{Math.round(chartZoom * 100)}%</span>
                    <Button type="button" variant="outline" size="sm" onClick={() => setChartZoom((prev) => Math.min(2.2, Number((prev + 0.15).toFixed(2))))}>+</Button>
                  </div>
                </div>
                <div className="max-h-[78vh] overflow-auto rounded-xl border border-border bg-background p-3">
                  <div style={{ width: `${chartZoom * 100}%`, minWidth: 920 }}>
                    <div className="h-[720px]">
                      <ResponsiveContainer>
                        <ScatterChart margin={{ top: 16, right: 30, bottom: 36, left: 16 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" dataKey="xCoord" domain={["auto", "auto"]}
                            label={{ value: xAxisLabel, position: "insideBottom", offset: -16 }} />
                          <YAxis type="number" dataKey="yCoord" domain={["auto", "auto"]}
                            label={{ value: yAxisLabel, angle: -90, position: "insideLeft" }} />
                          <ZAxis range={[80, 80]} />
                          <ReferenceLine x={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
                          <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
                          <Tooltip
                            cursor={{ strokeDasharray: "3 3" }}
                            content={({ active, payload }) => {
                              if (!active || !payload?.length) return null;
                              const d = payload[0].payload as {
                                genotype?: string;
                                environment?: string;
                                code?: string;
                                mean: number;
                                pc1: number;
                                pc2: number;
                                xCoord: number;
                                yCoord: number;
                                kind: string;
                              };
                              const name = d.genotype ?? d.environment ?? d.code ?? "—";
                              return (
                                <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                                  <div className="font-semibold">{name} <span className="text-muted-foreground font-normal">({d.kind})</span></div>
                                  <div>Mean: {fmt(d.mean, 2)}</div>
                                  {biplotType === "mean-stability" ? (
                                    <>
                                      <div>AEC abscissa: {fmt(d.xCoord, 3)}</div>
                                      <div>AEC ordinate: {fmt(d.yCoord, 3)}</div>
                                      <div className="text-muted-foreground">PC1: {fmt(d.pc1, 3)}; PC2: {fmt(d.pc2, 3)}</div>
                                    </>
                                  ) : (
                                    <>
                                      <div>PC1: {fmt(d.pc1, 3)}</div>
                                      <div>PC2: {fmt(d.pc2, 3)}</div>
                                    </>
                                  )}
                                </div>
                              );
                            }}
                          />
                          <Legend />
                          <Scatter name="Genotypes" data={chartGenos} fill="hsl(var(--primary))" shape="circle" />
                          <Scatter name="Environments" data={chartEnvs} fill="hsl(42 85% 45%)" shape="diamond" />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </CardHeader>
        <CardContent>
          <div ref={biplotRef} className="w-full h-[520px]">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 16, right: 30, bottom: 36, left: 16 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="xCoord" domain={["auto", "auto"]}
                  label={{ value: xAxisLabel, position: "insideBottom", offset: -16 }} />
                <YAxis type="number" dataKey="yCoord" domain={["auto", "auto"]}
                  label={{ value: yAxisLabel, angle: -90, position: "insideLeft" }} />
                <ZAxis range={[80, 80]} />
                <ReferenceLine x={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
                <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload as {
                      genotype?: string;
                      environment?: string;
                      code?: string;
                      mean: number;
                      pc1: number;
                      pc2: number;
                      xCoord: number;
                      yCoord: number;
                      kind: string;
                    };
                    const name = d.genotype ?? d.environment ?? d.code ?? "—";
                    return (
                      <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                        <div className="font-semibold">{name} <span className="text-muted-foreground font-normal">({d.kind})</span></div>
                        <div>Mean: {fmt(d.mean, 2)}</div>
                        {biplotType === "mean-stability" ? (
                          <>
                            <div>AEC abscissa: {fmt(d.xCoord, 3)}</div>
                            <div>AEC ordinate: {fmt(d.yCoord, 3)}</div>
                            <div className="text-muted-foreground">PC1: {fmt(d.pc1, 3)}; PC2: {fmt(d.pc2, 3)}</div>
                          </>
                        ) : (
                          <>
                            <div>PC1: {fmt(d.pc1, 3)}</div>
                            <div>PC2: {fmt(d.pc2, 3)}</div>
                          </>
                        )}
                      </div>
                    );
                  }}
                />
                <Legend />

                {/* Convex hull polygon for which-won-where */}
                {biplotType === "which-won-where" && (
                  <Customized
                    component={(props: { xAxisMap?: Record<string, { scale: (v: number) => number }>; yAxisMap?: Record<string, { scale: (v: number) => number }> }) => {
                      const xMap = props.xAxisMap;
                      const yMap = props.yAxisMap;
                      if (!xMap || !yMap) return null;
                      const xScale = Object.values(xMap)[0]?.scale;
                      const yScale = Object.values(yMap)[0]?.scale;
                      if (!xScale || !yScale) return null;
                      const hull = convexHull(biplotGenos.map((g) => ({ x: g.pc1, y: g.pc2, ref: g })));
                      if (hull.length < 3) return null;
                      const points = hull.map((p) => `${xScale(p.x)},${yScale(p.y)}`).join(" ");
                      const ox = xScale(0);
                      const oy = yScale(0);
                      // Spokes from origin to each hull vertex
                      return (
                        <g>
                          <polygon points={points} fill="hsl(var(--primary) / 0.06)" stroke="hsl(var(--primary))" strokeWidth={1.5} strokeDasharray="4 3" />
                          {hull.map((p, i) => (
                            <line key={i} x1={ox} y1={oy} x2={xScale(p.x)} y2={yScale(p.y)} stroke="hsl(var(--primary))" strokeWidth={1} opacity={0.4} />
                          ))}
                        </g>
                      );
                    }}
                  />
                )}

                {/* Ideal-genotype circle for mean-stability */}
                {biplotType === "mean-stability" && gge.mean_vs_stability && (
                  <Customized
                    component={(props: { xAxisMap?: Record<string, { scale: (v: number) => number }>; yAxisMap?: Record<string, { scale: (v: number) => number }> }) => {
                      const xMap = props.xAxisMap;
                      const yMap = props.yAxisMap;
                      if (!xMap || !yMap || !meanStabilityModel) return null;
                      const xScale = Object.values(xMap)[0]?.scale;
                      const yScale = Object.values(yMap)[0]?.scale;
                      if (!xScale || !yScale) return null;

                      const { ideal, projections } = meanStabilityModel;
                      const iCx = xScale(ideal.idealAbscissaScore);
                      const iCy = yScale(0);

                      const arrowSize = 7;
                      const toX = xScale(Math.max(...projections.map((p) => p.abscissaScore), 1));
                      const toY = yScale(0);
                      const fromX = xScale(Math.min(...projections.map((p) => p.abscissaScore), -1));
                      const fromY = yScale(0);

                      const angle = Math.atan2(toY - fromY, toX - fromX);
                      const leftX = toX - arrowSize * Math.cos(angle - Math.PI / 6);
                      const leftY = toY - arrowSize * Math.sin(angle - Math.PI / 6);
                      const rightX = toX - arrowSize * Math.cos(angle + Math.PI / 6);
                      const rightY = toY - arrowSize * Math.sin(angle + Math.PI / 6);

                      return (
                        <g>
                          <line
                            x1={fromX}
                            y1={fromY}
                            x2={toX}
                            y2={toY}
                            stroke="hsl(210 22% 35%)"
                            strokeWidth={1.8}
                            opacity={0.9}
                          />
                          <line
                            x1={xScale(0)}
                            y1={yScale(Math.min(...projections.map((p) => p.ordinateScore), -1))}
                            x2={xScale(0)}
                            y2={yScale(Math.max(...projections.map((p) => p.ordinateScore), 1))}
                            stroke="hsl(210 16% 48%)"
                            strokeWidth={1.4}
                            strokeDasharray="5 4"
                            opacity={0.85}
                          />

                          <polygon
                            points={`${toX},${toY} ${leftX},${leftY} ${rightX},${rightY}`}
                            fill="hsl(210 22% 35%)"
                          />
                          <text x={toX + 8} y={toY - 6} fontSize={10} fontWeight={600} fill="hsl(210 22% 30%)">
                            AEC +mean
                          </text>
                          <text
                            x={xScale(0) + 8}
                            y={yScale(Math.max(...projections.map((p) => p.ordinateScore), 1)) + 3}
                            fontSize={10}
                            fontWeight={500}
                            fill="hsl(210 16% 36%)"
                          >
                            Stability axis
                          </text>

                          {projections.map((p) => (
                            <line
                              key={`proj-${p.genotype}`}
                              x1={xScale(p.abscissaScore)}
                              y1={yScale(p.ordinateScore)}
                              x2={xScale(p.abscissaScore)}
                              y2={yScale(0)}
                              stroke="hsl(210 10% 45%)"
                              strokeWidth={1.1}
                              strokeDasharray="3 3"
                              opacity={0.5}
                            />
                          ))}

                          {ideal.ringRadii.map((r, i) => {
                            const pxR = Math.abs(xScale(ideal.idealAbscissaScore + r) - xScale(ideal.idealAbscissaScore));
                            return (
                              <circle
                                key={`ideal-ring-${i}`}
                                cx={iCx}
                                cy={iCy}
                                r={pxR}
                                fill="none"
                                stroke="hsl(150 58% 34% / 0.45)"
                                strokeWidth={1}
                                strokeDasharray="4 4"
                              />
                            );
                          })}

                          <circle cx={iCx} cy={iCy} r={5.2} fill="hsl(150 58% 34%)" stroke="white" strokeWidth={1} />
                          <text x={iCx + 10} y={iCy - 9} fontSize={11} fontWeight={700} fill="hsl(150 58% 25%)">
                            Ideal genotype
                          </text>
                        </g>
                      );
                    }}
                  />
                )}

                <Scatter
                  name="Genotypes"
                  data={chartGenos}
                  shape={(props) => {
                    const { cx, cy, payload } = props as { cx: number; cy: number; payload: typeof chartGenos[number] };
                    const winning = biplotType === "which-won-where" && gge.which_won_where
                      ? Object.values(gge.which_won_where.winning_genotypes).includes(payload.genotype)
                      : false;
                    return (
                      <g>
                        <circle cx={cx} cy={cy} r={winning ? 8 : 5} fill={winning ? "hsl(45 95% 50%)" : "hsl(var(--primary))"} stroke="hsl(var(--foreground))" strokeOpacity={0.4} />
                        <text x={cx + 8} y={cy + 3} fontSize={10} fontWeight={winning ? 700 : 400} fill="hsl(var(--foreground))">{payload.genotype}</text>
                      </g>
                    );
                  }}
                />
                <Scatter
                  name="Environments"
                  data={chartEnvs}
                  shape={(props) => {
                    const { cx, cy, payload, xAxis, yAxis } = props as {
                      cx: number; cy: number; payload: typeof chartEnvs[number];
                      xAxis?: { scale: (v: number) => number }; yAxis?: { scale: (v: number) => number };
                    };
                    const ox = xAxis?.scale ? xAxis.scale(0) : cx;
                    const oy = yAxis?.scale ? yAxis.scale(0) : cy;
                    return (
                      <g>
                        <line x1={ox} y1={oy} x2={cx} y2={cy} stroke="hsl(217 91% 45%)" strokeWidth={1.5} opacity={0.7} />
                        <polygon points={`${cx},${cy - 7} ${cx - 6},${cy + 4} ${cx + 6},${cy + 4}`} fill="hsl(217 91% 45%)" />
                        <text x={cx + 8} y={cy + 3} fontSize={10} fontWeight={600} fill="hsl(217 91% 30%)">{payload.environment}</text>
                      </g>
                    );
                  }}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 rounded-xl border border-border bg-muted/30 px-4 py-3">
            <p className="text-sm text-muted-foreground leading-relaxed">
              {biplotType === "mean-stability"
                ? "GGE Biplot Mean vs Stability view showing genotype performance, stability, and ideal genotype ranking across environments."
                : biplotType === "which-won-where"
                ? "GGE Biplot Which-Won-Where view showing sector winners and crossover interaction patterns across environments."
                : "GGE Biplot Environment Evaluation view showing discriminating ability and representativeness of test environments."}
            </p>
          </div>
        </CardContent>
      </Card>
      ) : (
        <Card>
          <CardContent className="p-4 text-sm text-muted-foreground">
            GGE biplot data is not available for this response.
          </CardContent>
        </Card>
      )}

      {/* Which-won-where mega-environments */}
      {biplotType === "which-won-where" && gge.which_won_where && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <CardTitle className="font-serif text-lg">Mega-environments</CardTitle>
            <ExportToolbar
              onCsv={() => {
                const rows = gge.which_won_where!.mega_environments.map((m) => ({
                  mega_env_id: m.id,
                  environments: m.environments.join("; "),
                  best_genotype: m.best_genotype,
                  mean_yield: m.mean_yield,
                }));
                downloadCsv(`gge_mega_environments_${trait}.csv`, rows as unknown as Record<string, unknown>[]);
              }}
            />
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {gge.which_won_where.mega_environments.map((m) => (
                <Card key={m.id} className="border-primary/30 bg-primary/[0.03]">
                  <CardContent className="p-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <Badge variant="secondary">Mega-Env {m.id}</Badge>
                      <div className="flex items-center gap-1 text-amber-600">
                        <Star className="h-4 w-4 fill-current" />
                        <span className="text-sm font-semibold">{m.best_genotype}</span>
                      </div>
                    </div>
                    <div className="text-sm">
                      <span className="text-muted-foreground">Environments: </span>
                      <span className="font-medium">{m.environments.join(", ")}</span>
                    </div>
                    <div className="text-sm">
                      <span className="text-muted-foreground">Mean yield: </span>
                      <span className="font-medium tabular-nums">{fmt(m.mean_yield, 2)}</span>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
            {gge.which_won_where.interpretation && (
              <p className="text-sm text-muted-foreground mt-4 leading-relaxed">{gge.which_won_where.interpretation}</p>
            )}
          </CardContent>
        </Card>
      )}

      {/* Mean-vs-stability ranking */}
      {biplotType === "mean-stability" && gge.mean_vs_stability && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="font-serif text-lg">Distance from ideal genotype</CardTitle>
            <ExportToolbar
              onCsv={() => {
                const rows = meanStabilityModel
                  ? meanStabilityModel.ideal.rankings
                  : gge.mean_vs_stability!.genotype_distances;
                downloadCsv(`gge_distance_${trait}.csv`, rows as unknown as Record<string, unknown>[]);
              }}
            />
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-right">Rank</TableHead>
                  <TableHead>Genotype</TableHead>
                  <TableHead className="text-right">Distance</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {[
                  ...(meanStabilityModel
                    ? meanStabilityModel.ideal.rankings
                    : gge.mean_vs_stability.genotype_distances),
                ]
                  .sort((a, b) => a.rank - b.rank)
                  .map((r) => {
                  const idealGenotype = meanStabilityModel
                    ? meanStabilityModel.ideal.idealGenotype
                    : gge.mean_vs_stability!.ideal_genotype;
                  const isIdeal = r.genotype === idealGenotype;
                  const top5 = r.rank <= 5;
                  return (
                    <TableRow key={r.genotype} className={isIdeal ? "bg-emerald-50/40 dark:bg-emerald-900/10" : ""}>
                      <TableCell className="text-right tabular-nums">{r.rank}</TableCell>
                      <TableCell className="font-medium flex items-center gap-1.5">
                        {isIdeal && <Crown className="h-3.5 w-3.5 text-amber-500" />}
                        {r.genotype}
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{fmt(r.distance_from_ideal, 3)}</TableCell>
                      <TableCell>
                        {top5 ? (
                          <Badge variant="outline" className="border-emerald-400 text-emerald-700 dark:text-emerald-300">Recommended</Badge>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
