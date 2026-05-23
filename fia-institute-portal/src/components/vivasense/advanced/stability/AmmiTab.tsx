import { useEffect, useMemo, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
  ScatterChart, Scatter, ZAxis, ReferenceLine, LabelList,
} from "recharts";
import { fmt, InterpretationPanel, ExportToolbar, downloadCsv, exportChartPng } from "../shared";
import type { AMMIResults } from "@/types/advancedAnalysis";

type AmmiChartPoint = {
  code: string;
  mean: number;
  pc1: number;
  pc2: number;
  kind: "genotype" | "environment";
};

function toFiniteNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

function pColor(p: number): string {
  if (p < 0.001) return "bg-emerald-200/70 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-100";
  if (p < 0.01) return "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/30 dark:text-emerald-100";
  if (p < 0.05) return "bg-amber-100 text-amber-900 dark:bg-amber-900/30 dark:text-amber-100";
  return "";
}
function asvBadge(c: string): string {
  const s = c.toLowerCase();
  if (s.includes("highly")) return "bg-emerald-200/70 text-emerald-900 border-emerald-400 dark:bg-emerald-900/40 dark:text-emerald-100";
  if (s.includes("moderate")) return "bg-amber-100 text-amber-900 border-amber-300 dark:bg-amber-900/30";
  if (s.includes("unstable")) return "bg-red-100 text-red-900 border-red-300 dark:bg-red-900/30";
  if (s.includes("stable")) return "bg-emerald-100 text-emerald-900 border-emerald-300 dark:bg-emerald-900/30";
  return "bg-muted text-muted-foreground";
}

export function AmmiTab({ ammi, trait }: { ammi: AMMIResults; trait: string }) {
  const varRef = useRef<HTMLDivElement>(null);
  const biplotRef = useRef<HTMLDivElement>(null);

  const anovaRows = useMemo(() => {
    const t = ammi?.anova_table;
    if (!t || !Array.isArray(t.source)) return [];
    return t.source
      .map((s, i) => {
        if (s == null) return null;
        return {
          source: s,
          df: t.df?.[i],
          sum_sq: t.sum_sq?.[i],
          mean_sq: t.mean_sq?.[i],
          f_value: t.f_value?.[i],
          p_value: t.p_value?.[i],
        };
      })
      .filter((r): r is NonNullable<typeof r> => r != null);
  }, [ammi]);

  const variance = Array.isArray(ammi?.variance_explained) ? ammi.variance_explained : [];
  const cumulative = Array.isArray(ammi?.cumulative_variance) ? ammi.cumulative_variance : [];

  const varScale = useMemo(() => {
    if (variance.length === 0 && cumulative.length === 0) return 100;
    const maxV = Math.max(...(variance.length ? variance : [0]), ...(cumulative.length ? cumulative : [0]));
    return maxV > 1 ? 1 : 100;
  }, [variance, cumulative]);
  const toPct = (v: number) => (typeof v === "number" ? v * varScale : 0);

  const varianceData = useMemo(() => {
    return variance.map((v, i) => ({
      axis: `IPCA${i + 1}`,
      individual: toPct(v),
      cumulative: toPct(cumulative[i] ?? 0),
    }));
  }, [variance, cumulative, varScale]);

  // Biplot: normalize backend key variants and drop invalid points.
  const rawGenotypes = useMemo(() => (Array.isArray(ammi?.biplot_data?.genotypes) ? ammi.biplot_data.genotypes : []), [ammi]);
  const rawEnvironments = useMemo(() => (Array.isArray(ammi?.biplot_data?.environments) ? ammi.biplot_data.environments : []), [ammi]);

  const normalizedGenotypes = useMemo(() => {
    return rawGenotypes
      .filter(Boolean)
      .map((g: any): AmmiChartPoint | null => {
        const code = String(g?.code ?? g?.genotype ?? "").trim();
        const pc1 = toFiniteNumber(g?.pc1 ?? g?.ipca1, Number.NaN);
        const pc2 = toFiniteNumber(g?.pc2 ?? g?.ipca2, Number.NaN);
        const mean = toFiniteNumber(g?.mean, 0);
        if (!code || !Number.isFinite(pc1) || !Number.isFinite(pc2)) return null;
        return { code, mean, pc1, pc2, kind: "genotype" };
      })
      .filter((p): p is AmmiChartPoint => p != null);
  }, [rawGenotypes]);

  const normalizedEnvironments = useMemo(() => {
    return rawEnvironments
      .filter(Boolean)
      .map((e: any): AmmiChartPoint | null => {
        const code = String(e?.code ?? e?.environment ?? "").trim();
        const pc1 = toFiniteNumber(e?.pc1 ?? e?.ipca1, Number.NaN);
        const pc2 = toFiniteNumber(e?.pc2 ?? e?.ipca2, Number.NaN);
        const mean = toFiniteNumber(e?.mean, 0);
        if (!code || !Number.isFinite(pc1) || !Number.isFinite(pc2)) return null;
        return { code, mean, pc1, pc2, kind: "environment" };
      })
      .filter((p): p is AmmiChartPoint => p != null);
  }, [rawEnvironments]);

  const biplotGenos = normalizedGenotypes;
  const biplotEnvs = normalizedEnvironments;
  const plotData = useMemo(
    () => ({
      genotypePoints: biplotGenos,
      envPoints: biplotEnvs,
      allPoints: [...biplotGenos, ...biplotEnvs],
    }),
    [biplotGenos, biplotEnvs]
  );

  const invalidGenoCount = Math.max(0, rawGenotypes.length - biplotGenos.length);
  const invalidEnvCount = Math.max(0, rawEnvironments.length - biplotEnvs.length);
  const stabilityMeasure = Array.isArray(ammi?.stability_measure) ? ammi.stability_measure : [];
  const hasBiplot = biplotGenos.length > 0 || biplotEnvs.length > 0;
  const hasVariance = varianceData.length > 0;
  const hasAnova = anovaRows.length > 0;
  const meanRange = useMemo(() => {
    const ms = biplotGenos.map((g) => g.mean).filter((v): v is number => typeof v === "number");
    if (ms.length === 0) return [0, 1] as const;
    return [Math.min(...ms), Math.max(...ms)] as const;
  }, [biplotGenos]);
  const meanColor = (m: number) => {
    const [lo, hi] = meanRange;
    if (hi === lo) return "hsl(142 71% 38%)";
    const t = (m - lo) / (hi - lo); // 0..1
    // red(0) → amber(0.5) → green(1)
    if (t < 0.5) {
      const k = t / 0.5; // 0..1
      const hue = 0 + (35 - 0) * k;
      return `hsl(${hue.toFixed(0)} 80% 50%)`;
    }
    const k = (t - 0.5) / 0.5;
    const hue = 35 + (142 - 35) * k;
    return `hsl(${hue.toFixed(0)} 70% 42%)`;
  };

  const ipca1Var = toPct(variance[0] ?? 0);
  const ipca2Var = toPct(variance[1] ?? 0);

  useEffect(() => {
    console.log("AMMI payload", ammi);
    console.log("Genotype coords", biplotGenos);
    console.log("Environment coords", biplotEnvs);
    console.log("AMMI plot payload", plotData);

    plotData.allPoints.forEach((p) => {
      console.log(p.code, p.pc1, p.pc2);
    });

    if (!biplotGenos.length && !biplotEnvs.length) {
      console.warn("AMMI biplot render skipped: no valid coordinates", {
        rawGenotypes: rawGenotypes.length,
        rawEnvironments: rawEnvironments.length,
        invalidGenoCount,
        invalidEnvCount,
      });
    }
  }, [
    ammi,
    biplotGenos,
    biplotEnvs,
    plotData,
    rawGenotypes.length,
    rawEnvironments.length,
    invalidGenoCount,
    invalidEnvCount,
  ]);

  return (
    <div className="space-y-6">
      {ammi?.interpretation && <InterpretationPanel text={ammi.interpretation} />}

      {/* Variance explained chart */}
      {hasVariance && (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2">
          <CardTitle className="font-serif text-lg">Variance explained by IPCA axes</CardTitle>
          <div className="flex flex-wrap items-center gap-2">
            <ExportToolbar
              onCsv={() => downloadCsv(`ammi_variance_${trait}.csv`, varianceData as unknown as Record<string, unknown>[])}
            />
            <ExportToolbar onPng={() => exportChartPng(varRef.current, `ammi_variance_${trait}.png`)} />
          </div>
        </CardHeader>
        <CardContent>
          <div ref={varRef} className="w-full h-[300px]">
            <ResponsiveContainer>
              <ComposedChart data={varianceData} margin={{ top: 16, right: 24, bottom: 16, left: 8 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="axis" />
                <YAxis label={{ value: "% Variance", angle: -90, position: "insideLeft" }} domain={[0, 100]} />
                <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
                <Legend />
                <Bar dataKey="individual" name="Individual %" fill="hsl(var(--primary))">
                  <LabelList dataKey="individual" position="top" formatter={(v: number) => `${v.toFixed(1)}%`} />
                </Bar>
                <Line type="monotone" dataKey="cumulative" name="Cumulative %" stroke="hsl(25 95% 53%)" strokeWidth={2} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
      )}

      {/* ANOVA table */}
      {hasAnova && (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="font-serif text-lg">AMMI ANOVA</CardTitle>
          <ExportToolbar onCsv={() => downloadCsv(`ammi_anova_${trait}.csv`, anovaRows as unknown as Record<string, unknown>[])} />
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Source</TableHead>
                <TableHead className="text-right">DF</TableHead>
                <TableHead className="text-right">Sum Sq</TableHead>
                <TableHead className="text-right">Mean Sq</TableHead>
                <TableHead className="text-right">F</TableHead>
                <TableHead className="text-right">p-value</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {anovaRows.map((r) => {
                const sig = r.p_value < 0.05;
                return (
                  <TableRow key={r.source} className={sig ? "font-medium" : ""}>
                    <TableCell>{r.source}</TableCell>
                    <TableCell className="text-right tabular-nums">{r.df}</TableCell>
                    <TableCell className="text-right tabular-nums">{fmt(r.sum_sq, 2)}</TableCell>
                    <TableCell className="text-right tabular-nums">{fmt(r.mean_sq, 2)}</TableCell>
                    <TableCell className="text-right tabular-nums">{fmt(r.f_value, 2)}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      <span className={`px-2 py-0.5 rounded ${pColor(r.p_value)}`}>{fmt(r.p_value, 4)}</span>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      )}


      {/* ASV table */}
      {stabilityMeasure.length > 0 && (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="font-serif text-lg">AMMI Stability Value (ASV)</CardTitle>
          <ExportToolbar onCsv={() => downloadCsv(`ammi_asv_${trait}.csv`, stabilityMeasure as unknown as Record<string, unknown>[])} />
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-right">Rank</TableHead>
                <TableHead>Genotype</TableHead>
                <TableHead className="text-right">Mean</TableHead>
                <TableHead className="text-right">ASV</TableHead>
                <TableHead>Class</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...stabilityMeasure].sort((a, b) => a.rank - b.rank).map((r) => (
                <TableRow key={r.genotype}>
                  <TableCell className="text-right tabular-nums">{r.rank}</TableCell>
                  <TableCell className="font-medium">{r.genotype}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.mean != null ? fmt(r.mean, 2) : "—"}</TableCell>
                  <TableCell className="text-right tabular-nums">{fmt(r.asv, 3)}</TableCell>
                  <TableCell><Badge variant="outline" className={asvBadge(r.stability_class)}>{r.stability_class}</Badge></TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      )}


      {/* Biplot */}
      {hasBiplot ? (
      <Card>
        <CardHeader className="flex flex-row items-center justify-between gap-2">
          <CardTitle className="font-serif text-lg">AMMI biplot — IPCA1 × IPCA2</CardTitle>
          <div className="flex flex-wrap items-center gap-2">
            <ExportToolbar
              onCsv={() =>
                downloadCsv(`ammi_biplot_genotypes_${trait}.csv`, biplotGenos as unknown as Record<string, unknown>[])
              }
              csvLabel="Genotype scores CSV"
            />
            <ExportToolbar
              onCsv={() =>
                downloadCsv(`ammi_biplot_environments_${trait}.csv`, biplotEnvs as unknown as Record<string, unknown>[])
              }
              csvLabel="Environment scores CSV"
            />
            <ExportToolbar onPng={() => exportChartPng(biplotRef.current, `ammi_biplot_${trait}.png`)} />
          </div>
        </CardHeader>
        <CardContent>
          <div ref={biplotRef} className="w-full min-h-[500px] h-[500px] border border-dashed border-muted-foreground/30 rounded-md">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 16, right: 30, bottom: 36, left: 16 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="pc1" domain={["auto", "auto"]}
                  label={{ value: `IPCA1 (${ipca1Var.toFixed(1)}%)`, position: "insideBottom", offset: -16 }} />
                <YAxis type="number" dataKey="pc2" domain={["auto", "auto"]}
                  label={{ value: `IPCA2 (${ipca2Var.toFixed(1)}%)`, angle: -90, position: "insideLeft" }} />
                <ZAxis range={[80, 80]} />
                <ReferenceLine x={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
                <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload as { code: string; mean: number; pc1: number; pc2: number; kind: string };
                    return (
                      <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                        <div className="font-semibold">{d.code} <span className="text-muted-foreground font-normal">({d.kind})</span></div>
                        <div>Mean: {fmt(d.mean, 2)}</div>
                        <div>IPCA1: {fmt(d.pc1, 3)}</div>
                        <div>IPCA2: {fmt(d.pc2, 3)}</div>
                      </div>
                    );
                  }}
                />
                <Legend />
                <Scatter
                  name="Genotypes"
                  data={biplotGenos}
                  shape={(props) => {
                    const { cx, cy, payload } = props as { cx: number; cy: number; payload: typeof biplotGenos[number] };
                    return (
                      <g>
                        <circle cx={cx} cy={cy} r={6} fill={meanColor(payload.mean)} stroke="hsl(var(--foreground))" strokeOpacity={0.4} />
                        <text x={cx + 8} y={cy + 3} fontSize={10} fill="hsl(var(--foreground))">{payload.code}</text>
                      </g>
                    );
                  }}
                />
                <Scatter
                  name="Environments"
                  data={biplotEnvs}
                  shape={(props) => {
                    const { cx, cy, payload, xAxis, yAxis } = props as {
                      cx: number; cy: number; payload: typeof biplotEnvs[number];
                      xAxis?: { scale: (v: number) => number };
                      yAxis?: { scale: (v: number) => number };
                    };
                    const ox = xAxis?.scale ? xAxis.scale(0) : cx;
                    const oy = yAxis?.scale ? yAxis.scale(0) : cy;
                    return (
                      <g>
                        <line x1={ox} y1={oy} x2={cx} y2={cy} stroke="hsl(217 91% 45%)" strokeWidth={1.5} opacity={0.7} />
                        <polygon
                          points={`${cx},${cy - 7} ${cx - 6},${cy + 4} ${cx + 6},${cy + 4}`}
                          fill="hsl(217 91% 45%)"
                        />
                        <text x={cx + 8} y={cy + 3} fontSize={10} fill="hsl(217 91% 30%)" fontWeight={600}>{payload.code}</text>
                      </g>
                    );
                  }}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          {(invalidGenoCount > 0 || invalidEnvCount > 0) && (
            <div className="mt-3 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
              AMMI diagnostics: filtered {invalidGenoCount} invalid genotype point(s) and {invalidEnvCount} invalid environment point(s)
              due to missing/non-numeric IPCA1/IPCA2 values.
            </div>
          )}
          <p className="text-xs text-muted-foreground mt-2">
            Circles = genotypes (color → mean performance); blue triangles with vectors = environments. Points near origin are most stable.
          </p>
        </CardContent>
      </Card>
      ) : (
        <Card>
          <CardContent className="p-4 text-sm text-muted-foreground">
            <div>No AMMI coordinates available</div>
            AMMI coordinates unavailable for rendering. Expected valid numeric IPCA1/IPCA2 pairs for genotype and/or environment points.
          </CardContent>
        </Card>
      )}
    </div>
  );
}
