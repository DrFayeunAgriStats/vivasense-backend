import { useMemo, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Legend, Tooltip,
} from "recharts";
import { fmt, ExportToolbar, downloadCsv, exportChartPng } from "../shared";
import type { StabilityResponse } from "@/types/advancedAnalysis";
import { spearman } from "./geometry";

const PALETTE = [
  "hsl(var(--primary))", "hsl(142 71% 38%)", "hsl(217 91% 55%)", "hsl(25 95% 53%)",
  "hsl(280 60% 55%)", "hsl(0 84% 55%)", "hsl(195 80% 45%)", "hsl(80 60% 45%)",
  "hsl(330 70% 55%)", "hsl(255 60% 55%)",
];

export function ComparisonTab({ result }: { result: StabilityResponse }) {
  const radarRef = useRef<HTMLDivElement>(null);
  // Build per-genotype rank vector across methods that exist.
  const consensus = useMemo(() => {
    const genoSet = new Set<string>();
    result.genotype_stability.forEach((g) => genoSet.add(g.genotype));
    result.ammi_results?.stability_measure.forEach((g) => genoSet.add(g.genotype));
    result.gge_results?.mean_vs_stability?.genotype_distances.forEach((g) => genoSet.add(g.genotype));

    const erRank = new Map<string, number>();
    const shuklaRank = new Map<string, number>();
    result.genotype_stability.forEach((g) => erRank.set(g.genotype, g.rank));
    // Shukla rank: derive from shukla_variance ascending if present
    const withShukla = result.genotype_stability
      .filter((g) => g.shukla_variance != null)
      .sort((a, b) => (a.shukla_variance! - b.shukla_variance!));
    withShukla.forEach((g, i) => shuklaRank.set(g.genotype, i + 1));

    const asvRank = new Map<string, number>();
    result.ammi_results?.stability_measure.forEach((g) => asvRank.set(g.genotype, g.rank));

    const ggeRank = new Map<string, number>();
    result.gge_results?.mean_vs_stability?.genotype_distances.forEach((g) => ggeRank.set(g.genotype, g.rank));

    const meanByGeno = new Map<string, number>();
    result.genotype_stability.forEach((g) => meanByGeno.set(g.genotype, g.mean));

    const erClass = new Map<string, string>();
    result.genotype_stability.forEach((g) => erClass.set(g.genotype, g.stability_class));

    const rows = Array.from(genoSet).map((g) => {
      const ranks = [erRank.get(g), shuklaRank.get(g), asvRank.get(g), ggeRank.get(g)].filter(
        (v): v is number => typeof v === "number"
      );
      const overall = ranks.length ? ranks.reduce((s, v) => s + v, 0) / ranks.length : Number.POSITIVE_INFINITY;
      return {
        genotype: g,
        mean: meanByGeno.get(g) ?? null,
        er_class: erClass.get(g) ?? "—",
        shukla_rank: shuklaRank.get(g) ?? null,
        asv_rank: asvRank.get(g) ?? null,
        gge_rank: ggeRank.get(g) ?? null,
        er_rank: erRank.get(g) ?? null,
        overall_score: overall,
      };
    });
    rows.sort((a, b) => a.overall_score - b.overall_score);
    return rows;
  }, [result]);

  const top10 = consensus.slice(0, 10);

  // Method agreement matrix using Spearman on common genotypes.
  const methods = useMemo(() => {
    const m: { key: string; label: string; rankByGeno: Map<string, number> }[] = [];
    if (consensus.some((r) => r.er_rank != null))
      m.push({ key: "er", label: "Eberhart-Russell", rankByGeno: new Map(consensus.filter((r) => r.er_rank != null).map((r) => [r.genotype, r.er_rank as number])) });
    if (consensus.some((r) => r.shukla_rank != null))
      m.push({ key: "shukla", label: "Shukla", rankByGeno: new Map(consensus.filter((r) => r.shukla_rank != null).map((r) => [r.genotype, r.shukla_rank as number])) });
    if (consensus.some((r) => r.asv_rank != null))
      m.push({ key: "asv", label: "AMMI ASV", rankByGeno: new Map(consensus.filter((r) => r.asv_rank != null).map((r) => [r.genotype, r.asv_rank as number])) });
    if (consensus.some((r) => r.gge_rank != null))
      m.push({ key: "gge", label: "GGE", rankByGeno: new Map(consensus.filter((r) => r.gge_rank != null).map((r) => [r.genotype, r.gge_rank as number])) });
    return m;
  }, [consensus]);

  const agreement = useMemo(() => {
    return methods.map((mi) =>
      methods.map((mj) => {
        if (mi.key === mj.key) return 1;
        const common: { a: number; b: number }[] = [];
        mi.rankByGeno.forEach((v, k) => {
          const w = mj.rankByGeno.get(k);
          if (typeof w === "number") common.push({ a: v, b: w });
        });
        if (common.length < 3) return 0;
        return spearman(common.map((c) => c.a), common.map((c) => c.b));
      })
    );
  }, [methods]);

  // Radar chart data: each genotype = one dataset with axes per method (normalized 0-100, lower rank = better → invert)
  const maxRanks = useMemo(() => {
    const arr: Record<string, number> = {};
    methods.forEach((m) => {
      arr[m.key] = Math.max(1, ...Array.from(m.rankByGeno.values()));
    });
    return arr;
  }, [methods]);

  const radarData = useMemo(() => {
    return methods.map((m) => {
      const obj: Record<string, number | string> = { method: m.label };
      top10.forEach((r) => {
        const rk = m.rankByGeno.get(r.genotype);
        // Normalize: best rank → 100, worst → 0
        obj[r.genotype] = rk != null ? Math.round(100 * (1 - (rk - 1) / Math.max(1, maxRanks[m.key] - 1))) : 0;
      });
      return obj;
    });
  }, [methods, top10, maxRanks]);

  const heatColor = (v: number) => {
    // -1 red → 0 grey → 1 green
    const t = (v + 1) / 2;
    const hue = 0 + (142 - 0) * t;
    return `hsl(${hue.toFixed(0)} 70% ${50 + 10 * (1 - Math.abs(v))}%)`;
  };

  const consensusList = top10.slice(0, 5);

  return (
    <div className="space-y-6">
      {/* Summary card */}
      <Card className="border-primary/30 bg-primary/[0.03]">
        <CardHeader>
          <CardTitle className="font-serif text-lg">Consensus stable genotypes</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-3">
            Genotypes ranked by average rank across all available methods. Lower average rank = more stable across approaches.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            {consensusList.map((r, i) => (
              <Card key={r.genotype} className="border-emerald-300/60">
                <CardContent className="p-3 text-center space-y-1">
                  <Badge variant="outline" className="border-emerald-400 text-emerald-700 dark:text-emerald-300">#{i + 1}</Badge>
                  <div className="font-serif text-lg font-semibold">{r.genotype}</div>
                  <div className="text-xs text-muted-foreground">
                    Mean: <span className="tabular-nums font-medium text-foreground">{r.mean != null ? fmt(r.mean, 2) : "—"}</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Avg rank: <span className="tabular-nums font-medium text-foreground">{fmt(r.overall_score, 1)}</span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Comparison table */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="font-serif text-lg">Method comparison</CardTitle>
          <ExportToolbar onCsv={() => downloadCsv(`stability_comparison_${result.trait}.csv`, consensus as unknown as Record<string, unknown>[])} />
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Genotype</TableHead>
                <TableHead className="text-right">Mean</TableHead>
                <TableHead>E-R class</TableHead>
                <TableHead className="text-right">E-R rank</TableHead>
                <TableHead className="text-right">Shukla rank</TableHead>
                <TableHead className="text-right">AMMI ASV rank</TableHead>
                <TableHead className="text-right">GGE rank</TableHead>
                <TableHead className="text-right">Overall</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {consensus.slice(0, 30).map((r, i) => (
                <TableRow key={r.genotype} className={i < 10 ? "bg-emerald-50/30 dark:bg-emerald-900/5" : ""}>
                  <TableCell className="font-medium">{r.genotype}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.mean != null ? fmt(r.mean, 2) : "—"}</TableCell>
                  <TableCell><Badge variant="outline" className="text-xs">{r.er_class}</Badge></TableCell>
                  <TableCell className="text-right tabular-nums">{r.er_rank ?? "—"}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.shukla_rank ?? "—"}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.asv_rank ?? "—"}</TableCell>
                  <TableCell className="text-right tabular-nums">{r.gge_rank ?? "—"}</TableCell>
                  <TableCell className="text-right tabular-nums font-semibold">{fmt(r.overall_score, 2)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {consensus.length > 30 && (
            <p className="text-xs text-muted-foreground mt-2">Showing top 30 of {consensus.length} genotypes.</p>
          )}
        </CardContent>
      </Card>

      {/* Radar chart */}
      {methods.length >= 2 && top10.length > 0 && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <CardTitle className="font-serif text-lg">Stability consensus radar (top 10)</CardTitle>
            <ExportToolbar onPng={() => exportChartPng(radarRef.current, `stability_radar_${result.trait}.png`)} />
          </CardHeader>
          <CardContent>
            <div ref={radarRef} className="w-full h-[440px]">
              <ResponsiveContainer>
                <RadarChart data={radarData} outerRadius="75%">
                  <PolarGrid />
                  <PolarAngleAxis dataKey="method" />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} />
                  <Tooltip />
                  <Legend />
                  {top10.map((r, i) => (
                    <Radar
                      key={r.genotype}
                      name={r.genotype}
                      dataKey={r.genotype}
                      stroke={PALETTE[i % PALETTE.length]}
                      fill={PALETTE[i % PALETTE.length]}
                      fillOpacity={0.08}
                    />
                  ))}
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Higher = better rank for that method. Genotypes with consistently large polygons are stable across all methods.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Agreement matrix */}
      {methods.length >= 2 && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-2">
            <CardTitle className="font-serif text-lg">Method agreement (Spearman ρ)</CardTitle>
            <ExportToolbar
              onCsv={() => {
                const rows = methods.map((mi, i) => {
                  const row: Record<string, unknown> = { method: mi.label };
                  methods.forEach((mj, j) => {
                    row[mj.label] = Number(agreement[i][j].toFixed(4));
                  });
                  return row;
                });
                downloadCsv(`stability_method_agreement_${result.trait}.csv`, rows);
              }}
            />
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  <th className="text-left p-2"></th>
                  {methods.map((m) => (
                    <th key={m.key} className="p-2 text-center font-medium">{m.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {methods.map((mi, i) => (
                  <tr key={mi.key}>
                    <td className="p-2 font-medium">{mi.label}</td>
                    {methods.map((mj, j) => {
                      const v = agreement[i][j];
                      return (
                        <td key={mj.key} className="p-2 text-center">
                          <div
                            className="rounded-md py-2 px-3 tabular-nums font-medium"
                            style={{ background: heatColor(v), color: "white" }}
                          >
                            {v.toFixed(2)}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-muted-foreground mt-2">Green = high agreement, red = low agreement. Diagonal = 1.</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
