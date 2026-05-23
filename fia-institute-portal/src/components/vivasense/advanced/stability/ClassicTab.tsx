import { useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Legend,
} from "recharts";
import { ArrowUpDown } from "lucide-react";
import { fmt, InterpretationPanel, ExportToolbar, downloadCsv, exportChartPng } from "../shared";
import type { StabilityResponse, StabilityRow } from "@/types/advancedAnalysis";

type SortKey = "rank" | "genotype" | "mean" | "bi" | "s2di" | "shukla_variance";

const CLASS_BADGE: Record<string, string> = {
  stable: "bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-900/30 dark:text-emerald-200",
  responsive_favorable: "bg-blue-100 text-blue-800 border-blue-300 dark:bg-blue-900/30 dark:text-blue-200",
  responsive: "bg-blue-100 text-blue-800 border-blue-300 dark:bg-blue-900/30 dark:text-blue-200",
  responsive_poor: "bg-orange-100 text-orange-800 border-orange-300 dark:bg-orange-900/30 dark:text-orange-200",
  unpredictable: "bg-red-100 text-red-800 border-red-300 dark:bg-red-900/30 dark:text-red-200",
};
function classBadge(c: string) {
  return CLASS_BADGE[c] ?? "bg-muted text-muted-foreground border-border";
}
function classColor(c: string): string {
  if (c === "stable") return "hsl(142 71% 38%)";
  if (c.startsWith("responsive_fav") || c === "responsive") return "hsl(217 91% 55%)";
  if (c === "responsive_poor") return "hsl(25 95% 53%)";
  if (c === "unpredictable") return "hsl(0 84% 55%)";
  return "hsl(var(--muted-foreground))";
}

export function ClassicTab({ result }: { result: StabilityResponse }) {
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sortAsc, setSortAsc] = useState(true);
  const chartRef = useRef<HTMLDivElement>(null);

  const allRows: StabilityRow[] = useMemo(
    () => (Array.isArray(result?.genotype_stability) ? result.genotype_stability.filter(Boolean) : []),
    [result]
  );
  const bestSet = useMemo(
    () => new Set(Array.isArray(result?.best_stable_genotypes) ? result.best_stable_genotypes : []),
    [result]
  );

  const sortedRows: StabilityRow[] = useMemo(() => {
    const arr = [...allRows];
    arr.sort((a, b) => {
      const av = a[sortKey] as number | string | undefined;
      const bv = b[sortKey] as number | string | undefined;
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") return sortAsc ? av - bv : bv - av;
      return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
    return arr;
  }, [allRows, sortKey, sortAsc]);

  const scatterData = useMemo(() => {
    const maxS2 = Math.max(...allRows.map((g) => Math.abs(g.s2di) || 0), 1e-9);
    return allRows.map((g) => ({
      genotype: g.genotype,
      x: g.bi,
      y: g.mean,
      // Inverse stability → larger circle = more stable (lower s2di)
      z: Math.max(40, 320 * (1 - Math.min(1, Math.abs(g.s2di) / maxS2))),
      cls: g.stability_class,
      s2di: g.s2di,
      shukla: g.shukla_variance,
      isBest: bestSet.has(g.genotype),
    }));
  }, [allRows, bestSet]);

  const groups = useMemo(() => {
    const map = new Map<string, typeof scatterData>();
    scatterData.forEach((d) => {
      const arr = map.get(d.cls) ?? [];
      arr.push(d);
      map.set(d.cls, arr);
    });
    return Array.from(map.entries());
  }, [scatterData]);

  const toggleSort = (k: SortKey) => {
    if (sortKey === k) setSortAsc(!sortAsc);
    else { setSortKey(k); setSortAsc(true); }
  };

  return (
    <div className="space-y-6">
      <InterpretationPanel text={result.interpretation} />

      {/* Scatter: mean vs bi */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="font-serif text-lg">Mean × b<sub>i</sub> stability scatter</CardTitle>
          <ExportToolbar onPng={() => exportChartPng(chartRef.current, `stability_scatter_${result.trait}.png`)} />
        </CardHeader>
        <CardContent>
          <div ref={chartRef} className="w-full h-[460px]">
            <ResponsiveContainer>
              <ScatterChart margin={{ top: 16, right: 30, bottom: 36, left: 16 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  type="number" dataKey="x" name="bi" domain={["auto", "auto"]}
                  label={{ value: "Regression coefficient (bᵢ)", position: "insideBottom", offset: -16 }}
                />
                <YAxis
                  type="number" dataKey="y" name="Mean"
                  label={{ value: `Mean ${result.trait}`, angle: -90, position: "insideLeft" }}
                />
                <ZAxis type="number" dataKey="z" range={[40, 320]} />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload as typeof scatterData[number];
                    return (
                      <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
                        <div className="font-semibold">{d.genotype}</div>
                        <div>Mean: {fmt(d.y, 2)}</div>
                        <div>bᵢ: {fmt(d.x, 3)}</div>
                        <div>S²di: {fmt(d.s2di, 3)}</div>
                        {d.shukla != null && <div>Shukla σ²: {fmt(d.shukla, 3)}</div>}
                        <div className="mt-1 text-muted-foreground">{d.cls}</div>
                      </div>
                    );
                  }}
                />
                <Legend />
                <ReferenceLine x={1} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" label={{ value: "bᵢ=1", position: "top" }} />
                <ReferenceLine y={result.grand_mean} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" label={{ value: "Grand mean", position: "right" }} />
                {groups.map(([cls, points]) => (
                  <Scatter key={cls} name={cls} data={points} fill={classColor(cls)} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Larger circles indicate higher stability (lower S²<sub>di</sub>). Top-left quadrant = high-yielding & stable.
          </p>
        </CardContent>
      </Card>

      {/* Table */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="font-serif text-lg">Per-genotype stability</CardTitle>
          <ExportToolbar
            onCsv={() => downloadCsv(`stability_${result.trait}.csv`, result.genotype_stability as unknown as Record<string, unknown>[])}
            onCopy={() => navigator.clipboard.writeText(result.interpretation)}
          />
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead><SortBtn k="genotype" cur={sortKey} asc={sortAsc} onClick={toggleSort}>Genotype</SortBtn></TableHead>
                <TableHead className="text-right"><SortBtn k="mean" cur={sortKey} asc={sortAsc} onClick={toggleSort}>Mean</SortBtn></TableHead>
                <TableHead className="text-right"><SortBtn k="bi" cur={sortKey} asc={sortAsc} onClick={toggleSort}>bᵢ</SortBtn></TableHead>
                <TableHead className="text-right"><SortBtn k="s2di" cur={sortKey} asc={sortAsc} onClick={toggleSort}>S²<sub>di</sub></SortBtn></TableHead>
                <TableHead className="text-right"><SortBtn k="shukla_variance" cur={sortKey} asc={sortAsc} onClick={toggleSort}>Shukla σ²</SortBtn></TableHead>
                <TableHead className="text-right"><SortBtn k="rank" cur={sortKey} asc={sortAsc} onClick={toggleSort}>Rank</SortBtn></TableHead>
                <TableHead>Class</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sortedRows.map((r) => {
                const isBest = bestSet.has(r.genotype);
                return (
                  <TableRow key={r.genotype} className={isBest ? "bg-emerald-50/40 dark:bg-emerald-900/10" : ""}>
                    <TableCell className="font-medium">
                      {r.genotype}
                      {isBest && <Badge variant="outline" className="ml-2 border-emerald-500 text-emerald-700 dark:text-emerald-300 text-[10px]">Top</Badge>}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{fmt(r.mean, 2)}</TableCell>
                    <TableCell className="text-right tabular-nums">{fmt(r.bi, 3)}</TableCell>
                    <TableCell className="text-right tabular-nums">{fmt(r.s2di, 3)}</TableCell>
                    <TableCell className="text-right tabular-nums">{r.shukla_variance != null ? fmt(r.shukla_variance, 3) : "—"}</TableCell>
                    <TableCell className="text-right tabular-nums">{r.rank}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className={classBadge(r.stability_class)}>
                        {r.stability_class}
                      </Badge>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}

function SortBtn({
  k, cur, asc, onClick, children,
}: { k: SortKey; cur: SortKey; asc: boolean; onClick: (k: SortKey) => void; children: React.ReactNode }) {
  const active = cur === k;
  return (
    <button
      type="button"
      onClick={() => onClick(k)}
      className={`inline-flex items-center gap-1 hover:text-foreground transition-colors ${active ? "text-foreground" : "text-muted-foreground"}`}
    >
      {children}
      <ArrowUpDown className={`h-3 w-3 ${active ? "opacity-100" : "opacity-50"}`} />
      {active && <span className="sr-only">{asc ? "ascending" : "descending"}</span>}
    </button>
  );
}
