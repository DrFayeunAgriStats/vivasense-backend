import React, { useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, ChevronDown, ChevronUp } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { TraitResult } from "@/services/geneticsUploadApi";

type NumericRecord = Record<string, unknown>;

type AssumptionTestResult = {
  statistic?: number | null;
  p_value?: number | null;
  pValue?: number | null;
  test_name?: string | null;
  conclusion?: string | null;
  passed?: boolean | null;
};

type GenotypeStat = {
  genotype?: string | null;
  mean?: number | null;
  median?: number | null;
  q1?: number | null;
  q3?: number | null;
  min?: number | null;
  max?: number | null;
  sd?: number | null;
  n?: number | null;
};

type BoxplotTooltipDatum = GenotypeStat & { genotypeLabel?: string };
type BoxplotTooltipItem = { payload?: BoxplotTooltipDatum };

interface AssumptionDiagnosticsDashboardProps {
  traitResult?: TraitResult | null;
  result?: Record<string, unknown> | null;
}

const BOX_PALETTE = ["#2563eb", "#0f766e", "#7c3aed", "#c2410c", "#059669", "#be185d", "#475569", "#8b5cf6"];

function isNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function asRecord(value: unknown): NumericRecord | null {
  return value && typeof value === "object" ? (value as NumericRecord) : null;
}

function asNumericArray(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value.map((entry) => Number(entry)).filter((entry) => Number.isFinite(entry));
}

function formatValue(value: unknown, digits = 3): string {
  if (!isNumber(Number(value))) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return "—";
    return parsed.toFixed(digits);
  }
  return Number(value).toFixed(digits);
}

function formatPValue(value: unknown): string {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "—";
  if (parsed < 0.0001) return "<0.0001";
  return parsed.toFixed(4);
}

function getTestRecord(tests: NumericRecord | null, keys: string[]): AssumptionTestResult | null {
  if (!tests) return null;
  for (const key of keys) {
    const candidate = asRecord(tests[key]);
    if (candidate) return candidate as AssumptionTestResult;
  }
  return null;
}

function getResultBadge(test: AssumptionTestResult | null) {
  const pValue = Number(test?.p_value ?? test?.pValue);
  if (!Number.isFinite(pValue)) {
    return (
      <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-600 hover:bg-slate-50">
        Unavailable
      </Badge>
    );
  }

  const passed = pValue > 0.05;
  return passed ? (
    <Badge className="bg-emerald-600 text-white hover:bg-emerald-600">
      <CheckCircle2 className="mr-1 h-3.5 w-3.5" />
      Passed
    </Badge>
  ) : (
    <Badge className="bg-amber-500 text-white hover:bg-amber-500">
      <AlertTriangle className="mr-1 h-3.5 w-3.5" />
      Violated
    </Badge>
  );
}

function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-border/70 bg-background p-4 shadow-sm">
      <h4 className="mb-3 text-sm font-semibold text-foreground">{title}</h4>
      {children}
    </div>
  );
}

function formatNumber(value: unknown, digits = 2): string {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return "—";
  return parsed.toFixed(digits);
}

function CustomBoxPlotTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: BoxplotTooltipItem[];
}) {
  if (!active || !payload?.[0]?.payload) return null;
  const d = payload[0].payload;

  return (
    <div className="rounded-lg border border-border bg-background p-3 text-sm shadow-md">
      <p className="mb-1 font-semibold text-foreground">{String(d.genotypeLabel ?? d.genotype ?? "Group")}</p>
      <p>Max: {formatNumber(d.max)}</p>
      <p>Q3: {formatNumber(d.q3)}</p>
      <p>Median: {formatNumber(d.median)}</p>
      <p>Q1: {formatNumber(d.q1)}</p>
      <p>Min: {formatNumber(d.min)}</p>
    </div>
  );
}

function BoxPlotShape(props: unknown) {
  const shapeProps = props as {
    x?: number;
    width?: number;
    payload?: GenotypeStat & { genotypeLabel?: string };
    yAxis?: { scale?: (value: number) => number };
  };
  const { x, width, payload, yAxis } = shapeProps;
  const scale = yAxis?.scale;

  if (!scale || x == null || width == null || !payload) return null;

  const min = Number(payload.min);
  const q1 = Number(payload.q1);
  const median = Number(payload.median);
  const q3 = Number(payload.q3);
  const max = Number(payload.max);

  if (![min, q1, median, q3, max].every(Number.isFinite)) return null;

  const centerX = x + width / 2;
  const boxWidth = width * 0.6;
  const boxX = centerX - boxWidth / 2;
  const capWidth = boxWidth * 0.4;
  const q1Y = scale(q1);
  const q3Y = scale(q3);
  const medianY = scale(median);
  const minY = scale(min);
  const maxY = scale(max);
  const boxY = Math.min(q1Y, q3Y);
  const boxHeight = Math.max(Math.abs(q1Y - q3Y), 1);

  return (
    <g>
      <rect
        x={boxX}
        y={boxY}
        width={boxWidth}
        height={boxHeight}
        fill="rgba(59,130,246,0.15)"
        stroke="#3b82f6"
        strokeWidth={1.5}
        rx={2}
      />
      <line
        x1={boxX}
        x2={boxX + boxWidth}
        y1={medianY}
        y2={medianY}
        stroke="#1e40af"
        strokeWidth={2}
      />
      <line x1={centerX} x2={centerX} y1={q3Y} y2={maxY} stroke="#3b82f6" strokeWidth={1} />
      <line
        x1={centerX - capWidth / 2}
        x2={centerX + capWidth / 2}
        y1={maxY}
        y2={maxY}
        stroke="#3b82f6"
        strokeWidth={1.5}
      />
      <line x1={centerX} x2={centerX} y1={q1Y} y2={minY} stroke="#3b82f6" strokeWidth={1} />
      <line
        x1={centerX - capWidth / 2}
        x2={centerX + capWidth / 2}
        y1={minY}
        y2={minY}
        stroke="#3b82f6"
        strokeWidth={1.5}
      />
    </g>
  );
}

function AssumptionSummaryTable({ tests }: { tests: NumericRecord | null }) {
  const shapiro = getTestRecord(tests, ["shapiro", "shapiro_wilk", "normality"]);
  const levene = getTestRecord(tests, ["levene", "homogeneity", "bartlett"]);

  const rows = [
    {
      name: "Shapiro-Wilk",
      label: "Normality",
      test: shapiro,
    },
    {
      name: "Levene",
      label: "Homogeneity of variance",
      test: levene,
    },
  ];

  return (
    <SectionCard title="Assumption Tests Summary">
      <div className="overflow-hidden rounded-lg border border-border/60">
        <table className="w-full text-sm">
          <thead className="bg-muted/60">
            <tr>
              <th className="px-3 py-2 text-left font-semibold text-foreground">Test Name</th>
              <th className="px-3 py-2 text-right font-semibold text-foreground">Statistic</th>
              <th className="px-3 py-2 text-right font-semibold text-foreground">p-value</th>
              <th className="px-3 py-2 text-right font-semibold text-foreground">Result</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={row.name} className={index % 2 === 0 ? "bg-background" : "bg-muted/25"}>
                <td className="px-3 py-2 text-foreground">
                  <div className="font-medium">{row.name}</div>
                  <div className="text-xs text-muted-foreground">{row.label}</div>
                </td>
                <td className="px-3 py-2 text-right font-mono text-xs text-foreground">
                  {formatValue(row.test?.statistic, 4)}
                </td>
                <td className="px-3 py-2 text-right font-mono text-xs text-foreground">
                  {formatPValue(row.test?.p_value ?? row.test?.pValue)}
                </td>
                <td className="px-3 py-2 text-right">{getResultBadge(row.test)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        A test is considered passed when p &gt; 0.05 and violated when p ≤ 0.05.
      </p>
    </SectionCard>
  );
}

function TreatmentBoxplots({ stats }: { stats: GenotypeStat[] }) {
  if (stats.length === 0) {
    return (
      <SectionCard title="Treatment Boxplots">
        <p className="text-sm text-muted-foreground">Boxplot data not available for this trait.</p>
      </SectionCard>
    );
  }

  const entries = stats
    .map((entry, index) => ({
      genotype: String(entry.genotype ?? `Group ${index + 1}`),
      min: Number(entry.min),
      q1: Number(entry.q1),
      median: Number(entry.median),
      q3: Number(entry.q3),
      max: Number(entry.max),
    }))
    .filter((entry) => [entry.min, entry.q1, entry.median, entry.q3, entry.max].every(Number.isFinite));

  if (entries.length === 0) {
    return (
      <SectionCard title="Treatment Boxplots">
        <p className="text-sm text-muted-foreground">Genotype summary statistics were incomplete.</p>
      </SectionCard>
    );
  }

  const rotateLabels = entries.length > 8;

  return (
    <SectionCard title="Treatment Boxplots">
      <p className="mb-3 text-xs text-muted-foreground">
        Boxplots summarize the observed trait distribution per genotype using min, quartiles, median, and max.
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={entries} margin={{ top: 12, right: 18, bottom: rotateLabels ? 72 : 34, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="genotype"
            tick={{ fontSize: 12 }}
            angle={rotateLabels ? -45 : 0}
            textAnchor={rotateLabels ? "end" : "middle"}
            interval={0}
            height={rotateLabels ? 64 : 32}
          />
          <YAxis
            label={{ value: "Trait value", angle: -90, position: "insideLeft" }}
            domain={[(dataMin: number) => dataMin * 0.95, (dataMax: number) => dataMax * 1.05]}
            tick={{ fontSize: 11 }}
            width={48}
          />
          <Tooltip content={<CustomBoxPlotTooltip />} />
          <Bar dataKey="q3" shape={<BoxPlotShape />} isAnimationActive={false} fill="transparent" />
        </ComposedChart>
      </ResponsiveContainer>
    </SectionCard>
  );
}

function ResidualHistogram({ residuals, shapiroPValue }: { residuals: number[]; shapiroPValue?: number | null }) {
  if (residuals.length === 0) {
    return (
      <SectionCard title="Residual Histogram">
        <p className="text-sm text-muted-foreground">Residual values are not available.</p>
      </SectionCard>
    );
  }

  const min = Math.min(...residuals);
  const max = Math.max(...residuals);
  const binCount = Math.max(1, Math.ceil(Math.log2(residuals.length) + 1));
  const range = max - min;
  const binWidth = range === 0 ? 1 : range / binCount;

  const bins = Array.from({ length: binCount }, (_, index) => {
    const start = min + index * binWidth;
    const end = index === binCount - 1 ? max : start + binWidth;
    return {
      bin: `${formatValue(start, 2)} to ${formatValue(end, 2)}`,
      start,
      end,
      count: residuals.filter((value) => (index === binCount - 1 ? value >= start && value <= end : value >= start && value < end)).length,
    };
  });

  return (
    <SectionCard title="Residual Histogram">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span>Sturges bins: {binCount}</span>
        <span>•</span>
        <span>Shapiro-Wilk p-value: {formatPValue(shapiroPValue)}</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={bins} margin={{ top: 12, right: 18, bottom: 48, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="bin" angle={-35} textAnchor="end" height={60} tick={{ fontSize: 11 }} interval={0} />
          <YAxis allowDecimals={false} tick={{ fontSize: 11 }} label={{ value: "Frequency", angle: -90, position: "insideLeft" }} />
          <Tooltip
            contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)" }}
            formatter={(value: unknown) => [String(value), "Count"]}
          />
          <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </SectionCard>
  );
}

function ResidualVsFittedPlot({
  residuals,
  fittedValues,
  levenePValue,
}: {
  residuals: number[];
  fittedValues: number[];
  levenePValue?: number | null;
}) {
  if (residuals.length === 0 || fittedValues.length === 0) {
    return (
      <SectionCard title="Residual vs Fitted Plot">
        <p className="text-sm text-muted-foreground">Residual or fitted values are not available.</p>
      </SectionCard>
    );
  }

  const pairCount = Math.min(residuals.length, fittedValues.length);
  const data = Array.from({ length: pairCount }, (_, index) => ({
    fitted: fittedValues[index],
    residual: residuals[index],
  })).filter((point) => Number.isFinite(point.fitted) && Number.isFinite(point.residual));

  if (data.length === 0) {
    return (
      <SectionCard title="Residual vs Fitted Plot">
        <p className="text-sm text-muted-foreground">Residual/fitted pairs could not be plotted.</p>
      </SectionCard>
    );
  }

  const fittedValuesOnly = data.map((point) => point.fitted);
  const residualValuesOnly = data.map((point) => point.residual);
  const xMin = Math.min(...fittedValuesOnly);
  const xMax = Math.max(...fittedValuesOnly);
  const yMin = Math.min(...residualValuesOnly, 0);
  const yMax = Math.max(...residualValuesOnly, 0);

  return (
    <SectionCard title="Residual vs Fitted Plot">
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span>Levene p-value: {formatPValue(levenePValue)}</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 12, right: 18, bottom: 40, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="fitted"
            domain={[xMin, xMax]}
            tick={{ fontSize: 11 }}
            label={{ value: "Fitted Values", position: "insideBottom", offset: -4 }}
          />
          <YAxis
            type="number"
            dataKey="residual"
            domain={[yMin, yMax]}
            tick={{ fontSize: 11 }}
            label={{ value: "Residuals", angle: -90, position: "insideLeft" }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)" }}
            formatter={(value: unknown, name: string) => [formatValue(value, 4), name === "fitted" ? "Fitted" : "Residual"]}
          />
          <ReferenceLine y={0} stroke="#dc2626" strokeDasharray="5 5" />
          <Scatter data={data} fill="#2563eb" fillOpacity={0.75} />
        </ScatterChart>
      </ResponsiveContainer>
    </SectionCard>
  );
}

export function AssumptionDiagnosticsDashboard({ traitResult, result }: AssumptionDiagnosticsDashboardProps) {
  const traitScopedResult = traitResult?.analysis_result?.result as NumericRecord | undefined;
  const directResult = (result ?? null) as NumericRecord | null;
  const resolvedResult = traitScopedResult ?? directResult ?? null;
  const [open, setOpen] = useState(false);

  const residuals = useMemo(() => {
    const diagnostics = asRecord(resolvedResult?.diagnostics);
    return asNumericArray(resolvedResult?.residuals ?? diagnostics?.residuals);
  }, [resolvedResult]);

  const fittedValues = useMemo(() => {
    const diagnostics = asRecord(resolvedResult?.diagnostics);
    return asNumericArray(
      resolvedResult?.fitted_values ??
      resolvedResult?.fitted ??
      diagnostics?.fitted_values ??
      diagnostics?.fitted
    );
  }, [resolvedResult]);

  const assumptionTests = asRecord(
    resolvedResult?.assumption_tests ??
    resolvedResult?.assumption_checks ??
    resolvedResult?.assumptions
  );
  const perGenotypeStats = Array.isArray(resolvedResult?.per_genotype_stats)
    ? (resolvedResult?.per_genotype_stats as GenotypeStat[])
    : [];

  const hasPlots = residuals.length > 0 && fittedValues.length > 0;
  const hasAssumptionTests = assumptionTests != null && Object.keys(assumptionTests).length > 0;

  if (!hasPlots && !hasAssumptionTests && perGenotypeStats.length === 0) return null;

  const shapiro = getTestRecord(assumptionTests, ["shapiro", "shapiro_wilk", "normality"]);
  const levene = getTestRecord(assumptionTests, ["levene", "homogeneity", "bartlett"]);

  const shapiroP = Number(shapiro?.p_value ?? shapiro?.pValue);
  const leveneP = Number(levene?.p_value ?? levene?.pValue);

  return (
    <Card className="border-border/70 shadow-sm">
      <CardHeader className="cursor-pointer select-none">
        <button
          type="button"
          className="flex w-full items-center justify-between gap-3 text-left"
          onClick={() => setOpen((current) => !current)}
          aria-expanded={open}
          aria-label="Assumption Diagnostics"
        >
          <CardTitle className="text-base font-semibold text-foreground">Assumption Diagnostics</CardTitle>
          {open ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
        </button>
      </CardHeader>
      {open && (
        <CardContent className="pt-0">
          <div className="grid gap-4 lg:grid-cols-2">
            {hasAssumptionTests && <AssumptionSummaryTable tests={assumptionTests} />}
            <TreatmentBoxplots stats={perGenotypeStats} />
            {hasPlots && <ResidualHistogram residuals={residuals} shapiroPValue={shapiroP} />}
            {hasPlots && (
              <ResidualVsFittedPlot residuals={residuals} fittedValues={fittedValues} levenePValue={leveneP} />
            )}
          </div>
        </CardContent>
      )}
    </Card>
  );
}
