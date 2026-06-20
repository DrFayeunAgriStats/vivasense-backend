import React, { useState } from "react";
import { ChevronDown, ChevronRight, Code, FlaskConical, ShieldCheck, ShieldX, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { HtmlTablesSection } from "./HtmlTablesSection";
import { generatePublishableHtmlTables } from "./utils/generatePublishableTables";
import { TableDownloadMenu } from "./results/TableDownloadMenu";
import { FigureDownloadMenu } from "./results/FigureDownloadMenu";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, ReferenceLine,
  ErrorBar, Cell, Legend, ComposedChart, Line,
} from "recharts";


/* ── Formatting helpers ────────────────────────────────────────────── */

const fmt0 = (v: unknown): string =>
  v == null || v === "" ? "—" : Math.round(Number(v)).toString();

const fmt2 = (v: unknown): string => {
  if (v == null || v === "") return "—";
  const n = Number(v);
  return typeof n === "number" && !isNaN(n) ? n.toFixed(2) : "N/A";
};

const fmt4 = (v: unknown): string => {
  if (v == null || v === "") return "—";
  const n = Number(v);
  return typeof n === "number" && !isNaN(n) ? n.toFixed(4) : "N/A";
};

const fmtN = (v: unknown, d = 2): string => {
  if (v == null || v === "") return "—";
  const n = Number(v);
  return typeof n === "number" && !isNaN(n) ? n.toFixed(d) : "N/A";
};

const stars = (p: unknown): string => {
  if (p == null) return "";
  const n = Number(p);
  if (n < 0.001) return " ***";
  if (n < 0.01) return " **";
  if (n < 0.05) return " *";
  return " ns";
};

const pDisplay = (p: unknown): string => {
  if (p == null) return "—";
  const n = Number(p);
  if (n < 0.0001) return "<0.0001" + stars(p);
  return fmt4(p) + stars(p);
};

const cleanSource = (s: string) => s.replace(/Q\('(.+?)'\)/g, "$1");

function toRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function normalizeAssumptions(input: unknown): Record<string, unknown> | null {
  const src = toRecord(input);
  if (!src) return null;

  // Already in the newer dashboard shape
  if (src.normality || src.homogeneity || src.overall) return src;

  // Legacy flat shape: shapiro_wilk / shapiro and bartlett / levene
  const shapiro = toRecord(src.shapiro_wilk) ?? toRecord(src.shapiro) ?? null;
  const homogeneity =
    toRecord(src.bartlett) ??
    toRecord(src.levene) ??
    toRecord(src.homogeneity) ??
    null;

  if (!shapiro && !homogeneity) return src;

  const normalityPassed = typeof shapiro?.passed === "boolean"
    ? (shapiro.passed as boolean)
    : Number(shapiro?.p_value) >= 0.05;
  const homogeneityPassed = typeof homogeneity?.passed === "boolean"
    ? (homogeneity.passed as boolean)
    : Number(homogeneity?.p_value) >= 0.05;
  const bothPassed = Boolean(normalityPassed) && Boolean(homogeneityPassed);

  return {
    normality: shapiro
      ? {
          ...shapiro,
          test: shapiro.test ?? shapiro.test_name ?? "Shapiro-Wilk",
          passed: normalityPassed,
        }
      : undefined,
    homogeneity: homogeneity
      ? {
          ...homogeneity,
          test: homogeneity.test ?? homogeneity.test_name ?? (src.bartlett ? "Bartlett" : "Levene"),
          passed: homogeneityPassed,
        }
      : undefined,
    overall: {
      passed: bothPassed,
      interpretation: bothPassed
        ? "Both normality and homogeneity assumptions are supported."
        : "One or more assumptions may be violated. Interpret inferential results with caution.",
    },
  };
}

function asDiagnosticsRows(value: unknown): Array<Record<string, unknown>> {
  if (!Array.isArray(value)) return [];
  return value.filter((x): x is Record<string, unknown> => !!x && typeof x === "object");
}

/* ── adaptBackendResult ────────────────────────────────────────────── */

function adaptBackendResult(raw: Record<string, unknown>): Record<string, unknown> {
  const r = { ...raw };

  // Normalize anova_table → anova if needed
  if (r.anova_table && !r.anova) r.anova = r.anova_table;

  // Normalize means_separation → means + letters
  if (r.means_separation && !r.means) {
    const sep = r.means_separation as Record<string, unknown>;
    if (sep.means) r.means = sep.means;
    if (sep.letters) r.letters = sep.letters;
  }

  // Normalize interaction_means
  if (r.interaction_means && !r.interactions) r.interactions = r.interaction_means;

  // Normalize assumption fields from multiple API response shapes.
  const tables = toRecord(r.tables);
  const rawAssumptions =
    r.assumptions ??
    r.assumption_checks ??
    r.assumption_tests ??
    tables?.assumptions ??
    tables?.assumption_checks ??
    tables?.assumption_tests;
  const normalizedAssumptions = normalizeAssumptions(rawAssumptions);
  if (normalizedAssumptions && !r.assumptions) r.assumptions = normalizedAssumptions;

  return r;
}

/* ── Reusable sub-components ───────────────────────────────────────── */

function PubTable({
  headers,
  rows,
  caption,
  downloadTitle,
}: {
  headers: string[];
  rows: React.ReactNode[][];
  caption?: string;
  downloadTitle?: string;
}) {
  // Build plain-text rows for download (strip React nodes to strings)
  const dlRows = rows.map((row) =>
    row.map((cell) => {
      if (cell == null) return "—";
      if (typeof cell === "object" && "props" in (cell as any)) {
        const props = (cell as any).props;
        return String(props?.children ?? "");
      }
      return String(cell);
    })
  );

  return (
    <div className="overflow-x-auto my-3">
      {downloadTitle && (
        <div className="flex justify-end mb-1">
          <TableDownloadMenu title={downloadTitle} headers={headers} rows={dlRows} />
        </div>
      )}
      <table className="w-full text-sm border-collapse border border-border">
        <thead>
          <tr className="bg-muted">
            {headers.map((h, i) => (
              <th
                key={i}
                className="border border-border px-3 py-2 text-left font-semibold text-foreground"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? "bg-background" : "bg-muted/40"}>
              {row.map((cell, j) => (
                <td
                  key={j}
                  className={`border border-border/60 px-3 py-1.5 text-foreground ${
                    j > 0 ? "text-right font-mono text-xs" : ""
                  }`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {caption && (
        <p className="text-xs text-muted-foreground mt-1 italic">{caption}</p>
      )}
    </div>
  );
}

function CollapsibleSection({
  title,
  icon,
  defaultOpen = true,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="bg-card rounded-lg border border-border shadow-sm mb-4 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full bg-muted border-b border-border px-4 py-2.5 flex items-center gap-2 hover:bg-muted/80 transition-colors text-left"
      >
        {open ? (
          <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
        )}
        {icon}
        <h3 className="font-semibold text-foreground text-sm">{title}</h3>
      </button>
      {open && <div className="px-4 py-3">{children}</div>}
    </div>
  );
}

/* ── Section renderers ─────────────────────────────────────────────── */

function ExperimentSummary({ result }: { result: Record<string, unknown> }) {
  const anova = result.anova as Record<string, Record<string, unknown>> | undefined;
  const descStats = result.descriptive_stats as Record<string, unknown> | undefined;
  const overall = descStats?.overall as Record<string, unknown> | undefined;

  // Extract grand mean & CV from descriptive stats
  const grandMean = overall?.mean;
  const cv = overall?.cv;

  // Extract overall F and p from anova
  let overallF: unknown = result.f_value;
  let overallP: unknown = result.f_pvalue;
  if (!overallF && anova?.F) {
    const fObj = anova.F as Record<string, unknown>;
    const firstSource = Object.keys(fObj).find((k) => k !== "Residual");
    if (firstSource) {
      overallF = fObj[firstSource];
      overallP = (anova["PR(>F)"] as Record<string, unknown>)?.[firstSource];
    }
  }

  const sig = overallP != null ? Number(overallP) < 0.05 : (result.significant as boolean);

  return (
    <div
      className={`rounded-lg p-4 mb-4 text-sm border ${
        sig
          ? "bg-green-50 border-green-200 dark:bg-green-950/30 dark:border-green-800"
          : "bg-yellow-50 border-yellow-200 dark:bg-yellow-950/30 dark:border-yellow-800"
      }`}
    >
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-x-6 gap-y-2">
        {result.design && (
          <div><span className="font-semibold text-foreground">Design:</span> <span className="text-muted-foreground">{String(result.design).replace(/_/g, " ")}</span></div>
        )}
        {result.response && (
          <div><span className="font-semibold text-foreground">Response:</span> <span className="text-muted-foreground">{String(result.response)}</span></div>
        )}
        {result.treatment && (
          <div><span className="font-semibold text-foreground">Treatment:</span> <span className="text-muted-foreground">{String(result.treatment)}</span></div>
        )}
        {result.n != null && (
          <div><span className="font-semibold text-foreground">n:</span> <span className="text-muted-foreground">{String(result.n)}</span></div>
        )}
        {grandMean != null && (
          <div><span className="font-semibold text-foreground">Grand Mean:</span> <span className="text-muted-foreground">{fmt2(grandMean)}</span></div>
        )}
        {cv != null && (
          <div><span className="font-semibold text-foreground">CV%:</span> <span className="text-muted-foreground">{fmt2(cv)}%</span></div>
        )}
        {result.r_squared != null && (
          <div><span className="font-semibold text-foreground">R²:</span> <span className="text-muted-foreground">{fmt4(result.r_squared)}</span></div>
        )}
        {overallF != null && (
          <div><span className="font-semibold text-foreground">F:</span> <span className="text-muted-foreground">{fmt2(overallF)}</span></div>
        )}
        {overallP != null && (
          <div><span className="font-semibold text-foreground">p-value:</span> <span className="text-muted-foreground">{pDisplay(overallP)}</span></div>
        )}
      </div>
      {result.formula && (
        <div className="mt-2 pt-2 border-t border-border/40">
          <span className="font-semibold text-foreground">Formula:</span>{" "}
          <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono text-muted-foreground">
            {String(result.formula)}
          </code>
        </div>
      )}
    </div>
  );
}

function ANOVATableDisplay({ anova }: { anova: Record<string, unknown> }) {
  if (!anova || typeof anova !== "object") return null;
  const dfObj = (anova.df ?? {}) as Record<string, unknown>;
  const sumSqObj = (anova.sum_sq ?? {}) as Record<string, unknown>;
  const msObj = (anova.mean_sq ?? anova.MS ?? {}) as Record<string, unknown>;
  const fObj = (anova.F ?? {}) as Record<string, unknown>;
  const prObj = (anova["PR(>F)"] ?? {}) as Record<string, unknown>;

  const sources = Object.keys(dfObj);
  if (!sources.length) return null;

  // Compute MS if not provided
  const getMS = (src: string): unknown => {
    if (msObj[src] != null) return msObj[src];
    const ss = sumSqObj[src] as number | null;
    const df = dfObj[src] as number | null;
    if (ss != null && df != null && df > 0) return ss / df;
    return null;
  };

  const rows: React.ReactNode[][] = sources.map((src) => [
    cleanSource(src),
    fmt0(dfObj[src]),
    fmt2(sumSqObj[src]),
    fmt2(getMS(src)),
    fObj[src] != null ? fmt2(fObj[src]) : "—",
    prObj[src] != null ? pDisplay(prObj[src]) : "—",
  ]);

  return (
    <PubTable
      headers={["Source", "DF", "SS", "MS", "F", "p-value"]}
      rows={rows}
      caption="Significance: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant"
      downloadTitle="ANOVA_Table"
    />
  );
}

function TukeyHSDDisplay({
  means,
  letters,
  treatment,
  descStats,
}: {
  means: Record<string, unknown>;
  letters?: Record<string, unknown> | null;
  treatment?: string;
  descStats?: Record<string, unknown> | null;
}) {
  if (!means || typeof means !== "object") return null;
  const treatKey =
    treatment && means[treatment] ? treatment : Object.keys(means)[0];
  const meansObj = (means[treatKey] ?? means) as Record<string, number>;
  const lettersObj = ((letters && ((letters as any)[treatKey] || letters)) ||
    {}) as Record<string, string>;

  // Try to get per-group stats for SE
  const groupStats = descStats
    ? ((descStats as any)[treatKey] as Record<string, Record<string, unknown>> | undefined)
    : null;

  const levels = Object.keys(meansObj);
  if (!levels.length) return null;

  const hasGroupStats = groupStats && Object.keys(groupStats).length > 0;

  const headers = hasGroupStats
    ? [treatKey || "Group", "n", "Mean", "SE", "Tukey Group"]
    : [treatKey || "Group", "Mean", "Tukey Group"];

  const rows: React.ReactNode[][] = levels
    .sort((a, b) => (meansObj[b] ?? 0) - (meansObj[a] ?? 0))
    .map((lvl) => {
      const gs = groupStats?.[lvl];
      if (hasGroupStats) {
        return [
          lvl,
          gs ? String(gs.count ?? gs.n ?? "—") : "—",
          fmt2(meansObj[lvl]),
          gs?.sem != null ? fmt2(gs.sem) : "—",
          <span className="font-bold text-primary">{lettersObj[lvl] || "—"}</span>,
        ];
      }
      return [
        lvl,
        fmt2(meansObj[lvl]),
        <span className="font-bold text-primary">{lettersObj[lvl] || "—"}</span>,
      ];
    });

  return (
    <PubTable
      headers={headers}
      rows={rows}
      caption="Means sharing the same letter are not significantly different (Tukey HSD, α=0.05)"
      downloadTitle="Treatment_Means_Tukey"
    />
  );
}

function InteractionMeansDisplay({ interactions }: { interactions: unknown }) {
  if (!interactions || typeof interactions !== "object") return null;

  // interactions can be: { "A:B": { "a1:b1": value, ... } } or array
  const entries = Object.entries(interactions as Record<string, unknown>);
  if (!entries.length) return null;

  return (
    <>
      {entries.map(([key, val]) => {
        if (!val || typeof val !== "object") return null;
        const meansObj = val as Record<string, unknown>;
        const levels = Object.keys(meansObj);
        if (!levels.length) return null;

        const rows: React.ReactNode[][] = levels
          .sort((a, b) => Number(meansObj[b] ?? 0) - Number(meansObj[a] ?? 0))
          .map((lvl) => [lvl, fmt2(meansObj[lvl])]);

        return (
          <div key={key} className="mb-3">
            <p className="text-xs font-semibold text-muted-foreground mb-1">
              Interaction: {cleanSource(key)}
            </p>
            <PubTable headers={["Combination", "Mean"]} rows={rows} downloadTitle={`Interaction_Means_${cleanSource(key)}`} />
          </div>
        );
      })}
    </>
  );
}

function EffectSizesBox({
  effectSizes,
  rSquared,
}: {
  effectSizes: Record<string, unknown>;
  rSquared?: unknown;
}) {
  if (!effectSizes || typeof effectSizes !== "object") return null;
  const entries = Object.entries(effectSizes).filter(([src]) => src !== "Residual");
  if (!entries.length) return null;

  return (
    <div className="space-y-3">
      {rSquared != null && (
        <div className="flex items-center gap-2 text-sm mb-2">
          <span className="font-semibold text-foreground">R²:</span>
          <span className="text-muted-foreground font-mono">{fmt4(rSquared)}</span>
          <span className="text-xs text-muted-foreground">
            ({(Number(rSquared) * 100).toFixed(1)}% variance explained)
          </span>
        </div>
      )}
      <PubTable
        headers={["Source", "η²", "ω²", "Cohen's f", "Interpretation"]}
        rows={entries.map(([src, es]) => {
          const obj = (typeof es === "object" && es ? es : {}) as Record<string, unknown>;
          return [
            cleanSource(src),
            fmt4(obj.eta_squared),
            fmt4(obj.omega_squared),
            fmt2(obj.cohens_f),
            <span
              className={`font-medium ${
                obj.interpretation === "large"
                  ? "text-green-600"
                  : obj.interpretation === "medium"
                  ? "text-yellow-600"
                  : "text-muted-foreground"
              }`}
            >
              {(obj.interpretation as string) || "—"}
            </span>,
          ];
        })}
        downloadTitle="Effect_Sizes"
      />
    </div>
  );
}

function AssumptionTests({ assumptions }: { assumptions: Record<string, unknown> }) {
  if (!assumptions || typeof assumptions !== "object") return null;

  const rows: React.ReactNode[][] = Object.entries(assumptions).map(
    ([key, val]) => {
      const obj = (typeof val === "object" && val ? val : {}) as Record<string, unknown>;
      return [
        key.charAt(0).toUpperCase() + key.slice(1),
        (obj.test_name as string) || "—",
        obj.statistic != null ? fmt4(obj.statistic) : "—",
        obj.p_value != null ? pDisplay(obj.p_value) : "—",
        obj.passed != null ? (
          obj.passed ? (
            <span className="text-green-600 font-medium">✓ Passed</span>
          ) : (
            <span className="text-destructive font-medium">✗ Failed</span>
          )
        ) : (
          "—"
        ),
        (obj.message as string) || "—",
      ];
    }
  );

  return (
    <PubTable
      headers={["Test", "Method", "Statistic", "p-value", "Result", "Note"]}
      rows={rows}
      downloadTitle="Assumption_Tests"
    />
  );
}

function DescriptiveStatsDisplay({ descStats }: { descStats: Record<string, unknown> }) {
  if (!descStats || typeof descStats !== "object") return null;

  const overall = descStats.overall as Record<string, unknown> | undefined;
  const groupKey = Object.keys(descStats).find((k) => k !== "overall");
  const groups = groupKey
    ? (descStats[groupKey] as Record<string, Record<string, unknown>>)
    : null;

  return (
    <>
      {overall && (
        <>
          <p className="text-xs font-semibold text-muted-foreground mb-1">Overall</p>
          <PubTable
            headers={["n", "Mean", "SD", "SEM", "CV (%)", "Min", "Max"]}
            rows={[
              [
                (overall.n as string) ?? "—",
                fmt2(overall.mean),
                fmt2(overall.std),
                fmt2(overall.sem),
                fmt2(overall.cv),
                fmt2(overall.min),
                fmt2(overall.max),
              ],
            ]}
            downloadTitle="Descriptive_Stats_Overall"
          />
        </>
      )}
      {groups && (
        <>
          <p className="text-xs font-semibold text-muted-foreground mt-3 mb-1">
            By {groupKey}
          </p>
          <PubTable
            headers={[groupKey!, "n", "Mean", "SD", "SEM", "Min", "Max", "CV (%)"]}
            rows={Object.entries(groups)
              .sort(
                ([, a], [, b]) =>
                  ((b.mean as number) ?? 0) - ((a.mean as number) ?? 0)
              )
              .map(([lvl, s]) => [
                lvl,
                (s.count ?? s.n ?? "—") as React.ReactNode,
                fmt2(s.mean),
                fmt2(s.std),
                fmt2(s.sem),
                fmt2(s.min),
                fmt2(s.max),
                fmt2(s.cv),
              ])}
            downloadTitle={`Descriptive_Stats_By_${groupKey}`}
          />
        </>
      )}
    </>
  );
}

/* ── Technical JSON toggle ─────────────────────────────────────────── */

function TechnicalJSON({ data }: { data: unknown }) {
  const [show, setShow] = useState(false);
  return (
    <div className="mt-6 border-t border-border pt-4">
      <button
        onClick={() => setShow(!show)}
        className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <Code className="w-3.5 h-3.5" />
        {show ? "Hide" : "Show"} technical JSON
      </button>
      {show && (
        <pre className="mt-2 p-3 bg-muted rounded-lg text-xs font-mono overflow-x-auto max-h-96 text-muted-foreground">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

/* ── Assumption Diagnostics Panel ────────────────────────────────────── */

function AssumptionDiagnosticsPanel({ data }: { data: Record<string, unknown> }) {
  const normalized = normalizeAssumptions(data) ?? data;
  const normality = normalized.normality as Record<string, unknown> | undefined;
  const homogeneity = normalized.homogeneity as Record<string, unknown> | undefined;
  const overall = normalized.overall as Record<string, unknown> | undefined;
  const reviewerMode = normalized.reviewer_mode as Record<string, unknown> | undefined;
  const reviewerSummary = String(
    reviewerMode?.summary ?? overall?.reviewer_summary ?? ""
  ).trim();

  if (!normality && !homogeneity) return null;

  const normalityPassed = (normality?.passed as boolean) ?? false;
  const homogeneityPassed = (homogeneity?.passed as boolean) ?? false;
  const bothPass = normalityPassed && homogeneityPassed;

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        {normality && (
          <div className="p-4 rounded-lg border bg-card">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium text-foreground">Normality (Shapiro-Wilk)</h4>
              {normalityPassed ? (
                <ShieldCheck className="w-5 h-5 text-green-600" />
              ) : (
                <ShieldX className="w-5 h-5 text-red-600" />
              )}
            </div>
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>p-value: <span className={`font-mono font-medium ${normalityPassed ? "text-green-600" : "text-red-600"}`}>{fmt4(normality.p_value)}</span></p>
              <p className="text-xs italic leading-relaxed">{String(normality.interpretation || "")}</p>
            </div>
          </div>
        )}
        {homogeneity && (
          <div className="p-4 rounded-lg border bg-card">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium text-foreground">Homogeneity of Variance ({String(homogeneity.test || "Test")})</h4>
              {homogeneityPassed ? (
                <ShieldCheck className="w-5 h-5 text-green-600" />
              ) : (
                <ShieldX className="w-5 h-5 text-red-600" />
              )}
            </div>
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>p-value: <span className={`font-mono font-medium ${homogeneityPassed ? "text-green-600" : "text-red-600"}`}>{fmt4(homogeneity.p_value)}</span></p>
              <p className="text-xs italic leading-relaxed">{String(homogeneity.interpretation || "")}</p>
            </div>
          </div>
        )}
      </div>
      {overall && (
        <div className={`p-4 rounded-lg border-l-4 ${bothPass ? "bg-green-50 border-green-300 dark:bg-green-950/30" : "bg-amber-50 border-amber-300 dark:bg-amber-950/30"}`}>
          {reviewerSummary && (
            <p className={`text-sm font-semibold mb-1 ${bothPass ? "text-green-900 dark:text-green-200" : "text-amber-900 dark:text-amber-200"}`}>
              {reviewerSummary}
            </p>
          )}
          <p className={`text-sm font-medium ${bothPass ? "text-green-900 dark:text-green-200" : "text-amber-900 dark:text-amber-200"}`}>
            {String(overall.interpretation || "Overall assessment unavailable")}
          </p>
        </div>
      )}
    </div>
  );
}

/* ── Treatment Boxplots ────────────────────────────────────────────── */

function TreatmentBoxplots({ stats }: { stats: Record<string, unknown>[] | undefined }) {
  if (!stats || stats.length === 0) return null;

  const data = (stats as Array<any>).map((s, idx) => ({
    index: idx,
    genotype: String(s.genotype),
    mean: Number(s.mean) ?? 0,
    se: Number(s.se) ?? 0,
    min: Number(s.min) ?? 0,
    max: Number(s.max) ?? 0,
    median: Number(s.median) ?? 0,
    q1: Number(s.q1) ?? 0,
    q3: Number(s.q3) ?? 0,
    n_reps: Number(s.n_reps) ?? 0,
  }));

  // Compute y-axis range with padding
  const allValues = data.flatMap(d => [d.min, d.q1, d.median, d.q3, d.max]);
  const minVal = Math.min(...allValues);
  const maxVal = Math.max(...allValues);
  const padding = (maxVal - minVal) * 0.1;

  // Custom SVG rendering for each box
  const BoxplotShape = (props: any) => {
    const { x, y, width, height, payload } = props;
    if (!payload) return null;

    const chartHeight = height;
    const yMin = payload.min;
    const yQ1 = payload.q1;
    const yMedian = payload.median;
    const yQ3 = payload.q3;
    const yMax = payload.max;

    // Convert data values to pixel coordinates (assume y-axis range is minVal to maxVal with padding)
    const range = maxVal + padding - (minVal - padding);
    const yToPixel = (val: number) => chartHeight - ((val - (minVal - padding)) / range) * chartHeight;

    const boxX = x + width * 0.25;
    const boxWidth = width * 0.5;

    const yPixelMin = yToPixel(yMin);
    const yPixelQ1 = yToPixel(yQ1);
    const yPixelMedian = yToPixel(yMedian);
    const yPixelQ3 = yToPixel(yQ3);
    const yPixelMax = yToPixel(yMax);

    return (
      <g>
        {/* Whisker lines (min to Q1, Q3 to max) */}
        <line x1={x + width / 2} y1={yPixelMin} x2={x + width / 2} y2={yPixelQ1} stroke="var(--muted-foreground)" strokeWidth={1} />
        <line x1={x + width / 2} y1={yPixelQ3} x2={x + width / 2} y2={yPixelMax} stroke="var(--muted-foreground)" strokeWidth={1} />

        {/* Whisker caps (min and max) */}
        <line x1={x + width * 0.4} y1={yPixelMin} x2={x + width * 0.6} y2={yPixelMin} stroke="var(--muted-foreground)" strokeWidth={1} />
        <line x1={x + width * 0.4} y1={yPixelMax} x2={x + width * 0.6} y2={yPixelMax} stroke="var(--muted-foreground)" strokeWidth={1} />

        {/* Box (Q1 to Q3) */}
        <rect
          x={boxX}
          y={yPixelQ3}
          width={boxWidth}
          height={yPixelQ1 - yPixelQ3}
          fill="rgba(59, 130, 246, 0.2)"
          stroke="var(--primary)"
          strokeWidth={1.5}
        />

        {/* Median line (bold) */}
        <line
          x1={boxX}
          y1={yPixelMedian}
          x2={boxX + boxWidth}
          y2={yPixelMedian}
          stroke="var(--foreground)"
          strokeWidth={2.5}
        />

        {/* Mean as small diamond */}
        <circle
          cx={x + width / 2}
          cy={yToPixel(payload.mean)}
          r={2.5}
          fill="var(--foreground)"
          opacity={0.7}
        />
      </g>
    );
  };

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground italic">
        Box spans Q1 to Q3, median line shown in bold, whiskers extend to min/max, mean marked as dot.
      </p>
      <div className="w-full overflow-x-auto">
        <ResponsiveContainer width={Math.max(600, data.length * 120)} height={350}>
          <ComposedChart data={data} margin={{ top: 20, right: 30, left: 60, bottom: 100 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="genotype"
              angle={-45}
              textAnchor="end"
              height={120}
              tick={{ fontSize: 12 }}
            />
            <YAxis
              domain={[minVal - padding, maxVal + padding]}
              label={{ value: "Trait Value", angle: -90, position: "insideLeft" }}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)" }}
              cursor={false}
              content={({ active, payload }) => {
                if (!active || !payload?.[0]) return null;
                const d = payload[0].payload;
                return (
                  <div className="p-2 bg-card border border-border rounded text-xs">
                    <p className="font-semibold">{d.genotype}</p>
                    <p>Min: {fmt2(d.min)}</p>
                    <p>Q1: {fmt2(d.q1)}</p>
                    <p className="font-semibold">Median: {fmt2(d.median)}</p>
                    <p>Q3: {fmt2(d.q3)}</p>
                    <p>Max: {fmt2(d.max)}</p>
                    <p className="text-muted-foreground">n = {d.n_reps ?? "—"}</p>
                  </div>
                );
              }}
            />
            <Bar dataKey="index" shape={<BoxplotShape />} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="text-xs text-muted-foreground">
        Five-number summary (min, Q1, median, Q3, max) per genotype with mean marked as dot.
      </div>
    </div>
  );
}

/* ── Residual Histogram ────────────────────────────────────────────── */

function ResidualHistogram({ residuals }: { residuals: unknown[] | undefined }) {
  if (!residuals || residuals.length === 0) return null;

  const residVals = (residuals as Array<unknown>).map((r) => Number(r)).filter((r) => isFinite(r));
  if (residVals.length === 0) return null;

  const min = Math.min(...residVals);
  const max = Math.max(...residVals);
  const binCount = Math.min(15, Math.ceil(Math.sqrt(residVals.length)));
  const binWidth = (max - min) / binCount;

  const bins: Array<{ bin: string; count: number }> = [];
  for (let i = 0; i < binCount; i++) {
    const binStart = min + i * binWidth;
    const binEnd = binStart + binWidth;
    const count = residVals.filter((v) => v >= binStart && v < binEnd).length;
    bins.push({
      bin: `${fmt2(binStart)}–${fmt2(binEnd)}`,
      count,
    });
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground italic">
        Residuals should approximately follow a normal distribution (bell-shaped curve centered near zero).
      </p>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={bins} margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="bin"
            angle={-45}
            textAnchor="end"
            height={80}
            tick={{ fontSize: 11 }}
          />
          <YAxis label={{ value: "Frequency", angle: -90, position: "insideLeft" }} />
          <Tooltip contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)" }} />
          <Bar dataKey="count" fill="var(--primary)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Residual vs Fitted Plot ────────────────────────────────────────── */

function ResidualVsFittedPlot({
  residuals,
  fitted,
}: {
  residuals: unknown[] | undefined;
  fitted: unknown[] | undefined;
}) {
  if (!residuals || !fitted || residuals.length === 0 || fitted.length === 0) return null;
  if (residuals.length !== fitted.length) return null;

  const data = residuals.map((r, idx) => ({
    fitted: Number(fitted[idx]),
    residual: Number(r),
  })).filter((d) => isFinite(d.fitted) && isFinite(d.residual));

  if (data.length === 0) return null;

  const fittedMin = Math.min(...data.map((d) => d.fitted));
  const fittedMax = Math.max(...data.map((d) => d.fitted));

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground italic">
        A random scatter around zero (horizontal line at residual = 0) supports the model assumptions. Patterns suggest potential violations.
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="fitted"
            type="number"
            label={{ value: "Fitted Values", position: "insideBottomRight", offset: -10 }}
          />
          <YAxis label={{ value: "Residuals", angle: -90, position: "insideLeft" }} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)" }}
            formatter={(value: unknown) => fmt4(value)}
          />
          <Scatter
            name="Residuals"
            data={data}
            fill="var(--primary)"
            fillOpacity={0.6}
          />
          <ReferenceLine
            y={0}
            stroke="var(--destructive)"
            strokeDasharray="5 5"
            label={{ value: "y = 0", position: "right", fill: "var(--foreground)" }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function ResidualsVsTreatmentPlot({ observations }: { observations: Array<Record<string, unknown>> }) {
  if (!observations.length) return null;
  const rows = observations
    .map((o, idx) => ({
      treatment: String(o.treatment ?? `Obs ${idx + 1}`),
      residual: Number(o.residual),
    }))
    .filter((r) => Number.isFinite(r.residual));
  if (!rows.length) return null;

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ScatterChart margin={{ top: 20, right: 30, left: 0, bottom: 80 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="treatment" angle={-45} textAnchor="end" height={90} interval={0} />
        <YAxis label={{ value: "Residual", angle: -90, position: "insideLeft" }} />
        <Tooltip contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)" }} />
        <ReferenceLine y={0} stroke="var(--destructive)" strokeDasharray="5 5" />
        <Scatter data={rows} fill="var(--primary)" fillOpacity={0.65} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function ScaleLocationPlot({ observations }: { observations: Array<Record<string, unknown>> }) {
  if (!observations.length) return null;
  const rows = observations
    .map((o) => {
      const fitted = Number(o.fitted);
      const std = Number(o.standardized_residual);
      return {
        fitted,
        sqrtAbsStd: Number.isFinite(std) ? Math.sqrt(Math.abs(std)) : NaN,
      };
    })
    .filter((r) => Number.isFinite(r.fitted) && Number.isFinite(r.sqrtAbsStd));
  if (!rows.length) return null;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="fitted" type="number" label={{ value: "Fitted Values", position: "insideBottomRight", offset: -10 }} />
        <YAxis label={{ value: "Sqrt(|Std Resid|)", angle: -90, position: "insideLeft" }} />
        <Tooltip contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)" }} formatter={(v: unknown) => fmt4(v)} />
        <Scatter data={rows} fill="#0f766e" fillOpacity={0.65} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

function CooksDistancePlot({ observations, threshold }: { observations: Array<Record<string, unknown>>; threshold: number }) {
  if (!observations.length) return null;
  const rows = observations
    .map((o, idx) => ({
      observation: Number(o.observation ?? idx + 1),
      cooksDistance: Number(o.cooks_distance),
    }))
    .filter((r) => Number.isFinite(r.cooksDistance));
  if (!rows.length) return null;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={rows} margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="observation" label={{ value: "Observation", position: "insideBottomRight", offset: -10 }} />
        <YAxis label={{ value: "Cook's Distance", angle: -90, position: "insideLeft" }} />
        <Tooltip contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)" }} formatter={(v: unknown) => fmt4(v)} />
        <Bar dataKey="cooksDistance" fill="#c2410c" radius={[3, 3, 0, 0]} />
        {Number.isFinite(threshold) && threshold > 0 && (
          <ReferenceLine y={threshold} stroke="var(--destructive)" strokeDasharray="5 5" />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function StandardizedResidualPlot({ observations }: { observations: Array<Record<string, unknown>> }) {
  if (!observations.length) return null;
  const rows = observations
    .map((o, idx) => ({
      observation: Number(o.observation ?? idx + 1),
      stdResidual: Number(o.standardized_residual),
    }))
    .filter((r) => Number.isFinite(r.stdResidual));
  if (!rows.length) return null;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={rows} margin={{ top: 20, right: 30, left: 0, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="observation" label={{ value: "Observation", position: "insideBottomRight", offset: -10 }} />
        <YAxis label={{ value: "Standardized Residual", angle: -90, position: "insideLeft" }} />
        <Tooltip contentStyle={{ backgroundColor: "var(--background)", border: "1px solid var(--border)" }} formatter={(v: unknown) => fmt4(v)} />
        <Line dataKey="stdResidual" stroke="#1d4ed8" strokeWidth={2} dot={false} />
        <ReferenceLine y={0} stroke="var(--foreground)" strokeDasharray="3 3" />
        <ReferenceLine y={3} stroke="var(--destructive)" strokeDasharray="5 5" />
        <ReferenceLine y={-3} stroke="var(--destructive)" strokeDasharray="5 5" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

/* ── Main component ────────────────────────────────────────────────── */

export interface VivaSenseResultsDisplayProps {
  result: Record<string, unknown> | null;
}


export default function VivaSenseResultsDisplay({
  result,
}: VivaSenseResultsDisplayProps) {
  if (!result) return null;

  // Multi-trait: result.results is a dict keyed by trait name
  if (result.results && typeof result.results === "object") {
    return (
      <div className="space-y-6">
        {Object.entries(result.results as Record<string, unknown>).map(
          ([trait, traitResult]) => (
            <div key={trait}>
              <h2 className="text-base font-bold text-foreground border-b border-border pb-1 mb-3">
                Trait: {trait}
              </h2>
              <VivaSenseResultsDisplay
                result={{ ...(traitResult as Record<string, unknown>), response: trait }}
              />
            </div>
          )
        )}
      </div>
    );
  }

  // Normalize backend keys
  const adapted = adaptBackendResult(result);

  const anova = adapted.anova as Record<string, unknown> | undefined;
  const means = adapted.means as Record<string, unknown> | undefined;
  const letters = adapted.letters as Record<string, unknown> | undefined;
  const effectSizes = adapted.effect_sizes as Record<string, unknown> | undefined;
  const assumptions = adapted.assumptions as Record<string, unknown> | undefined;
  const descStats = adapted.descriptive_stats as Record<string, unknown> | undefined;
  const plots = adapted.plots as Record<string, string> | undefined;
  const posthoc = adapted.posthoc as Record<string, Record<string, unknown>> | undefined;
  const interactions = adapted.interactions as unknown;
  const diagnosticObservations = asDiagnosticsRows(adapted.diagnostic_observations);
  const outlierSummary = toRecord(adapted.outlier_summary) ?? toRecord(assumptions?.outlier_detection);
  const cooksThreshold = Number(outlierSummary?.cooks_distance_threshold ?? 0);

  return (
    <div className="space-y-2">

      {/* Experiment Summary */}
      <CollapsibleSection title="Experiment Summary" defaultOpen={true}>
        <ExperimentSummary result={adapted} />
      </CollapsibleSection>

      {/* Dr. Fayeun Interpretation */}
      {adapted.interpretation && (
        <CollapsibleSection
          title="Dr. Fayeun's Interpretation"
          icon={<FlaskConical className="w-4 h-4 text-primary" />}
          defaultOpen={true}
        >
          <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
            {String(adapted.interpretation)}
          </p>
        </CollapsibleSection>
      )}

      {/* Descriptive Statistics */}
      {descStats && (
        <CollapsibleSection title="Descriptive Statistics" defaultOpen={true}>
          <DescriptiveStatsDisplay descStats={descStats} />
        </CollapsibleSection>
      )}

      {/* Effect Sizes */}
      {effectSizes && (
        <CollapsibleSection title="Effect Sizes" defaultOpen={true}>
          <EffectSizesBox effectSizes={effectSizes} rSquared={adapted.r_squared} />
        </CollapsibleSection>
      )}

      {/* ANOVA Table */}
      {anova && Object.keys(anova).length > 0 && (
        <CollapsibleSection title="Analysis of Variance (ANOVA)" defaultOpen={true}>
          <ANOVATableDisplay anova={anova} />
        </CollapsibleSection>
      )}

      {/* Assumption Diagnostics Dashboard (NEW) */}
      {assumptions && (
        <CollapsibleSection title="Assumption Diagnostics" defaultOpen={true} icon={<FlaskConical className="w-4 h-4 text-primary" />}>
          <AssumptionDiagnosticsPanel data={assumptions} />
        </CollapsibleSection>
      )}

      {/* Treatment Boxplots (NEW) */}
      {adapted.per_genotype_stats && (
        <CollapsibleSection title="Treatment Boxplots" defaultOpen={true}>
          <TreatmentBoxplots stats={adapted.per_genotype_stats as Record<string, unknown>[] | undefined} />
        </CollapsibleSection>
      )}

      {/* Residual Histogram (NEW) */}
      {adapted.residuals && (
        <CollapsibleSection title="Residual Histogram" defaultOpen={true}>
          <ResidualHistogram residuals={adapted.residuals as unknown[] | undefined} />
        </CollapsibleSection>
      )}

      {/* Residual vs Fitted Plot (NEW) */}
      {adapted.residuals && adapted.fitted_values && (
        <CollapsibleSection title="Residual vs Fitted Plot" defaultOpen={true}>
          <ResidualVsFittedPlot
            residuals={adapted.residuals as unknown[] | undefined}
            fitted={adapted.fitted_values as unknown[] | undefined}
          />
        </CollapsibleSection>
      )}

      {/* Residuals vs Treatment (NEW) */}
      {diagnosticObservations.length > 0 && (
        <CollapsibleSection title="Residuals vs Treatment" defaultOpen={true}>
          <ResidualsVsTreatmentPlot observations={diagnosticObservations} />
        </CollapsibleSection>
      )}

      {/* Scale-Location Plot (NEW) */}
      {diagnosticObservations.length > 0 && (
        <CollapsibleSection title="Scale-Location Plot" defaultOpen={true}>
          <ScaleLocationPlot observations={diagnosticObservations} />
        </CollapsibleSection>
      )}

      {/* Cook's Distance Plot (NEW) */}
      {diagnosticObservations.length > 0 && (
        <CollapsibleSection title="Cook's Distance" defaultOpen={true}>
          <CooksDistancePlot observations={diagnosticObservations} threshold={cooksThreshold} />
        </CollapsibleSection>
      )}

      {/* Standardized Residual Plot (NEW) */}
      {diagnosticObservations.length > 0 && (
        <CollapsibleSection title="Standardized Residual Plot" defaultOpen={true}>
          <StandardizedResidualPlot observations={diagnosticObservations} />
        </CollapsibleSection>
      )}

      {/* Treatment Means & Tukey HSD */}
      {means && (
        <CollapsibleSection title="Treatment Means & Tukey HSD" defaultOpen={true}>
          <TukeyHSDDisplay
            means={means}
            letters={letters}
            treatment={adapted.treatment as string | undefined}
            descStats={descStats}
          />
        </CollapsibleSection>
      )}

      {/* Interaction Means (factorial designs) */}
      {interactions && (
        <CollapsibleSection title="Interaction Means" defaultOpen={true}>
          <InteractionMeansDisplay interactions={interactions} />
        </CollapsibleSection>
      )}

      {/* Non-parametric test statistic */}
      {(adapted.H_statistic != null || adapted.chi2_statistic != null) && (
        <CollapsibleSection title="Test Statistic" defaultOpen={true}>
          <PubTable
            headers={["Statistic", "Value", "p-value", "Significant"]}
            rows={[
              [
                adapted.H_statistic != null ? "Kruskal-Wallis H" : "Friedman χ²",
                fmt4(adapted.H_statistic ?? adapted.chi2_statistic),
                pDisplay(adapted.p_value),
                (adapted.significant as boolean) ? (
                  <span className="text-green-600 font-medium">Yes</span>
                ) : (
                  <span className="text-muted-foreground">No</span>
                ),
              ],
            ]}
            downloadTitle="Test_Statistic"
          />
        </CollapsibleSection>
      )}

      {/* Post-hoc Comparisons */}
      {posthoc && Object.keys(posthoc).length > 0 && (
        <CollapsibleSection title="Post-hoc Comparisons" defaultOpen={true}>
          <PubTable
            headers={["Pair", "Statistic", "p (adjusted)", "Significant"]}
            rows={Object.entries(posthoc).map(([pair, ph]) => [
              pair,
              fmt4(ph.U ?? ph.W),
              pDisplay(ph.p_adjusted),
              (ph.significant as boolean) ? (
                <span className="text-green-600">Yes</span>
              ) : (
                <span className="text-muted-foreground">No</span>
              ),
            ])}
            downloadTitle="Post_Hoc_Comparisons"
          />
        </CollapsibleSection>
      )}

      {/* Publication plots */}
      {plots?.publication_bar && (
        <CollapsibleSection title="Publication Bar Chart" defaultOpen={true}>
          <div className="flex justify-end mb-2">
            <FigureDownloadMenu title="Publication_Bar_Chart" base64={plots.publication_bar} />
          </div>
          <img
            src={`data:image/png;base64,${plots.publication_bar}`}
            alt="Publication bar chart"
            className="max-w-full rounded"
          />
        </CollapsibleSection>
      )}

      {plots?.bar && !plots?.publication_bar && (
        <CollapsibleSection title="Bar Chart" defaultOpen={true}>
          <div className="flex justify-end mb-2">
            <FigureDownloadMenu title="Bar_Chart" base64={plots.bar} />
          </div>
          <img
            src={`data:image/png;base64,${plots.bar}`}
            alt="Bar chart"
            className="max-w-full rounded"
          />
        </CollapsibleSection>
      )}

      {/* Publishable HTML Tables */}
      {(() => {
        const htmlTables = (adapted.html_tables as Record<string, string>) ||
          generatePublishableHtmlTables(adapted);
        const hasHtmlTables = htmlTables && Object.keys(htmlTables).length > 0;
        return hasHtmlTables ? (
          <CollapsibleSection title="Publishable Tables (Word Export)" defaultOpen={false}>
            <HtmlTablesSection htmlTables={htmlTables} />
          </CollapsibleSection>
        ) : null;
      })()}

      {/* Technical JSON toggle */}
      <TechnicalJSON data={result} />
    </div>
  );
}
