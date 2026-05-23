import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Trash2, BarChart3, Image, CheckCircle2, XCircle, Table2, Sparkles, Loader2, AlertCircle, ShieldCheck, ChevronDown, Download, Users } from "lucide-react";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import ReactMarkdown from "react-markdown";
import { HtmlTablesSection } from "./HtmlTablesSection";
import { generatePublishableHtmlTables } from "./utils/generatePublishableTables";
import { DescriptivePublicationTables } from "./DescriptivePublicationTables";
import { computeDescriptivePublicationTables } from "./utils/computeDescriptivePublicationTables";
import { getSignificanceStars, formatPValue, SignificanceLegend } from "./results/SignificanceStars";
import { GenericDataTable } from "./results/GenericDataTable";
import { downloadPlotAsPng } from "./results/ExportButtons";
import { TableDownloadMenu, normalizeForDownload } from "./results/TableDownloadMenu";
import { FigureDownloadMenu } from "./results/FigureDownloadMenu";
import type { GroundingCheck } from "./VivaSenseMultiTraitInterpretation";

export interface MultiTraitResultsData {
  summary_table?: Record<string, unknown>[] | unknown[][];
  significance_heatmap?: string;
  correlation_heatmap?: string;
  pca_biplot?: string;
  plots?: Record<string, string>;
  html_tables?: Record<string, string>;
  trait_results?: Record<string, TraitResult>;
  per_trait?: Record<string, TraitResult>;
  meta?: Record<string, unknown>;
  interpretation?: string;
}

interface TraitResult {
  significant?: boolean;
  p_value?: number;
  tables?: Record<string, unknown>;
  plots?: Record<string, string>;
  [key: string]: unknown;
}

interface Props {
  results: MultiTraitResultsData;
  onClear: () => void;
  traitInterpretations?: Record<string, string>;
  completedTraits?: Set<string>;
  traitGroundingChecks?: Record<string, GroundingCheck>;
}

// ── Utility functions ──────────────────────────────────────────────────

function fmt(v: unknown, decimals = 3): string {
  if (v == null) return "—";
  const n = typeof v === "number" ? v : Number.parseFloat(String(v));
  if (Number.isNaN(n)) return String(v);
  return n.toFixed(decimals);
}

function fmtInt(v: unknown): string {
  if (v == null) return "—";
  const n = typeof v === "number" ? v : Number.parseFloat(String(v));
  if (Number.isNaN(n)) return String(v);
  return Number.isInteger(n) ? String(n) : n.toFixed(0);
}

function toFiniteNumber(v: unknown): number | null {
  if (v == null || v === "") return null;
  const n = typeof v === "number" ? v : Number.parseFloat(String(v));
  return Number.isFinite(n) ? n : null;
}

function cleanFormulaLabel(raw: string): string {
  return raw
    .replace(/Q\(\s*['"]([^'"]+)['"]\s*\)/g, "$1")
    .replace(/C\(\s*([^\)]+)\s*\)/g, "$1")
    .replace(/`/g, "")
    .trim();
}

/** Map source row labels to standard agronomy names */
function standardizeSource(src: string): string {
  const cleaned = cleanFormulaLabel(src);
  const s = cleaned.toLowerCase();
  if (/^(genotype|treatment|variety|cultivar|factor)$/i.test(s)) return "Genotype";
  if (/^(block|rep|replication|replicate)$/i.test(s)) return "Block";
  if (/^(residual|error|within)$/i.test(s)) return "Error";
  if (/^total$/i.test(s)) return "Total";
  return cleaned.replace(/\s*:\s*/g, " × ");
}

const COL_MAP: Record<string, string> = {
  source: "Source", Source: "Source",
  df: "DF", Df: "DF", DF: "DF",
  sum_sq: "SS", "Sum Sq": "SS", SS: "SS", ss: "SS",
  mean_sq: "MS", "Mean Sq": "MS", MS: "MS", ms: "MS",
  f_value: "F", "F value": "F", F: "F", "F-value": "F", "F Value": "F",
  p_value: "p-value", "Pr(>F)": "p-value", "p-value": "p-value", "P-value": "p-value", pvalue: "p-value", p: "p-value",
};

const ANOVA_ORDER = ["Source", "DF", "SS", "MS", "F", "p-value"];

const LOWER_IS_BETTER_PATTERNS = [
  /days?\s*(to|of)?\s*(flower|maturity|heading|anthesis|silk)/i,
  /disease|severity|incidence|lodging|sterility/i,
];

function isLowerBetterTrait(traitName: string): boolean {
  return LOWER_IS_BETTER_PATTERNS.some((pattern) => pattern.test(traitName));
}

interface ParsedAnovaRow {
  Source: string;
  DF: number | null;
  SS: number | null;
  MS: number | null;
  F: number | null;
  "p-value": number | null;
}

interface ParsedAnova {
  rows: ParsedAnovaRow[];
  hasRequiredFields: boolean;
  missingCriticalFields: boolean;
  hasErrorRow: boolean;
  hasModelRow: boolean;
  mse: number | null;
  totalDf: number | null;
}

interface ParsedMeanRow {
  Genotype: string;
  Mean: number | null;
  SE: number | null;
  "Tukey Group": string;
  Rank: number;
}

interface ParsedMeans {
  rows: ParsedMeanRow[];
  hasLetters: boolean;
  missingLetters: boolean;
  nLevels: number;
}

interface ExperimentalPrecision {
  grandMean: number | null;
  mse: number | null;
  sem: number | null;
  cvPercent: number | null;
  replications: number | null;
}

function sortAnovaRows(rows: ParsedAnovaRow[]): ParsedAnovaRow[] {
  const category = (source: string): number => {
    const s = source.toLowerCase();
    if (/^total$/.test(s)) return 40;
    if (/error|residual|within/.test(s)) return 30;
    if (/block|rep/.test(s)) return 20;
    return 10;
  };
  return [...rows].sort((a, b) => category(a.Source) - category(b.Source));
}

/** Try to parse pandas dict-of-dicts ANOVA: { df: { Source1: val, ... }, sum_sq: { ... }, ... } */
function tryParsePandasAnova(data: unknown): { headers: string[]; rows: Record<string, unknown>[] } | null {
  if (!data || typeof data !== "object" || Array.isArray(data)) return null;
  const obj = data as Record<string, unknown>;
  const colKeys = Object.keys(obj);
  const dfKey = colKeys.find((k) => /^(df|Df|DF)$/i.test(k));
  if (!dfKey) return null;
  const dfObj = obj[dfKey];
  if (!dfObj || typeof dfObj !== "object" || Array.isArray(dfObj)) return null;
  const sources = Object.keys(dfObj as Record<string, unknown>);
  if (sources.length === 0) return null;

  const headers = ["source", ...colKeys.filter((k) => k !== "source" && k !== "Source")];
  const rows = sources.map((src) => {
    const row: Record<string, unknown> = { source: src };
    colKeys.forEach((ck) => {
      if (ck === "source" || ck === "Source") return;
      const colData = obj[ck];
      if (colData && typeof colData === "object" && !Array.isArray(colData)) {
        row[ck] = (colData as Record<string, unknown>)[src];
      }
    });
    const ss = row["sum_sq"] ?? row["SS"] ?? row["Sum Sq"] ?? row["ss"];
    const df = row[dfKey];
    if (row["mean_sq"] == null && row["MS"] == null && row["Mean Sq"] == null && ss != null && df != null && Number(df) > 0) {
      row["mean_sq"] = Number(ss) / Number(df);
    }
    return row;
  });
  return { headers, rows };
}

function parseAnovaData(data: unknown): ParsedAnova | null {
  const parsed = tryParsePandasAnova(data) || normalizeTableRows(data);
  if (!parsed || parsed.rows.length === 0) return null;

  const mappedRows = parsed.rows.map((row) => {
    const mapped: Record<string, unknown> = {};
    parsed.headers.forEach((origH) => {
      mapped[COL_MAP[origH] || origH] = row[origH];
    });

    const source = standardizeSource(String(mapped.Source ?? row.source ?? row.Source ?? ""));
    const df = toFiniteNumber(mapped.DF);
    const ss = toFiniteNumber(mapped.SS);
    const ms = toFiniteNumber(mapped.MS) ?? ((ss != null && df != null && df > 0) ? ss / df : null);

    return {
      Source: source,
      DF: df,
      SS: ss,
      MS: ms,
      F: toFiniteNumber(mapped.F),
      "p-value": toFiniteNumber(mapped["p-value"]),
    } satisfies ParsedAnovaRow;
  });

  const hasModelRow = mappedRows.some((row) => {
    const s = row.Source.toLowerCase();
    return !/error|residual|total|within/.test(s);
  });
  const hasErrorRow = mappedRows.some((row) => /error|residual|within/i.test(row.Source));

  const missingCriticalFields = mappedRows.some((row) => {
    if (/^total$/i.test(row.Source)) return false;
    return row.DF == null || row.SS == null;
  });

  let rows = [...mappedRows];
  const hasTotalRow = rows.some((row) => /^total$/i.test(row.Source));
  if (!hasTotalRow) {
    const sumDf = rows.reduce((acc, row) => acc + (row.DF ?? 0), 0);
    const sumSS = rows.reduce((acc, row) => acc + (row.SS ?? 0), 0);
    if (sumDf > 0 && sumSS >= 0) {
      rows.push({
        Source: "Total",
        DF: sumDf,
        SS: sumSS,
        MS: sumDf > 0 ? sumSS / sumDf : null,
        F: null,
        "p-value": null,
      });
    }
  }

  rows = sortAnovaRows(rows);
  const errorRow = rows.find((row) => /error|residual|within/i.test(row.Source));
  const totalRow = rows.find((row) => /^total$/i.test(row.Source));

  return {
    rows,
    hasRequiredFields: hasModelRow && hasErrorRow,
    missingCriticalFields,
    hasErrorRow,
    hasModelRow,
    mse: errorRow?.MS ?? (errorRow?.SS != null && errorRow?.DF != null && errorRow.DF > 0 ? errorRow.SS / errorRow.DF : null),
    totalDf: totalRow?.DF ?? null,
  };
}

function parseMeanSeparationData(data: unknown, traitName: string): ParsedMeans | null {
  const parsed = normalizeTableRows(data);
  if (!parsed || parsed.rows.length === 0) return null;

  const treatCol = parsed.headers.find((h) => /treatment|genotype|group|factor|variety/i.test(h)) || parsed.headers[0];
  const meanCol = parsed.headers.find((h) => /(^mean$|mean_|_mean|average)/i.test(h)) || parsed.headers.find((h) => /mean/i.test(h));
  const seCol = parsed.headers.find((h) => /se$|std_err|std_error|standard_error/i.test(h));
  const letterCol = parsed.headers.find((h) => /letter|tukey|group/i.test(h) && !/genotype|treatment|factor|variety/i.test(h));

  if (!treatCol || !meanCol) return null;

  const lowerIsBetter = isLowerBetterTrait(traitName);
  const rows = parsed.rows.map((row) => ({
    Genotype: String(row[treatCol] ?? ""),
    Mean: toFiniteNumber(row[meanCol]),
    SE: toFiniteNumber(row[seCol ?? "se"] ?? row.std_err),
    "Tukey Group": letterCol ? String(row[letterCol] ?? "").trim() : "",
    Rank: 0,
  } satisfies ParsedMeanRow));

  const sorted = [...rows].sort((a, b) => {
    const ma = a.Mean ?? (lowerIsBetter ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY);
    const mb = b.Mean ?? (lowerIsBetter ? Number.POSITIVE_INFINITY : Number.NEGATIVE_INFINITY);
    return lowerIsBetter ? ma - mb : mb - ma;
  }).map((row, idx) => ({ ...row, Rank: idx + 1 }));

  const hasLetters = Boolean(letterCol);
  const missingLetters = hasLetters
    ? sorted.some((row) => !row["Tukey Group"])
    : true;

  return {
    rows: sorted,
    hasLetters,
    missingLetters,
    nLevels: sorted.length,
  };
}

function resolveReplications(
  trait: TraitResult,
  tables: Record<string, unknown> | undefined,
  meta: Record<string, unknown> | undefined,
  totalDf: number | null,
  nLevels: number,
): number | null {
  const candidates = [
    (trait as Record<string, unknown>).n_reps,
    (trait as Record<string, unknown>).replications,
    tables?.n_reps,
    tables?.replications,
    meta?.n_reps,
    meta?.replications,
  ];

  for (const candidate of candidates) {
    const n = toFiniteNumber(candidate);
    if (n != null && n >= 2) return n;
  }

  if (totalDf != null && nLevels > 0) {
    const nObs = totalDf + 1;
    const inferred = nObs / nLevels;
    if (Number.isFinite(inferred) && inferred >= 2) return inferred;
  }

  return null;
}

function resolveGrandMean(
  trait: TraitResult,
  tables: Record<string, unknown> | undefined,
  meta: Record<string, unknown> | undefined,
  means: ParsedMeans | null,
): number | null {
  const candidates = [
    (trait as Record<string, unknown>).grand_mean,
    tables?.grand_mean,
    meta?.grand_mean,
  ];
  for (const candidate of candidates) {
    const n = toFiniteNumber(candidate);
    if (n != null) return n;
  }

  if (!means || means.rows.length === 0) return null;
  const validMeans = means.rows.map((row) => row.Mean).filter((v): v is number => v != null);
  if (validMeans.length === 0) return null;
  const sum = validMeans.reduce((acc, value) => acc + value, 0);
  return sum / validMeans.length;
}

function computeExperimentalPrecision(
  trait: TraitResult,
  tables: Record<string, unknown> | undefined,
  meta: Record<string, unknown> | undefined,
  anova: ParsedAnova | null,
  means: ParsedMeans | null,
): ExperimentalPrecision | null {
  if (!anova) return null;

  const mse = anova.mse;
  const replications = resolveReplications(trait, tables, meta, anova.totalDf, means?.nLevels ?? 0);
  const grandMean = resolveGrandMean(trait, tables, meta, means);

  const sem = (mse != null && replications != null && replications > 0) ? Math.sqrt(mse / replications) : null;
  const cvPercent = (mse != null && grandMean != null && grandMean !== 0)
    ? (Math.sqrt(mse) / grandMean) * 100
    : null;

  return { grandMean, mse, sem, cvPercent, replications };
}

function PerTraitAnovaTable({ anova, precision }: { anova: ParsedAnova; precision: ExperimentalPrecision | null }) {
  const dlHeaders = ANOVA_ORDER;
  const dlRows = anova.rows.map((row) => dlHeaders.map((h) => {
    if (h === "Source") return row.Source;
    if (h === "DF") return fmtInt(row.DF);
    if (h === "SS") return fmt(row.SS);
    if (h === "MS") return fmt(row.MS);
    if (h === "F") return fmt(row.F);
    if (h === "p-value") {
      const stars = row["p-value"] != null ? ` ${getSignificanceStars(row["p-value"])}` : "";
      return `${formatPValue(row["p-value"])}${stars}`.trim();
    }
    return "—";
  }));

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-foreground flex items-center gap-2">
          <Table2 className="w-4 h-4 text-primary" />
          ANOVA Table
        </h4>
        <TableDownloadMenu title="ANOVA_Table" headers={dlHeaders} rows={dlRows} />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse font-mono">
          <thead>
            <tr className="border-b-2 border-foreground/30">
              {ANOVA_ORDER.map((h) => (
                <th key={h} className={`px-3 py-2 font-semibold text-foreground ${h === "Source" ? "text-left" : "text-right"}`}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {anova.rows.map((row, ri) => {
              const isErrorOrTotal = /error|residual|total|within/i.test(row.Source);
              return (
                <tr key={`${row.Source}-${ri}`} className="border-b border-border/50 hover:bg-muted/20">
                  <td className="px-3 py-2 text-left font-medium text-foreground">{row.Source}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmtInt(row.DF)}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(row.SS)}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(row.MS)}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{isErrorOrTotal ? "—" : fmt(row.F)}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                    {isErrorOrTotal ? "—" : `${formatPValue(row["p-value"])} ${getSignificanceStars(row["p-value"] ?? null)}`.trim()}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {precision && (
        <div className="mt-4 rounded-lg border border-border bg-muted/30 p-4">
          <h5 className="text-sm font-semibold text-foreground mb-3">Experimental Precision</h5>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide">Grand Mean</p>
              <p className="font-semibold text-foreground font-mono">{fmt(precision.grandMean)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide">MSE</p>
              <p className="font-semibold text-foreground font-mono">{fmt(precision.mse)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide">SEm (±)</p>
              <p className="font-semibold text-foreground font-mono">{fmt(precision.sem)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide">CV (%)</p>
              <p className="font-semibold text-foreground font-mono">{precision.cvPercent != null ? `${fmt(precision.cvPercent, 2)}%` : "—"}</p>
            </div>
          </div>
        </div>
      )}
      <SignificanceLegend />
    </div>
  );
}

function PerTraitMeansTable({ means }: { means: ParsedMeans }) {
  const dlHeaders = ["Genotype", "Mean", "SE", "Tukey Group", "Rank"];
  const dlRows = means.rows.map((row) => [
    row.Genotype,
    fmt(row.Mean),
    fmt(row.SE),
    row["Tukey Group"] || "—",
    String(row.Rank),
  ]);

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-semibold text-foreground flex items-center gap-2">
          <Users className="w-4 h-4 text-primary" />
          Mean Separation Table
        </h4>
        <TableDownloadMenu title="Mean_Separation_Tukey" headers={dlHeaders} rows={dlRows} />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse font-mono">
          <thead>
            <tr className="border-b-2 border-foreground/30">
              {dlHeaders.map((h) => (
                <th key={h} className={`px-3 py-2 font-semibold text-foreground ${h === "Genotype" || h === "Tukey Group" ? "text-left" : "text-right"}`}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {means.rows.map((row) => (
              <tr key={row.Genotype} className="border-b border-border/50 hover:bg-muted/20">
                <td className="px-3 py-2 text-left font-medium text-foreground">{row.Genotype}</td>
                <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(row.Mean)}</td>
                <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(row.SE)}</td>
                <td className="px-3 py-2 text-left font-bold text-primary">{row["Tukey Group"] || "—"}</td>
                <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{row.Rank}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted-foreground mt-2 italic">
        Bars represent mean ± SE. Means sharing the same letter are not significantly different according to Tukey HSD (α = 0.05).
      </p>
    </div>
  );
}

function ResultMessage({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-border bg-muted/30 p-3 text-sm text-muted-foreground">
      {message}
    </div>
  );
}

/** Safely convert any value to string — never render [object Object] */
function safeString(val: unknown): string {
  if (val == null) return "—";
  if (typeof val === "object") {
    try { return JSON.stringify(val); } catch { return "—"; }
  }
  return String(val);
}

/** Normalize array-of-arrays, array-of-objects, or pandas dict-of-dicts into a common shape */
function normalizeTableRows(data: unknown): { headers: string[]; rows: Record<string, unknown>[] } | null {
  if (Array.isArray(data) && data.length > 0) {
    if (Array.isArray(data[0])) {
      const headers = (data[0] as unknown[]).map(String);
      const rows = data.slice(1).map((row: unknown) => {
        const r: Record<string, unknown> = {};
        (row as unknown[]).forEach((cell, i) => { r[headers[i]] = cell; });
        return r;
      });
      return { headers, rows };
    }
    if (typeof data[0] === "object" && data[0] !== null) {
      const headers = Object.keys(data[0] as Record<string, unknown>);
      return { headers, rows: data as Record<string, unknown>[] };
    }
    return null;
  }
  // Handle pandas dict-of-dicts: { "ColName": { "0": val, "1": val, ... }, ... }
  if (data && typeof data === "object" && !Array.isArray(data)) {
    const obj = data as Record<string, unknown>;
    const headers = Object.keys(obj);
    if (headers.length === 0) return null;
    const firstCol = obj[headers[0]];
    if (firstCol && typeof firstCol === "object" && !Array.isArray(firstCol)) {
      const rowKeys = Object.keys(firstCol as Record<string, unknown>).sort((a, b) => Number(a) - Number(b));
      const rows = rowKeys.map((rk) => {
        const row: Record<string, unknown> = {};
        headers.forEach((h) => {
          const col = obj[h] as Record<string, unknown> | undefined;
          row[h] = col ? col[rk] : undefined;
        });
        return row;
      });
      return { headers, rows };
    }
  }
  return null;
}

/** Derive a statistically correct conclusion from a p-value */
function deriveConclusion(testType: string, pVal: number | null): string {
  if (pVal == null) return "";
  const isNormality = /normality|shapiro|kolmogorov|anderson/i.test(testType);
  const isHomogeneity = /homogeneity|levene|bartlett|brown.?forsythe/i.test(testType);
  if (isNormality) {
    return pVal < 0.05
      ? "Normality assumption violated."
      : "Normality assumption satisfied.";
  }
  if (isHomogeneity) {
    return pVal < 0.05
      ? "Homogeneity of variance assumption violated."
      : "Homogeneity of variance assumption satisfied.";
  }
  return pVal < 0.05 ? "Significant (p < 0.05)." : "Not significant (p ≥ 0.05).";
}

/** Format a single assumption test object into readable text */
function formatTestResult(val: unknown): string {
  if (val == null) return "—";
  if (typeof val === "string") {
    // Sanitize software error language
    return val
      .replace(/failed\s+due\s+to\s+software\s+error/gi, "could not be computed")
      .replace(/software\s+error/gi, "computation issue")
      .replace(/internal\s+error/gi, "could not be computed");
  }
  if (typeof val === "object" && !Array.isArray(val)) {
    const sub = val as Record<string, unknown>;
    const testName = sub.test ?? sub.test_name ?? sub.method;
    const stat = sub.statistic ?? sub.stat ?? sub.W ?? sub.test_statistic;
    const pVal = sub.p_value ?? sub.pvalue ?? sub.p;

    const pNum = pVal != null ? Number(pVal) : null;
    const testStr = String(testName ?? "");

    if (testName || stat != null || pNum != null) {
      const parts: string[] = [];
      if (testName) parts.push(`Test: ${testStr}`);
      if (stat != null) parts.push(`Statistic: ${Number(stat).toFixed(4)}`);
      if (pNum != null) parts.push(`p = ${pNum.toFixed(4)}`);
      // Add statistically correct conclusion
      const conclusion = deriveConclusion(testStr, pNum);
      if (conclusion) parts.push(conclusion);
      return parts.join(", ");
    }
    // Fallback for simple shapes
    const status = sub.status ?? sub.result ?? sub.conclusion;
    if (status) return `${String(status)}${pNum != null ? ` (p = ${pNum.toFixed(4)})` : ""}`;
    try { return JSON.stringify(val); } catch { return "—"; }
  }
  return String(val);
}

/** Parse assumption guidance from various backend shapes into renderable parts */
function parseAssumptionGuidance(raw: unknown): { normality?: string; homogeneity?: string; message?: string; recommendation?: string; text?: string } | null {
  if (!raw) return null;
  if (typeof raw === "string") return { text: raw };
  if (typeof raw === "object" && !Array.isArray(raw)) {
    const obj = raw as Record<string, unknown>;
    const inner = (obj.overall ?? obj.guidance ?? obj) as Record<string, unknown>;
    if (typeof inner === "string") return { text: inner };
    const result: Record<string, string> = {};
    for (const [key, val] of Object.entries(inner)) {
      if (val == null) continue;
      result[key] = formatTestResult(val);
    }
    if (Object.keys(result).length === 0) return null;
    return {
      normality: result.normality ?? result.normality_status ?? result.normality_test,
      homogeneity: result.homogeneity ?? result.homogeneity_status ?? result.homogeneity_test ?? result.levene,
      message: result.message ?? result.overall_message,
      recommendation: result.recommendation,
      text: result.text ?? result.verdict ?? result.assumption_verdict,
    };
  }
  return { text: String(raw) };
}

function extractFigureMeanMap(
  traitObj: Record<string, unknown>,
  tables: Record<string, unknown> | undefined,
): Record<string, number> | null {
  const candidates = [
    traitObj.figure_means,
    traitObj.plot_means,
    traitObj.mean_plot_values,
    tables?.figure_means,
    tables?.plot_means,
    tables?.mean_plot_values,
  ];

  for (const candidate of candidates) {
    if (!candidate) continue;

    if (Array.isArray(candidate)) {
      const map: Record<string, number> = {};
      for (const row of candidate) {
        if (!row || typeof row !== "object") continue;
        const record = row as Record<string, unknown>;
        const genotype = String(record.genotype ?? record.treatment ?? record.label ?? "").trim();
        const mean = toFiniteNumber(record.mean ?? record.value);
        if (genotype && mean != null) map[genotype] = mean;
      }
      if (Object.keys(map).length > 0) return map;
    }

    if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
      const entries = Object.entries(candidate as Record<string, unknown>);
      const map: Record<string, number> = {};
      for (const [genotype, meanRaw] of entries) {
        const mean = toFiniteNumber(meanRaw);
        if (mean != null) map[genotype] = mean;
      }
      if (Object.keys(map).length > 0) return map;
    }
  }

  return null;
}

function meansMatchFigure(parsedMeans: ParsedMeans | null, figureMap: Record<string, number> | null): boolean {
  if (!parsedMeans || !figureMap) return true;
  const tableMap = Object.fromEntries(
    parsedMeans.rows
      .filter((row) => row.Genotype && row.Mean != null)
      .map((row) => [row.Genotype, row.Mean as number]),
  );

  const overlappingKeys = Object.keys(tableMap).filter((k) => figureMap[k] != null);
  if (overlappingKeys.length === 0) return true;

  return overlappingKeys.every((key) => Math.abs(tableMap[key] - figureMap[key]) <= 0.01);
}

// ── Publication-style figure with download ────────────────────────────

/** Check if a string looks like valid base64 image data */
function isValidBase64(str: string): boolean {
  if (!str || str.length < 100) return false;
  return /^[A-Za-z0-9+/\n\r]+=*$/.test(str.slice(0, 200));
}

function PublicationFigure({ label, base64, altText }: { label: string; base64: string; altText: string }) {
  if (!isValidBase64(base64)) return null;
  return (
    <Card>
      <CardContent className="pt-6">
        <div className="bg-background rounded-lg p-4 border border-border">
          <img
            src={`data:image/png;base64,${base64}`}
            alt={altText}
            className="max-w-full mx-auto rounded"
          />
        </div>
        <div className="flex items-center justify-between mt-3">
          <p className="text-sm text-muted-foreground italic">{label}</p>
          <FigureDownloadMenu title={altText} base64={base64} />
        </div>
      </CardContent>
    </Card>
  );
}

// ── Summary Table with journal formatting ─────────────────────────────

function JournalSummaryTable({ data }: { data: Record<string, unknown>[] | unknown[][] }) {
  const parsed = normalizeTableRows(data as unknown[]);
  if (!parsed) return null;

  const dlHeaders = parsed.headers.map((h) => h.replace(/_/g, " "));
  const dlRows = parsed.rows.map((row) => parsed.headers.map((h) => {
    const val = row[h];
    const isNum = val != null && !isNaN(Number(val)) && String(val).trim() !== "";
    return isNum ? fmt(val) : String(val ?? "—");
  }));

  return (
    <div>
      <div className="flex justify-end mb-2">
        <TableDownloadMenu title="Trait_Summary" headers={dlHeaders} rows={dlRows} />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse font-mono">
          <thead>
            <tr className="border-b-2 border-foreground/30">
              {parsed.headers.map((h) => (
                <th key={h} className="px-3 py-2.5 font-semibold text-foreground text-left">
                  {h.replace(/_/g, " ")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {parsed.rows.map((row, ri) => (
              <tr key={ri} className="border-b border-border/50 hover:bg-muted/20">
                {parsed.headers.map((h, ci) => {
                  const val = row[h];
                  const isNum = val != null && !isNaN(Number(val)) && String(val).trim() !== "";
                  return (
                    <td key={h} className={`px-3 py-2 ${ci === 0 ? "font-medium text-foreground" : "text-muted-foreground"} ${isNum ? "tabular-nums text-right" : "text-left"}`}>
                      {isNum ? fmt(val) : String(val ?? "—")}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────

export function VivaSenseMultiTraitResults({ results, onClear, traitInterpretations = {}, completedTraits = new Set(), traitGroundingChecks = {} }: Props) {
  const { summary_table, meta, plots } = results;
  const traitData = results.per_trait || results.trait_results;

  const sigHeatmap = results.significance_heatmap || plots?.significance_heatmap;
  const corrHeatmap = results.correlation_heatmap || plots?.correlation_heatmap;
  const pcaBiplot = results.pca_biplot || plots?.pca_biplot;

  // Compute publication tables for ALL multi-trait analyses (not just descriptive)
  const pubTablesData = traitData && Object.keys(traitData).length > 0
    ? computeDescriptivePublicationTables(results as Record<string, unknown>)
    : null;

  return (
    <section className="py-20 bg-muted/30" id="results">
      <div className="container-wide">
        <div className="max-w-5xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center mb-8">
            <h2 className="font-serif text-3xl lg:text-4xl font-bold text-foreground mb-4">
              Multi-Trait Analysis Results
            </h2>
            <p className="text-muted-foreground mb-4">Journal-ready statistical output</p>
            <Button variant="outline" size="sm" onClick={onClear}>
              <Trash2 className="w-4 h-4 mr-2" />
              Clear Results
            </Button>
          </div>

          {/* Meta summary */}
          {meta && Object.keys(meta).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Sparkles className="w-6 h-6 text-primary" />
                  Experiment Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-3">
                  {Object.entries(meta).map(([key, value]) => {
                    if (typeof value === "object") return null;
                    return (
                      <div key={key} className="flex flex-col">
                        <dt className="text-xs text-muted-foreground font-medium uppercase tracking-wide">
                          {key.replace(/_/g, " ")}
                        </dt>
                        <dd className="text-foreground font-semibold font-mono">{String(value)}</dd>
                      </div>
                    );
                  })}
                </dl>
              </CardContent>
            </Card>
          )}

          {/* Summary Table */}
          {summary_table && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Table2 className="w-6 h-6 text-primary" />
                  Trait Summary Table
                </CardTitle>
              </CardHeader>
              <CardContent>
                <JournalSummaryTable data={summary_table} />
              </CardContent>
            </Card>
          )}

          {/* Publication Tables for descriptive multi-trait */}
          {pubTablesData && (
            <DescriptivePublicationTables data={pubTablesData} />
          )}

          {/* Heatmaps & PCA — publication style */}
          {(sigHeatmap || corrHeatmap) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {sigHeatmap && (
                <PublicationFigure
                  label="Figure 1. Significance heatmap across traits"
                  base64={sigHeatmap}
                  altText="Significance_Heatmap"
                />
              )}
              {corrHeatmap && (
                <PublicationFigure
                  label={`Figure ${sigHeatmap ? 2 : 1}. Correlation matrix`}
                  base64={corrHeatmap}
                  altText="Correlation_Matrix"
                />
              )}
            </div>
          )}

          {pcaBiplot && (
            <PublicationFigure
              label={`Figure ${(sigHeatmap ? 1 : 0) + (corrHeatmap ? 1 : 0) + 1}. PCA Biplot`}
              base64={pcaBiplot}
              altText="PCA_Biplot"
            />
          )}

          {/* Per-Trait Detailed Results */}
          {traitData && Object.keys(traitData).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <BarChart3 className="w-6 h-6 text-primary" />
                  Per-Trait Detailed Results
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Accordion type="multiple" className="w-full">
                  {Object.entries(traitData).map(([traitName, trait]) => {
                    const traitObj = trait as Record<string, unknown>;
                    const isSignificant = trait.significant === true || (trait.p_value != null && Number(trait.p_value) <= 0.05);
                    const pVal = trait.p_value;
                    const tables = trait.tables as Record<string, unknown> | undefined;
                    const traitPlots = trait.plots;
                    const traitMeta = (traitObj.meta && typeof traitObj.meta === "object")
                      ? (traitObj.meta as Record<string, unknown>)
                      : meta;

                    const anovaTable = tables?.anova
                      ?? ((traitObj.anova as Record<string, unknown> | undefined)?.table)
                      ?? traitObj.anova
                      ?? tables?.anova_table
                      ?? tables?.combined_anova;

                    const meansTable = tables?.means
                      ?? ((traitObj.mean_separation as Record<string, unknown> | undefined)?.table)
                      ?? traitObj.mean_separation
                      ?? tables?.genotype_means
                      ?? tables?.tukey
                      ?? tables?.mean_separation
                      ?? tables?.letters
                      ?? traitObj.letters
                      ?? tables?.tukey_hsd;

                    const rawAssumption = tables?.assumption_guidance
                      ?? traitObj.assumptions
                      ?? traitObj.assumption_guidance
                      ?? tables?.assumptions
                      ?? tables?.assumption_tests;
                    const assumptionGuidance = parseAssumptionGuidance(rawAssumption);

                    const parsedAnova = parseAnovaData(anovaTable);
                    const parsedMeans = parseMeanSeparationData(meansTable, traitName);
                    const precision = computeExperimentalPrecision(trait, tables, traitMeta, parsedAnova, parsedMeans);

                    const meanPlot = traitPlots?.mean_plot && isValidBase64(traitPlots.mean_plot)
                      ? traitPlots.mean_plot
                      : (traitPlots?.bar_chart && isValidBase64(traitPlots.bar_chart) ? traitPlots.bar_chart : null);
                    const hasAnyPlot = Boolean(traitPlots && Object.values(traitPlots).some((plot) => isValidBase64(plot)));

                    const figureMeanMap = extractFigureMeanMap(traitObj, tables);
                    const integrityErrors: string[] = [];
                    const integrityWarnings: string[] = [];

                    // Missing data = warnings (don't block interpretation)
                    if (!parsedAnova || !parsedAnova.hasRequiredFields || parsedAnova.missingCriticalFields) {
                      integrityWarnings.push("ANOVA results unavailable.");
                    }
                    if (!parsedMeans || parsedMeans.rows.length === 0) {
                      integrityWarnings.push("Tukey HSD could not be computed for this trait.");
                    }
                    if (!meanPlot) {
                      integrityWarnings.push("Figure could not be generated.");
                    }
                    if (isSignificant && parsedMeans && (!parsedMeans.hasLetters || parsedMeans.missingLetters)) {
                      integrityWarnings.push("Mean separation results were computed but Tukey group letters could not be displayed.");
                      if (hasAnyPlot) {
                        integrityWarnings.push("Figure rendered without Tukey annotations. Check plotting layer.");
                      }
                    }
                    // Actual data mismatch = error (blocks interpretation)
                    if (!meansMatchFigure(parsedMeans, figureMeanMap)) {
                      integrityErrors.push("Result integrity validation failed. Tables and figures do not match.");
                    }

                    const integrityPassed = integrityErrors.length === 0;

                    return (
                      <AccordionItem key={traitName} value={traitName}>
                        <AccordionTrigger className="hover:no-underline">
                          <div className="flex items-center gap-3">
                            {completedTraits.has(traitName) ? (
                              <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
                            ) : isSignificant ? (
                              <CheckCircle2 className="w-5 h-5 text-primary/70 flex-shrink-0" />
                            ) : (
                              <XCircle className="w-5 h-5 text-destructive flex-shrink-0" />
                            )}
                            <span className="font-semibold text-foreground">{traitName}</span>
                            {traitGroundingChecks[traitName]?.passed && (
                              <Badge variant="outline" className="text-primary border-primary/30 text-[10px] px-1.5 py-0">
                                <ShieldCheck className="w-3 h-3 mr-0.5" />
                                Grounded ✓
                              </Badge>
                            )}
                            {completedTraits.has(traitName) && !traitGroundingChecks[traitName] && (
                              <span className="text-xs text-primary font-medium">✓ interpreted</span>
                            )}
                            {pVal != null && (
                              <span className="text-xs text-muted-foreground ml-2 font-mono">
                                (p = {formatPValue(pVal)}) {getSignificanceStars(pVal)}
                              </span>
                            )}
                          </div>
                        </AccordionTrigger>
                        <AccordionContent>
                          <div className="space-y-6 pt-2">
                            {Array.from(new Set(integrityErrors)).map((message) => (
                              <div key={message} className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
                                {message}
                              </div>
                            ))}
                            {Array.from(new Set(integrityWarnings)).map((message) => (
                              <div key={message} className="rounded-lg border border-border bg-muted/30 p-3 text-sm text-muted-foreground italic">
                                {message}
                              </div>
                            ))}

                            <div>
                              <h4 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                                <Sparkles className="w-4 h-4 text-primary" />
                                Assumption Checks
                              </h4>
                              {assumptionGuidance ? (
                                <div className="text-sm bg-muted/50 rounded-lg p-3 space-y-1">
                                  {assumptionGuidance.normality && (
                                    <p className="text-muted-foreground"><span className="font-medium text-foreground">Normality:</span> {assumptionGuidance.normality}</p>
                                  )}
                                  {assumptionGuidance.homogeneity && (
                                    <p className="text-muted-foreground"><span className="font-medium text-foreground">Homogeneity:</span> {assumptionGuidance.homogeneity}</p>
                                  )}
                                  {assumptionGuidance.message && (
                                    <p className="text-muted-foreground"><span className="font-medium text-foreground">Message:</span> {assumptionGuidance.message}</p>
                                  )}
                                  {assumptionGuidance.recommendation && (
                                    <p className="text-muted-foreground"><span className="font-medium text-foreground">Recommendation:</span> {assumptionGuidance.recommendation}</p>
                                  )}
                                  {assumptionGuidance.text && !assumptionGuidance.message && (
                                    <p className="text-muted-foreground whitespace-pre-wrap">{assumptionGuidance.text}</p>
                                  )}
                                </div>
                              ) : (
                                <ResultMessage message="Result could not be generated." />
                              )}
                            </div>

                            {parsedAnova && parsedAnova.hasRequiredFields && !parsedAnova.missingCriticalFields
                              ? <PerTraitAnovaTable anova={parsedAnova} precision={precision} />
                              : <ResultMessage message="ANOVA results unavailable." />}

                            {parsedMeans && parsedMeans.rows.length > 0
                              ? <PerTraitMeansTable means={parsedMeans} />
                              : <ResultMessage message="Mean separation results unavailable." />}

                            {meanPlot ? (
                              <div>
                                <div className="flex items-center justify-between mb-2">
                                  <h4 className="text-sm font-semibold text-foreground flex items-center gap-2">
                                    <BarChart3 className="w-4 h-4 text-primary" />
                                    Mean Plot
                                  </h4>
                                  <FigureDownloadMenu title={`${traitName}_Mean_Plot`} base64={meanPlot} />
                                </div>
                                <div className="bg-background rounded-lg p-4 border border-border">
                                  <img
                                    src={`data:image/png;base64,${meanPlot}`}
                                    alt={`${traitName} mean plot`}
                                    className="max-w-full mx-auto rounded"
                                  />
                                </div>
                                <p className="text-xs text-muted-foreground mt-2 italic">
                                  Bars represent mean ± SE. Means sharing the same letter are not significantly different according to Tukey HSD (α = 0.05).
                                </p>
                              </div>
                            ) : (
                              <ResultMessage message="Figure could not be generated." />
                            )}

                            {tables && Object.entries(tables)
                              .filter(([k, v]) =>
                                k !== "anova" && k !== "means" && k !== "assumption_guidance"
                                && k !== "anova_table" && k !== "combined_anova"
                                && k !== "genotype_means" && k !== "tukey" && k !== "mean_separation"
                                && k !== "letters" && k !== "tukey_hsd"
                                && k !== "assumptions" && k !== "assumption_tests"
                                && !/error|traceback|exception/i.test(k)
                                && v != null
                                && (typeof v !== "object" || Object.keys(v as Record<string, unknown>).length > 0)
                              )
                              .map(([tName, tData]) => (
                                <GenericDataTable key={tName} label={tName.replace(/_/g, " ")} data={tData} />
                              ))}

                            {traitPlots && Object.entries(traitPlots)
                              .filter(([k, b64]) => k !== "mean_plot" && !/error|traceback|exception/i.test(k) && b64 && isValidBase64(b64))
                              .map(([pName, b64]) => (
                                <div key={pName}>
                                  <div className="flex items-center justify-between mb-2">
                                    <h4 className="text-sm font-semibold text-foreground capitalize flex items-center gap-2">
                                      <Image className="w-4 h-4 text-primary" />
                                      {pName.replace(/_/g, " ")}
                                    </h4>
                                    <FigureDownloadMenu title={`${traitName}_${pName}`} base64={b64} />
                                  </div>
                                  <div className="bg-background rounded-lg p-4 border border-border">
                                    <img
                                      src={`data:image/png;base64,${b64}`}
                                      alt={pName}
                                      className="max-w-full mx-auto rounded"
                                    />
                                  </div>
                                </div>
                              ))}

                            {integrityPassed && traitInterpretations[traitName] ? (
                              <div className="mt-4 pt-4 border-t border-border">
                                {traitGroundingChecks[traitName] && !traitGroundingChecks[traitName].passed && (
                                  <Collapsible defaultOpen className="mb-3">
                                    <div className="rounded-lg border border-accent/40 bg-accent/10 p-3">
                                      <CollapsibleTrigger className="w-full flex items-center justify-between text-xs font-semibold text-accent-foreground">
                                        <span className="flex items-center gap-1.5">
                                          <AlertCircle className="w-3.5 h-3.5" />
                                          Statistical Review Alerts ({traitGroundingChecks[traitName].warning_count} warning{traitGroundingChecks[traitName].warning_count !== 1 ? "s" : ""})
                                        </span>
                                        <ChevronDown className="w-3.5 h-3.5 transition-transform duration-200 [[data-state=open]>&]:rotate-180" />
                                      </CollapsibleTrigger>
                                      <CollapsibleContent>
                                        <ul className="list-disc list-inside space-y-0.5 mt-1">
                                          {traitGroundingChecks[traitName].warnings.map((w, i) => (
                                            <li key={i} className="text-xs text-accent-foreground/90">{w}</li>
                                          ))}
                                        </ul>
                                      </CollapsibleContent>
                                    </div>
                                  </Collapsible>
                                )}
                                <h4 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                                  <Sparkles className="w-4 h-4 text-primary" />
                                  Dr. Fayeun's Interpretation
                                </h4>
                                <div className="prose prose-sm max-w-none dark:prose-invert bg-muted/30 rounded-lg p-4">
                                  <ReactMarkdown>{traitInterpretations[traitName]}</ReactMarkdown>
                                </div>
                              </div>
                            ) : completedTraits.size > 0 && !completedTraits.has(traitName) && integrityPassed ? (
                              <div className="mt-4 pt-4 border-t border-border flex items-center gap-2 text-muted-foreground text-sm">
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Interpretation pending…
                              </div>
                            ) : traitInterpretations[traitName] && !integrityPassed ? (
                              <ResultMessage message="Result integrity validation failed. Tables and figures do not match." />
                            ) : null}
                          </div>
                        </AccordionContent>
                      </AccordionItem>
                    );
                  })}
                </Accordion>
              </CardContent>
            </Card>
          )}

          {/* Publishable HTML Tables */}
          {(() => {
            const htmlTables = results.html_tables && Object.keys(results.html_tables).length > 0
              ? results.html_tables
              : generatePublishableHtmlTables(results as Record<string, unknown>);
            return htmlTables && Object.keys(htmlTables).length > 0 ? (
              <HtmlTablesSection htmlTables={htmlTables} />
            ) : null;
          })()}
        </div>
      </div>
    </section>
  );
}
