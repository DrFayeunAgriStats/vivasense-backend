import { useMemo, useState } from "react";
import * as XLSX from "xlsx";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Loader2, TrendingUp, AlertTriangle, LineChart, Info, ArrowRight, CheckCircle2, XCircle, Activity, Download, Ban } from "lucide-react";
import {
  VS_SECTIONS,
  vsP,
  vsBullet,
  vsSectionHeading,
  vsKvTable,
  vsCautionParagraphs,
  vsChatHistory,
  vsInterpretationNote,
  downloadVivaSenseDocument,
  vsFmtP,
  type VsChatMsg,
} from "@/lib/vivasenseWordReport";
import type { Paragraph as DocxParagraph, Table as DocxTable } from "docx";
import { useToast } from "@/hooks/use-toast";
import { WordExportPreviewModal, type WordPreviewSection } from "@/components/vivasense/WordExportPreviewModal";
import { Eye } from "lucide-react";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RTooltip,
  Line,
  ComposedChart,
} from "recharts";
import type { DatasetContext } from "@/types/geneticsUpload";

const MODULE = "relationship" as const;

interface Props {
  datasetContext: DatasetContext;
}

interface RegressionResult {
  n: number;
  slope: number;
  intercept: number;
  r2: number;
  r: number;
  pValue: number;
  seSlope: number;
  ciSlopeLow: number;
  ciSlopeHigh: number;
  points: { x: number; y: number; yhat: number }[];
  xMin: number;
  xMax: number;
}

// Two-sided t critical value (95%) approximation via inverse-t (Hill 1970 simplified)
function tCritical95(df: number): number {
  if (df <= 0) return NaN;
  if (df > 30) return 1.96;
  // Lookup for small df, otherwise interpolate
  const table: Record<number, number> = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
  };
  return table[Math.round(df)] ?? 2.0;
}

// Student-t two-sided p-value via series approximation (good enough for UI)
function tCdf(t: number, df: number): number {
  // Abramowitz & Stegun approximation
  const x = df / (df + t * t);
  // regularized incomplete beta I_x(df/2, 1/2)
  const a = df / 2;
  const b = 0.5;
  // Use continued fraction via lanczos? Simpler: use approximation
  // Fall back to normal approx for df > 30
  if (df > 30) {
    // normal approximation
    const z = t;
    return 0.5 * (1 + erf(z / Math.SQRT2));
  }
  // crude but acceptable: incomplete beta via series
  const ibeta = incompleteBeta(x, a, b);
  const cdf = 1 - 0.5 * ibeta;
  return t >= 0 ? cdf : 1 - cdf;
}

function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function logGamma(z: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = logGamma(a) + logGamma(b) - logGamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
  // Lentz continued fraction
  let f = 1, c = 1, d = 0;
  for (let i = 0; i < 200; i++) {
    const m = i;
    let numerator: number;
    if (i === 0) numerator = 1;
    else if (i % 2 === 0) {
      const k = i / 2;
      numerator = (k * (b - k) * x) / ((a + 2 * k - 1) * (a + 2 * k));
    } else {
      const k = (i - 1) / 2;
      numerator = -((a + k) * (a + b + k) * x) / ((a + 2 * k) * (a + 2 * k + 1));
    }
    d = 1 + numerator * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + numerator / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < 1e-10) break;
  }
  return front * (f - 1);
}

function computeRegression(xs: number[], ys: number[]): RegressionResult | null {
  const pairs = xs.map((x, i) => [x, ys[i]] as [number, number]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  const n = pairs.length;
  if (n < 3) return null;
  const xMean = pairs.reduce((s, [x]) => s + x, 0) / n;
  const yMean = pairs.reduce((s, [, y]) => s + y, 0) / n;
  let sxx = 0, syy = 0, sxy = 0;
  pairs.forEach(([x, y]) => {
    sxx += (x - xMean) ** 2;
    syy += (y - yMean) ** 2;
    sxy += (x - xMean) * (y - yMean);
  });
  if (sxx === 0) return null;
  const slope = sxy / sxx;
  const intercept = yMean - slope * xMean;
  const ssRes = pairs.reduce((s, [x, y]) => s + (y - (intercept + slope * x)) ** 2, 0);
  const r2 = syy === 0 ? 0 : 1 - ssRes / syy;
  const r = syy === 0 || sxx === 0 ? 0 : sxy / Math.sqrt(sxx * syy);
  // standard error of slope and t-stat
  const df = n - 2;
  const seSlope = Math.sqrt(ssRes / df) / Math.sqrt(sxx);
  const tStat = seSlope === 0 ? 0 : slope / seSlope;
  const pValue = df > 0 ? 2 * (1 - tCdf(Math.abs(tStat), df)) : 1;

  const tCrit = tCritical95(df);
  const ciSlopeLow = slope - tCrit * seSlope;
  const ciSlopeHigh = slope + tCrit * seSlope;

  const xMin = Math.min(...pairs.map(([x]) => x));
  const xMax = Math.max(...pairs.map(([x]) => x));
  const points = pairs
    .map(([x, y]) => ({ x, y, yhat: intercept + slope * x }))
    .sort((a, b) => a.x - b.x);
  return { n, slope, intercept, r2, r, pValue, seSlope, ciSlopeLow, ciSlopeHigh, points, xMin, xMax };
}

function parseDataset(base64: string, fileType: string): Record<string, unknown>[] {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const wb = fileType === "csv"
    ? XLSX.read(new TextDecoder().decode(bytes), { type: "string" })
    : XLSX.read(bytes, { type: "array" });
  const sheet = wb.Sheets[wb.SheetNames[0]];
  return XLSX.utils.sheet_to_json<Record<string, unknown>>(sheet, { defval: null });
}

// ============================================================
// Word export — uses shared VivaSense branded report builder
// ============================================================
interface ChatMsg { role: "user" | "assistant" | "system"; content: string; }

interface WordExportPayload {
  xVar: string;
  yVar: string;
  datasetName: string;
  result: RegressionResult;
  interpretation: {
    directionLabel: string;
    directionSentence: string;
    strengthLabel: string;
    strengthSentence: string;
    significanceLabel: string;
    significanceSentence: string;
    practicalMeaningDisplay: boolean;
    practicalSentence: string;
    significant: boolean;
  };
  warnings: string[];
  finalInsight: string;
  chatHistory: ChatMsg[];
}

function buildEquation(r: RegressionResult): string {
  const sign = r.slope >= 0 ? "+" : "−";
  return `Y = ${r.intercept.toFixed(3)} ${sign} ${Math.abs(r.slope).toFixed(3)} · X`;
}

async function downloadWordReport(payload: WordExportPayload) {
  const { xVar, yVar, datasetName, result, interpretation, warnings, finalInsight, chatHistory } = payload;
  const dateStr = new Date().toISOString().slice(0, 10);

  const summaryTable = vsKvTable([
    ["Equation", buildEquation(result)],
    ["Slope (β)", result.slope.toFixed(4)],
    ["Intercept", result.intercept.toFixed(4)],
    ["R²", result.r2.toFixed(4)],
    ["p-value", vsFmtP(result.pValue)],
    ["Slope (95% CI)", `${result.ciSlopeLow.toFixed(4)} to ${result.ciSlopeHigh.toFixed(4)}`],
    ["Correlation (r)", result.r.toFixed(4)],
    ["Sample size (n)", String(result.n)],
  ]);

  const body: (DocxParagraph | DocxTable)[] = [];

  // 1. Dataset & Variables
  body.push(vsSectionHeading(VS_SECTIONS.dataset));
  body.push(vsP(`Dataset: ${datasetName}`));
  body.push(vsP(`Predictor (X): ${xVar}`));
  body.push(vsP(`Response (Y): ${yVar}`));

  // 2. Model Summary
  body.push(vsSectionHeading(VS_SECTIONS.model));
  body.push(summaryTable);
  body.push(vsP(""));

  // 3. Results — visual chart not embedded; reference what's on screen
  body.push(vsSectionHeading(VS_SECTIONS.results));
  body.push(vsP(
    "A scatter plot with the fitted regression line is shown in the application. Coefficient estimates and fit statistics are tabulated above."
  ));

  // 4. Interpretation
  body.push(vsSectionHeading(VS_SECTIONS.interpretation));
  body.push(vsP("Direction of Relationship", { bold: true }));
  body.push(vsP(`${interpretation.directionLabel} — ${interpretation.directionSentence}`));
  body.push(vsP("Strength of Relationship", { bold: true }));
  body.push(vsP(`${interpretation.strengthLabel} — ${interpretation.strengthSentence}`));
  body.push(vsP("Statistical Significance", { bold: true }));
  body.push(vsP(`${interpretation.significanceLabel} — ${interpretation.significanceSentence} (p ${vsFmtP(result.pValue)})`));
  body.push(vsP("Practical Meaning", { bold: true }));
  body.push(vsP(interpretation.practicalSentence));

  // 5. Caution & Reliability
  body.push(vsSectionHeading(VS_SECTIONS.caution));
  if (!interpretation.significant) {
    body.push(vsBullet(
      "No statistically reliable linear relationship detected. This model is NOT suitable for prediction.",
      "B91C1C",
    ));
  }
  vsCautionParagraphs(warnings.filter((w) =>
    !w.startsWith("No statistically reliable linear relationship")
  )).forEach((p) => body.push(p));

  // 6. Final Insight Summary
  body.push(vsSectionHeading(VS_SECTIONS.finalInsight));
  body.push(vsP(finalInsight));

  // 7. Chat History
  body.push(vsSectionHeading(VS_SECTIONS.chat));
  vsChatHistory(chatHistory as VsChatMsg[]).forEach((p) => body.push(p));

  // 8. Interpretation Note
  body.push(vsSectionHeading(VS_SECTIONS.note));
  body.push(vsInterpretationNote());

  const safe = (s: string) => s.replace(/[^a-zA-Z0-9_-]+/g, "_");
  await downloadVivaSenseDocument(
    {
      header: {
        moduleType: "Regression",
        contextLine: `Linear model · ${xVar} → ${yVar} · n = ${result.n}`,
      },
      body,
    },
    `VivaSense_Regression_${safe(xVar)}_vs_${safe(yVar)}_${dateStr}.docx`,
  );
}

export function RegressionAnalysisTab({ datasetContext }: Props) {
  const { toast } = useToast();
  const [xVar, setXVar] = useState<string>("");
  const [yVar, setYVar] = useState<string>("");
  const [model, setModel] = useState<string>("linear");
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<RegressionResult | null>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(false);

  const handleRun = async () => {
    if (!xVar || !yVar) {
      toast({ title: "Select both X and Y variables", variant: "destructive" });
      return;
    }
    if (xVar === yVar) {
      toast({ title: "X and Y must differ", variant: "destructive" });
      return;
    }
    setIsRunning(true);
    setResult(null);
    try {
      console.log("[MODULE]", MODULE, "[SUBMODULE] regression", { xVar, yVar, model });
      const rows = parseDataset(datasetContext.base64Content, datasetContext.fileType);
      const xs = rows.map((r) => Number(r[xVar]));
      const ys = rows.map((r) => Number(r[yVar]));
      const res = computeRegression(xs, ys);
      if (!res) {
        toast({ title: "Insufficient numeric data", description: "Need at least 3 valid (X, Y) pairs.", variant: "destructive" });
        return;
      }
      setResult(res);
      toast({ title: "Regression computed" });
    } catch (err: any) {
      toast({ title: "Regression failed", description: err.message, variant: "destructive" });
    } finally {
      setIsRunning(false);
    }
  };

  const interpretation = useMemo(() => {
    if (!result) return null;

    // === STATISTICAL HIERARCHY: Significance gates Direction, Strength, Practical Meaning ===
    const significant = result.pValue < 0.05;

    // 1) DIRECTION — only meaningful if significant
    let directionClass: "positive" | "negative" | "no_clear_relationship";
    let directionLabel: string;
    let directionSentence: string;
    if (!significant) {
      directionClass = "no_clear_relationship";
      directionLabel = "No Directional Relationship";
      directionSentence = `No directional relationship was established between ${xVar} and ${yVar}.`;
    } else if (result.slope > 0) {
      directionClass = "positive";
      directionLabel = "Positive Relationship";
      directionSentence = `Higher values of ${xVar} are associated with higher values of ${yVar}.`;
    } else {
      directionClass = "negative";
      directionLabel = "Negative Relationship";
      directionSentence = `Higher values of ${xVar} are associated with lower values of ${yVar}.`;
    }

    // 2) STRENGTH — only meaningful if significant
    let strengthClass: "negligible" | "weak" | "moderate" | "strong" | "very_strong";
    let strengthLabel: string;
    let strengthSentence: string;
    if (!significant) {
      strengthClass = "negligible";
      strengthLabel = "Negligible / Unreliable";
      strengthSentence = "The model explains negligible variation and is not statistically reliable.";
    } else if (result.r2 < 0.25) {
      strengthClass = "weak";
      strengthLabel = "Weak";
      strengthSentence = "The fitted model explains a small proportion of the variation in the response.";
    } else if (result.r2 < 0.50) {
      strengthClass = "moderate";
      strengthLabel = "Moderate";
      strengthSentence = "The fitted model explains a moderate proportion of the variation in the response.";
    } else if (result.r2 < 0.75) {
      strengthClass = "strong";
      strengthLabel = "Strong";
      strengthSentence = "The fitted model explains a substantial proportion of the variation in the response.";
    } else {
      strengthClass = "very_strong";
      strengthLabel = "Very Strong";
      strengthSentence = "The fitted model explains a very large proportion of the variation in the response.";
    }

    // 3) SIGNIFICANCE
    const significanceLabel = significant ? "Statistically Significant" : "Not Statistically Significant";
    const significanceSentence = significant
      ? "Evidence of a linear relationship was detected (p < 0.05)."
      : "Not statistically reliable (p ≥ 0.05).";

    // 4) PRACTICAL MEANING — HIDDEN if not significant
    const practicalMeaningDisplay = significant;
    const practicalSentence = significant
      ? (result.slope >= 0
          ? `For every 1-unit increase in ${xVar}, ${yVar} is expected to change by ${result.slope.toFixed(3)} units on average, based on the fitted linear model.`
          : `For every 1-unit increase in ${xVar}, ${yVar} is expected to decrease by ${Math.abs(result.slope).toFixed(3)} units on average, based on the fitted linear model.`)
      : "Because the relationship is not statistically reliable, the slope must NOT be interpreted.";

    // === Consistency assertions (dev safety) ===
    if (!significant) {
      console.assert(directionClass === "no_clear_relationship", "Direction must be no_clear_relationship when p ≥ 0.05");
      console.assert(strengthClass === "negligible", "Strength must be negligible when p ≥ 0.05");
      console.assert(practicalMeaningDisplay === false, "Practical meaning must be hidden when p ≥ 0.05");
    }

    return {
      directionClass, directionLabel, directionSentence,
      strengthClass, strengthLabel, strengthSentence,
      significant, significanceLabel, significanceSentence,
      practicalMeaningDisplay, practicalSentence,
    };
  }, [result, xVar, yVar]);

  const reliabilityWarnings = useMemo(() => {
    if (!result) return [] as string[];
    const w: string[] = [];
    if (result.pValue >= 0.05) {
      w.push("No statistically reliable linear relationship detected. This model is NOT suitable for prediction.");
    }
    if (result.n < 10) w.push("Small sample size — results may be unstable.");
    if (result.r2 > 0.9 && result.n < 15) w.push("Very high model fit with small sample size — may be misleading.");
    return w;
  }, [result]);

  // Final Insight Summary — replaces previous "What this means"
  const finalInsight = useMemo(() => {
    if (!result || !interpretation) return "";
    if (!interpretation.significant) {
      return `The analysis did not establish a statistically reliable relationship between ${xVar} and ${yVar} (p = ${result.pValue.toFixed(3)}). The model explains negligible variation and should not be used for prediction.`;
    }
    const dir = interpretation.directionClass;
    const str = interpretation.strengthClass;
    const dirPhrase = dir === "positive" ? "move together in a positive direction" : "move in opposite directions";
    const fitPhrase =
      str === "very_strong" || str === "strong" ? "the fitted model captures much of that pattern"
      : str === "moderate" ? "the fitted model captures a moderate share of that pattern"
      : "the fitted model captures only a small share of that pattern";
    return `The two variables ${dirPhrase}, and ${fitPhrase}. The relationship is statistically significant, but the result should be interpreted as association rather than cause-and-effect.`;
  }, [result, interpretation, xVar, yVar]);

  const traits = datasetContext.availableTraitColumns;

  const datasetName = (datasetContext as { fileName?: string }).fileName ?? datasetContext?.file?.name ?? "dataset";

  const previewSections: WordPreviewSection[] = useMemo(() => {
    if (!result || !interpretation) return [];
    const sections: WordPreviewSection[] = [
      {
        heading: VS_SECTIONS.dataset,
        bullets: [
          `Dataset: ${datasetName}`,
          `Predictor (X): ${xVar}`,
          `Response (Y): ${yVar}`,
        ],
      },
      {
        heading: VS_SECTIONS.model,
        rows: [
          ["Equation", buildEquation(result)],
          ["Slope (β)", result.slope.toFixed(4)],
          ["Intercept", result.intercept.toFixed(4)],
          ["R²", result.r2.toFixed(4)],
          ["p-value", vsFmtP(result.pValue)],
          ["Slope (95% CI)", `${result.ciSlopeLow.toFixed(4)} to ${result.ciSlopeHigh.toFixed(4)}`],
          ["Correlation (r)", result.r.toFixed(4)],
          ["Sample size (n)", String(result.n)],
        ],
      },
      {
        heading: VS_SECTIONS.results,
        paragraph:
          "A scatter plot with the fitted regression line is shown in the application. Coefficient estimates and fit statistics are tabulated above.",
      },
      {
        heading: VS_SECTIONS.interpretation,
        bullets: [
          `Direction — ${interpretation.directionLabel}: ${interpretation.directionSentence}`,
          `Strength — ${interpretation.strengthLabel}: ${interpretation.strengthSentence}`,
          `Significance — ${interpretation.significanceLabel}: ${interpretation.significanceSentence} (p ${vsFmtP(result.pValue)})`,
          `Practical Meaning — ${interpretation.practicalSentence}`,
        ],
      },
      {
        heading: VS_SECTIONS.finalInsight,
        paragraph: finalInsight,
      },
      {
        heading: VS_SECTIONS.note,
        paragraph:
          "Regression describes statistical association, not causation. Any domain-specific conclusion should be made in the context of study design, measurement quality, and subject-matter knowledge.",
      },
    ];
    return sections;
  }, [result, interpretation, finalInsight, xVar, yVar, datasetName]);

  const previewWarnings: string[] = useMemo(() => {
    const w: string[] = [];
    if (interpretation && !interpretation.significant) {
      w.push("No statistically reliable linear relationship detected. This model is NOT suitable for prediction.");
    }
    reliabilityWarnings.forEach((rw) => {
      if (!rw.startsWith("No statistically reliable linear relationship")) w.push(rw);
    });
    return w;
  }, [interpretation, reliabilityWarnings]);

  const canExport = !!result && !!interpretation;

  if (import.meta.env.DEV) {
    // eslint-disable-next-line no-console
    console.debug("[REGRESSION][preview]", { canExport, sections: previewSections.length, warnings: previewWarnings.length });
  }

  const handleExportWord = async () => {
    if (!result || !interpretation) return;
    setIsExporting(true);
    try {
      await downloadWordReport({
        xVar,
        yVar,
        datasetName: (datasetContext as { fileName?: string }).fileName ?? "dataset",
        result,
        interpretation: {
          directionLabel: interpretation.directionLabel,
          directionSentence: interpretation.directionSentence,
          strengthLabel: interpretation.strengthLabel,
          strengthSentence: interpretation.strengthSentence,
          significanceLabel: interpretation.significanceLabel,
          significanceSentence: interpretation.significanceSentence,
          practicalMeaningDisplay: interpretation.practicalMeaningDisplay,
          practicalSentence: interpretation.practicalSentence,
          significant: interpretation.significant,
        },
        warnings: reliabilityWarnings,
        finalInsight,
        chatHistory: [],
      });
      toast({ title: "Word report downloaded" });
    } catch (err: any) {
      toast({ title: "Word export failed", description: err.message, variant: "destructive" });
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* Cross-reference note */}
      <div className="rounded-md border border-amber-300/60 bg-amber-50/70 dark:bg-amber-900/10 dark:border-amber-700/40 p-3 text-sm flex items-start gap-2">
        <Info className="h-4 w-4 text-amber-700 dark:text-amber-300 mt-0.5 shrink-0" />
        <div className="text-amber-900 dark:text-amber-100">
          Looking for trait-based interpretation? Use{" "}
          <span className="font-medium">Trait Influence Analysis</span> in the{" "}
          <span className="font-medium">Genetic Analysis</span> module.
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <LineChart className="h-5 w-5 text-primary" />
            Regression Analysis
          </CardTitle>
          <p className="text-sm text-muted-foreground mt-1">
            Examines the relationship between two variables and fits a regression model.
          </p>
        </CardHeader>
        <CardContent className="space-y-5">
          <p className="text-sm text-muted-foreground">
            Select a predictor (X) and response (Y) variable to examine their relationship. A regression model will be fitted and visualized.
          </p>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">X Variable (Predictor)</label>
              <select
                value={xVar}
                onChange={(e) => setXVar(e.target.value)}
                className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm"
              >
                <option value="">Select variable…</option>
                {traits.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Y Variable (Response)</label>
              <select
                value={yVar}
                onChange={(e) => setYVar(e.target.value)}
                className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm"
              >
                <option value="">Select variable…</option>
                {traits.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium">Regression Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm"
              >
                <option value="linear">Linear (Y = a + bX)</option>
                <option value="quadratic" disabled>Quadratic (coming soon)</option>
              </select>
            </div>
          </div>

          {!result && (
            <div className="rounded-md border border-dashed border-border bg-muted/30 p-6 text-sm text-muted-foreground space-y-2">
              <LineChart className="h-8 w-8 mx-auto text-muted-foreground/50" />
              <p className="font-medium text-foreground text-center">After running the analysis, you will see:</p>
              <ul className="list-disc pl-6 space-y-1 max-w-md mx-auto">
                <li>Scatter plot with fitted regression line</li>
                <li>Model equation</li>
                <li>Strength and significance of the relationship</li>
                <li>Interpretation of how the predictor influences the response</li>
                <li>Reliability assessment of the model</li>
              </ul>
            </div>
          )}

          <Button onClick={handleRun} disabled={isRunning || !xVar || !yVar} className="gap-2">
            {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <TrendingUp className="h-4 w-4" />}
            Run Regression
            <ArrowRight className="h-4 w-4" />
          </Button>

          <p className="text-xs text-muted-foreground italic">
            This module provides general statistical interpretation. Domain-specific conclusions should be made within the appropriate analytical context.
          </p>
        </CardContent>
      </Card>

      {result && interpretation && (
        <div className="space-y-4">
          {/* RED CAUTION BOX — top priority when not significant */}
          {!interpretation.significant && (
            <div className="rounded-md border-2 border-destructive bg-destructive/10 p-4 flex items-start gap-3">
              <Ban className="h-5 w-5 text-destructive mt-0.5 shrink-0" />
              <div className="text-sm">
                <p className="font-semibold text-destructive mb-1">
                  No statistically reliable linear relationship detected.
                </p>
                <p className="text-destructive/90">
                  This model is <strong>NOT</strong> suitable for prediction. The slope and direction must not be interpreted (p = {result.pValue.toFixed(3)}).
                </p>
              </div>
            </div>
          )}

          {/* Model Output Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <Activity className="h-4 w-4 text-primary" />
                Model Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
                <SummaryItem label="Equation" value={`Y = ${result.intercept.toFixed(3)} ${result.slope >= 0 ? "+" : "−"} ${Math.abs(result.slope).toFixed(3)}·X`} mono />
                <SummaryItem label="Slope (β)" value={result.slope.toFixed(4)} mono />
                <SummaryItem label="Intercept" value={result.intercept.toFixed(4)} mono />
                <SummaryItem label="R²" value={result.r2.toFixed(4)} mono />
                <SummaryItem label="p-value" value={result.pValue < 0.001 ? "<0.001" : result.pValue.toFixed(4)} mono />
                <SummaryItem
                  label="Slope (95% CI)"
                  value={`${result.ciSlopeLow.toFixed(4)} to ${result.ciSlopeHigh.toFixed(4)}`}
                  mono
                />
              </div>
              <div className="mt-3 text-xs text-muted-foreground">
                Correlation coefficient (r): <span className="font-mono">{result.r.toFixed(4)}</span> · Sample size (n) = {result.n}
              </div>
            </CardContent>
          </Card>

          {/* Scatter + line */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <LineChart className="h-4 w-4 text-primary" />
                Scatter Plot with Fitted Regression Line
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={result.points} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis dataKey="x" type="number" name={xVar} domain={["dataMin", "dataMax"]} tick={{ fontSize: 12 }} label={{ value: xVar, position: "insideBottom", offset: -5, fontSize: 12 }} />
                    <YAxis dataKey="y" type="number" name={yVar} tick={{ fontSize: 12 }} label={{ value: yVar, angle: -90, position: "insideLeft", fontSize: 12 }} />
                    <RTooltip cursor={{ strokeDasharray: "3 3" }} />
                    <Scatter data={result.points} fill="hsl(var(--primary))" />
                    <Line type="linear" dataKey="yhat" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} activeDot={false} isAnimationActive={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Interpretation Panel */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Interpretation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <InterpretBlock title="Direction of Relationship">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant={
                    interpretation.directionClass === "no_clear_relationship" ? "outline"
                    : interpretation.directionClass === "negative" ? "secondary"
                    : "default"
                  }>
                    {interpretation.directionLabel}
                  </Badge>
                </div>
                <p className="text-muted-foreground mt-1.5">{interpretation.directionSentence}</p>
              </InterpretBlock>

              <InterpretBlock title="Strength of Relationship">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant={interpretation.strengthClass === "negligible" ? "destructive" : "outline"}>
                    {interpretation.strengthLabel}
                  </Badge>
                  {interpretation.significant && (
                    <span className="text-muted-foreground">
                      This model explains <span className="font-medium text-foreground">{(result.r2 * 100).toFixed(1)}%</span> of the variation in {yVar}.
                    </span>
                  )}
                </div>
                <p className="text-muted-foreground mt-1.5">{interpretation.strengthSentence}</p>
              </InterpretBlock>

              <InterpretBlock title="Statistical Significance">
                <div className="flex items-center gap-2">
                  {interpretation.significant ? (
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                  ) : (
                    <XCircle className="h-4 w-4 text-destructive" />
                  )}
                  <span className="text-foreground">
                    {interpretation.significanceSentence} (p {result.pValue < 0.001 ? "< 0.001" : `= ${result.pValue.toFixed(3)}`})
                  </span>
                </div>
              </InterpretBlock>

              <InterpretBlock title="Practical Meaning">
                {interpretation.practicalMeaningDisplay ? (
                  <>
                    <p className="text-muted-foreground">{interpretation.practicalSentence}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Slope 95% CI: <span className="font-mono text-foreground">{result.ciSlopeLow.toFixed(3)}</span> to{" "}
                      <span className="font-mono text-foreground">{result.ciSlopeHigh.toFixed(3)}</span>
                    </p>
                  </>
                ) : (
                  <div className="rounded-md border border-destructive/40 bg-destructive/5 p-3 flex items-start gap-2">
                    <Ban className="h-4 w-4 text-destructive mt-0.5 shrink-0" />
                    <p className="text-destructive">{interpretation.practicalSentence}</p>
                  </div>
                )}
              </InterpretBlock>

              {/* Caution and Reliability */}
              <InterpretBlock title="Caution and Reliability">
                {reliabilityWarnings.length > 0 ? (
                  <div className="rounded-md border border-amber-300/60 bg-amber-50/70 dark:bg-amber-900/10 dark:border-amber-700/40 p-3">
                    <div className="flex items-center gap-2 text-amber-800 dark:text-amber-200 font-medium mb-1.5">
                      <AlertTriangle className="h-4 w-4" />
                      Model Reliability
                    </div>
                    <ul className="list-disc pl-5 space-y-1 text-amber-900 dark:text-amber-100">
                      {reliabilityWarnings.map((w, i) => <li key={i}>{w}</li>)}
                    </ul>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 rounded-md border border-emerald-300/60 bg-emerald-50/70 dark:bg-emerald-900/10 dark:border-emerald-700/40 p-3 text-emerald-900 dark:text-emerald-100">
                    <CheckCircle2 className="h-4 w-4" />
                    No major reliability warnings were detected for this model.
                  </div>
                )}
              </InterpretBlock>

              {/* Interpretation Note */}
              <InterpretBlock title="Interpretation Note">
                <p className="text-muted-foreground italic">
                  Regression describes statistical association, not causation. Any domain-specific conclusion should be made in the context of study design, measurement quality, and subject-matter knowledge.
                </p>
              </InterpretBlock>

              {/* What this means */}
              <div className="rounded-md border border-border bg-muted/30 p-3">
                <p className="font-medium text-foreground mb-1">Final Insight Summary</p>
                <p className="text-muted-foreground">{finalInsight}</p>
              </div>
            </CardContent>
          </Card>

          {/* Export buttons */}
          <div className="flex flex-wrap justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => setPreviewOpen(true)}
              disabled={!canExport || isExporting}
              className="gap-2"
            >
              <Eye className="h-4 w-4" />
              Preview Report
            </Button>
            <Button
              variant="outline"
              onClick={handleExportWord}
              disabled={!canExport || isExporting}
              className="gap-2"
            >
              {isExporting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
              Export to Word
            </Button>
          </div>

          <WordExportPreviewModal
            open={previewOpen}
            onOpenChange={setPreviewOpen}
            moduleName="Regression"
            reportTitle={`Regression Analysis · ${xVar} → ${yVar}`}
            datasetSummary={`Dataset: ${datasetName} · Linear model · n = ${result?.n ?? 0}`}
            sections={previewSections}
            warnings={previewWarnings}
            notes={[
              "Regression describes statistical association, not causation.",
            ]}
            canExport={canExport}
            cannotExportReason={!canExport ? "Run a regression first to enable export." : undefined}
            onConfirmExport={handleExportWord}
          />

          {/* Cross-module nav */}
          <div className="rounded-md border border-border bg-muted/30 p-3 text-sm flex items-start gap-2">
            <Info className="h-4 w-4 text-primary mt-0.5 shrink-0" />
            <div>
              Need deeper trait-based interpretation? Use{" "}
              <span className="font-medium">Trait Influence Analysis</span> in the{" "}
              <span className="font-medium">Genetic Analysis</span> module.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function SummaryItem({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-md border border-border bg-muted/20 px-3 py-2">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`text-sm font-medium ${mono ? "font-mono" : ""}`}>{value}</div>
    </div>
  );
}

function InterpretBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <p className="font-medium text-foreground">{title}</p>
      <div>{children}</div>
    </div>
  );
}
