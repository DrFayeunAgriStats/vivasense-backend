import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, TrendingUp, ChevronDown, CheckCircle, XCircle, Lightbulb, BarChart3, Shield } from "lucide-react";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { TableDownloadMenu } from "../results/TableDownloadMenu";
import ReactMarkdown from "react-markdown";

// Types matching the exact backend response
interface RegressionCoefficient {
  term: string;
  estimate: number;
  std_error: number;
  t_statistic: number;
  p_value: number;
  p_display: string;
  significance: string;
  ci_lower: number;
  ci_upper: number;
}

interface ModelFit {
  r_squared: number;
  adj_r_squared: number;
  f_statistic: number;
  f_p_value: number;
  f_p_display: string;
  rmse: number;
  n_obs: number;
  n_predictors: number;
  aic: number;
  bic: number;
}

interface VIFEntry {
  vif: number;
  flag: string;
  interpretation: string;
}

interface AssumptionEntry {
  test: string;
  statistic: number;
  p_value: number;
  passed: boolean;
  interpretation: string;
}

interface RegressionData {
  model?: string;
  response?: string;
  predictors?: string[];
  coefficients?: RegressionCoefficient[];
  model_fit?: ModelFit;
  vif?: Record<string, VIFEntry>;
  assumptions?: Record<string, AssumptionEntry>;
  warnings?: string[];
}

interface Props {
  data: {
    tables?: { regression?: RegressionData; [key: string]: unknown };
    interpretation?: string;
    intelligence?: { executive_insight?: string; [key: string]: unknown };
    [key: string]: unknown;
  };
}

const fmt4 = (v: unknown): string =>
  v == null || v === "" ? "—" : Number(v).toFixed(4);
const fmt2 = (v: unknown): string =>
  v == null || v === "" ? "—" : Number(v).toFixed(2);

// Local CollapsibleCard delegates to the shared VsResultSection so the
// regression UI matches the Word report's section hierarchy + typography.
import { VsResultSection } from "@/components/vivasense/results/VsResultSection";

function CollapsibleCard({
  title,
  icon,
  defaultOpen = true,
  children,
}: {
  title: string;
  icon: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  // VsResultSection accepts an icon component; wrap a node icon in a tiny
  // functional component so the API stays compatible with existing callers.
  const IconCmp = React.useMemo<React.ElementType>(
    () => () => <>{icon}</>,
    [icon],
  );
  return (
    <VsResultSection title={title} icon={IconCmp} defaultOpen={defaultOpen}>
      {children}
    </VsResultSection>
  );
}

export function RegressionResultsDisplay({ data }: Props) {
  // Find regression data - check tables.regression or any key matching /regress/i
  const tables = data.tables ?? {};
  let reg: RegressionData | null = null;
  if (tables.regression && typeof tables.regression === "object") {
    reg = tables.regression as RegressionData;
  } else {
    const key = Object.keys(tables).find((k) => /regress/i.test(k));
    if (key) reg = tables[key] as RegressionData;
  }

  if (!reg) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground">
          Regression results unavailable.
        </CardContent>
      </Card>
    );
  }

  const coefficients = reg.coefficients ?? [];
  const modelFit = reg.model_fit;
  const vifData = reg.vif;
  const assumptions = reg.assumptions;
  const warnings = reg.warnings ?? [];
  const rawInterpretation = data.interpretation || data.intelligence?.executive_insight;
  const interpretation: string | null = rawInterpretation
    ? Array.isArray(rawInterpretation)
      ? rawInterpretation.map((item) => typeof item === "string" ? `- ${item}` : `- ${JSON.stringify(item)}`).join("\n")
      : typeof rawInterpretation === "object"
      ? JSON.stringify(rawInterpretation, null, 2)
      : String(rawInterpretation)
    : null;

  const isSignificant = modelFit && modelFit.f_p_value < 0.05;

  return (
    <div className="space-y-4">
      {/* Header — Trait Influence Analysis */}
      <div className="space-y-1">
        <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-primary" />
          Trait Influence Analysis
        </h3>
        <p className="text-sm text-muted-foreground">
          Evaluate how traits influence a response variable using regression.
        </p>
      </div>

      {/* Card 1 — Model Summary */}
      {modelFit && (
        <Card className={`border-2 ${isSignificant ? "border-emerald-300 dark:border-emerald-700" : "border-amber-300 dark:border-amber-700"}`}>
          <CardContent className="p-5">
            <div className="flex items-start gap-3 mb-4">
              <TrendingUp className={`w-6 h-6 mt-0.5 ${isSignificant ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"}`} />
              <div className="flex-1">
                <p className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Model Summary</p>
                {reg.model && (
                  <p className="font-mono text-sm font-semibold text-foreground mb-2">
                    {reg.model}
                  </p>
                )}
                <div className="flex flex-wrap gap-3 items-center">
                  <Badge variant="outline">n = {modelFit.n_obs}</Badge>
                  <Badge variant="outline">{modelFit.n_predictors} predictor{modelFit.n_predictors !== 1 ? "s" : ""}</Badge>
                  <div className="text-sm">
                    <span className="font-semibold text-foreground">R² = </span>
                    <span className="font-mono text-foreground">{fmt4(modelFit.r_squared)}</span>
                    <span className="text-muted-foreground ml-1">({(modelFit.r_squared * 100).toFixed(1)}%)</span>
                  </div>
                  <div className="text-sm">
                    <span className="font-semibold text-foreground">Adj. R² = </span>
                    <span className="font-mono text-foreground">{fmt4(modelFit.adj_r_squared)}</span>
                  </div>
                  <div className="text-sm">
                    <span className="font-semibold text-foreground">RMSE = </span>
                    <span className="font-mono text-foreground">{fmt2(modelFit.rmse)}</span>
                  </div>
                </div>
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-sm font-semibold text-foreground">F = {fmt2(modelFit.f_statistic)}</span>
                  <span className="text-sm text-muted-foreground">({modelFit.f_p_display})</span>
                  <Badge className={isSignificant
                    ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300"
                    : "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300"
                  }>
                    {isSignificant ? "✓ Significant" : "Not Significant"}
                  </Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Card 2 — Coefficient Table */}
      {coefficients.length > 0 && (
        <CollapsibleCard
          title="Trait Coefficients & Effect Direction"
          icon={<BarChart3 className="w-5 h-5 text-primary" />}
          defaultOpen
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-border">
                  {["Term", "Estimate (β)", "Std Error", "t-statistic", "p-value", "95% CI", "Sig."].map((h) => (
                    <th key={h} className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50 whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {coefficients.map((c, i) => {
                  const isIntercept = c.term === "Intercept" || c.term === "(Intercept)";
                  const isSig = c.p_value < 0.05 && !isIntercept;
                  return (
                    <tr
                      key={i}
                      className={`border-b border-border/50 ${i % 2 !== 0 ? "bg-muted/20" : ""} ${isSig ? "border-l-4 border-l-emerald-500" : ""}`}
                    >
                      <td className={`px-3 py-2 font-medium whitespace-nowrap ${isIntercept ? "italic text-muted-foreground" : "text-foreground"}`}>
                        {c.term}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-xs">{fmt4(c.estimate)}</td>
                      <td className="px-3 py-2 text-right font-mono text-xs">{fmt4(c.std_error)}</td>
                      <td className="px-3 py-2 text-right font-mono text-xs">{fmt2(c.t_statistic)}</td>
                      <td className="px-3 py-2 text-right font-mono text-xs">{c.p_display}</td>
                      <td className="px-3 py-2 text-right font-mono text-xs whitespace-nowrap">
                        [{fmt4(c.ci_lower)}, {fmt4(c.ci_upper)}]
                      </td>
                      <td className="px-3 py-2 text-center font-bold">
                        {c.significance}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted-foreground mt-3 italic">
            Significance: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, ns = not significant.
          </p>
          <div className="flex justify-end mt-2">
            <TableDownloadMenu
              title="Regression_Coefficients"
              headers={["Term", "Estimate", "Std Error", "t-statistic", "p-value", "CI Lower", "CI Upper", "Sig"]}
              rows={coefficients.map((c) => [c.term, String(c.estimate), String(c.std_error), String(c.t_statistic), c.p_display, String(c.ci_lower), String(c.ci_upper), c.significance])}
            />
          </div>
        </CollapsibleCard>
      )}

      {/* Card 3 — Model Fit Statistics */}
      {modelFit && (
        <CollapsibleCard
          title="Model Fit Statistics"
          icon={<BarChart3 className="w-5 h-5 text-primary" />}
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "R²", value: fmt4(modelFit.r_squared) },
              { label: "Adjusted R²", value: fmt4(modelFit.adj_r_squared) },
              { label: "F-statistic", value: fmt2(modelFit.f_statistic) },
              { label: "F p-value", value: modelFit.f_p_display },
              { label: "RMSE", value: fmt2(modelFit.rmse) },
              { label: "AIC", value: fmt2(modelFit.aic) },
              { label: "BIC", value: fmt2(modelFit.bic) },
              { label: "n observations", value: String(modelFit.n_obs) },
            ].map((item) => (
              <div key={item.label} className="rounded-lg border border-border p-3 text-center">
                <p className="text-xs text-muted-foreground mb-1">{item.label}</p>
                <p className="font-mono font-semibold text-foreground">{item.value}</p>
              </div>
            ))}
          </div>
        </CollapsibleCard>
      )}

      {/* Card 4 — VIF */}
      {vifData && Object.keys(vifData).length > 0 && (
        <CollapsibleCard
          title="Multicollinearity (VIF)"
          icon={<Shield className="w-5 h-5 text-primary" />}
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-border">
                  {["Predictor", "VIF", "Assessment"].map((h) => (
                    <th key={h} className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50 whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(vifData).map(([predictor, entry], i) => {
                  const flagColors: Record<string, string> = {
                    ok: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300",
                    moderate: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300",
                    severe: "bg-destructive/10 text-destructive",
                  };
                  const flagLabels: Record<string, string> = {
                    ok: "✓ Acceptable",
                    moderate: "⚠ Moderate (VIF 5–10)",
                    severe: "❌ Severe (VIF > 10)",
                  };
                  return (
                    <tr key={predictor} className={`border-b border-border/50 ${i % 2 !== 0 ? "bg-muted/20" : ""}`}>
                      <td className="px-3 py-2 font-medium text-foreground">{predictor}</td>
                      <td className="px-3 py-2 text-right font-mono text-xs">{fmt2(entry.vif)}</td>
                      <td className="px-3 py-2">
                        <Badge variant="outline" className={flagColors[entry.flag] ?? ""}>
                          {flagLabels[entry.flag] ?? entry.flag}
                        </Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted-foreground mt-3 italic">
            VIF &gt; 10 indicates severe multicollinearity — consider removing or combining predictors.
          </p>
        </CollapsibleCard>
      )}

      {/* Card 5 — Assumption Tests */}
      {assumptions && Object.keys(assumptions).length > 0 && (
        <CollapsibleCard
          title="Assumption Tests"
          icon={<CheckCircle className="w-5 h-5 text-primary" />}
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-border">
                  {["Test", "Statistic", "p-value", "Passed", "Interpretation"].map((h) => (
                    <th key={h} className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50 whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(assumptions).map(([key, entry], i) => (
                  <tr key={key} className={`border-b border-border/50 ${i % 2 !== 0 ? "bg-muted/20" : ""}`}>
                    <td className="px-3 py-2 font-medium text-foreground">{entry.test}</td>
                    <td className="px-3 py-2 text-right font-mono text-xs">{fmt4(entry.statistic)}</td>
                    <td className="px-3 py-2 text-right font-mono text-xs">{fmt4(entry.p_value)}</td>
                    <td className="px-3 py-2">
                      {entry.passed ? (
                        <span className="text-emerald-600 dark:text-emerald-400 font-medium flex items-center gap-1">
                          <CheckCircle className="w-4 h-4" /> Passed
                        </span>
                      ) : (
                        <span className="text-destructive font-medium flex items-center gap-1">
                          <XCircle className="w-4 h-4" /> Failed
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-2 text-sm text-muted-foreground">{entry.interpretation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CollapsibleCard>
      )}

      {/* Card 6 — Interpretation */}
      {interpretation && (
        <CollapsibleCard
          title="Interpretation"
          icon={<Lightbulb className="w-5 h-5 text-primary" />}
        >
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <ReactMarkdown>{String(interpretation)}</ReactMarkdown>
          </div>
          <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-700 p-4">
            <p className="text-sm text-amber-800 dark:text-amber-200">
              ⚠️ <strong>Academic Integrity Reminder:</strong> Verify all numbers against the tables above. Adapt text to your own words.
            </p>
          </div>
        </CollapsibleCard>
      )}

      {/* Card 7 — Warnings */}
      {warnings.length > 0 && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-700 p-4">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-amber-800 dark:text-amber-200 mb-2">Warnings</p>
              <ul className="list-disc list-inside space-y-1">
                {warnings.map((w, i) => (
                  <li key={i} className="text-sm text-amber-700 dark:text-amber-300">{w}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
