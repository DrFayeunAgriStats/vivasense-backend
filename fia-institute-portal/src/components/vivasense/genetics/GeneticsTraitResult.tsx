import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Download, FileText, Image as ImageIcon, BarChart3, Info, Lightbulb, AlertTriangle, BookOpen, ChevronDown, Target, FlaskConical, Beaker } from "lucide-react";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import ReactMarkdown from "react-markdown";
import { GeneticsFormulaPanel } from "./GeneticsFormulaPanel";
import { PathAnalysisDisplay } from "./PathAnalysisDisplay";
import { PathAnalysisResultsDisplay } from "./PathAnalysisResultsDisplay";
import { CorrelationMatrixDisplay } from "./CorrelationMatrixDisplay";
import { RegressionDisplay } from "./RegressionDisplay";
import { RegressionResultsDisplay } from "./RegressionResultsDisplay";

export interface TraitResultData {
  trait_name?: string;
  n?: number;
  grand_mean?: number;
  variance_components?: {
    genetic_variance?: number;
    environmental_variance?: number;
    gxl_variance?: number;
    phenotypic_variance?: number;
    gcv?: number;
    pcv?: number;
    ecv?: number;
  };
  heritability?: {
    h2?: number;
    interpretation?: string;
  };
  genetic_advance?: {
    ga?: number;
    ga_percent?: number;
  };
  anova_table?: unknown;
  plots?: Record<string, string>;
  interpretation?: string;
  intelligence?: {
    executive_insight?: string;
    reviewer_radar?: string;
    decision_rules?: string;
    formulas_used?: string;
  };
  tables?: Record<string, unknown>;
  meta?: Record<string, unknown>;
}

interface Props {
  data: TraitResultData;
  traitName: string;
  onExportCSV: () => void;
  onExportPNG: (plotName: string, base64: string) => void;
}

function classifyH2(value: number): { label: string; color: string } {
  if (value >= 0.6) return { label: "High", color: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200" };
  if (value >= 0.3) return { label: "Moderate", color: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200" };
  return { label: "Low", color: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200" };
}

function classifyGA(value: number): { label: string; color: string } {
  if (value >= 20) return { label: "High", color: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200" };
  if (value >= 10) return { label: "Moderate", color: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200" };
  return { label: "Low", color: "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200" };
}

function renderAnovaTable(data: unknown) {
  if (Array.isArray(data) && data.length > 0) {
    if (Array.isArray(data[0])) {
      const headers = data[0] as unknown[];
      const rows = data.slice(1) as unknown[][];
      return (
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-border">
                {headers.map((h, i) => (
                  <th key={i} className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50 whitespace-nowrap">{String(h)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, ri) => (
                <tr key={ri} className={`border-b border-border/50 ${ri % 2 === 0 ? "" : "bg-muted/20"}`}>
                  {(row as unknown[]).map((cell, ci) => (
                    <td key={ci} className="px-3 py-2 text-muted-foreground whitespace-nowrap">{String(cell ?? "")}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    if (typeof data[0] === "object" && data[0] !== null) {
      const headers = Object.keys(data[0] as Record<string, unknown>);
      return (
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-border">
                {headers.map((h) => (
                  <th key={h} className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50 whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(data as Record<string, unknown>[]).map((row, ri) => (
                <tr key={ri} className={`border-b border-border/50 ${ri % 2 === 0 ? "" : "bg-muted/20"}`}>
                  {headers.map((h) => (
                    <td key={h} className="px-3 py-2 text-muted-foreground whitespace-nowrap">{row[h] != null ? String(row[h]) : ""}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
  }
  return <pre className="text-xs text-muted-foreground overflow-x-auto">{JSON.stringify(data, null, 2)}</pre>;
}

const INTELLIGENCE_SECTIONS = [
  { key: "executive_insight", label: "Executive Insight", icon: Lightbulb, color: "text-primary" },
  { key: "reviewer_radar", label: "Reviewer Radar", icon: AlertTriangle, color: "text-amber-600 dark:text-amber-400" },
  { key: "decision_rules", label: "Decision Rules", icon: BookOpen, color: "text-emerald-600 dark:text-emerald-400" },
  { key: "formulas_used", label: "Formulas Used", icon: FileText, color: "text-blue-600 dark:text-blue-400" },
] as const;

export function GeneticsTraitResult({ data, traitName, onExportCSV, onExportPNG }: Props) {
  const [showStatDetails, setShowStatDetails] = useState(false);

  const vc = data.variance_components;
  const h2 = data.heritability;
  const ga = data.genetic_advance;
  const meta = data.meta;
  const tables = data.tables;
  const plots = data.plots;
  const intelligence = data.intelligence;
  const interpretation = data.interpretation;

  const h2Value = h2?.h2 ?? (meta?.heritability as number | undefined);
  const h2Class = h2Value != null ? classifyH2(h2Value) : null;

  const gaPercent = ga?.ga_percent ?? (meta?.ga_percent as number | undefined);
  const gaClass = gaPercent != null ? classifyGA(gaPercent) : null;

  return (
    <div className="space-y-6">
      {/* Trait Header */}
      <div className="flex flex-wrap items-center gap-4">
        <h3 className="font-serif text-2xl font-bold text-foreground">{traitName}</h3>
        {data.n != null && <Badge variant="outline">n = {data.n}</Badge>}
        {(data.grand_mean ?? meta?.grand_mean) != null && (
          <Badge variant="secondary">Grand Mean = {Number(data.grand_mean ?? meta?.grand_mean).toFixed(2)}</Badge>
        )}
      </div>

      {/* ─── SECTION 1: Summary (h², GCV, PCV, GA) ─── */}
      {(h2Value != null || gaPercent != null || (meta && Object.keys(meta).length > 0 && !vc)) && (
        <Card className="border-primary/30">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Target className="h-5 w-5 text-primary" />
              Summary — Key Genetic Parameters
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Meta Summary (fallback when no structured vc) */}
            {meta && Object.keys(meta).length > 0 && !vc && (
              <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-3">
                {Object.entries(meta).map(([key, value]) => (
                  <div key={key} className="flex flex-col">
                    <dt className="text-sm text-muted-foreground font-medium capitalize">{key.replace(/_/g, " ")}</dt>
                    <dd className="text-foreground font-semibold">{String(value)}</dd>
                  </div>
                ))}
              </dl>
            )}

            {/* Heritability & GA cards */}
            {(h2Value != null || gaPercent != null) && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {h2Value != null && (
                  <div className="rounded-lg border border-border p-6 text-center space-y-3">
                    <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Broad-sense Heritability</p>
                    <p className="text-4xl font-bold text-foreground">{(h2Value * 100).toFixed(1)}%</p>
                    {h2Class && <Badge className={h2Class.color}>{h2Class.label}</Badge>}
                    {h2?.interpretation && <p className="text-sm text-muted-foreground">{h2.interpretation}</p>}
                    <p className="text-xs font-mono text-muted-foreground mt-2">H² = σ²g / σ²p</p>
                  </div>
                )}
                {ga && (
                  <div className="rounded-lg border border-border p-6 text-center space-y-3">
                    <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">Genetic Advance</p>
                    {ga.ga != null && <p className="text-2xl font-bold text-foreground">GA = {Number(ga.ga).toFixed(2)}</p>}
                    {gaPercent != null && (
                      <>
                        <p className="text-4xl font-bold text-foreground">GA% = {gaPercent.toFixed(1)}%</p>
                        {gaClass && <Badge className={gaClass.color}>{gaClass.label}</Badge>}
                      </>
                    )}
                    <p className="text-xs font-mono text-muted-foreground mt-2">GA = k × √σ²p × H²</p>
                    <p className="text-xs font-mono text-muted-foreground">GA% = (GA / X̄) × 100</p>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* ─── SECTION 2: Statistical Analysis (ANOVA + Mean Separation) ─── */}
      {(data.anova_table || tables) && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2 text-lg">
                <FlaskConical className="h-5 w-5 text-primary" />
                Statistical Analysis
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="h-4 w-4 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent className="max-w-xs text-sm">
                      ANOVA analysis determines genetic and environmental variance. Mean separation (Tukey HSD) identifies significantly different genotypes.
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </CardTitle>
              <div className="flex items-center gap-2">
                <Checkbox
                  id={`stat-details-${traitName}`}
                  checked={showStatDetails}
                  onCheckedChange={(v) => setShowStatDetails(!!v)}
                />
                <label htmlFor={`stat-details-${traitName}`} className="text-sm text-muted-foreground cursor-pointer select-none">
                  Show Statistical Details
                </label>
              </div>
            </div>
            <p className="text-sm text-muted-foreground mt-1">
              Variance partitioning via ANOVA underpins all genetic parameter estimates shown above.
            </p>
          </CardHeader>

          {showStatDetails && (
            <CardContent className="space-y-6">
              {/* ANOVA Table */}
              {data.anova_table && (
                <div>
                  <h4 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                    <BarChart3 className="h-4 w-4 text-primary" />
                    ANOVA Table
                  </h4>
                  {renderAnovaTable(data.anova_table)}
                </div>
              )}

              {/* Remaining tables (mean separation, etc.) */}
              {tables && Object.entries(tables)
                .filter(([name]) => !["correlations", "path_analysis", "selection_index", "combined_anova", "variance_components", "genotype_means", "assumptions", "assumption_guidance"].includes(name) && !/regress/i.test(name))
                .map(([name, tData]) => {
                  if (!tData) return null;
                  return (
                    <div key={name}>
                      <h4 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                        <BarChart3 className="h-4 w-4 text-primary" />
                        {name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                      </h4>
                      {renderAnovaTable(tData)}
                    </div>
                  );
                })}
            </CardContent>
          )}
        </Card>
      )}

      {/* ─── SECTION 3: Variance Components Breakdown ─── */}
      {vc && (
        <Collapsible defaultOpen>
          <Card>
            <CardHeader className="pb-3">
              <CollapsibleTrigger className="w-full flex items-center justify-between">
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Beaker className="h-5 w-5 text-primary" />
                  Variance Components Breakdown
                </CardTitle>
                <ChevronDown className="h-5 w-5 text-muted-foreground transition-transform" />
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50">Component</th>
                        <th className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50">Symbol</th>
                        <th className="text-right px-3 py-2 font-semibold text-foreground bg-muted/50">Value</th>
                        <th className="text-right px-3 py-2 font-semibold text-foreground bg-muted/50">% of σ²p</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { label: "Genetic Variance", sym: "σ²g", val: vc.genetic_variance },
                        { label: "Environmental Variance", sym: "σ²e", val: vc.environmental_variance },
                        { label: "G×E Interaction", sym: "σ²ge", val: vc.gxl_variance },
                        { label: "Phenotypic Variance", sym: "σ²p", val: vc.phenotypic_variance },
                        { label: "GCV (%)", sym: "GCV", val: vc.gcv },
                        { label: "PCV (%)", sym: "PCV", val: vc.pcv },
                        { label: "ECV (%)", sym: "ECV", val: vc.ecv },
                      ]
                        .filter((r) => r.val != null)
                        .map((r, i) => {
                          const pct = vc.phenotypic_variance && vc.phenotypic_variance > 0 && ["σ²g", "σ²e", "σ²ge"].includes(r.sym)
                            ? ((r.val! / vc.phenotypic_variance) * 100).toFixed(1)
                            : "—";
                          return (
                            <tr key={r.sym} className={`border-b border-border/50 ${i % 2 === 0 ? "" : "bg-muted/20"}`}>
                              <td className="px-3 py-2 text-foreground font-medium">{r.label}</td>
                              <td className="px-3 py-2 font-mono text-muted-foreground">{r.sym}</td>
                              <td className="px-3 py-2 text-right font-mono text-foreground">{Number(r.val).toFixed(4)}</td>
                              <td className="px-3 py-2 text-right text-muted-foreground">{pct}</td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* ─── SECTION 4: Trait Relationships (Correlations, Path, Regression) ─── */}
      {tables?.correlations && (
        <CorrelationMatrixDisplay data={tables.correlations as Record<string, unknown>} />
      )}

      {tables?.path_analysis && (() => {
        const pa = tables.path_analysis as Record<string, unknown>;
        if (pa.path_matrix || pa.effect_decomposition || pa.path_diagram) {
          return <PathAnalysisResultsDisplay data={pa as any} traitName={traitName} />;
        }
        return <PathAnalysisDisplay data={pa} traitName={traitName} />;
      })()}

      {(() => {
        if (!tables) return null;
        const regressionKey = Object.keys(tables).find((k) => /regress/i.test(k));
        const regressionData = regressionKey ? tables[regressionKey] : null;
        if (!regressionData) return null;
        const rd = regressionData as Record<string, unknown>;
        if (rd.model_fit || (Array.isArray(rd.coefficients) && rd.coefficients.length > 0 && (rd.coefficients[0] as any)?.term)) {
          return <RegressionResultsDisplay data={{ tables: { regression: rd as any }, interpretation: data.interpretation, intelligence: data.intelligence as any }} />;
        }
        return <RegressionDisplay data={rd} />;
      })()}

      {tables?.selection_index && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <BarChart3 className="w-5 h-5 text-primary" />
              Selection Index
            </CardTitle>
          </CardHeader>
          <CardContent>{renderAnovaTable(tables.selection_index)}</CardContent>
        </Card>
      )}

      {/* Plots */}
      {plots && Object.entries(plots).map(([plotName, base64]) => (
        <Card key={plotName}>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-3">
                <ImageIcon className="w-5 h-5 text-primary" />
                {plotName.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
              </CardTitle>
              <Button variant="outline" size="sm" onClick={() => onExportPNG(plotName, base64)}>
                <Download className="w-4 h-4 mr-1" /> PNG
              </Button>
            </div>
          </CardHeader>
          <CardContent className="flex justify-center">
            <img src={`data:image/png;base64,${base64}`} alt={plotName} className="max-w-full rounded-lg shadow-md" />
          </CardContent>
        </Card>
      ))}

      {/* ─── SECTION 5: Interpretation & Recommendations ─── */}
      {intelligence && INTELLIGENCE_SECTIONS.map(({ key, label, icon: Icon, color }) => {
        const rawContent = intelligence[key as keyof typeof intelligence];
        if (!rawContent) return null;
        const content: string = Array.isArray(rawContent)
          ? rawContent.map((item) => typeof item === "string" ? `- ${item}` : `- ${JSON.stringify(item)}`).join("\n")
          : typeof rawContent === "object"
          ? JSON.stringify(rawContent, null, 2)
          : String(rawContent);
        return (
          <Collapsible key={key} defaultOpen>
            <Card>
              <CardHeader className="pb-2">
                <CollapsibleTrigger className="w-full flex items-center justify-between">
                  <CardTitle className="flex items-center gap-3 text-lg">
                    <Icon className={`w-5 h-5 ${color}`} />
                    {label}
                  </CardTitle>
                  <ChevronDown className="w-5 h-5 text-muted-foreground" />
                </CollapsibleTrigger>
              </CardHeader>
              <CollapsibleContent>
                <CardContent>
                  <div className="prose prose-sm max-w-none dark:prose-invert">
                    <ReactMarkdown>{content}</ReactMarkdown>
                  </div>
                </CardContent>
              </CollapsibleContent>
            </Card>
          </Collapsible>
        );
      })}

      {interpretation && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <Lightbulb className="w-5 h-5 text-primary" />
              Interpretation &amp; Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="prose prose-sm max-w-none dark:prose-invert">
              <ReactMarkdown>{interpretation}</ReactMarkdown>
            </div>
            <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-700 p-4">
              <p className="text-sm text-amber-800 dark:text-amber-200">
                ⚠️ <strong>Academic Integrity Reminder:</strong> Verify all numbers against the tables above. Adapt text to your own words.
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Formulas */}
      <GeneticsFormulaPanel />

      {/* Export Buttons */}
      <div className="flex flex-wrap gap-3">
        <Button variant="outline" size="sm" onClick={onExportCSV}>
          <Download className="w-4 h-4 mr-2" /> Export CSV
        </Button>
      </div>
    </div>
  );
}
