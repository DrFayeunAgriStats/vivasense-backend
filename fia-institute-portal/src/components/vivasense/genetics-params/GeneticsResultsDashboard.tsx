import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Beaker,
  BarChart3,
  AlertTriangle,
  BookOpen,
  Lightbulb,
  Download,
  Loader2,
  Info,
  ChevronDown,
  FlaskConical,
  Target,
} from "lucide-react";
import { toast } from "sonner";
import type { GeneticsAnalysisResult } from "@/types/genetics";

interface Props {
  result: GeneticsAnalysisResult;
}

/* ── Helpers ──────────────────────────────────────────── */

export function fmtNum(v: unknown): string {
  if (v == null || v === "") return "—";
  const n = Number(v);
  return isNaN(n) ? String(v) : n.toFixed(4);
}

export function formatP(v: unknown): string {
  if (v == null || v === "") return "—";
  const n = Number(v);
  if (isNaN(n)) return String(v);
  if (n < 0.001) return "<0.001 ***";
  if (n < 0.01) return n.toFixed(4) + " **";
  if (n < 0.05) return n.toFixed(4) + " *";
  return n.toFixed(4) + " ns";
}

export function extractRows(data: unknown): Record<string, unknown>[] {
  const isInterceptRow = (row: Record<string, unknown>): boolean => {
    const label = String(row.source ?? row.Source ?? row.term ?? "").trim().toLowerCase();
    return label === "(intercept)" || label === "intercept";
  };

  if (!data) return [];
  // Already array of row objects
  if (Array.isArray(data)) {
    if (data.length > 0 && typeof data[0] === "object" && data[0] !== null) {
      return (data as Record<string, unknown>[]).filter((row) => !isInterceptRow(row));
    }
    return [];
  }
  if (typeof data === "object" && data !== null) {
    const obj = data as Record<string, unknown>;
    const keys = Object.keys(obj);
    if (keys.length === 0) return [];

    // Parallel-array format: { source: [...], df: [...], ss: [...], ... }
    const first = obj[keys[0]];
    if (Array.isArray(first)) {
      const len = (first as unknown[]).length;
      const rows: Record<string, unknown>[] = [];
      for (let i = 0; i < len; i++) {
        const row: Record<string, unknown> = {};
        keys.forEach(k => {
          const arr = obj[k];
          row[k] = Array.isArray(arr) ? (arr as unknown[])[i] : undefined;
        });
        rows.push(row);
      }
      return rows.filter((row) => !isInterceptRow(row));
    }

    // Dict-of-dicts (pandas-style)
    if (typeof first === "object" && first !== null && !Array.isArray(first)) {
      const cols = keys;
      const rowKeys = Object.keys(first as Record<string, unknown>);
      return rowKeys.map(rk => {
        const row: Record<string, unknown> = { source: rk };
        cols.forEach(col => {
          row[col.toLowerCase()] = ((obj[col] as Record<string, unknown>) ?? {})[rk];
        });
        return row;
      }).filter((row) => !isInterceptRow(row));
    }
  }
  return [];
}

export function GeneticsResultsDashboard({ result }: Props) {
  const [isDownloading, setIsDownloading] = useState(false);
  const [showStatDetails, setShowStatDetails] = useState(false);

  // Extract ANOVA and mean separation rows from various response shapes
  const anovaRows = extractRows(result.anova_table);
  const meanSepRows = extractRows(result.mean_separation);

  const handleDownloadWord = async () => {
    setIsDownloading(true);
    try {
      const traitName = result.computation_mode || "trait";
      const payload = {
        trait_name: traitName,
        analysis_results: result,
        dataset_name: result.dataset_name || `${traitName} Analysis`,
        variance_components: result.variance_components,
        genetic_parameters: result.genetic_parameters,
        interpretation: result.interpretation_paragraph,
        classification_summary: result.classification_summary,
        breeding_implication: result.breeding_implication,
        caution_note: result.caution_note,
        computation_mode: result.computation_mode,
        estimation_basis: result.estimation_basis,
        anova_table: result.anova_table,
        mean_separation: result.mean_separation,
      };

      console.log("[COMPONENT] GeneticsResultsDashboard -> download-results", payload);

      const res = await fetch(
        "https://vivasense-genetics-docker.onrender.com/genetics/download-results",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );

      if (!res.ok) {
        const text = await res.text();
        console.error("[DOWNLOAD ERROR RAW]", text);
        let msg = text;
        try {
          const parsed = JSON.parse(text);
          msg = parsed.detail || parsed.error || parsed.message || text;
        } catch {}
        throw new Error(msg);
      }

      const blob = await res.blob();
      const date = new Date().toISOString().slice(0, 10);
      const safeTrait = traitName.replace(/\s+/g, "_").replace(/[^a-zA-Z0-9_]/g, "");
      const filename = `VivaSense_Genetics_${safeTrait}_${date}.docx`;

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);

      toast.success("Downloaded successfully");
    } catch (err: any) {
      console.error("[DOWNLOAD ERROR]", err);
      toast.error(err.message || "Download failed. Try again.");
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header badges */}
      <div className="flex flex-wrap gap-2">
        <Badge variant="outline" className="text-xs">
          Mode: {result.computation_mode}
        </Badge>
        <Badge variant="outline" className="text-xs">
          Basis: {result.estimation_basis}
        </Badge>
      </div>

      {/* ─── SECTION 1: Summary (h², GCV, PCV, GA) ─── */}
      {result.genetic_parameters && result.genetic_parameters.length > 0 && (
        <Card className="border border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Target className="h-5 w-5 text-primary" />
              Summary — Key Genetic Parameters
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Parameter</TableHead>
                  <TableHead>Symbol</TableHead>
                  <TableHead className="text-right">Value</TableHead>
                  <TableHead>Unit</TableHead>
                  <TableHead>Classification</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {result.genetic_parameters.map((gp, i) => (
                  <TableRow key={i}>
                    <TableCell className="font-medium">{gp.label}</TableCell>
                    <TableCell className="font-mono text-sm">{gp.symbol}</TableCell>
                    <TableCell className="text-right font-mono">
                      {typeof gp.value === "number" ? gp.value.toFixed(4) : gp.value}
                    </TableCell>
                    <TableCell className="text-xs">{gp.unit ?? "—"}</TableCell>
                    <TableCell>
                      {gp.classification ? (
                        <Badge variant="secondary" className="text-xs">{gp.classification}</Badge>
                      ) : "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* ─── SECTION 2: Statistical Analysis (ANOVA + Mean Separation) ─── */}
      <Card className="border border-primary/20">
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
                id="stat-details"
                checked={showStatDetails}
                onCheckedChange={(v) => setShowStatDetails(!!v)}
              />
              <label htmlFor="stat-details" className="text-sm text-muted-foreground cursor-pointer select-none">
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
            {anovaRows.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                  <BarChart3 className="h-4 w-4 text-primary" />
                  ANOVA Table
                </h4>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Source</TableHead>
                        <TableHead className="text-right">DF</TableHead>
                        <TableHead className="text-right">SS</TableHead>
                        <TableHead className="text-right">MS</TableHead>
                        <TableHead className="text-right">F</TableHead>
                        <TableHead className="text-right">p-value</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {anovaRows.map((row, i) => {
                        const pVal = row.p_value ?? row.pvalue ?? row["Pr(>F)"] ?? row["PR(>F)"];
                        const pNum = pVal != null ? Number(pVal) : NaN;
                        const isSig = !isNaN(pNum) && pNum < 0.05;
                        return (
                          <TableRow key={i}>
                            <TableCell className="font-medium">{String(row.source ?? row.Source ?? row.term ?? "")}</TableCell>
                            <TableCell className="text-right font-mono">{String(row.df ?? row.DF ?? "—")}</TableCell>
                            <TableCell className="text-right font-mono">{fmtNum(row.ss ?? row.SS ?? row.sum_sq)}</TableCell>
                            <TableCell className="text-right font-mono">{fmtNum(row.ms ?? row.MS ?? row.mean_sq)}</TableCell>
                            <TableCell className="text-right font-mono">{fmtNum(row.f_value ?? row.F ?? row.f)}</TableCell>
                            <TableCell className={`text-right font-mono ${isSig ? "text-green-600 font-semibold" : ""}`}>
                              {pVal != null ? formatP(pVal) : "—"}
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
                <p className="text-xs text-muted-foreground mt-1 italic">
                  Significance: * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001, ns = not significant
                </p>
              </div>
            )}

            {/* Mean Separation */}
            {meanSepRows.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                  <Target className="h-4 w-4 text-primary" />
                  Mean Separation (Tukey HSD)
                </h4>
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Genotype</TableHead>
                        <TableHead className="text-right">Mean ± SE</TableHead>
                        <TableHead className="text-center">Group</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {meanSepRows.map((row, i) => {
                        const grp = String(row.group ?? row.Group ?? row.letter ?? row.tukey_group ?? "—").trim().charAt(0).toLowerCase();
                        const groupColors: Record<string, string> = {
                          a: "bg-emerald-50 dark:bg-emerald-900/20",
                          b: "bg-orange-50 dark:bg-orange-900/20",
                          c: "bg-red-50 dark:bg-red-900/20",
                          d: "bg-gray-100 dark:bg-gray-800/30",
                          e: "bg-gray-200 dark:bg-gray-700/30",
                        };
                        return (
                          <TableRow key={i} className={groupColors[grp] ?? ""}>
                            <TableCell className="font-medium">{String(row.genotype ?? row.Genotype ?? row.treatment ?? row.level ?? "")}</TableCell>
                            <TableCell className="text-right font-mono">
                              {fmtNum(row.mean ?? row.Mean)} ± {fmtNum(row.se ?? row.SE ?? row.std_err)}
                            </TableCell>
                            <TableCell className="text-center font-bold text-lg">
                              {String(row.group ?? row.Group ?? row.letter ?? row.tukey_group ?? "—")}
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
                <p className="text-xs text-muted-foreground mt-1 italic">
                  Means with the same letter are not significantly different (Tukey HSD, α = 0.05)
                </p>
              </div>
            )}

            {/* Classification Summary */}
            {result.classification_summary && result.classification_summary.length > 0 && (
              <div>
                <h4 className="text-sm font-semibold text-foreground mb-2 flex items-center gap-2">
                  <BookOpen className="h-4 w-4 text-primary" />
                  Classification Summary
                </h4>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Parameter</TableHead>
                      <TableHead className="text-right">Value</TableHead>
                      <TableHead>Category</TableHead>
                      <TableHead>Reference</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.classification_summary.map((cs, i) => (
                      <TableRow key={i}>
                        <TableCell className="font-medium">{cs.parameter}</TableCell>
                        <TableCell className="text-right font-mono">
                          {typeof cs.value === "number" ? cs.value.toFixed(4) : cs.value}
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline" className="text-xs">{cs.category}</Badge>
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">{cs.reference ?? "—"}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        )}
      </Card>

      {/* ─── SECTION 3: Genetic Parameters (Variance Components Breakdown) ─── */}
      {result.variance_components && result.variance_components.length > 0 && (
        <Collapsible defaultOpen>
          <Card className="border border-primary/20">
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
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Component</TableHead>
                      <TableHead>Symbol</TableHead>
                      <TableHead className="text-right">Value</TableHead>
                      <TableHead>Formula</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {result.variance_components.map((vc, i) => (
                      <TableRow key={i}>
                        <TableCell className="font-medium">{vc.label}</TableCell>
                        <TableCell className="font-mono text-sm">{vc.symbol}</TableCell>
                        <TableCell className="text-right font-mono">
                          {typeof vc.value === "number" ? vc.value.toFixed(4) : vc.value}
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">{vc.formula ?? "—"}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}

      {/* ─── SECTION 4: Interpretation & Recommendations ─── */}
      {result.interpretation_paragraph && (
        <Card className="border border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Lightbulb className="h-5 w-5 text-primary" />
              Interpretation &amp; Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent className="prose prose-sm max-w-none text-foreground space-y-4">
            <p className="leading-relaxed">{result.interpretation_paragraph}</p>
          </CardContent>
        </Card>
      )}

      {/* Breeding Implication */}
      {result.breeding_implication && (
        <Card className="border-l-4 border-l-primary bg-primary/5">
          <CardContent className="p-4">
            <p className="text-sm font-semibold text-primary mb-1">Breeding Implication</p>
            <p className="text-sm text-foreground leading-relaxed">{result.breeding_implication}</p>
          </CardContent>
        </Card>
      )}

      {/* Caution Note */}
      {result.caution_note && (
        <Card className="border-l-4 border-l-amber-500 bg-amber-50">
          <CardContent className="p-4 flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
            <div>
              <p className="text-sm font-semibold text-amber-800 mb-1">Academic Caution</p>
              <p className="text-sm text-amber-900 leading-relaxed">{result.caution_note}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Download Button */}
      <div className="flex justify-end">
        <Button
          onClick={handleDownloadWord}
          disabled={isDownloading}
          className="bg-emerald-600 hover:bg-emerald-700 text-white gap-2"
        >
          {isDownloading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Download className="h-4 w-4" />
          )}
          {isDownloading ? "Downloading..." : "📥 Download Results as Word"}
        </Button>
      </div>

      {/* Disclaimer */}
      <div className="rounded-md border border-border bg-muted/40 p-4 text-xs text-muted-foreground leading-relaxed">
        <strong>Disclaimer:</strong> VivaSense™ is an academic support platform developed by Field-to-Insight Academy
        © Dr. Fayeun Lawrence Stephen. It does not replace supervision, authorship, or institutional regulations.
        All outputs must be validated against your experimental records before inclusion in any thesis or publication.
      </div>
    </div>
  );
}
