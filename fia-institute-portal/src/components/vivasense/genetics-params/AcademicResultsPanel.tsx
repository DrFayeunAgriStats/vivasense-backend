/**
 * AcademicResultsPanel – Insight-first results display
 * Follows: INSIGHT → EVIDENCE → DETAILS hierarchy
 */
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  ChevronDown, Shield, Sprout, Target, BookOpen, Info, BarChart3, FlaskConical,
} from "lucide-react";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { extractRows, fmtNum, formatP } from "./GeneticsResultsDashboard";

/* ── Types ─────────────────────────────────────────── */

interface ClassificationCard {
  label: string;
  value: string;
  tooltip: string;
}

interface StatisticalNote {
  text: string;
}

interface AcademicResultsProps {
  /** Module label, e.g. "Genetic Parameters" */
  moduleLabel: string;
  /** When true, suppress breeding/genotype-specific labels (use treatment-level wording) */
  domainNeutral?: boolean;
  /** Selection reliability: High / Moderate / Low */
  selectionReliability?: string;
  /** One-line insight summary */
  insightSummary?: string;
  /** Basis sub-text */
  insightBasis?: string;
  /** Full interpretation paragraph */
  interpretation?: string;
  /** Breeding / action recommendation */
  recommendation?: string;
  /** Classification summary cards */
  classifications?: ClassificationCard[];
  /** Statistical footnotes */
  statisticalNotes?: StatisticalNote[];
  /** Raw ANOVA table data */
  anovaTable?: unknown;
  /** Raw mean separation data */
  meanSeparation?: unknown;
  /** Additional collapsible table sections */
  extraTables?: { title: string; data: unknown }[];
  /** Descriptive stats grid */
  descriptiveStats?: { label: string; value: string }[];
}

/* ── Helpers ───────────────────────────────────────── */

function reliabilityColor(level?: string) {
  const l = (level ?? "").toLowerCase();
  if (l === "high") return "bg-emerald-100 text-emerald-800 border-emerald-300 dark:bg-emerald-900/30 dark:text-emerald-300 dark:border-emerald-700";
  if (l === "moderate") return "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-900/30 dark:text-amber-300 dark:border-amber-700";
  return "bg-red-100 text-red-800 border-red-300 dark:bg-red-900/30 dark:text-red-300 dark:border-red-700";
}

/**
 * Strip genetics-specific terminology from free-text interpretation/recommendation
 * when the panel is rendered in domain-neutral mode (e.g., generic ANOVA).
 * Removes whole sentences mentioning heritability, GCV, PCV, GAM, genotype, breeding, selection.
 */
function sanitizeDomainNeutralText(text?: string): string | undefined {
  if (!text) return text;
  const GENETICS_RE = /\b(heritabilit(y|ies)|broad[-\s]?sense|narrow[-\s]?sense|h2|h²|gcv|pcv|gam|genetic advance|genotyp(e|ic|es)|breed(ing|er)|selection (intensity|reliability|gain|differential)|allel(e|ic)|cultivar|germplasm)\b/i;
  // Split into sentences, drop any containing genetics terms
  const sentences = text.split(/(?<=[.!?])\s+/);
  const kept = sentences.filter((s) => !GENETICS_RE.test(s));
  const cleaned = kept.join(" ").replace(/\s+/g, " ").trim();
  return cleaned.length > 0 ? cleaned : undefined;
}

function classificationColor(value: string) {
  const v = value.toLowerCase();
  if (v === "high" || v === "small") return "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300";
  if (v === "moderate") return "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300";
  return "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300";
}

/* ── Component ─────────────────────────────────────── */

export function AcademicResultsPanel({
  moduleLabel,
  domainNeutral,
  selectionReliability,
  insightSummary,
  insightBasis,
  interpretation,
  recommendation,
  classifications,
  statisticalNotes,
  anovaTable,
  meanSeparation,
  extraTables,
  descriptiveStats,
}: AcademicResultsProps) {
  const anovaRows = extractRows(anovaTable);
  const msRows = extractRows(meanSeparation);
  const hasDetailedStats = anovaRows.length > 0 || msRows.length > 0 || (extraTables && extraTables.length > 0) || (descriptiveStats && descriptiveStats.length > 0);

  // In domain-neutral mode, strip genetics-specific sentences from free text and hide reliability/classifications/recommendation.
  const safeInterpretation = domainNeutral ? sanitizeDomainNeutralText(interpretation) : interpretation;
  const safeRecommendation = domainNeutral ? sanitizeDomainNeutralText(recommendation) : recommendation;
  const safeReliability = domainNeutral ? undefined : selectionReliability;
  const safeClassifications = domainNeutral ? undefined : classifications;

  return (
    <div className="space-y-5">
      {/* ─── SECTION 1: KEY INSIGHT ─── */}
      {(safeReliability || insightSummary) && (
        <Card className="border-2 border-primary/20 shadow-md">
          <CardContent className="py-6 px-6 space-y-3">
            <div className="flex items-center gap-3 flex-wrap">
              <Shield className="h-6 w-6 text-primary shrink-0" />
              <h3 className="text-lg font-semibold text-foreground">Key Insight</h3>
              {safeReliability && (
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-semibold border ${reliabilityColor(safeReliability)}`}>
                  Selection Reliability: {safeReliability}
                </span>
              )}
            </div>
            {insightSummary && (
              <p className="text-base text-foreground leading-relaxed">{insightSummary}</p>
            )}
            {insightBasis && (
              <p className="text-xs text-muted-foreground">{insightBasis}</p>
            )}
          </CardContent>
        </Card>
      )}

      {/* ─── SECTION 2: INTERPRETATION ─── */}
      {safeInterpretation && (
        <Card className="bg-emerald-50/60 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-800">
          <CardContent className="py-5 px-6">
            <div className="flex items-start gap-3">
              <BookOpen className="h-5 w-5 text-emerald-700 dark:text-emerald-400 mt-0.5 shrink-0" />
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-emerald-800 dark:text-emerald-300 uppercase tracking-wide">
                  {domainNeutral ? "ANOVA Interpretation" : "Interpretation"}
                </h4>
                <p className="text-sm text-emerald-900 dark:text-emerald-200 leading-relaxed whitespace-pre-line">
                  {safeInterpretation}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ─── SECTION 3: RECOMMENDATION ─── */}
      {safeRecommendation && (
        <Card className="bg-blue-50/60 dark:bg-blue-950/20 border-blue-200 dark:border-blue-800">
          <CardContent className="py-5 px-6">
            <div className="flex items-start gap-3">
              <Sprout className="h-5 w-5 text-blue-700 dark:text-blue-400 mt-0.5 shrink-0" />
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-blue-800 dark:text-blue-300 uppercase tracking-wide">
                  {domainNeutral ? "Practical Implication" : "Decision Guidance"}
                </h4>
                <p className="text-sm text-blue-900 dark:text-blue-200 leading-relaxed whitespace-pre-line">
                  {safeRecommendation}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* ─── SECTION 4: CLASSIFICATION SUMMARY ─── */}
      {safeClassifications && safeClassifications.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <TooltipProvider>
            {safeClassifications.map((c) => (
              <Tooltip key={c.label}>
                <TooltipTrigger asChild>
                  <Card className="cursor-help hover:shadow-sm transition-shadow">
                    <CardContent className="py-4 px-5 flex items-center justify-between">
                      <span className="text-sm font-medium text-foreground">{c.label}</span>
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${classificationColor(c.value)}`}>
                        {c.value}
                      </span>
                    </CardContent>
                  </Card>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-sm">{c.tooltip}</TooltipContent>
              </Tooltip>
            ))}
          </TooltipProvider>
        </div>
      )}

      {/* ─── SECTION 5: STATISTICAL NOTES ─── */}
      {statisticalNotes && statisticalNotes.length > 0 && (
        <div className="space-y-1.5">
          {statisticalNotes.map((note, i) => (
            <div key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
              <Info className="h-3.5 w-3.5 mt-0.5 shrink-0" />
              <span>{note.text}</span>
            </div>
          ))}
        </div>
      )}

      {/* ─── SECTION 7: FULL TABLES (Collapsible) ─── */}
      {hasDetailedStats && (
        <Collapsible>
          <Card>
            <CardHeader className="pb-2">
              <CollapsibleTrigger className="w-full flex items-center justify-between group">
                <CardTitle className="flex items-center gap-2 text-base">
                  <FlaskConical className="h-5 w-5 text-muted-foreground" />
                  Show Detailed Statistics
                </CardTitle>
                <ChevronDown className="h-5 w-5 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="space-y-6 pt-2">
                {/* Descriptive stats */}
                {descriptiveStats && descriptiveStats.length > 0 && (
                  <div>
                    <h4 className="text-sm font-semibold mb-2">Descriptive Statistics</h4>
                    <div className="grid gap-2 sm:grid-cols-3 text-sm">
                      {descriptiveStats.map((d) => (
                        <div key={d.label} className="flex justify-between border-b border-border/50 pb-1">
                          <span className="text-muted-foreground">{d.label}</span>
                          <span className="font-mono">{d.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* ANOVA Table */}
                {anovaRows.length > 0 && (
                  <div>
                    <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
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
                                <TableCell className={`text-right font-mono ${isSig ? "text-emerald-700 dark:text-emerald-400 font-semibold" : ""}`}>
                                  {pVal != null ? formatP(pVal) : "—"}
                                </TableCell>
                              </TableRow>
                            );
                          })}
                        </TableBody>
                      </Table>
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 italic">* p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001</p>
                  </div>
                )}

                {/* Mean Separation */}
                {msRows.length > 0 && (
                  <div>
                    <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                      <Target className="h-4 w-4 text-primary" />
                      Mean Separation (Tukey HSD)
                    </h4>
                    <div className="overflow-x-auto">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>{domainNeutral ? "Treatment" : "Genotype"}</TableHead>
                            <TableHead className="text-right">Mean ± SE</TableHead>
                            <TableHead className="text-center">Group</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {msRows.map((row, i) => {
                            const grp = String(row.group ?? row.Group ?? row.letter ?? row.tukey_group ?? "—").trim().charAt(0).toLowerCase();
                            const colors: Record<string, string> = {
                              a: "bg-emerald-50 dark:bg-emerald-900/20",
                              b: "bg-orange-50 dark:bg-orange-900/20",
                              c: "bg-red-50 dark:bg-red-900/20",
                            };
                            return (
                              <TableRow key={i} className={colors[grp] ?? ""}>
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

                {/* Extra tables */}
                {extraTables?.map((et, idx) => {
                  const rows = extractRows(et.data);
                  if (rows.length === 0) return null;
                  const headers = Object.keys(rows[0]);
                  return (
                    <div key={idx}>
                      <h4 className="text-sm font-semibold mb-2">{et.title}</h4>
                      <div className="overflow-x-auto">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              {headers.map((h) => (
                                <TableHead key={h}>{h}</TableHead>
                              ))}
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {rows.map((row, ri) => (
                              <TableRow key={ri}>
                                {headers.map((h) => (
                                  <TableCell key={h} className="font-mono text-sm">
                                    {row[h] != null ? String(row[h]) : "—"}
                                  </TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}
    </div>
  );
}
