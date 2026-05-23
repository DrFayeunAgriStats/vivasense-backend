import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Download, FileText, AlertCircle, Trophy } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  PublicationTablesData,
  publicationTablesToHtml,
} from "./utils/computeDescriptivePublicationTables";
import { HtmlTablesSection } from "./HtmlTablesSection";

interface Props {
  data: PublicationTablesData;
}

function fmt(v: number, d = 2): string {
  return typeof v === "number" && !isNaN(v)
    ? v.toFixed(d)
    : "N/A";
}

function performanceBadgeColor(cls: string): string {
  switch (cls) {
    case "Best": return "bg-green-100 text-green-800 border-green-300 dark:bg-green-900/40 dark:text-green-300 dark:border-green-700";
    case "Strong": return "bg-blue-100 text-blue-800 border-blue-300 dark:bg-blue-900/40 dark:text-blue-300 dark:border-blue-700";
    case "Intermediate": return "bg-yellow-100 text-yellow-800 border-yellow-300 dark:bg-yellow-900/40 dark:text-yellow-300 dark:border-yellow-700";
    case "Below average": return "bg-orange-100 text-orange-800 border-orange-300 dark:bg-orange-900/40 dark:text-orange-300 dark:border-orange-700";
    case "Lowest": return "bg-red-100 text-red-800 border-red-300 dark:bg-red-900/40 dark:text-red-300 dark:border-red-700";
    default: return "bg-muted text-muted-foreground border-border";
  }
}

function assessmentColor(assessment: string): string {
  if (assessment.includes("Superior")) return "text-green-700 dark:text-green-400 font-bold";
  if (assessment.includes("strong")) return "text-blue-700 dark:text-blue-400 font-semibold";
  if (assessment.includes("Intermediate")) return "text-yellow-700 dark:text-yellow-400";
  if (assessment.includes("Below")) return "text-orange-700 dark:text-orange-400";
  if (assessment.includes("Weak")) return "text-red-700 dark:text-red-400";
  return "text-muted-foreground";
}

export function DescriptivePublicationTables({ data }: Props) {
  const htmlTables = publicationTablesToHtml(data);
  let tableNum = 1;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="font-serif text-xl font-bold text-foreground flex items-center gap-2">
          <FileText className="w-5 h-5 text-primary" />
          Publication Tables
        </h3>
      </div>

      {/* Table 1: Overall Descriptive Statistics */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base flex items-center gap-2">
            <FileText className="w-4 h-4 text-primary" />
            Table {tableNum++}. Overall descriptive statistics of the evaluated traits
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse font-mono">
              <thead>
                <tr className="border-b-2 border-foreground/30">
                  <th className="px-3 py-2 text-left font-semibold text-foreground">Trait</th>
                  <th className="px-3 py-2 text-right font-semibold text-foreground">Mean</th>
                  <th className="px-3 py-2 text-right font-semibold text-foreground">Minimum</th>
                  <th className="px-3 py-2 text-right font-semibold text-foreground">Maximum</th>
                  <th className="px-3 py-2 text-right font-semibold text-foreground">Std. Dev.</th>
                  <th className="px-3 py-2 text-right font-semibold text-foreground">CV (%)</th>
                </tr>
              </thead>
              <tbody>
                {data.overall_descriptive_statistics.map((row, i) => (
                  <tr key={i} className="border-b border-border/50 hover:bg-muted/20">
                    <td className="px-3 py-2 text-left font-medium text-foreground">{row.trait}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(row.mean)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(row.minimum)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(row.maximum)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(row.standard_deviation)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(row.cv_percent, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Per-trait genotype performance tables */}
      {Object.entries(data.trait_performance_tables).map(([trait, rows]) => {
        if (rows.length === 0) return null;
        const num = tableNum++;
        return (
          <Card key={trait}>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <FileText className="w-4 h-4 text-primary" />
                Table {num}. Descriptive performance of genotypes for {trait}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse font-mono">
                  <thead>
                    <tr className="border-b-2 border-foreground/30">
                      <th className="px-3 py-2 text-left font-semibold text-foreground">Genotype</th>
                      <th className="px-3 py-2 text-right font-semibold text-foreground">Mean</th>
                      <th className="px-3 py-2 text-right font-semibold text-foreground">Std. Dev.</th>
                      <th className="px-3 py-2 text-right font-semibold text-foreground">CV (%)</th>
                      <th className="px-3 py-2 text-right font-semibold text-foreground">Rank</th>
                      <th className="px-3 py-2 text-left font-semibold text-foreground">Performance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r, i) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-muted/20">
                        <td className="px-3 py-2 text-left font-medium text-foreground">{r.genotype}</td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(r.mean)}</td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(r.standard_deviation)}</td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{fmt(r.cv_percent, 1)}</td>
                        <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">{r.rank}</td>
                        <td className="px-3 py-2 text-left">
                          <span className={`inline-block px-2 py-0.5 rounded-full text-xs border ${performanceBadgeColor(r.performance_class)}`}>
                            {r.performance_class}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        );
      })}

      {/* Multi-trait ranking table */}
      {data.multi_trait_ranking.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Trophy className="w-4 h-4 text-primary" />
              Table {tableNum}. Multi-trait ranking of evaluated genotypes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse font-mono">
                <thead>
                  <tr className="border-b-2 border-foreground/30">
                    <th className="px-3 py-2 text-left font-semibold text-foreground">Genotype</th>
                    {data.trait_names
                      .filter((t) => data.trait_performance_tables[t]?.length > 0)
                      .map((t) => (
                        <th key={t} className="px-3 py-2 text-right font-semibold text-foreground">
                          {t} Rank
                        </th>
                      ))}
                    <th className="px-3 py-2 text-right font-semibold text-foreground">Rank Sum</th>
                    <th className="px-3 py-2 text-left font-semibold text-foreground">Assessment</th>
                  </tr>
                </thead>
                <tbody>
                  {data.multi_trait_ranking.map((r, i) => (
                    <tr key={i} className={`border-b border-border/50 hover:bg-muted/20 ${i === 0 ? "bg-green-50/50 dark:bg-green-950/20" : ""}`}>
                      <td className="px-3 py-2 text-left font-medium text-foreground">
                        {r.genotype}
                        {i === 0 && <span className="ml-1 text-xs">🏆</span>}
                      </td>
                      {data.trait_names
                        .filter((t) => data.trait_performance_tables[t]?.length > 0)
                        .map((t) => (
                          <td key={t} className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                            {r.trait_ranks[t] ?? "—"}
                          </td>
                        ))}
                      <td className="px-3 py-2 text-right tabular-nums font-semibold text-foreground">{r.rank_sum}</td>
                      <td className={`px-3 py-2 text-left text-xs ${assessmentColor(r.overall_assessment)}`}>
                        {r.overall_assessment}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Word export section */}
      <HtmlTablesSection htmlTables={htmlTables} />
    </div>
  );
}
