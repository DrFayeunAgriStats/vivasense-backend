import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Trash2, Download } from "lucide-react";
import { GeneticsTraitResult, type TraitResultData } from "./genetics/GeneticsTraitResult";
import { HtmlTablesSection } from "./HtmlTablesSection";
import { generatePublishableHtmlTables } from "./utils/generatePublishableTables";

/**
 * The backend can return results in two shapes:
 * A) Flat single-trait: { meta, tables, plots, interpretation, intelligence }
 * B) Multi-trait keyed: { per_trait: { "Trait1": {...}, "Trait2": {...} }, ... }
 * 
 * We normalise both into Record<string, TraitResultData>.
 */
export interface GeneticsMultiTraitResults {
  per_trait?: Record<string, TraitResultData>;
  // flat single-trait fallback fields
  meta?: Record<string, unknown>;
  tables?: Record<string, unknown>;
  plots?: Record<string, string>;
  interpretation?: string;
  intelligence?: Record<string, string>;
  html_tables?: Record<string, string>;
  [key: string]: unknown;
}

interface Props {
  results: GeneticsMultiTraitResults;
  analysisType: string;
  selectedTraits: string[];
  onClear: () => void;
}

function formatTitle(name: string) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function buildTraitMap(results: GeneticsMultiTraitResults, selectedTraits: string[]): Record<string, TraitResultData> {
  // Shape B: per_trait keyed
  if (results.per_trait && typeof results.per_trait === "object") {
    return results.per_trait as Record<string, TraitResultData>;
  }

  // Shape A: single-trait, wrap in map keyed by first selectedTrait or "Result"
  const traitName = selectedTraits[0] || (results.meta?.trait as string) || "Result";
  return {
    [traitName]: {
      meta: results.meta,
      tables: results.tables,
      plots: results.plots,
      interpretation: results.interpretation,
      intelligence: results.intelligence as TraitResultData["intelligence"],
    },
  };
}

function exportTraitCSV(traitName: string, data: TraitResultData) {
  try {
    let csv = `Trait,${traitName}\n\n`;

    // Variance Components
    const vc = data.variance_components;
    if (vc) {
      csv += "Component,Symbol,Value\n";
      const rows = [
        ["Genetic Variance", "σ²g", vc.genetic_variance],
        ["Environmental Variance", "σ²e", vc.environmental_variance],
        ["G×E Interaction", "σ²ge", vc.gxl_variance],
        ["Phenotypic Variance", "σ²p", vc.phenotypic_variance],
        ["GCV (%)", "GCV", vc.gcv],
        ["PCV (%)", "PCV", vc.pcv],
        ["ECV (%)", "ECV", vc.ecv],
      ];
      rows.filter((r) => r[2] != null).forEach(([l, s, v]) => {
        csv += `${l},${s},${Number(v).toFixed(4)}\n`;
      });
      csv += "\n";
    }

    // Heritability
    if (data.heritability?.h2 != null) {
      csv += `Heritability (H²),${(data.heritability.h2 * 100).toFixed(2)}%\n`;
    }
    if (data.genetic_advance?.ga != null) {
      csv += `Genetic Advance (GA),${Number(data.genetic_advance.ga).toFixed(2)}\n`;
    }
    if (data.genetic_advance?.ga_percent != null) {
      csv += `GA%,${data.genetic_advance.ga_percent.toFixed(2)}%\n`;
    }

    // Fallback: tables
    if (data.tables) {
      Object.entries(data.tables).forEach(([name, tData]) => {
        csv += `\n${name}\n`;
        if (Array.isArray(tData) && tData.length > 0) {
          if (Array.isArray(tData[0])) {
            tData.forEach((row: unknown) => {
              csv += (row as unknown[]).map(String).join(",") + "\n";
            });
          } else if (typeof tData[0] === "object") {
            const headers = Object.keys(tData[0] as Record<string, unknown>);
            csv += headers.join(",") + "\n";
            (tData as Record<string, unknown>[]).forEach((row) => {
              csv += headers.map((h) => String(row[h] ?? "")).join(",") + "\n";
            });
          }
        }
      });
    }

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `VivaSense_Genetics_${traitName.replace(/\s+/g, "_")}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error("CSV export failed:", err);
  }
}

function exportPlotPNG(plotName: string, base64: string) {
  try {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${base64}`;
    link.download = `VivaSense_${plotName.replace(/\s+/g, "_")}.png`;
    link.click();
  } catch (err) {
    console.error("PNG export failed:", err);
  }
}

function exportFullReport(analysisType: string, traitMap: Record<string, TraitResultData>) {
  try {
    let text = `VivaSense™ – Plant Breeding Genetics Analysis Report\n`;
    text += `Analysis: ${formatTitle(analysisType)}\n`;
    text += `Generated: ${new Date().toLocaleString()}\n`;
    text += `${"=".repeat(60)}\n\n`;

    Object.entries(traitMap).forEach(([traitName, data]) => {
      text += `\n${"═".repeat(60)}\nTRAIT: ${traitName}\n${"═".repeat(60)}\n\n`;

      const vc = data.variance_components;
      if (vc) {
        text += "VARIANCE COMPONENTS\n" + "─".repeat(40) + "\n";
        [
          ["Genetic Variance (σ²g)", vc.genetic_variance],
          ["Environmental Variance (σ²e)", vc.environmental_variance],
          ["G×E Interaction (σ²ge)", vc.gxl_variance],
          ["Phenotypic Variance (σ²p)", vc.phenotypic_variance],
          ["GCV (%)", vc.gcv],
          ["PCV (%)", vc.pcv],
        ].filter(([, v]) => v != null).forEach(([l, v]) => {
          text += `${l}: ${Number(v).toFixed(4)}\n`;
        });
        text += "\n";
      }

      if (data.heritability?.h2 != null) text += `Heritability (H²): ${(data.heritability.h2 * 100).toFixed(2)}%\n`;
      if (data.genetic_advance?.ga != null) text += `Genetic Advance: ${Number(data.genetic_advance.ga).toFixed(2)}\n`;
      if (data.genetic_advance?.ga_percent != null) text += `GA%: ${data.genetic_advance.ga_percent.toFixed(2)}%\n`;

      if (data.interpretation) text += `\nINTERPRETATION\n${"─".repeat(40)}\n${data.interpretation}\n`;
    });

    text += `\n${"=".repeat(60)}\n`;
    text += `VivaSense™ – A Statistical Intelligence Engine\nby Field-to-Insight Academy © Dr. Fayeun Lawrence Stephen\nhttps://fieldtoinsightacademy.com.ng/vivasense\n`;

    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `VivaSense_Genetics_${analysisType}_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error("Report download failed:", err);
  }
}

export function VivaSenseGeneticsResultsTabs({ results, analysisType, selectedTraits, onClear }: Props) {
  const traitMap = useMemo(() => buildTraitMap(results, selectedTraits), [results, selectedTraits]);
  const traitNames = Object.keys(traitMap);

  if (traitNames.length === 0) return null;

  return (
    <section className="py-20 bg-muted/30" id="genetics-results">
      <div className="container-wide">
        <div className="max-w-5xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center mb-8">
            <h2 className="font-serif text-3xl lg:text-4xl font-bold text-foreground mb-2">
              {formatTitle(analysisType)} Results
            </h2>
            <p className="text-muted-foreground mb-4">
              Plant Breeding Genetics Analysis — {traitNames.length} trait{traitNames.length > 1 ? "s" : ""} analyzed
            </p>
            <div className="flex items-center justify-center gap-3 flex-wrap">
              <Button variant="outline" size="sm" onClick={() => exportFullReport(analysisType, traitMap)}>
                <Download className="w-4 h-4 mr-2" /> Download Full Report
              </Button>
              <Button variant="outline" size="sm" onClick={onClear}>
                <Trash2 className="w-4 h-4 mr-2" /> Clear Results
              </Button>
            </div>
          </div>

          {/* Tabbed Results */}
          {traitNames.length === 1 ? (
            <GeneticsTraitResult
              data={traitMap[traitNames[0]]}
              traitName={traitNames[0]}
              onExportCSV={() => exportTraitCSV(traitNames[0], traitMap[traitNames[0]])}
              onExportPNG={exportPlotPNG}
            />
          ) : (
            <Tabs defaultValue={traitNames[0]} className="w-full">
              <TabsList className="flex flex-wrap h-auto gap-1 bg-muted p-1">
                {traitNames.map((name) => (
                  <TabsTrigger key={name} value={name} className="text-sm font-medium px-4 py-2">
                    {name}
                  </TabsTrigger>
                ))}
              </TabsList>
              {traitNames.map((name) => (
                <TabsContent key={name} value={name} className="mt-6">
                  <GeneticsTraitResult
                    data={traitMap[name]}
                    traitName={name}
                    onExportCSV={() => exportTraitCSV(name, traitMap[name])}
                    onExportPNG={exportPlotPNG}
                  />
                </TabsContent>
              ))}
            </Tabs>
          )}

          {/* Publishable HTML Tables — from backend or auto-generated */}
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
