import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { Trash2, BarChart3, Image, Info, Lightbulb, AlertTriangle, BookOpen, ChevronDown, Download, FileText } from "lucide-react";
import ReactMarkdown from "react-markdown";

export interface GeneticsResultsData {
  meta?: Record<string, unknown>;
  tables?: Record<string, unknown>;
  plots?: Record<string, string>;
  interpretation?: string;
  intelligence?: {
    executive_insight?: string;
    reviewer_radar?: string;
    decision_rules?: string;
    formulas_used?: string;
  };
}

interface Props {
  results: GeneticsResultsData;
  analysisType: string;
  onClear: () => void;
}

function renderTable(tableName: string, data: unknown) {
  if (Array.isArray(data) && data.length > 0 && Array.isArray(data[0])) {
    const headers = data[0] as unknown[];
    const rows = data.slice(1) as unknown[][];
    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-border">
              {headers.map((h, i) => (
                <th key={i} className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50 whitespace-nowrap">
                  {String(h)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => (
              <tr key={ri} className="border-b border-border/50 hover:bg-muted/30">
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

  if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object" && data[0] !== null) {
    const headers = Object.keys(data[0] as Record<string, unknown>);
    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-border">
              {headers.map((h) => (
                <th key={h} className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50 whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(data as Record<string, unknown>[]).map((row, ri) => (
              <tr key={ri} className="border-b border-border/50 hover:bg-muted/30">
                {headers.map((h) => (
                  <td key={h} className="px-3 py-2 text-muted-foreground whitespace-nowrap">
                    {row[h] != null ? String(row[h]) : ""}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return <pre className="text-xs text-muted-foreground overflow-x-auto">{JSON.stringify(data, null, 2)}</pre>;
}

const INTELLIGENCE_SECTIONS = [
  { key: "executive_insight", label: "Executive Insight", icon: Lightbulb, color: "text-primary" },
  { key: "reviewer_radar", label: "Reviewer Radar", icon: AlertTriangle, color: "text-amber-600 dark:text-amber-400" },
  { key: "decision_rules", label: "Decision Rules", icon: BookOpen, color: "text-emerald-600 dark:text-emerald-400" },
  { key: "formulas_used", label: "Formulas Used", icon: FileText, color: "text-blue-600 dark:text-blue-400" },
] as const;

function formatTitle(name: string) {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function VivaSenseGeneticsResults({ results, analysisType, onClear }: Props) {
  const { meta, tables, plots, interpretation, intelligence } = results;

  const handleDownload = () => {
    try {
      let text = `VivaSense™ – Plant Breeding Genetics Analysis Report\n`;
      text += `Analysis: ${formatTitle(analysisType)}\n`;
      text += `Generated: ${new Date().toLocaleString()}\n`;
      text += `${"=".repeat(60)}\n\n`;

      if (meta) {
        text += `SUMMARY\n${"─".repeat(40)}\n`;
        Object.entries(meta).forEach(([k, v]) => {
          text += `${k.replace(/_/g, " ")}: ${String(v)}\n`;
        });
        text += "\n";
      }

      if (tables) {
        Object.entries(tables).forEach(([name, data]) => {
          text += `${formatTitle(name)}\n${"─".repeat(40)}\n`;
          if (Array.isArray(data) && data.length > 0) {
            if (Array.isArray(data[0])) {
              data.forEach((row: unknown) => {
                text += (row as unknown[]).map(String).join("\t") + "\n";
              });
            } else if (typeof data[0] === "object") {
              const headers = Object.keys(data[0] as Record<string, unknown>);
              text += headers.join("\t") + "\n";
              (data as Record<string, unknown>[]).forEach((row) => {
                text += headers.map((h) => String(row[h] ?? "")).join("\t") + "\n";
              });
            }
          } else {
            text += JSON.stringify(data, null, 2) + "\n";
          }
          text += "\n";
        });
      }

      if (interpretation) {
        text += `INTERPRETATION\n${"─".repeat(40)}\n${interpretation}\n\n`;
      }

      if (intelligence) {
        INTELLIGENCE_SECTIONS.forEach(({ key, label }) => {
          const content = intelligence[key as keyof typeof intelligence];
          if (content) {
            text += `${label.toUpperCase()}\n${"─".repeat(40)}\n${content}\n\n`;
          }
        });
      }

      text += `${"=".repeat(60)}\n`;
      text += `VivaSense™ – A Statistical Intelligence Engine\nby Field-to-Insight Academy © Dr. Fayeun Lawrence Stephen\n`;
      text += `https://fieldtoinsightacademy.com.ng/vivasense\n`;

      const blob = new Blob([text], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `VivaSense_Genetics_${analysisType}_${new Date().toISOString().slice(0, 10)}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Download failed:", err);
    }
  };

  return (
    <section className="py-20 bg-muted/30" id="genetics-results">
      <div className="container-wide">
        <div className="max-w-5xl mx-auto space-y-8">
          <div className="text-center mb-8">
            <h2 className="font-serif text-3xl lg:text-4xl font-bold text-foreground mb-2">
              {formatTitle(analysisType)} Results
            </h2>
            <p className="text-muted-foreground mb-4">Plant Breeding Genetics Analysis</p>
            <div className="flex items-center justify-center gap-3">
              <Button variant="outline" size="sm" onClick={handleDownload}>
                <Download className="w-4 h-4 mr-2" />
                Download Report
              </Button>
              <Button variant="outline" size="sm" onClick={onClear}>
                <Trash2 className="w-4 h-4 mr-2" />
                Clear Results
              </Button>
            </div>
          </div>

          {/* Meta Summary */}
          {meta && Object.keys(meta).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Info className="w-6 h-6 text-primary" />
                  Analysis Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-x-6 gap-y-3">
                  {Object.entries(meta).map(([key, value]) => (
                    <div key={key} className="flex flex-col">
                      <dt className="text-sm text-muted-foreground font-medium capitalize">
                        {key.replace(/_/g, " ")}
                      </dt>
                      <dd className="text-foreground font-semibold">{String(value)}</dd>
                    </div>
                  ))}
                </dl>
              </CardContent>
            </Card>
          )}

          {/* Tables */}
          {tables &&
            Object.entries(tables).map(([tableName, tableData]) => {
              if (!tableData) return null;
              return (
                <Card key={tableName}>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-3">
                      <BarChart3 className="w-6 h-6 text-primary" />
                      {formatTitle(tableName)}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {renderTable(tableName, tableData)}
                  </CardContent>
                </Card>
              );
            })}

          {/* Plots */}
          {plots &&
            Object.entries(plots).map(([plotName, base64]) => (
              <Card key={plotName}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-3">
                    <Image className="w-6 h-6 text-primary" />
                    {formatTitle(plotName)}
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex justify-center">
                  <img
                    src={`data:image/png;base64,${base64}`}
                    alt={plotName}
                    className="max-w-full rounded-lg shadow-md"
                  />
                </CardContent>
              </Card>
            ))}

          {/* Intelligence Blocks */}
          {intelligence && (
            <div className="space-y-4">
              <h3 className="font-serif text-2xl font-bold text-foreground text-center">
                Intelligence Blocks
              </h3>
              {INTELLIGENCE_SECTIONS.map(({ key, label, icon: Icon, color }) => {
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
                          <ChevronDown className="w-5 h-5 text-muted-foreground transition-transform duration-200 [[data-state=open]>&]:rotate-180" />
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
            </div>
          )}

          {/* Interpretation */}
          {interpretation && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  <Lightbulb className="w-6 h-6 text-primary" />
                  Interpretation
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="prose prose-sm max-w-none dark:prose-invert">
                  <ReactMarkdown>{Array.isArray(interpretation) ? interpretation.join("\n\n") : String(interpretation)}</ReactMarkdown>
                </div>
                <div className="mt-4 rounded-lg border border-amber-300 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-700 p-4">
                  <p className="text-sm text-amber-800 dark:text-amber-200">
                    ⚠️ <strong>Academic Integrity Reminder:</strong> This interpretation is a starting point for your own analysis. Verify all numbers against the tables above. Adapt any suggested text to your own words and field context. Discuss with your supervisor before submitting.
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </section>
  );
}
