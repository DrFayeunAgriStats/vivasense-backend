import React, { useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { AlertTriangle, ArrowRight, Target, ChevronDown, Download, Lightbulb, Image as ImageIcon } from "lucide-react";
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from "@/components/ui/collapsible";
import { TableDownloadMenu } from "../results/TableDownloadMenu";
import { FigureDownloadMenu } from "../results/FigureDownloadMenu";
import ReactMarkdown from "react-markdown";

const fmt4 = (v: unknown): string =>
  v == null || v === "" ? "—" : Number(v).toFixed(4);
const fmt2 = (v: unknown): string =>
  v == null || v === "" ? "—" : Number(v).toFixed(2);

function coeffColor(val: number): string {
  return val >= 0
    ? "text-[#1565C0] dark:text-blue-400"
    : "text-[#C62828] dark:text-red-400";
}

function classifyVIF(vif: number): { label: string; className: string } {
  if (vif >= 10) return { label: "❌ Severe", className: "bg-destructive/10 text-destructive" };
  if (vif >= 5) return { label: "⚠ Moderate", className: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300" };
  return { label: "✓ OK", className: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300" };
}

interface PathAnalysisData {
  target_trait?: string;
  R_squared?: number;
  residual_effect?: { value: number };
  direct_effects?: Record<string, { value: number; correlation_with_target?: number }>;
  indirect_effects?: Record<string, Record<string, { value: number }>>;
  path_matrix?: Record<string, Record<string, number>>;
  effect_decomposition?: Record<string, {
    direct: number;
    indirect: number;
    total: number;
    pct_direct: number;
    pct_indirect: number;
    vif?: number;
  }>;
  model_fit?: {
    r_squared: number;
    residual_path: number;
    residual_path_formula?: string;
    residual_interpretation?: string;
  };
  path_diagram?: {
    png_base64?: string;
    plotly_json?: string;
  };
  interpretation?: string;
}

interface Props {
  data: PathAnalysisData;
  traitName?: string;
}

// Local CollapsibleCard delegates to the shared VsResultSection so the
// path-analysis UI matches the Word report's section hierarchy + typography.
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

function PlotlyDiagram({ plotlyJson, pngBase64 }: { plotlyJson?: string; pngBase64?: string }) {
  const plotRef = useRef<HTMLDivElement>(null);
  const [usedPlotly, setUsedPlotly] = useState(false);

  useEffect(() => {
    if (!plotlyJson || !plotRef.current) return;
    try {
      const parsed = typeof plotlyJson === "string" ? JSON.parse(plotlyJson) : plotlyJson;
      const Plotly = (window as any).Plotly;
      if (Plotly && plotRef.current) {
        Plotly.react(plotRef.current, parsed.data, parsed.layout ?? {}, { responsive: true });
        setUsedPlotly(true);
      }
    } catch (e) {
      console.warn("Plotly render failed, falling back to PNG", e);
    }
  }, [plotlyJson]);

  if (usedPlotly) {
    return (
      <div>
        <div ref={plotRef} className="w-full" />
        <div className="flex gap-2 mt-2 justify-end">
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              const Plotly = (window as any).Plotly;
              if (Plotly && plotRef.current) {
                Plotly.downloadImage(plotRef.current, { format: "png", filename: "path_diagram", width: 1200, height: 800 });
              }
            }}
          >
            <Download className="w-4 h-4 mr-1" /> PNG
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              const Plotly = (window as any).Plotly;
              if (Plotly && plotRef.current) {
                Plotly.downloadImage(plotRef.current, { format: "svg", filename: "path_diagram", width: 1200, height: 800 });
              }
            }}
          >
            <Download className="w-4 h-4 mr-1" /> SVG
          </Button>
        </div>
      </div>
    );
  }

  // Fallback to PNG
  if (pngBase64) {
    return (
      <div>
        <img
          src={`data:image/png;base64,${pngBase64}`}
          alt="Path Diagram"
          className="max-w-full rounded-lg shadow-md mx-auto"
        />
        <div className="flex justify-end mt-2">
          <FigureDownloadMenu title="Path_Diagram" base64={pngBase64} />
        </div>
      </div>
    );
  }

  return <p className="text-sm text-muted-foreground text-center">No path diagram available.</p>;
}

export function PathAnalysisResultsDisplay({ data, traitName }: Props) {
  const {
    target_trait,
    path_matrix,
    effect_decomposition,
    model_fit,
    path_diagram,
    interpretation,
  } = data;

  const displayTrait = traitName || target_trait;

  // Get all predictor names (exclude Residual for decomposition)
  const predictorNames = path_matrix
    ? Object.keys(path_matrix).filter((k) => !/residual/i.test(k))
    : [];
  const allMatrixKeys = path_matrix ? Object.keys(path_matrix) : [];

  // Get indirect column names from the path matrix
  const indirectCols = path_matrix && predictorNames.length > 0
    ? Object.keys(path_matrix[predictorNames[0]] ?? {}).filter(
        (k) => k !== "Direct" && k !== "Total_r" && !k.startsWith("Total")
      )
    : [];

  return (
    <div className="space-y-4">
      {/* Summary badges */}
      <div className="flex flex-wrap gap-3 items-center">
        {displayTrait && (
          <Badge variant="secondary" className="gap-1">
            <Target className="w-3 h-3" /> Target: {displayTrait}
          </Badge>
        )}
        {model_fit && (
          <>
            <Badge variant="outline">
              R² = {fmt4(model_fit.r_squared)} ({(model_fit.r_squared * 100).toFixed(1)}%)
            </Badge>
            <Badge variant="outline">
              Residual = {fmt4(model_fit.residual_path)}
            </Badge>
          </>
        )}
      </div>

      {/* Card 1 — Path Diagram */}
      {path_diagram && (path_diagram.plotly_json || path_diagram.png_base64) && (
        <CollapsibleCard
          title="Path Diagram"
          icon={<ImageIcon className="w-5 h-5 text-primary" />}
          defaultOpen
        >
          <PlotlyDiagram plotlyJson={path_diagram.plotly_json} pngBase64={path_diagram.png_base64} />
          <p className="text-xs text-muted-foreground mt-3 italic">
            Blue arrow = positive effect, Red arrow = negative effect, Dashed = residual path.
          </p>
        </CollapsibleCard>
      )}

      {/* Card 2 — Path Coefficient Matrix */}
      {path_matrix && allMatrixKeys.length > 0 && (
        <CollapsibleCard
          title={`Path Coefficient Matrix${displayTrait ? ` — ${displayTrait}` : ""}`}
          icon={<ArrowRight className="w-5 h-5 text-primary" />}
          defaultOpen
        >
          {(() => {
            const headers = ["Causal Variable", "Direct (p)", ...indirectCols.map((c) => c.replace("Indirect_via_", "via ")), "Total r"];
            const rows: React.ReactNode[][] = allMatrixKeys.map((name) => {
              const row = path_matrix[name] ?? {};
              const isResidual = /residual/i.test(name);
              const cells: React.ReactNode[] = [
                <span className={isResidual ? "italic font-bold text-muted-foreground" : "font-medium text-foreground"}>
                  {name}
                </span>,
                <span className={`font-mono ${row.Direct != null ? coeffColor(row.Direct) : ""}`}>
                  {fmt4(row.Direct)}
                </span>,
                ...indirectCols.map((col) => (
                  <span className={`font-mono ${row[col] != null ? coeffColor(Number(row[col])) : "text-muted-foreground"}`}>
                    {fmt4(row[col])}
                  </span>
                )),
                <span className={`font-mono font-semibold ${row.Total_r != null ? coeffColor(row.Total_r) : ""}`}>
                  {fmt4(row.Total_r)}
                </span>,
              ];
              return cells;
            });

            const dlRows = rows.map((r) =>
              r.map((c) => {
                if (c == null) return "—";
                if (typeof c === "object" && "props" in (c as any)) {
                  const props = (c as any).props;
                  return typeof props?.children === "string" ? props.children : String(props?.children ?? "");
                }
                return String(c);
              })
            );

            return (
              <>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr className="border-b border-border">
                        {headers.map((h, i) => (
                          <th key={i} className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50 whitespace-nowrap">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, ri) => {
                        const isResidual = /residual/i.test(allMatrixKeys[ri]);
                        return (
                          <tr key={ri} className={`border-b border-border/50 ${isResidual ? "bg-muted/40" : ri % 2 !== 0 ? "bg-muted/20" : ""}`}>
                            {row.map((cell, ci) => (
                              <td key={ci} className={`px-3 py-2 whitespace-nowrap ${ci > 0 ? "text-right text-xs" : ""}`}>
                                {cell}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {model_fit && (
                  <div className="mt-3 p-3 rounded-lg bg-muted/30 border border-border text-sm">
                    <p className="text-foreground">
                      <strong>R² = {fmt4(model_fit.r_squared)}</strong> — {model_fit.residual_interpretation}
                    </p>
                    {model_fit.residual_path_formula && (
                      <p className="font-mono text-xs text-muted-foreground mt-1">{model_fit.residual_path_formula}</p>
                    )}
                  </div>
                )}
                <p className="text-xs text-muted-foreground mt-2 italic">
                  Standardised path coefficients. Direct + Σ Indirect = Total r.
                </p>
                <div className="flex justify-end mt-2">
                  <TableDownloadMenu title="Path_Coefficient_Matrix" headers={headers} rows={dlRows} />
                </div>
              </>
            );
          })()}
        </CollapsibleCard>
      )}

      {/* Card 3 — Effect Decomposition */}
      {effect_decomposition && Object.keys(effect_decomposition).length > 0 && (
        <CollapsibleCard
          title="Effect Decomposition"
          icon={<ArrowRight className="w-5 h-5 text-primary" />}
        >
          {(() => {
            const hasVIF = Object.values(effect_decomposition).some((e) => e.vif != null);
            const headers = ["Predictor", "Direct (p)", "Indirect (Σ)", "Total (r)", "% Direct", "% Indirect", ...(hasVIF ? ["VIF"] : [])];
            const entries = Object.entries(effect_decomposition);

            const rows: React.ReactNode[][] = entries.map(([name, e]) => {
              const row: React.ReactNode[] = [
                <span className="font-medium text-foreground">{name}</span>,
                <span className={`font-mono ${coeffColor(e.direct)}`}>{fmt4(e.direct)}</span>,
                <span className={`font-mono ${coeffColor(e.indirect)}`}>{fmt4(e.indirect)}</span>,
                <span className={`font-mono font-semibold ${coeffColor(e.total)}`}>{fmt4(e.total)}</span>,
                <span className="font-mono">{fmt2(e.pct_direct)}%</span>,
                <span className="font-mono">{fmt2(e.pct_indirect)}%</span>,
              ];
              if (hasVIF && e.vif != null) {
                const vifClass = classifyVIF(e.vif);
                row.push(
                  <Badge variant="outline" className={vifClass.className}>
                    {fmt2(e.vif)} {vifClass.label}
                  </Badge>
                );
              } else if (hasVIF) {
                row.push("—");
              }
              return row;
            });

            const dlRows = rows.map((r) =>
              r.map((c) => {
                if (c == null) return "—";
                if (typeof c === "object" && "props" in (c as any)) {
                  const props = (c as any).props;
                  return typeof props?.children === "string" ? props.children : String(props?.children ?? "");
                }
                return String(c);
              })
            );

            return (
              <>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr className="border-b border-border">
                        {headers.map((h, i) => (
                          <th key={i} className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50 whitespace-nowrap">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((row, ri) => (
                        <tr key={ri} className={`border-b border-border/50 ${ri % 2 !== 0 ? "bg-muted/20" : ""}`}>
                          {row.map((cell, ci) => (
                            <td key={ci} className={`px-3 py-2 whitespace-nowrap ${ci > 0 ? "text-right text-xs" : ""}`}>{cell}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="flex justify-end mt-2">
                  <TableDownloadMenu title="Effect_Decomposition" headers={headers} rows={dlRows} />
                </div>
              </>
            );
          })()}
        </CollapsibleCard>
      )}

      {/* Card 4 — Interpretation */}
      {interpretation && (
        <CollapsibleCard
          title="Path Analysis Interpretation"
          icon={<Lightbulb className="w-5 h-5 text-primary" />}
        >
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <ReactMarkdown>{interpretation}</ReactMarkdown>
          </div>
        </CollapsibleCard>
      )}
    </div>
  );
}
