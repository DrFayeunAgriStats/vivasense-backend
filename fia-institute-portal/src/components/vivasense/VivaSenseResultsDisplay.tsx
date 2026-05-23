import React, { useState } from "react";
import { ChevronDown, ChevronRight, Code, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { HtmlTablesSection } from "./HtmlTablesSection";
import { generatePublishableHtmlTables } from "./utils/generatePublishableTables";
import { TableDownloadMenu } from "./results/TableDownloadMenu";
import { FigureDownloadMenu } from "./results/FigureDownloadMenu";


/* ── Formatting helpers ────────────────────────────────────────────── */

const fmt0 = (v: unknown): string =>
  v == null || v === "" ? "—" : Math.round(Number(v)).toString();

const fmt2 = (v: unknown): string => {
  if (v == null || v === "") return "—";
  const n = Number(v);
  return typeof n === "number" && !isNaN(n) ? n.toFixed(2) : "N/A";
};

const fmt4 = (v: unknown): string => {
  if (v == null || v === "") return "—";
  const n = Number(v);
  return typeof n === "number" && !isNaN(n) ? n.toFixed(4) : "N/A";
};

const fmtN = (v: unknown, d = 2): string => {
  if (v == null || v === "") return "—";
  const n = Number(v);
  return typeof n === "number" && !isNaN(n) ? n.toFixed(d) : "N/A";
};

const stars = (p: unknown): string => {
  if (p == null) return "";
  const n = Number(p);
  if (n < 0.001) return " ***";
  if (n < 0.01) return " **";
  if (n < 0.05) return " *";
  return " ns";
};

const pDisplay = (p: unknown): string => {
  if (p == null) return "—";
  const n = Number(p);
  if (n < 0.0001) return "<0.0001" + stars(p);
  return fmt4(p) + stars(p);
};

const cleanSource = (s: string) => s.replace(/Q\('(.+?)'\)/g, "$1");

/* ── adaptBackendResult ────────────────────────────────────────────── */

function adaptBackendResult(raw: Record<string, unknown>): Record<string, unknown> {
  const r = { ...raw };

  // Normalize anova_table → anova if needed
  if (r.anova_table && !r.anova) r.anova = r.anova_table;

  // Normalize means_separation → means + letters
  if (r.means_separation && !r.means) {
    const sep = r.means_separation as Record<string, unknown>;
    if (sep.means) r.means = sep.means;
    if (sep.letters) r.letters = sep.letters;
  }

  // Normalize interaction_means
  if (r.interaction_means && !r.interactions) r.interactions = r.interaction_means;

  return r;
}

/* ── Reusable sub-components ───────────────────────────────────────── */

function PubTable({
  headers,
  rows,
  caption,
  downloadTitle,
}: {
  headers: string[];
  rows: React.ReactNode[][];
  caption?: string;
  downloadTitle?: string;
}) {
  // Build plain-text rows for download (strip React nodes to strings)
  const dlRows = rows.map((row) =>
    row.map((cell) => {
      if (cell == null) return "—";
      if (typeof cell === "object" && "props" in (cell as any)) {
        const props = (cell as any).props;
        return String(props?.children ?? "");
      }
      return String(cell);
    })
  );

  return (
    <div className="overflow-x-auto my-3">
      {downloadTitle && (
        <div className="flex justify-end mb-1">
          <TableDownloadMenu title={downloadTitle} headers={headers} rows={dlRows} />
        </div>
      )}
      <table className="w-full text-sm border-collapse border border-border">
        <thead>
          <tr className="bg-muted">
            {headers.map((h, i) => (
              <th
                key={i}
                className="border border-border px-3 py-2 text-left font-semibold text-foreground"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? "bg-background" : "bg-muted/40"}>
              {row.map((cell, j) => (
                <td
                  key={j}
                  className={`border border-border/60 px-3 py-1.5 text-foreground ${
                    j > 0 ? "text-right font-mono text-xs" : ""
                  }`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {caption && (
        <p className="text-xs text-muted-foreground mt-1 italic">{caption}</p>
      )}
    </div>
  );
}

function CollapsibleSection({
  title,
  icon,
  defaultOpen = true,
  children,
}: {
  title: string;
  icon?: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="bg-card rounded-lg border border-border shadow-sm mb-4 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full bg-muted border-b border-border px-4 py-2.5 flex items-center gap-2 hover:bg-muted/80 transition-colors text-left"
      >
        {open ? (
          <ChevronDown className="w-4 h-4 text-muted-foreground shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-muted-foreground shrink-0" />
        )}
        {icon}
        <h3 className="font-semibold text-foreground text-sm">{title}</h3>
      </button>
      {open && <div className="px-4 py-3">{children}</div>}
    </div>
  );
}

/* ── Section renderers ─────────────────────────────────────────────── */

function ExperimentSummary({ result }: { result: Record<string, unknown> }) {
  const anova = result.anova as Record<string, Record<string, unknown>> | undefined;
  const descStats = result.descriptive_stats as Record<string, unknown> | undefined;
  const overall = descStats?.overall as Record<string, unknown> | undefined;

  // Extract grand mean & CV from descriptive stats
  const grandMean = overall?.mean;
  const cv = overall?.cv;

  // Extract overall F and p from anova
  let overallF: unknown = result.f_value;
  let overallP: unknown = result.f_pvalue;
  if (!overallF && anova?.F) {
    const fObj = anova.F as Record<string, unknown>;
    const firstSource = Object.keys(fObj).find((k) => k !== "Residual");
    if (firstSource) {
      overallF = fObj[firstSource];
      overallP = (anova["PR(>F)"] as Record<string, unknown>)?.[firstSource];
    }
  }

  const sig = overallP != null ? Number(overallP) < 0.05 : (result.significant as boolean);

  return (
    <div
      className={`rounded-lg p-4 mb-4 text-sm border ${
        sig
          ? "bg-green-50 border-green-200 dark:bg-green-950/30 dark:border-green-800"
          : "bg-yellow-50 border-yellow-200 dark:bg-yellow-950/30 dark:border-yellow-800"
      }`}
    >
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-x-6 gap-y-2">
        {result.design && (
          <div><span className="font-semibold text-foreground">Design:</span> <span className="text-muted-foreground">{String(result.design).replace(/_/g, " ")}</span></div>
        )}
        {result.response && (
          <div><span className="font-semibold text-foreground">Response:</span> <span className="text-muted-foreground">{String(result.response)}</span></div>
        )}
        {result.treatment && (
          <div><span className="font-semibold text-foreground">Treatment:</span> <span className="text-muted-foreground">{String(result.treatment)}</span></div>
        )}
        {result.n != null && (
          <div><span className="font-semibold text-foreground">n:</span> <span className="text-muted-foreground">{String(result.n)}</span></div>
        )}
        {grandMean != null && (
          <div><span className="font-semibold text-foreground">Grand Mean:</span> <span className="text-muted-foreground">{fmt2(grandMean)}</span></div>
        )}
        {cv != null && (
          <div><span className="font-semibold text-foreground">CV%:</span> <span className="text-muted-foreground">{fmt2(cv)}%</span></div>
        )}
        {result.r_squared != null && (
          <div><span className="font-semibold text-foreground">R²:</span> <span className="text-muted-foreground">{fmt4(result.r_squared)}</span></div>
        )}
        {overallF != null && (
          <div><span className="font-semibold text-foreground">F:</span> <span className="text-muted-foreground">{fmt2(overallF)}</span></div>
        )}
        {overallP != null && (
          <div><span className="font-semibold text-foreground">p-value:</span> <span className="text-muted-foreground">{pDisplay(overallP)}</span></div>
        )}
      </div>
      {result.formula && (
        <div className="mt-2 pt-2 border-t border-border/40">
          <span className="font-semibold text-foreground">Formula:</span>{" "}
          <code className="text-xs bg-muted px-1.5 py-0.5 rounded font-mono text-muted-foreground">
            {String(result.formula)}
          </code>
        </div>
      )}
    </div>
  );
}

function ANOVATableDisplay({ anova }: { anova: Record<string, unknown> }) {
  if (!anova || typeof anova !== "object") return null;
  const dfObj = (anova.df ?? {}) as Record<string, unknown>;
  const sumSqObj = (anova.sum_sq ?? {}) as Record<string, unknown>;
  const msObj = (anova.mean_sq ?? anova.MS ?? {}) as Record<string, unknown>;
  const fObj = (anova.F ?? {}) as Record<string, unknown>;
  const prObj = (anova["PR(>F)"] ?? {}) as Record<string, unknown>;

  const sources = Object.keys(dfObj);
  if (!sources.length) return null;

  // Compute MS if not provided
  const getMS = (src: string): unknown => {
    if (msObj[src] != null) return msObj[src];
    const ss = sumSqObj[src] as number | null;
    const df = dfObj[src] as number | null;
    if (ss != null && df != null && df > 0) return ss / df;
    return null;
  };

  const rows: React.ReactNode[][] = sources.map((src) => [
    cleanSource(src),
    fmt0(dfObj[src]),
    fmt2(sumSqObj[src]),
    fmt2(getMS(src)),
    fObj[src] != null ? fmt2(fObj[src]) : "—",
    prObj[src] != null ? pDisplay(prObj[src]) : "—",
  ]);

  return (
    <PubTable
      headers={["Source", "DF", "SS", "MS", "F", "p-value"]}
      rows={rows}
      caption="Significance: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant"
      downloadTitle="ANOVA_Table"
    />
  );
}

function TukeyHSDDisplay({
  means,
  letters,
  treatment,
  descStats,
}: {
  means: Record<string, unknown>;
  letters?: Record<string, unknown> | null;
  treatment?: string;
  descStats?: Record<string, unknown> | null;
}) {
  if (!means || typeof means !== "object") return null;
  const treatKey =
    treatment && means[treatment] ? treatment : Object.keys(means)[0];
  const meansObj = (means[treatKey] ?? means) as Record<string, number>;
  const lettersObj = ((letters && ((letters as any)[treatKey] || letters)) ||
    {}) as Record<string, string>;

  // Try to get per-group stats for SE
  const groupStats = descStats
    ? ((descStats as any)[treatKey] as Record<string, Record<string, unknown>> | undefined)
    : null;

  const levels = Object.keys(meansObj);
  if (!levels.length) return null;

  const hasGroupStats = groupStats && Object.keys(groupStats).length > 0;

  const headers = hasGroupStats
    ? [treatKey || "Group", "n", "Mean", "SE", "Tukey Group"]
    : [treatKey || "Group", "Mean", "Tukey Group"];

  const rows: React.ReactNode[][] = levels
    .sort((a, b) => (meansObj[b] ?? 0) - (meansObj[a] ?? 0))
    .map((lvl) => {
      const gs = groupStats?.[lvl];
      if (hasGroupStats) {
        return [
          lvl,
          gs ? String(gs.count ?? gs.n ?? "—") : "—",
          fmt2(meansObj[lvl]),
          gs?.sem != null ? fmt2(gs.sem) : "—",
          <span className="font-bold text-primary">{lettersObj[lvl] || "—"}</span>,
        ];
      }
      return [
        lvl,
        fmt2(meansObj[lvl]),
        <span className="font-bold text-primary">{lettersObj[lvl] || "—"}</span>,
      ];
    });

  return (
    <PubTable
      headers={headers}
      rows={rows}
      caption="Means sharing the same letter are not significantly different (Tukey HSD, α=0.05)"
      downloadTitle="Treatment_Means_Tukey"
    />
  );
}

function InteractionMeansDisplay({ interactions }: { interactions: unknown }) {
  if (!interactions || typeof interactions !== "object") return null;

  // interactions can be: { "A:B": { "a1:b1": value, ... } } or array
  const entries = Object.entries(interactions as Record<string, unknown>);
  if (!entries.length) return null;

  return (
    <>
      {entries.map(([key, val]) => {
        if (!val || typeof val !== "object") return null;
        const meansObj = val as Record<string, unknown>;
        const levels = Object.keys(meansObj);
        if (!levels.length) return null;

        const rows: React.ReactNode[][] = levels
          .sort((a, b) => Number(meansObj[b] ?? 0) - Number(meansObj[a] ?? 0))
          .map((lvl) => [lvl, fmt2(meansObj[lvl])]);

        return (
          <div key={key} className="mb-3">
            <p className="text-xs font-semibold text-muted-foreground mb-1">
              Interaction: {cleanSource(key)}
            </p>
            <PubTable headers={["Combination", "Mean"]} rows={rows} downloadTitle={`Interaction_Means_${cleanSource(key)}`} />
          </div>
        );
      })}
    </>
  );
}

function EffectSizesBox({
  effectSizes,
  rSquared,
}: {
  effectSizes: Record<string, unknown>;
  rSquared?: unknown;
}) {
  if (!effectSizes || typeof effectSizes !== "object") return null;
  const entries = Object.entries(effectSizes).filter(([src]) => src !== "Residual");
  if (!entries.length) return null;

  return (
    <div className="space-y-3">
      {rSquared != null && (
        <div className="flex items-center gap-2 text-sm mb-2">
          <span className="font-semibold text-foreground">R²:</span>
          <span className="text-muted-foreground font-mono">{fmt4(rSquared)}</span>
          <span className="text-xs text-muted-foreground">
            ({(Number(rSquared) * 100).toFixed(1)}% variance explained)
          </span>
        </div>
      )}
      <PubTable
        headers={["Source", "η²", "ω²", "Cohen's f", "Interpretation"]}
        rows={entries.map(([src, es]) => {
          const obj = (typeof es === "object" && es ? es : {}) as Record<string, unknown>;
          return [
            cleanSource(src),
            fmt4(obj.eta_squared),
            fmt4(obj.omega_squared),
            fmt2(obj.cohens_f),
            <span
              className={`font-medium ${
                obj.interpretation === "large"
                  ? "text-green-600"
                  : obj.interpretation === "medium"
                  ? "text-yellow-600"
                  : "text-muted-foreground"
              }`}
            >
              {(obj.interpretation as string) || "—"}
            </span>,
          ];
        })}
        downloadTitle="Effect_Sizes"
      />
    </div>
  );
}

function AssumptionTests({ assumptions }: { assumptions: Record<string, unknown> }) {
  if (!assumptions || typeof assumptions !== "object") return null;

  const rows: React.ReactNode[][] = Object.entries(assumptions).map(
    ([key, val]) => {
      const obj = (typeof val === "object" && val ? val : {}) as Record<string, unknown>;
      return [
        key.charAt(0).toUpperCase() + key.slice(1),
        (obj.test_name as string) || "—",
        obj.statistic != null ? fmt4(obj.statistic) : "—",
        obj.p_value != null ? pDisplay(obj.p_value) : "—",
        obj.passed != null ? (
          obj.passed ? (
            <span className="text-green-600 font-medium">✓ Passed</span>
          ) : (
            <span className="text-destructive font-medium">✗ Failed</span>
          )
        ) : (
          "—"
        ),
        (obj.message as string) || "—",
      ];
    }
  );

  return (
    <PubTable
      headers={["Test", "Method", "Statistic", "p-value", "Result", "Note"]}
      rows={rows}
      downloadTitle="Assumption_Tests"
    />
  );
}

function DescriptiveStatsDisplay({ descStats }: { descStats: Record<string, unknown> }) {
  if (!descStats || typeof descStats !== "object") return null;

  const overall = descStats.overall as Record<string, unknown> | undefined;
  const groupKey = Object.keys(descStats).find((k) => k !== "overall");
  const groups = groupKey
    ? (descStats[groupKey] as Record<string, Record<string, unknown>>)
    : null;

  return (
    <>
      {overall && (
        <>
          <p className="text-xs font-semibold text-muted-foreground mb-1">Overall</p>
          <PubTable
            headers={["n", "Mean", "SD", "SEM", "CV (%)", "Min", "Max"]}
            rows={[
              [
                (overall.n as string) ?? "—",
                fmt2(overall.mean),
                fmt2(overall.std),
                fmt2(overall.sem),
                fmt2(overall.cv),
                fmt2(overall.min),
                fmt2(overall.max),
              ],
            ]}
            downloadTitle="Descriptive_Stats_Overall"
          />
        </>
      )}
      {groups && (
        <>
          <p className="text-xs font-semibold text-muted-foreground mt-3 mb-1">
            By {groupKey}
          </p>
          <PubTable
            headers={[groupKey!, "n", "Mean", "SD", "SEM", "Min", "Max", "CV (%)"]}
            rows={Object.entries(groups)
              .sort(
                ([, a], [, b]) =>
                  ((b.mean as number) ?? 0) - ((a.mean as number) ?? 0)
              )
              .map(([lvl, s]) => [
                lvl,
                (s.count ?? s.n ?? "—") as React.ReactNode,
                fmt2(s.mean),
                fmt2(s.std),
                fmt2(s.sem),
                fmt2(s.min),
                fmt2(s.max),
                fmt2(s.cv),
              ])}
            downloadTitle={`Descriptive_Stats_By_${groupKey}`}
          />
        </>
      )}
    </>
  );
}

/* ── Technical JSON toggle ─────────────────────────────────────────── */

function TechnicalJSON({ data }: { data: unknown }) {
  const [show, setShow] = useState(false);
  return (
    <div className="mt-6 border-t border-border pt-4">
      <button
        onClick={() => setShow(!show)}
        className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <Code className="w-3.5 h-3.5" />
        {show ? "Hide" : "Show"} technical JSON
      </button>
      {show && (
        <pre className="mt-2 p-3 bg-muted rounded-lg text-xs font-mono overflow-x-auto max-h-96 text-muted-foreground">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

/* ── Main component ────────────────────────────────────────────────── */

export interface VivaSenseResultsDisplayProps {
  result: Record<string, unknown> | null;
}


export default function VivaSenseResultsDisplay({
  result,
}: VivaSenseResultsDisplayProps) {
  if (!result) return null;

  // Multi-trait: result.results is a dict keyed by trait name
  if (result.results && typeof result.results === "object") {
    return (
      <div className="space-y-6">
        {Object.entries(result.results as Record<string, unknown>).map(
          ([trait, traitResult]) => (
            <div key={trait}>
              <h2 className="text-base font-bold text-foreground border-b border-border pb-1 mb-3">
                Trait: {trait}
              </h2>
              <VivaSenseResultsDisplay
                result={{ ...(traitResult as Record<string, unknown>), response: trait }}
              />
            </div>
          )
        )}
      </div>
    );
  }

  // Normalize backend keys
  const adapted = adaptBackendResult(result);

  const anova = adapted.anova as Record<string, unknown> | undefined;
  const means = adapted.means as Record<string, unknown> | undefined;
  const letters = adapted.letters as Record<string, unknown> | undefined;
  const effectSizes = adapted.effect_sizes as Record<string, unknown> | undefined;
  const assumptions = adapted.assumptions as Record<string, unknown> | undefined;
  const descStats = adapted.descriptive_stats as Record<string, unknown> | undefined;
  const plots = adapted.plots as Record<string, string> | undefined;
  const posthoc = adapted.posthoc as Record<string, Record<string, unknown>> | undefined;
  const interactions = adapted.interactions as unknown;

  return (
    <div className="space-y-2">

      {/* Experiment Summary */}
      <CollapsibleSection title="Experiment Summary" defaultOpen={true}>
        <ExperimentSummary result={adapted} />
      </CollapsibleSection>

      {/* Dr. Fayeun Interpretation */}
      {adapted.interpretation && (
        <CollapsibleSection
          title="Dr. Fayeun's Interpretation"
          icon={<FlaskConical className="w-4 h-4 text-primary" />}
          defaultOpen={true}
        >
          <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
            {String(adapted.interpretation)}
          </p>
        </CollapsibleSection>
      )}

      {/* Descriptive Statistics */}
      {descStats && (
        <CollapsibleSection title="Descriptive Statistics" defaultOpen={true}>
          <DescriptiveStatsDisplay descStats={descStats} />
        </CollapsibleSection>
      )}

      {/* Effect Sizes */}
      {effectSizes && (
        <CollapsibleSection title="Effect Sizes" defaultOpen={true}>
          <EffectSizesBox effectSizes={effectSizes} rSquared={adapted.r_squared} />
        </CollapsibleSection>
      )}

      {/* ANOVA Table */}
      {anova && Object.keys(anova).length > 0 && (
        <CollapsibleSection title="Analysis of Variance (ANOVA)" defaultOpen={true}>
          <ANOVATableDisplay anova={anova} />
        </CollapsibleSection>
      )}

      {/* Treatment Means & Tukey HSD */}
      {means && (
        <CollapsibleSection title="Treatment Means & Tukey HSD" defaultOpen={true}>
          <TukeyHSDDisplay
            means={means}
            letters={letters}
            treatment={adapted.treatment as string | undefined}
            descStats={descStats}
          />
        </CollapsibleSection>
      )}

      {/* Interaction Means (factorial designs) */}
      {interactions && (
        <CollapsibleSection title="Interaction Means" defaultOpen={true}>
          <InteractionMeansDisplay interactions={interactions} />
        </CollapsibleSection>
      )}

      {/* Assumption Tests */}
      {assumptions && (
        <CollapsibleSection title="Assumption Tests" defaultOpen={false}>
          <AssumptionTests assumptions={assumptions} />
        </CollapsibleSection>
      )}

      {/* Non-parametric test statistic */}
      {(adapted.H_statistic != null || adapted.chi2_statistic != null) && (
        <CollapsibleSection title="Test Statistic" defaultOpen={true}>
          <PubTable
            headers={["Statistic", "Value", "p-value", "Significant"]}
            rows={[
              [
                adapted.H_statistic != null ? "Kruskal-Wallis H" : "Friedman χ²",
                fmt4(adapted.H_statistic ?? adapted.chi2_statistic),
                pDisplay(adapted.p_value),
                (adapted.significant as boolean) ? (
                  <span className="text-green-600 font-medium">Yes</span>
                ) : (
                  <span className="text-muted-foreground">No</span>
                ),
              ],
            ]}
            downloadTitle="Test_Statistic"
          />
        </CollapsibleSection>
      )}

      {/* Post-hoc Comparisons */}
      {posthoc && Object.keys(posthoc).length > 0 && (
        <CollapsibleSection title="Post-hoc Comparisons" defaultOpen={true}>
          <PubTable
            headers={["Pair", "Statistic", "p (adjusted)", "Significant"]}
            rows={Object.entries(posthoc).map(([pair, ph]) => [
              pair,
              fmt4(ph.U ?? ph.W),
              pDisplay(ph.p_adjusted),
              (ph.significant as boolean) ? (
                <span className="text-green-600">Yes</span>
              ) : (
                <span className="text-muted-foreground">No</span>
              ),
            ])}
            downloadTitle="Post_Hoc_Comparisons"
          />
        </CollapsibleSection>
      )}

      {/* Publication plots */}
      {plots?.publication_bar && (
        <CollapsibleSection title="Publication Bar Chart" defaultOpen={true}>
          <div className="flex justify-end mb-2">
            <FigureDownloadMenu title="Publication_Bar_Chart" base64={plots.publication_bar} />
          </div>
          <img
            src={`data:image/png;base64,${plots.publication_bar}`}
            alt="Publication bar chart"
            className="max-w-full rounded"
          />
        </CollapsibleSection>
      )}

      {plots?.bar && !plots?.publication_bar && (
        <CollapsibleSection title="Bar Chart" defaultOpen={true}>
          <div className="flex justify-end mb-2">
            <FigureDownloadMenu title="Bar_Chart" base64={plots.bar} />
          </div>
          <img
            src={`data:image/png;base64,${plots.bar}`}
            alt="Bar chart"
            className="max-w-full rounded"
          />
        </CollapsibleSection>
      )}

      {/* Publishable HTML Tables */}
      {(() => {
        const htmlTables = (adapted.html_tables as Record<string, string>) ||
          generatePublishableHtmlTables(adapted);
        const hasHtmlTables = htmlTables && Object.keys(htmlTables).length > 0;
        return hasHtmlTables ? (
          <CollapsibleSection title="Publishable Tables (Word Export)" defaultOpen={false}>
            <HtmlTablesSection htmlTables={htmlTables} />
          </CollapsibleSection>
        ) : null;
      })()}

      {/* Technical JSON toggle */}
      <TechnicalJSON data={result} />
    </div>
  );
}
