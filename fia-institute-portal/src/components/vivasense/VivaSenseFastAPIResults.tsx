import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Trash2, Info, Download } from "lucide-react";
import { HtmlTablesSection } from "./HtmlTablesSection";
import { generatePublishableHtmlTables } from "./utils/generatePublishableTables";
import { ExperimentSummary } from "./results/ExperimentSummary";
import { DescriptiveStatsTable } from "./results/DescriptiveStatsTable";
import { EnhancedAnovaTable } from "./results/EnhancedAnovaTable";
import { EnhancedMeanSeparation } from "./results/EnhancedMeanSeparation";
import { PublicationPlots } from "./results/PublicationPlots";
import { ExportButtons } from "./results/ExportButtons";
import { GenericDataTable } from "./results/GenericDataTable";

export interface FastAPIResultsData {
  status?: string;
  design?: string;
  response?: string;
  treatment?: string;
  error?: string;
  formula?: string;
  grand_mean?: number;
  cv_percent?: number;
  anova?: Record<string, unknown>;
  means?: Record<string, unknown>;
  letters?: Record<string, unknown>;
  effect_sizes?: Record<string, unknown>;
  assumptions?: Record<string, unknown>;
  descriptive_stats?: Record<string, unknown>;
  plots?: Record<string, string>;
  html_tables?: Record<string, string>;
  n_treatments?: number;
  n_reps?: number;
  n_observations?: number;
  meta?: Record<string, unknown>;
  tables?: Record<string, unknown>;
  interpretation?: string;
}

interface Props {
  results: FastAPIResultsData;
  onClear: () => void;
}

function isNonEmpty(val: unknown): boolean {
  if (val == null) return false;
  if (Array.isArray(val)) return val.length > 0;
  if (typeof val === "object") return Object.keys(val as Record<string, unknown>).length > 0;
  return false;
}

const META_KEYS = ["status", "design", "response", "treatment", "formula"];
const EXTRA_TABLE_KEYS = ["effect_sizes", "assumptions"];
const EXTRA_LABELS: Record<string, string> = {
  effect_sizes: "Effect Sizes",
  assumptions: "Assumption Tests",
};

/** Extract MSE from ANOVA data for SE computation */
function extractMSE(anovaData: unknown): number | null {
  if (!Array.isArray(anovaData)) return null;
  for (const row of anovaData) {
    if (typeof row !== "object" || !row) continue;
    const r = row as Record<string, unknown>;
    const src = String(r.source ?? r.Source ?? "").toLowerCase();
    if (/error|residual|within/.test(src)) {
      const ms = r.mean_sq ?? r.MS ?? r["Mean Sq"] ?? r.ms;
      if (ms != null) return Number(ms);
    }
  }
  return null;
}

/** Count unique treatments and replications from data */
function extractCounts(results: FastAPIResultsData): {
  nTreatments: number | null;
  nReps: number | null;
  nObservations: number | null;
} {
  let nTreatments = results.n_treatments ?? (results.meta?.n_treatments as number | undefined) ?? null;
  let nReps = results.n_reps ?? (results.meta?.n_reps as number | undefined) ?? null;
  let nObservations = results.n_observations ?? (results.meta?.n_observations as number | undefined) ?? null;

  // Try to infer from means/letters data
  if (nTreatments == null) {
    const meansArr = results.means ?? results.letters;
    if (Array.isArray(meansArr)) nTreatments = meansArr.length;
  }

  // Try to infer n_obs from ANOVA total DF + 1
  if (nObservations == null && Array.isArray(results.anova)) {
    for (const row of results.anova) {
      if (typeof row !== "object" || !row) continue;
      const r = row as Record<string, unknown>;
      const src = String(r.source ?? r.Source ?? "").toLowerCase();
      if (/total/.test(src)) {
        const df = Number(r.df ?? r.DF ?? r.Df);
        if (!isNaN(df)) nObservations = df + 1;
      }
    }
  }

  // Infer reps if we have treatments and observations
  if (nReps == null && nTreatments != null && nObservations != null && nTreatments > 0) {
    nReps = Math.round(nObservations / nTreatments);
  }

  return { nTreatments, nReps, nObservations };
}

export function VivaSenseFastAPIResults({ results, onClear }: Props) {
  const metaEntries: Array<[string, unknown]> = META_KEYS
    .filter((k) => (results as Record<string, unknown>)[k] != null && (results as Record<string, unknown>)[k] !== "")
    .map((k) => [k, (results as Record<string, unknown>)[k]]);

  if (results.grand_mean != null) metaEntries.push(["grand_mean", Number(results.grand_mean).toFixed(3)]);
  if (results.cv_percent != null) metaEntries.push(["cv_percent", `${Number(results.cv_percent).toFixed(1)}%`]);

  if (metaEntries.length === 0 && results.meta && isNonEmpty(results.meta)) {
    Object.entries(results.meta).forEach(([k, v]) => {
      if (v != null && v !== "") metaEntries.push([k, v]);
    });
  }

  const grandMean = results.grand_mean ?? (results.meta?.grand_mean as number | undefined) ?? null;
  const cvPercent = results.cv_percent ?? (results.meta?.cv_percent as number | undefined) ?? null;
  const anovaData = results.anova ?? results.tables?.anova ?? null;
  const meansData = results.means ?? results.tables?.means ?? null;
  const lettersData = results.letters ?? results.tables?.letters ?? null;
  const descStats = results.descriptive_stats ?? results.tables?.descriptive_stats ?? null;
  const plots = results.plots;

  const mse = extractMSE(anovaData);
  const { nTreatments, nReps, nObservations } = extractCounts(results);

  const extraTables = EXTRA_TABLE_KEYS
    .filter((k) => isNonEmpty((results as Record<string, unknown>)[k]) || isNonEmpty(results.tables?.[k]))
    .map((k) => ({
      key: k,
      label: EXTRA_LABELS[k] || k.replace(/_/g, " "),
      data: (results as Record<string, unknown>)[k] || results.tables?.[k],
    }));

  const htmlTables = results.html_tables && Object.keys(results.html_tables).length > 0
    ? results.html_tables
    : generatePublishableHtmlTables(results as Record<string, unknown>);
  const hasHtmlTables = htmlTables && Object.keys(htmlTables).length > 0;

  return (
    <section className="py-20 bg-muted/30" id="results">
      <div className="container-wide">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center mb-8">
            <h2 className="font-serif text-3xl lg:text-4xl font-bold text-foreground mb-4">
              Analysis Results
            </h2>
            <p className="text-muted-foreground mb-4">Journal-ready statistical output</p>
            <Button variant="outline" size="sm" onClick={onClear}>
              <Trash2 className="w-4 h-4 mr-2" />
              Clear Results
            </Button>
          </div>

          {/* Backend error notice */}
          {results.error && (
            <Card className="border-destructive/50">
              <CardContent className="pt-6">
                <p className="text-sm text-destructive flex items-center gap-2">
                  <Info className="w-4 h-4" />
                  Backend note: {String(results.error)}
                </p>
              </CardContent>
            </Card>
          )}

          {/* 1. Experiment Summary */}
          <ExperimentSummary
            metaEntries={metaEntries}
            nTreatments={nTreatments}
            nReps={nReps}
            nObservations={nObservations}
          />

          {/* 2. Descriptive Statistics */}
          {isNonEmpty(descStats) && <DescriptiveStatsTable data={descStats} />}

          {/* 3. ANOVA Table */}
          {isNonEmpty(anovaData) && (
            <EnhancedAnovaTable
              anovaData={anovaData}
              grandMean={grandMean}
              cvPercent={cvPercent}
            />
          )}

          {/* 4. Mean Separation */}
          {(isNonEmpty(meansData) || isNonEmpty(lettersData)) && (
            <EnhancedMeanSeparation
              meansData={meansData}
              lettersData={lettersData}
              mse={mse}
              nReps={nReps}
            />
          )}

          {/* 5. Additional tables */}
          {extraTables.map(({ key, label, data }) => (
            <GenericDataTable key={key} label={label} data={data} />
          ))}

          {/* 6. Figures */}
          {plots && Object.keys(plots).length > 0 && <PublicationPlots plots={plots} />}

          {/* 7. Publishable HTML Tables */}
          {hasHtmlTables && <HtmlTablesSection htmlTables={htmlTables!} />}

          {/* Legacy interpretation */}
          {results.interpretation && (
            <Card>
              <CardContent className="pt-6">
                <p className="text-muted-foreground whitespace-pre-wrap leading-relaxed">
                  {results.interpretation}
                </p>
              </CardContent>
            </Card>
          )}

          {/* 8. Download */}
          <Card>
            <CardContent className="pt-6">
              <h3 className="font-serif text-lg font-bold text-foreground mb-4 flex items-center gap-2">
                <Download className="w-5 h-5 text-primary" />
                Download Results
              </h3>
              <ExportButtons results={results} />
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}
