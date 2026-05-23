import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table2 } from "lucide-react";
import { getSignificanceStars, formatPValue, SignificanceLegend } from "./SignificanceStars";
import { TableDownloadMenu } from "./TableDownloadMenu";

interface Props {
  anovaData: unknown;
  grandMean?: number | null;
  cvPercent?: number | null;
}

function fmt(v: unknown, decimals = 3): string {
  if (v == null) return "—";
  const n = typeof v === "number" ? v : parseFloat(String(v));
  if (isNaN(n)) return String(v);
  return n.toFixed(decimals);
}

function fmtInt(v: unknown): string {
  if (v == null) return "—";
  const n = typeof v === "number" ? v : parseFloat(String(v));
  if (isNaN(n)) return String(v);
  return Number.isInteger(n) ? String(n) : n.toFixed(0);
}

function normalizeRows(data: unknown): { headers: string[]; rows: Record<string, unknown>[] } | null {
  if (Array.isArray(data) && data.length > 0) {
    if (Array.isArray(data[0])) {
      const headers = (data[0] as unknown[]).map(String);
      const rows = data.slice(1).map((row: unknown) => {
        const r: Record<string, unknown> = {};
        (row as unknown[]).forEach((cell, i) => { r[headers[i]] = cell; });
        return r;
      });
      return { headers, rows };
    }
    if (typeof data[0] === "object" && data[0] !== null) {
      const headers = Object.keys(data[0] as Record<string, unknown>);
      return { headers, rows: data as Record<string, unknown>[] };
    }
  }
  if (data && typeof data === "object" && !Array.isArray(data)) {
    const entries = Object.entries(data as Record<string, unknown>);
    if (entries.length === 0) return null;
    const headers = ["Source", ...entries.map(([k]) => k)];
    const row: Record<string, unknown> = { Source: "Value" };
    entries.forEach(([k, v]) => { row[k] = v; });
    return { headers, rows: [row] };
  }
  return null;
}

const COL_MAP: Record<string, string> = {
  source: "Source", Source: "Source",
  df: "DF", Df: "DF", DF: "DF",
  sum_sq: "SS", "Sum Sq": "SS", SS: "SS",
  mean_sq: "MS", "Mean Sq": "MS", MS: "MS",
  f_value: "F value", "F value": "F value", F: "F value", "F-value": "F value",
  p_value: "P value", "Pr(>F)": "P value", "p-value": "P value", "P-value": "P value", pvalue: "P value",
};

function standardizeSource(src: string): string {
  const s = src.trim().toLowerCase();
  if (/^(genotype|treatment|variety|cultivar|factor)$/i.test(s)) return "Genotype";
  if (/^(block|rep|replication|replicate)$/i.test(s)) return "Block (Replication)";
  if (/^(residual|error|within)$/i.test(s)) return "Error";
  if (/^total$/i.test(s)) return "Total";
  return src;
}

const DESIRED_ORDER = ["Source", "DF", "SS", "MS", "F value", "P value", "Sig."];

function isInterceptLabel(value: unknown): boolean {
  const label = String(value ?? "").trim().toLowerCase();
  return label === "(intercept)" || label === "intercept";
}

export function EnhancedAnovaTable({ anovaData, grandMean, cvPercent }: Props) {
  const parsed = normalizeRows(anovaData);
  if (!parsed) return null;

  const mappedHeaders = parsed.headers.map((h) => COL_MAP[h] || h);
  const pIdx = mappedHeaders.indexOf("P value");

  const orderedHeaders = DESIRED_ORDER.filter((h) => {
    if (h === "Sig.") return pIdx >= 0;
    return mappedHeaders.includes(h);
  });
  mappedHeaders.forEach((h) => {
    if (!orderedHeaders.includes(h) && h !== "Sig.") orderedHeaders.push(h);
  });

  const filteredRows = parsed.rows.filter((row) => {
    const mapped: Record<string, unknown> = {};
    parsed.headers.forEach((origH) => { mapped[COL_MAP[origH] || origH] = row[origH]; });
    return !isInterceptLabel(mapped["Source"]);
  });

  // Build download data
  const dlHeaders = orderedHeaders.filter((h) => h !== "Sig.");
  const dlRows = filteredRows.map((row) => {
    const mapped: Record<string, unknown> = {};
    parsed.headers.forEach((origH) => { mapped[COL_MAP[origH] || origH] = row[origH]; });
    const sourceLabel = mapped["Source"] ? standardizeSource(String(mapped["Source"])) : "";
    return dlHeaders.map((h) => {
      if (h === "Source") return sourceLabel;
      if (h === "DF") return fmtInt(mapped[h]);
      if (h === "P value") return formatPValue(mapped[h]);
      return fmt(mapped[h]);
    });
  });

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-3">
            <Table2 className="w-6 h-6 text-primary" />
            Analysis of Variance (ANOVA)
          </CardTitle>
          <TableDownloadMenu title="ANOVA_Table" headers={dlHeaders} rows={dlRows} />
        </div>
      </CardHeader>
      <CardContent>
        {(grandMean != null || cvPercent != null) && (
          <div className="flex flex-wrap gap-4 mb-5 p-3 rounded-lg bg-muted/40 border border-border">
            {grandMean != null && (
              <div className="flex flex-col">
                <span className="text-xs text-muted-foreground font-medium uppercase tracking-wide">Grand Mean</span>
                <span className="text-lg font-semibold text-foreground font-mono">{fmt(grandMean)}</span>
              </div>
            )}
            {cvPercent != null && (
              <div className="flex flex-col">
                <span className="text-xs text-muted-foreground font-medium uppercase tracking-wide">CV (%)</span>
                <span className="text-lg font-semibold text-foreground font-mono">{Number(cvPercent).toFixed(1)}%</span>
              </div>
            )}
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse font-mono">
            <thead>
              <tr className="border-b-2 border-foreground/30">
                {orderedHeaders.map((h) => (
                  <th key={h} className={`px-4 py-2.5 font-semibold text-foreground ${h === "Source" ? "text-left" : "text-right"}`}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((row, ri) => {
                const mapped: Record<string, unknown> = {};
                parsed.headers.forEach((origH) => { mapped[COL_MAP[origH] || origH] = row[origH]; });
                const pVal = mapped["P value"];
                const sigStars = pIdx >= 0 ? getSignificanceStars(pVal) : "";
                const sourceLabel = mapped["Source"] ? standardizeSource(String(mapped["Source"])) : "";
                const isErrorOrTotal = /error|total|residual/i.test(sourceLabel);

                return (
                  <tr key={ri} className="border-b border-border/50 hover:bg-muted/20">
                    {orderedHeaders.map((h) => {
                      if (h === "Sig.") return <td key={h} className="px-4 py-2.5 text-right font-bold text-primary whitespace-nowrap">{sigStars}</td>;
                      if (h === "Source") return <td key={h} className="px-4 py-2.5 text-left font-medium text-foreground">{sourceLabel}</td>;
                      if (h === "DF") return <td key={h} className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">{fmtInt(mapped[h])}</td>;
                      if (h === "P value") return <td key={h} className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">{isErrorOrTotal ? "—" : formatPValue(mapped[h])}</td>;
                      if (h === "F value") return <td key={h} className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">{isErrorOrTotal ? "—" : fmt(mapped[h])}</td>;
                      return <td key={h} className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">{fmt(mapped[h])}</td>;
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <SignificanceLegend />
      </CardContent>
    </Card>
  );
}
