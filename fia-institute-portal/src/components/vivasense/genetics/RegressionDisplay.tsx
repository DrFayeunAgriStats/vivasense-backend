import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, TrendingUp } from "lucide-react";
import { TableDownloadMenu } from "../results/TableDownloadMenu";

const fmt4 = (v: unknown): string =>
  v == null || v === "" ? "—" : Number(v).toFixed(4);
const fmt2 = (v: unknown): string =>
  v == null || v === "" ? "—" : Number(v).toFixed(2);

const pDisplay = (p: unknown): string => {
  if (p == null) return "—";
  const n = Number(p);
  if (n < 0.001) return "<0.001 ***";
  if (n < 0.01) return `${fmt4(n)} **`;
  if (n < 0.05) return `${fmt4(n)} *`;
  return fmt4(n) + " ns";
};

function classifyVIF(vif: number): { label: string; className: string } {
  if (vif >= 10) return { label: "Severe", className: "bg-destructive/10 text-destructive border-destructive/30" };
  if (vif >= 5) return { label: "Moderate", className: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300 border-amber-300" };
  return { label: "OK", className: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300 border-emerald-300" };
}

interface RegressionCoefficient {
  variable?: string;
  predictor?: string;
  coefficient?: number;
  beta?: number;
  std_error?: number;
  se?: number;
  t_value?: number;
  t_stat?: number;
  p_value?: number;
  vif?: number;
  [key: string]: unknown;
}

interface RegressionSummary {
  r_squared?: number;
  adj_r_squared?: number;
  f_statistic?: number;
  f_pvalue?: number;
  n?: number;
  aic?: number;
  bic?: number;
  dependent_variable?: string;
  [key: string]: unknown;
}

interface Props {
  data: Record<string, unknown>;
}

export function RegressionDisplay({ data }: Props) {
  // Extract coefficients
  const coefficients = (data.coefficients ?? data.parameters ?? data.betas ?? []) as RegressionCoefficient[] | Record<string, unknown>;
  const summary = (data.summary ?? data.model_summary ?? data) as RegressionSummary;

  // Normalize coefficients to array
  let coeffArray: RegressionCoefficient[];
  if (Array.isArray(coefficients)) {
    coeffArray = coefficients;
  } else if (typeof coefficients === "object" && coefficients) {
    coeffArray = Object.entries(coefficients).map(([key, val]) => {
      if (typeof val === "number") return { variable: key, coefficient: val };
      return { variable: key, ...(val as Record<string, unknown>) };
    });
  } else {
    coeffArray = [];
  }

  if (!coeffArray.length && !summary.r_squared) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground">
          Regression results unavailable.
        </CardContent>
      </Card>
    );
  }

  const hasVIF = coeffArray.some((c) => c.vif != null);
  const highVIF = coeffArray.filter((c) => c.vif != null && Number(c.vif) >= 5);

  const headers = [
    "Variable",
    "Coefficient",
    "Std. Error",
    "t-value",
    "p-value",
    ...(hasVIF ? ["VIF", "Status"] : []),
  ];

  const rows = coeffArray.map((c) => {
    const name = c.variable ?? c.predictor ?? "—";
    const coef = c.coefficient ?? c.beta;
    const se = c.std_error ?? c.se;
    const t = c.t_value ?? c.t_stat;
    const p = c.p_value;
    const vif = c.vif;

    const row: React.ReactNode[] = [
      <span className="font-medium text-foreground">{name}</span>,
      <span className="font-mono">{fmt4(coef)}</span>,
      <span className="font-mono">{fmt4(se)}</span>,
      <span className="font-mono">{fmt2(t)}</span>,
      <span className="font-mono">{pDisplay(p)}</span>,
    ];

    if (hasVIF) {
      if (vif != null) {
        const vifClass = classifyVIF(Number(vif));
        row.push(<span className="font-mono">{fmt2(vif)}</span>);
        row.push(<Badge variant="outline" className={vifClass.className}>{vifClass.label}</Badge>);
      } else {
        row.push("—");
        row.push("—");
      }
    }

    return row;
  });

  const dlRows = rows.map((row) =>
    row.map((cell) => {
      if (cell == null) return "—";
      if (typeof cell === "object" && "props" in (cell as any)) {
        const props = (cell as any).props;
        return typeof props?.children === "string" ? props.children : String(props?.children ?? "");
      }
      return String(cell);
    })
  );

  return (
    <div className="space-y-4">
      {/* Model Summary */}
      {(summary.r_squared != null || summary.f_statistic != null) && (
        <Card className="border-primary/20">
          <CardContent className="p-4">
            <div className="flex flex-wrap gap-4 items-center">
              <TrendingUp className="w-5 h-5 text-primary" />
              {summary.dependent_variable && (
                <Badge variant="secondary">Response: {summary.dependent_variable}</Badge>
              )}
              {summary.r_squared != null && (
                <div className="text-sm">
                  <span className="font-semibold text-foreground">R² = </span>
                  <span className="font-mono text-foreground">{fmt4(summary.r_squared)}</span>
                  <span className="text-muted-foreground ml-1">({(Number(summary.r_squared) * 100).toFixed(1)}%)</span>
                </div>
              )}
              {summary.adj_r_squared != null && (
                <div className="text-sm">
                  <span className="font-semibold text-foreground">Adj. R² = </span>
                  <span className="font-mono text-foreground">{fmt4(summary.adj_r_squared)}</span>
                </div>
              )}
              {summary.f_statistic != null && (
                <div className="text-sm">
                  <span className="font-semibold text-foreground">F = </span>
                  <span className="font-mono text-foreground">{fmt2(summary.f_statistic)}</span>
                  {summary.f_pvalue != null && (
                    <span className="text-muted-foreground ml-1">({pDisplay(summary.f_pvalue)})</span>
                  )}
                </div>
              )}
              {summary.n != null && <Badge variant="outline">n = {summary.n}</Badge>}
            </div>
          </CardContent>
        </Card>
      )}

      {/* VIF warning */}
      {highVIF.length > 0 && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-700 p-4">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-amber-800 dark:text-amber-200 mb-1">
                Multicollinearity Detected
              </p>
              <p className="text-sm text-amber-700 dark:text-amber-300">
                {highVIF.map((c) => c.variable ?? c.predictor).join(", ")} ha{highVIF.length > 1 ? "ve" : "s"} VIF ≥ 5.
                Regression coefficients may be unstable. Consider variable selection or regularization.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Coefficients Table */}
      {coeffArray.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">Regression Coefficients</CardTitle>
              <TableDownloadMenu title="Regression_Coefficients" headers={headers} rows={dlRows} />
            </div>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b border-border">
                    {headers.map((h, i) => (
                      <th key={i} className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50 whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, ri) => (
                    <tr key={ri} className={`border-b border-border/50 ${ri % 2 === 0 ? "" : "bg-muted/20"}`}>
                      {row.map((cell, ci) => (
                        <td key={ci} className={`px-3 py-2 whitespace-nowrap ${ci > 0 ? "text-right text-xs" : ""}`}>
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-muted-foreground mt-3 italic">
              Significance: * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001, ns = not significant.
              {hasVIF && " VIF &gt; 5 indicates moderate multicollinearity; VIF &gt; 10 indicates severe."}
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
