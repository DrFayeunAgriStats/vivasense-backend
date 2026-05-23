import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, ArrowRight, Target } from "lucide-react";
import { TableDownloadMenu } from "../results/TableDownloadMenu";

const fmt4 = (v: unknown): string =>
  v == null || v === "" ? "—" : Number(v).toFixed(4);
const fmt2 = (v: unknown): string =>
  v == null || v === "" ? "—" : Number(v).toFixed(2);
const asArray = <T,>(v: unknown): T[] => (Array.isArray(v) ? (v as T[]) : []);
const asObj = (v: unknown): Record<string, unknown> =>
  v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};

function classifyCoefficient(val: number): string {
  const abs = Math.abs(val);
  if (abs >= 0.6) return "text-emerald-700 dark:text-emerald-400 font-bold";
  if (abs >= 0.3) return "text-amber-700 dark:text-amber-400 font-semibold";
  return "text-muted-foreground";
}

function classifyVIF(vif: number): { label: string; className: string } {
  if (vif >= 10) return { label: "Severe", className: "bg-destructive/10 text-destructive border-destructive/30" };
  if (vif >= 5) return { label: "Moderate", className: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300 border-amber-300" };
  return { label: "OK", className: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300 border-emerald-300" };
}

interface PathCoefficient {
  trait?: string;
  variable?: string;
  predictor?: string;
  direct_effect?: number;
  indirect_effect?: number;
  indirect_effects?: Record<string, number>;
  total_effect?: number;
  correlation?: number;
  vif?: number;
  [key: string]: unknown;
}

interface Props {
  data: Record<string, unknown> | PathCoefficient[];
  traitName?: string;
}

function normalizePathData(data: Record<string, unknown> | PathCoefficient[]): {
  coefficients: PathCoefficient[];
  r2?: number;
  residual?: number;
  dependentVar?: string;
} {
  if (Array.isArray(data)) {
    return { coefficients: data };
  }

  const coefficients = (data.coefficients ?? data.path_coefficients ?? data.direct_effects ?? data.effects ?? []) as PathCoefficient[];
  const r2 = data.r_squared ?? data.R2 ?? data.r2;
  const residual = data.residual_effect ?? data.residual;
  const dependentVar = data.dependent_variable ?? data.response;

  // If coefficients is a dict of { trait: direct_effect }, convert
  if (!Array.isArray(coefficients) && typeof coefficients === "object") {
    const entries = Object.entries(coefficients as Record<string, unknown>);
    return {
      coefficients: entries.map(([key, val]) => ({
        trait: key,
        direct_effect: typeof val === "number" ? val : (val as any)?.direct_effect,
        indirect_effect: typeof val === "number" ? undefined : (val as any)?.indirect_effect,
        total_effect: typeof val === "number" ? undefined : (val as any)?.total_effect,
        correlation: typeof val === "number" ? undefined : (val as any)?.correlation,
        vif: typeof val === "number" ? undefined : (val as any)?.vif,
      })),
      r2: r2 as number | undefined,
      residual: residual as number | undefined,
      dependentVar: dependentVar as string | undefined,
    };
  }

  return {
    coefficients: Array.isArray(coefficients) ? coefficients : [],
    r2: r2 as number | undefined,
    residual: residual as number | undefined,
    dependentVar: dependentVar as string | undefined,
  };
}

export function PathAnalysisDisplay({ data, traitName }: Props) {
  const { coefficients: rawCoefficients, r2, residual, dependentVar } = normalizePathData(data);
  const coefficients = asArray<PathCoefficient>(rawCoefficients);

  if (coefficients.length === 0) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground">
          Path analysis data unavailable for this trait.
        </CardContent>
      </Card>
    );
  }

  const hasVIF = coefficients.some((c) => c.vif != null);
  const hasIndirect = coefficients.some((c) => c.indirect_effect != null);
  const hasCorrelation = coefficients.some((c) => c.correlation != null);

  const headers = [
    "Predictor",
    "Direct Effect",
    ...(hasIndirect ? ["Indirect Effect"] : []),
    ...(hasCorrelation ? ["Correlation (r)"] : []),
    "Total Effect",
    ...(hasVIF ? ["VIF", "Status"] : []),
  ];

  const rows = coefficients.map((c) => {
    const name = c.trait ?? c.variable ?? c.predictor ?? "—";
    const direct = c.direct_effect;
    const indirect = c.indirect_effect;
    const total = c.total_effect ?? (direct != null && indirect != null ? direct + indirect : direct);
    const corr = c.correlation;
    const vif = c.vif;

    const row: React.ReactNode[] = [
      name,
      <span className={direct != null ? classifyCoefficient(Number(direct)) : "text-muted-foreground"}>
        {fmt4(direct)}
      </span>,
    ];

    if (hasIndirect) {
      row.push(
        <span className={indirect != null ? classifyCoefficient(Number(indirect)) : "text-muted-foreground"}>
          {fmt4(indirect)}
        </span>
      );
    }

    if (hasCorrelation) {
      row.push(
        <span className={corr != null ? classifyCoefficient(Number(corr)) : "text-muted-foreground"}>
          {fmt4(corr)}
        </span>
      );
    }

    row.push(
      <span className={total != null ? classifyCoefficient(Number(total)) : "text-muted-foreground"}>
        {fmt4(total)}
      </span>
    );

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

  // Build downloadable rows (plain text)
  const dlRows = rows.map((row) =>
    row.map((cell) => {
      if (cell == null) return "—";
      if (typeof cell === "object" && "props" in (cell as any)) {
        const props = (cell as any).props;
        if (props?.children != null) {
          if (typeof props.children === "string") return props.children;
          return String(props.children);
        }
        return "";
      }
      return String(cell);
    })
  );

  const highVIFTraits = coefficients.filter((c) => c.vif != null && Number(c.vif) >= 5);

  return (
    <div className="space-y-4">
      {/* Summary metrics */}
      {(r2 != null || residual != null || dependentVar) && (
        <div className="flex flex-wrap gap-3 items-center">
          {dependentVar && (
            <Badge variant="secondary" className="gap-1">
              <Target className="w-3 h-3" /> Dependent: {String(dependentVar)}
            </Badge>
          )}
          {r2 != null && (
            <Badge variant="outline">R² = {fmt4(r2)} ({(Number(r2) * 100).toFixed(1)}%)</Badge>
          )}
          {residual != null && (
            <Badge variant="outline">Residual = {fmt4(residual)}</Badge>
          )}
        </div>
      )}

      {/* VIF warnings */}
      {highVIFTraits.length > 0 && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-700 p-4">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-amber-800 dark:text-amber-200 mb-1">
                Multicollinearity Warning
              </p>
              <p className="text-sm text-amber-700 dark:text-amber-300">
                {highVIFTraits.length} predictor{highVIFTraits.length > 1 ? "s" : ""} show{highVIFTraits.length === 1 ? "s" : ""} elevated VIF (≥5):
                {" "}{highVIFTraits.map((c) => c.trait ?? c.variable ?? c.predictor).join(", ")}.
                Consider removing redundant predictors or using ridge regression.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Path Coefficients Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-lg">
              <ArrowRight className="w-5 h-5 text-primary" />
              Path Coefficients {traitName ? `— ${traitName}` : ""}
            </CardTitle>
            <TableDownloadMenu title="Path_Coefficients" headers={headers} rows={dlRows} />
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-border">
                  {headers.map((h, i) => (
                    <th
                      key={i}
                      className="text-left px-3 py-2 font-semibold text-foreground bg-muted/50 whitespace-nowrap"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, ri) => (
                  <tr key={ri} className={`border-b border-border/50 ${ri % 2 === 0 ? "" : "bg-muted/20"}`}>
                    {row.map((cell, ci) => (
                      <td
                        key={ci}
                        className={`px-3 py-2 whitespace-nowrap ${ci > 0 ? "text-right font-mono text-xs" : "text-foreground font-medium"}`}
                      >
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted-foreground mt-3 italic">
            Direct effects represent the path coefficient from each predictor to the dependent variable.
            Indirect effects show influence mediated through other predictors.
            {hasVIF && " VIF &gt; 5 indicates moderate multicollinearity; VIF &gt; 10 indicates severe multicollinearity."}
          </p>
        </CardContent>
      </Card>

      {/* Indirect effects decomposition */}
      {coefficients.some((c) => Object.keys(asObj(c.indirect_effects)).length > 0) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Effect Decomposition (Indirect via)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              {coefficients
                .filter((c) => Object.keys(asObj(c.indirect_effects)).length > 0)
                .map((c, idx) => {
                  const name = c.trait ?? c.variable ?? c.predictor ?? `Predictor ${idx + 1}`;
                  const effects = asObj(c.indirect_effects);
                  return (
                    <div key={idx} className="mb-4 last:mb-0">
                      <p className="text-sm font-semibold text-foreground mb-2">{name}</p>
                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                        {Object.entries(effects).map(([via, val]) => (
                          <div
                            key={via}
                            className="rounded-md border border-border p-2 text-center"
                          >
                            <p className="text-xs text-muted-foreground">via {via}</p>
                            <p className={`font-mono text-sm ${classifyCoefficient(Number(val))}`}>
                              {fmt4(val)}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
