import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TableDownloadMenu } from "../results/TableDownloadMenu";

const fmt3 = (v: unknown): string =>
  v == null || v === "" ? "—" : Number(v).toFixed(3);

function correlationColor(val: number): string {
  const abs = Math.abs(val);
  if (abs >= 0.7) return val > 0
    ? "bg-emerald-200 dark:bg-emerald-900/60 text-emerald-900 dark:text-emerald-100"
    : "bg-red-200 dark:bg-red-900/60 text-red-900 dark:text-red-100";
  if (abs >= 0.5) return val > 0
    ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-800 dark:text-emerald-200"
    : "bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200";
  if (abs >= 0.3) return val > 0
    ? "bg-emerald-50 dark:bg-emerald-950/30 text-foreground"
    : "bg-red-50 dark:bg-red-950/30 text-foreground";
  return "text-muted-foreground";
}

function significanceStars(p: unknown): string {
  if (p == null) return "";
  const n = Number(p);
  if (n < 0.001) return "***";
  if (n < 0.01) return "**";
  if (n < 0.05) return "*";
  return "";
}

interface CorrelationEntry {
  trait_1?: string;
  trait_2?: string;
  row?: string;
  col?: string;
  r?: number;
  value?: number;
  coefficient?: number;
  p_value?: number;
  type?: string; // phenotypic, genotypic, environmental
  [key: string]: unknown;
}

interface Props {
  data: Record<string, unknown> | CorrelationEntry[];
  correlationType?: string;
}

function buildMatrix(data: Record<string, unknown> | CorrelationEntry[]): {
  traits: string[];
  matrix: Record<string, Record<string, number>>;
  pValues?: Record<string, Record<string, number>>;
  type?: string;
} {
  // Case 1: Already a matrix (dict of dicts)
  if (!Array.isArray(data) && typeof data === "object") {
    // Check if it's { phenotypic: {...}, genotypic: {...} } wrapper
    const keys = Object.keys(data);
    if (keys.some((k) => ["phenotypic", "genotypic", "environmental"].includes(k.toLowerCase()))) {
      // Use first available correlation type
      const firstType = keys.find((k) => typeof data[k] === "object" && data[k] !== null);
      if (firstType) {
        const inner = data[firstType] as Record<string, unknown>;
        const result = buildMatrix(inner);
        return { ...result, type: firstType };
      }
    }

    // Check if it's a correlation matrix { trait1: { trait2: value } }
    const firstVal = data[keys[0]];
    if (typeof firstVal === "object" && firstVal !== null && !Array.isArray(firstVal)) {
      const traits = keys;
      const matrix: Record<string, Record<string, number>> = {};
      traits.forEach((t1) => {
        matrix[t1] = {};
        const row = data[t1] as Record<string, number>;
        if (row) {
          traits.forEach((t2) => {
            matrix[t1][t2] = row[t2] ?? (t1 === t2 ? 1 : 0);
          });
        }
      });
      return { traits, matrix };
    }

    // Check if it's array-like entries
    if (data.data && Array.isArray(data.data)) {
      return buildMatrix(data.data as CorrelationEntry[]);
    }
  }

  // Case 2: Array of {trait_1, trait_2, r/value} entries
  if (Array.isArray(data)) {
    const traitSet = new Set<string>();
    const matrix: Record<string, Record<string, number>> = {};
    const pValues: Record<string, Record<string, number>> = {};

    (data as CorrelationEntry[]).forEach((entry) => {
      const t1 = entry.trait_1 ?? entry.row ?? "";
      const t2 = entry.trait_2 ?? entry.col ?? "";
      const val = entry.r ?? entry.value ?? entry.coefficient ?? 0;
      const pVal = entry.p_value;

      if (t1 && t2) {
        traitSet.add(t1);
        traitSet.add(t2);
        if (!matrix[t1]) matrix[t1] = {};
        if (!matrix[t2]) matrix[t2] = {};
        matrix[t1][t2] = Number(val);
        matrix[t2][t1] = Number(val);

        if (pVal != null) {
          if (!pValues[t1]) pValues[t1] = {};
          if (!pValues[t2]) pValues[t2] = {};
          pValues[t1][t2] = Number(pVal);
          pValues[t2][t1] = Number(pVal);
        }
      }
    });

    const traits = Array.from(traitSet);
    traits.forEach((t) => {
      if (!matrix[t]) matrix[t] = {};
      matrix[t][t] = 1;
    });

    return { traits, matrix, pValues: Object.keys(pValues).length > 0 ? pValues : undefined };
  }

  return { traits: [], matrix: {} };
}

export function CorrelationMatrixDisplay({ data, correlationType }: Props) {
  const { traits, matrix, pValues, type } = buildMatrix(data);

  if (!traits.length) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground">
          Correlation data unavailable for this analysis.
        </CardContent>
      </Card>
    );
  }

  const displayType = correlationType ?? type;
  const shortTraits = traits.map((t) => t.replace(/_/g, " "));

  // Build downloadable rows
  const headers = ["", ...shortTraits];
  const dlRows = traits.map((t1, i) =>
    [shortTraits[i], ...traits.map((t2) => fmt3(matrix[t1]?.[t2]))]
  );

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-lg">
            {displayType
              ? `${displayType.charAt(0).toUpperCase() + displayType.slice(1)} Correlation Matrix`
              : "Correlation Matrix"}
          </CardTitle>
          <TableDownloadMenu title={`Correlation_Matrix_${displayType ?? ""}`} headers={headers} rows={dlRows} />
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="text-xs border-collapse">
            <thead>
              <tr>
                <th className="px-2 py-1.5 text-left font-semibold text-foreground bg-muted/50 border border-border min-w-[100px]" />
                {shortTraits.map((t, i) => (
                  <th
                    key={i}
                    className="px-2 py-1.5 text-center font-semibold text-foreground bg-muted/50 border border-border min-w-[70px] whitespace-nowrap"
                    title={traits[i]}
                  >
                    {t.length > 15 ? t.slice(0, 13) + "…" : t}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {traits.map((t1, i) => (
                <tr key={i}>
                  <td className="px-2 py-1.5 font-medium text-foreground bg-muted/50 border border-border whitespace-nowrap">
                    {shortTraits[i].length > 15 ? shortTraits[i].slice(0, 13) + "…" : shortTraits[i]}
                  </td>
                  {traits.map((t2, j) => {
                    const val = matrix[t1]?.[t2];
                    const pVal = pValues?.[t1]?.[t2];
                    const isDiagonal = i === j;
                    const stars = significanceStars(pVal);

                    return (
                      <td
                        key={j}
                        className={`px-2 py-1.5 text-center font-mono border border-border ${
                          isDiagonal
                            ? "bg-muted/60 font-bold"
                            : j > i
                            ? correlationColor(val ?? 0)
                            : "bg-muted/20 text-muted-foreground"
                        }`}
                        title={`${traits[i]} × ${traits[j]}: r = ${fmt3(val)}${pVal != null ? `, p = ${fmt3(pVal)}` : ""}`}
                      >
                        {isDiagonal ? "1.000" : fmt3(val)}
                        {!isDiagonal && stars && (
                          <span className="text-primary ml-0.5">{stars}</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-3 mt-4 text-xs text-muted-foreground">
          <span>Significance: * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001</span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded bg-emerald-200 dark:bg-emerald-900/60" /> Strong positive (|r| ≥ 0.7)
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded bg-red-200 dark:bg-red-900/60" /> Strong negative (|r| ≥ 0.7)
          </span>
        </div>

        {/* Highlight strong correlations */}
        {(() => {
          const strong: { t1: string; t2: string; r: number }[] = [];
          for (let i = 0; i < traits.length; i++) {
            for (let j = i + 1; j < traits.length; j++) {
              const val = matrix[traits[i]]?.[traits[j]];
              if (val != null && Math.abs(val) >= 0.5) {
                strong.push({ t1: shortTraits[i], t2: shortTraits[j], r: val });
              }
            }
          }
          if (!strong.length) return null;
          strong.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));

          return (
            <div className="mt-4 rounded-lg border border-primary/20 bg-primary/5 p-3">
              <p className="text-sm font-semibold text-foreground mb-2">Notable Correlations (|r| ≥ 0.5)</p>
              <div className="flex flex-wrap gap-2">
                {strong.slice(0, 8).map((s, i) => (
                  <Badge
                    key={i}
                    variant="outline"
                    className={s.r > 0 ? "border-emerald-400 text-emerald-700 dark:text-emerald-300" : "border-red-400 text-red-700 dark:text-red-300"}
                  >
                    {s.t1} × {s.t2}: {fmt3(s.r)}
                  </Badge>
                ))}
              </div>
            </div>
          );
        })()}
      </CardContent>
    </Card>
  );
}
