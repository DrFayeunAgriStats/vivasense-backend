import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Users } from "lucide-react";
import { TableDownloadMenu } from "./TableDownloadMenu";

interface Props {
  meansData: unknown;
  lettersData: unknown;
  mse?: number | null;
  nReps?: number | null;
}

function fmt(v: unknown, decimals = 3): string {
  if (v == null) return "—";
  const n = typeof v === "number" ? v : parseFloat(String(v));
  if (isNaN(n)) return String(v);
  return n.toFixed(decimals);
}

function computeSE(mse: number | null | undefined, r: number | null | undefined): number | null {
  if (mse == null || r == null || r <= 0) return null;
  return Math.sqrt(mse / r);
}

function buildMergedRows(
  means: unknown,
  letters: unknown,
  computedSE: number | null,
): { headers: string[]; rows: Record<string, string>[] } | null {
  const meansArr = Array.isArray(means) && means.length > 0 && typeof means[0] === "object" ? means as Record<string, unknown>[] : null;
  const lettersArr = Array.isArray(letters) && letters.length > 0 && typeof letters[0] === "object" ? letters as Record<string, unknown>[] : null;

  if (lettersArr) {
    const cols = Object.keys(lettersArr[0]);
    const treatmentCol = cols.find((c) => /treatment|genotype|group|factor|variety/i.test(c)) || cols[0];
    const meanCol = cols.find((c) => /mean/i.test(c));
    const seCol = cols.find((c) => /se$|std_err|std_error|standard_error/i.test(c));
    const letterCol = cols.find((c) => /letter|group|tukey/i.test(c) && !/genotype|treatment/i.test(c));

    const headers = ["Genotype", "Mean", "SE", "Tukey Group"];
    const rows = lettersArr.map((row) => {
      const rawSE = row[seCol ?? "se"] ?? row["se"] ?? row["std_err"];
      const seValue = rawSE != null ? fmt(rawSE) : (computedSE != null ? fmt(computedSE) : "—");
      return {
        Genotype: String(row[treatmentCol] ?? ""),
        Mean: fmt(row[meanCol ?? "mean"]),
        SE: seValue,
        "Tukey Group": String(row[letterCol ?? "letter"] ?? row["letters"] ?? ""),
      };
    });
    return { headers, rows };
  }

  if (meansArr) {
    const cols = Object.keys(meansArr[0]);
    const treatmentCol = cols.find((c) => /treatment|genotype|group|factor|variety/i.test(c)) || cols[0];
    const meanCol = cols.find((c) => /mean/i.test(c));
    const seCol = cols.find((c) => /se$|std_err|std_error/i.test(c));

    const headers = ["Genotype", "Mean", "SE"];
    const rows = meansArr.map((row) => {
      const rawSE = row[seCol ?? "se"] ?? row["std_err"];
      const seValue = rawSE != null ? fmt(rawSE) : (computedSE != null ? fmt(computedSE) : "—");
      return {
        Genotype: String(row[treatmentCol] ?? ""),
        Mean: fmt(row[meanCol ?? "mean"]),
        SE: seValue,
      };
    });
    return { headers, rows };
  }

  const arr = (letters || means) as unknown;
  if (Array.isArray(arr) && arr.length > 1 && Array.isArray(arr[0])) {
    const hdrs = (arr[0] as unknown[]).map(String);
    const dataRows = arr.slice(1).map((r: unknown) => {
      const row: Record<string, string> = {};
      (r as unknown[]).forEach((cell, i) => { row[hdrs[i]] = String(cell ?? ""); });
      return row;
    });
    return { headers: hdrs, rows: dataRows };
  }

  return null;
}

export function EnhancedMeanSeparation({ meansData, lettersData, mse, nReps }: Props) {
  const computedSE = computeSE(mse, nReps);
  const parsed = buildMergedRows(meansData, lettersData, computedSE);
  if (!parsed) return null;

  const dlRows = parsed.rows.map((row) => parsed.headers.map((h) => row[h] ?? "—"));

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-3">
            <Users className="w-6 h-6 text-primary" />
            Mean Separation (Tukey's HSD)
          </CardTitle>
          <TableDownloadMenu title="Mean_Separation_Tukey" headers={parsed.headers} rows={dlRows} />
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse font-mono">
            <thead>
              <tr className="border-b-2 border-foreground/30">
                {parsed.headers.map((h) => (
                  <th key={h} className={`px-4 py-2.5 font-semibold text-foreground ${h === "Genotype" || h === "Tukey Group" ? "text-left" : "text-right"}`}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {parsed.rows.map((row, ri) => (
                <tr key={ri} className="border-b border-border/50 hover:bg-muted/20">
                  {parsed.headers.map((h) => (
                    <td key={h} className={`px-4 py-2.5 text-muted-foreground ${
                      h === "Genotype" ? "text-left font-medium text-foreground" :
                      h === "Tukey Group" ? "text-left font-bold text-primary" :
                      "text-right tabular-nums"
                    }`}>
                      {row[h] ?? "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {computedSE != null && (
          <p className="text-xs text-muted-foreground mt-2">
            SE = √(MSE/r) = √({fmt(mse!)}/{nReps}) = {fmt(computedSE)}
          </p>
        )}
        <p className="text-xs text-muted-foreground mt-1 italic">
          Means followed by the same letter are not significantly different (Tukey's HSD, α = 0.05).
        </p>
      </CardContent>
    </Card>
  );
}
