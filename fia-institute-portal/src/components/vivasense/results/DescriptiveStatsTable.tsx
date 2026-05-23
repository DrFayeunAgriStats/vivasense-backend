import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart3 } from "lucide-react";
import { TableDownloadMenu } from "./TableDownloadMenu";

interface Props {
  data: unknown;
}

function fmt(v: unknown, decimals = 3): string {
  if (v == null) return "—";
  const n = typeof v === "number" ? v : parseFloat(String(v));
  if (typeof n === "number" && !isNaN(n)) {
    return n.toFixed(decimals);
  }
  return String(v);
}

export function DescriptiveStatsTable({ data }: Props) {
  if (!data) return null;

  if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object" && data[0] !== null) {
    const headers = Object.keys(data[0] as Record<string, unknown>);
    const dlRows = (data as Record<string, unknown>[]).map((row) => headers.map((h) => {
      const v = row[h];
      return typeof v === "number" ? fmt(v) : String(v ?? "—");
    }));

    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-3">
              <BarChart3 className="w-6 h-6 text-primary" />
              Descriptive Statistics
            </CardTitle>
            <TableDownloadMenu title="Descriptive_Statistics" headers={headers.map((h) => h.replace(/_/g, " "))} rows={dlRows} />
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse font-mono">
              <thead>
                <tr className="border-b-2 border-foreground/30">
                  {headers.map((h) => (
                    <th key={h} className="px-4 py-2.5 font-semibold text-foreground text-right first:text-left">
                      {h.replace(/_/g, " ")}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(data as Record<string, unknown>[]).map((row, ri) => (
                  <tr key={ri} className="border-b border-border/50 hover:bg-muted/20">
                    {headers.map((h, ci) => (
                      <td key={h} className={`px-4 py-2.5 text-muted-foreground ${ci === 0 ? "text-left font-medium text-foreground" : "text-right tabular-nums"}`}>
                        {ci === 0 ? String(row[h] ?? "") : fmt(row[h])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (typeof data === "object" && !Array.isArray(data)) {
    const entries = Object.entries(data as Record<string, unknown>).filter(([, v]) => v != null);
    if (entries.length === 0) return null;
    const dlRows = entries.map(([k, v]) => [k.replace(/_/g, " "), typeof v === "number" ? fmt(v) : String(v ?? "—")]);

    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-3">
              <BarChart3 className="w-6 h-6 text-primary" />
              Descriptive Statistics
            </CardTitle>
            <TableDownloadMenu title="Descriptive_Statistics" headers={["Statistic", "Value"]} rows={dlRows} />
          </div>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse font-mono">
              <thead>
                <tr className="border-b-2 border-foreground/30">
                  <th className="px-4 py-2.5 font-semibold text-foreground text-left">Statistic</th>
                  <th className="px-4 py-2.5 font-semibold text-foreground text-right">Value</th>
                </tr>
              </thead>
              <tbody>
                {entries.map(([k, v]) => (
                  <tr key={k} className="border-b border-border/50 hover:bg-muted/20">
                    <td className="px-4 py-2.5 text-foreground font-medium">{k.replace(/_/g, " ")}</td>
                    <td className="px-4 py-2.5 text-muted-foreground text-right tabular-nums">{fmt(v)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    );
  }

  return null;
}
