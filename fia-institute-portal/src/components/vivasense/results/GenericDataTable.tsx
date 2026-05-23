import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart3 } from "lucide-react";
import { TableDownloadMenu } from "./TableDownloadMenu";

interface Props {
  label: string;
  data: unknown;
}

function fmt(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "object") {
    const entries = Object.entries(v as Record<string, unknown>);
    return entries.map(([k, val]) => `${k.replace(/_/g, " ")}: ${typeof val === "number" ? (val % 1 === 0 ? val : val.toFixed(3)) : String(val ?? "—")}`).join(", ");
  }
  if (typeof v === "number") return v % 1 === 0 ? String(v) : v.toFixed(3);
  return String(v);
}

function dictOfDictsToRows(data: Record<string, unknown>): Record<string, unknown>[] | null {
  const headers = Object.keys(data);
  if (headers.length === 0) return null;
  const firstCol = data[headers[0]];
  if (!firstCol || typeof firstCol !== "object" || Array.isArray(firstCol)) return null;
  const rowKeys = Object.keys(firstCol as Record<string, unknown>).sort((a, b) => Number(a) - Number(b));
  return rowKeys.map((rk) => {
    const row: Record<string, unknown> = {};
    headers.forEach((h) => {
      const col = data[h] as Record<string, unknown> | undefined;
      row[h] = col ? col[rk] : undefined;
    });
    return row;
  });
}

function buildDownloadData(headers: string[], rows: Record<string, unknown>[]): unknown[][] {
  return rows.map((row) => headers.map((h) => fmt(row[h])));
}

export function GenericDataTable({ label, data }: Props) {
  const safeName = label.replace(/\s+/g, "_").replace(/[^a-zA-Z0-9_]/g, "");

  // Array of objects
  if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object" && data[0] !== null && !Array.isArray(data[0])) {
    const headers = Object.keys(data[0] as Record<string, unknown>);
    const dlHeaders = headers.map((h) => h.replace(/_/g, " "));
    const dlRows = buildDownloadData(headers, data as Record<string, unknown>[]);

    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-3 text-base">
              <BarChart3 className="w-5 h-5 text-primary" />
              {label}
            </CardTitle>
            <TableDownloadMenu title={safeName} headers={dlHeaders} rows={dlRows} />
          </div>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-sm border-collapse font-mono">
            <thead>
              <tr className="border-b-2 border-foreground/30">
                {headers.map((h) => (
                  <th key={h} className="px-4 py-2.5 font-semibold text-foreground text-left">{h.replace(/_/g, " ")}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(data as Record<string, unknown>[]).map((row, ri) => (
                <tr key={ri} className="border-b border-border/50 hover:bg-muted/20">
                  {headers.map((h) => (
                    <td key={h} className="px-4 py-2.5 text-muted-foreground tabular-nums">{fmt(row[h])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    );
  }

  // Dict-of-dicts (pandas format)
  if (data && typeof data === "object" && !Array.isArray(data)) {
    const obj = data as Record<string, unknown>;
    const rows = dictOfDictsToRows(obj);
    if (rows && rows.length > 0) {
      const headers = Object.keys(obj);
      const dlHeaders = headers.map((h) => h.replace(/_/g, " "));
      const dlRows = buildDownloadData(headers, rows);

      return (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-3 text-base">
                <BarChart3 className="w-5 h-5 text-primary" />
                {label}
              </CardTitle>
              <TableDownloadMenu title={safeName} headers={dlHeaders} rows={dlRows} />
            </div>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <table className="w-full text-sm border-collapse font-mono">
              <thead>
                <tr className="border-b-2 border-foreground/30">
                  {headers.map((h) => (
                    <th key={h} className="px-4 py-2.5 font-semibold text-foreground text-left">{h.replace(/_/g, " ")}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, ri) => (
                  <tr key={ri} className="border-b border-border/50 hover:bg-muted/20">
                    {headers.map((h) => (
                      <td key={h} className="px-4 py-2.5 text-muted-foreground tabular-nums">{fmt(row[h])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      );
    }

    // Simple key-value object
    const entries = Object.entries(obj).filter(([, v]) => v != null);
    if (entries.length === 0) return null;
    const dlRows = entries.map(([k, v]) => [k.replace(/_/g, " "), fmt(v)]);

    return (
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-3 text-base">
              <BarChart3 className="w-5 h-5 text-primary" />
              {label}
            </CardTitle>
            <TableDownloadMenu title={safeName} headers={["Metric", "Value"]} rows={dlRows} />
          </div>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full text-sm border-collapse font-mono">
            <thead>
              <tr className="border-b-2 border-foreground/30">
                <th className="px-4 py-2.5 font-semibold text-foreground text-left">Metric</th>
                <th className="px-4 py-2.5 font-semibold text-foreground text-left">Value</th>
              </tr>
            </thead>
            <tbody>
              {entries.map(([k, v]) => (
                <tr key={k} className="border-b border-border/50 hover:bg-muted/20">
                  <td className="px-4 py-2.5 text-foreground font-medium">{k.replace(/_/g, " ")}</td>
                  <td className="px-4 py-2.5 text-muted-foreground tabular-nums">{fmt(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>
    );
  }

  return null;
}
