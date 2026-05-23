import { Button } from "@/components/ui/button";
import { Download, FileSpreadsheet, FileText } from "lucide-react";
import * as XLSX from "xlsx";
import type { FastAPIResultsData } from "../VivaSenseFastAPIResults";
import { formatPValue, getSignificanceStars } from "./SignificanceStars";

interface Props {
  results: FastAPIResultsData;
  analysisType?: string;
}

const TABLE_KEYS = ["descriptive_stats", "anova", "means", "letters", "effect_sizes", "assumptions"];
const TABLE_LABELS: Record<string, string> = {
  descriptive_stats: "Table 1. Descriptive Statistics",
  anova: "Table 2. Analysis of Variance",
  means: "Table 3. Treatment Means",
  letters: "Table 3. Mean Separation (Tukey)",
  effect_sizes: "Effect Sizes",
  assumptions: "Assumption Tests",
};

function toSheetData(data: unknown): unknown[][] | null {
  if (Array.isArray(data) && data.length > 0) {
    if (Array.isArray(data[0])) return data as unknown[][];
    if (typeof data[0] === "object" && data[0] !== null) {
      const headers = Object.keys(data[0] as Record<string, unknown>);
      const rows = (data as Record<string, unknown>[]).map((r) => headers.map((h) => r[h] ?? ""));
      return [headers, ...rows];
    }
  }
  if (data && typeof data === "object" && !Array.isArray(data)) {
    const entries = Object.entries(data as Record<string, unknown>);
    return [["Parameter", "Value"], ...entries.map(([k, v]) => [k, typeof v === "object" ? JSON.stringify(v) : v])];
  }
  return null;
}

function downloadExcel(results: FastAPIResultsData) {
  const wb = XLSX.utils.book_new();
  let sheetCount = 0;

  // Summary sheet
  const metaEntries: string[][] = [["Parameter", "Value"]];
  if (results.design) metaEntries.push(["Experimental Design", results.design]);
  if (results.response) metaEntries.push(["Trait Analysed", results.response]);
  if (results.treatment) metaEntries.push(["Treatment Factor", results.treatment]);
  if (results.grand_mean != null) metaEntries.push(["Grand Mean", Number(results.grand_mean).toFixed(3)]);
  if (results.cv_percent != null) metaEntries.push(["CV (%)", Number(results.cv_percent).toFixed(1) + "%"]);
  if (results.formula) metaEntries.push(["Formula", results.formula]);
  if (metaEntries.length > 1) {
    const ws = XLSX.utils.aoa_to_sheet(metaEntries);
    XLSX.utils.book_append_sheet(wb, ws, "Experiment Summary");
    sheetCount++;
  }

  TABLE_KEYS.forEach((key) => {
    const data = (results as Record<string, unknown>)[key];
    if (!data) return;
    let sheetData = toSheetData(data);
    if (!sheetData) return;

    // Special handling for ANOVA: map p-value column correctly
    if (key === "anova" && sheetData.length > 0) {
      const headers = sheetData[0] as string[];
      // Find the p-value column index (various backend names)
      const pColIdx = headers.findIndex((h) =>
        /^(p[_\-\s]?value|pr\(>f\)|p-value|pvalue|Pr\(>F\))$/i.test(String(h))
      );
      // Rename header to standard "p-value"
      if (pColIdx >= 0) {
        headers[pColIdx] = "p-value";
      }
      // Format p-values in data rows
      if (pColIdx >= 0) {
        for (let ri = 1; ri < sheetData.length; ri++) {
          const row = sheetData[ri] as unknown[];
          const raw = row[pColIdx];
          if (raw == null || raw === "" || raw === "—") {
            row[pColIdx] = "";
          } else {
            const num = typeof raw === "number" ? raw : parseFloat(String(raw));
            if (isNaN(num)) {
              row[pColIdx] = String(raw);
            } else if (num < 0.001) {
              row[pColIdx] = "<0.001";
            } else {
              row[pColIdx] = parseFloat(num.toFixed(4));
            }
          }
        }
      }
    }

    const ws = XLSX.utils.aoa_to_sheet(sheetData);
    const name = (TABLE_LABELS[key] || key).replace(/Table \d+\.\s*/g, "").slice(0, 31);
    XLSX.utils.book_append_sheet(wb, ws, name);
    sheetCount++;
  });

  if (sheetCount === 0) return;
  const dateStr = new Date().toISOString().slice(0, 10);
  XLSX.writeFile(wb, `VivaSense_Results_${dateStr}.xlsx`);
}

function downloadPlotAsPng(plotName: string, base64: string) {
  const a = document.createElement("a");
  a.href = `data:image/png;base64,${base64}`;
  a.download = `VivaSense_${plotName.replace(/\s+/g, "_")}_${new Date().toISOString().slice(0, 10)}.png`;
  a.click();
}

/** Generate Word-compatible HTML for all result tables */
function downloadWordTables(results: FastAPIResultsData) {
  const styles = `
    <style>
      body { font-family: 'Times New Roman', serif; font-size: 12pt; }
      table { border-collapse: collapse; width: 100%; margin-bottom: 20pt; }
      th, td { border: 1px solid #000; padding: 4pt 8pt; text-align: right; font-size: 11pt; }
      th { background-color: #f2f2f2; font-weight: bold; }
      td:first-child, th:first-child { text-align: left; }
      caption { text-align: left; font-weight: bold; margin-bottom: 6pt; font-size: 11pt; }
      .summary { margin-bottom: 16pt; }
      .summary td { border: none; padding: 2pt 8pt; }
      p.note { font-size: 10pt; font-style: italic; margin-top: 4pt; }
      p.footer { font-size: 9pt; color: #666; margin-top: 20pt; text-align: center; }
    </style>
  `;

  let html = `<html><head><meta charset="utf-8">${styles}</head><body>`;
  html += `<h2>VivaSense Statistical Analysis Report</h2>`;
  html += `<p>Generated: ${new Date().toLocaleDateString()}</p>`;

  // Experiment summary
  html += `<table class="summary"><caption>Experiment Summary</caption>`;
  if (results.design) html += `<tr><td><strong>Design</strong></td><td>${results.design}</td></tr>`;
  if (results.response) html += `<tr><td><strong>Trait</strong></td><td>${results.response}</td></tr>`;
  if (results.treatment) html += `<tr><td><strong>Treatment</strong></td><td>${results.treatment}</td></tr>`;
  if (results.grand_mean != null) html += `<tr><td><strong>Grand Mean</strong></td><td>${Number(results.grand_mean).toFixed(3)}</td></tr>`;
  if (results.cv_percent != null) html += `<tr><td><strong>CV (%)</strong></td><td>${Number(results.cv_percent).toFixed(1)}%</td></tr>`;
  html += `</table>`;

  // Helper to build table
  const buildTable = (data: unknown, caption: string, formatFn?: (h: string, v: unknown) => string) => {
    if (!data) return "";
    if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object" && data[0] !== null) {
      const rows = data as Record<string, unknown>[];
      const headers = Object.keys(rows[0]);
      let t = `<table><caption>${caption}</caption><thead><tr>`;
      headers.forEach(h => { t += `<th>${h.replace(/_/g, " ")}</th>`; });
      t += `</tr></thead><tbody>`;
      rows.forEach(row => {
        t += "<tr>";
        headers.forEach(h => {
          const val = formatFn ? formatFn(h, row[h]) : fmtWord(row[h]);
          t += `<td>${val}</td>`;
        });
        t += "</tr>";
      });
      t += `</tbody></table>`;
      return t;
    }
    return "";
  };

  // Format for Word output
  const fmtWord = (v: unknown): string => {
    if (v == null) return "—";
    const n = typeof v === "number" ? v : parseFloat(String(v));
    if (isNaN(n)) return String(v);
    return n.toFixed(3);
  };

  // ANOVA table with significance
  if (results.anova) {
    html += buildTable(results.anova, "Table 2. Analysis of Variance (ANOVA)", (h, v) => {
      if (/p.?value|pr/i.test(h)) return formatPValue(v);
      if (/df/i.test(h)) { const n = Number(v); return isNaN(n) ? String(v ?? "") : String(Math.round(n)); }
      return fmtWord(v);
    });
    html += `<p class="note">Significance: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant</p>`;
  }

  // Descriptive stats
  if (results.descriptive_stats) {
    html += buildTable(results.descriptive_stats, "Table 1. Descriptive Statistics");
  }

  // Mean separation
  const lettersData = results.letters || results.means;
  if (lettersData) {
    html += buildTable(lettersData, "Table 3. Mean Separation (Tukey's HSD)");
    html += `<p class="note">Means followed by the same letter are not significantly different (Tukey's HSD, α = 0.05).</p>`;
  }

  html += `<p class="footer">Generated by VivaSense — https://fieldtoinsightacademy.com.ng/vivasense</p>`;
  html += `</body></html>`;

  const blob = new Blob([html], { type: "application/msword" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `VivaSense_Tables_${new Date().toISOString().slice(0, 10)}.doc`;
  a.click();
  URL.revokeObjectURL(a.href);
}

export function ExportButtons({ results }: Props) {
  const hasPlots = results.plots && Object.keys(results.plots).length > 0;
  const hasTables = TABLE_KEYS.some((k) => (results as Record<string, unknown>)[k] != null);

  return (
    <div className="flex flex-wrap gap-3">
      {hasTables && (
        <>
          <Button variant="outline" size="sm" onClick={() => downloadWordTables(results)}>
            <FileText className="w-4 h-4 mr-2" />
            Download Word Tables
          </Button>
          <Button variant="outline" size="sm" onClick={() => downloadExcel(results)}>
            <FileSpreadsheet className="w-4 h-4 mr-2" />
            Download Excel
          </Button>
        </>
      )}
      {hasPlots &&
        Object.entries(results.plots!).map(([name, base64]) => (
          <Button
            key={name}
            variant="outline"
            size="sm"
            onClick={() => downloadPlotAsPng(name, base64)}
          >
            <Download className="w-4 h-4 mr-2" />
            {name.replace(/_/g, " ")} (PNG)
          </Button>
        ))}
    </div>
  );
}

export { downloadExcel, downloadPlotAsPng };
