/**
 * Generate and download a plain-text summary of VivaSense analysis results.
 */

function tableToText(data: unknown): string {
  try {
    if (Array.isArray(data) && data.length > 0 && Array.isArray(data[0])) {
      const rows = data as unknown[][];
      const colWidths = rows[0].map((_, ci) =>
        Math.max(...rows.map((r) => String(r[ci] ?? "").length))
      );
      return rows
        .map((row) =>
          row.map((cell, ci) => String(cell ?? "").padEnd(colWidths[ci])).join("  ")
        )
        .join("\n");
    }

  if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object" && data[0] !== null) {
    const headers = Object.keys(data[0] as Record<string, unknown>);
    const allRows = [headers, ...(data as Record<string, unknown>[]).map((r) => headers.map((h) => String(r[h] ?? "")))];
    const colWidths = headers.map((_, ci) =>
      Math.max(...allRows.map((r) => r[ci].length))
    );
    return allRows
      .map((row) => row.map((cell, ci) => cell.padEnd(colWidths[ci])).join("  "))
      .join("\n");
  }

    return JSON.stringify(data, null, 2);
  } catch {
    return JSON.stringify(data, null, 2);
  }
}

/** Format a single trait's tables into text lines */
function formatTraitSection(traitName: string, trait: Record<string, unknown>): string[] {
  const lines: string[] = [];
  const tables = trait.tables as Record<string, unknown> | undefined;

  lines.push(`--- ${traitName.toUpperCase()} ---`);
  lines.push("");

  // ANOVA table
  const anovaData = tables?.anova;
  if (anovaData) {
    lines.push("ANOVA TABLE");
    lines.push("-".repeat(40));
    lines.push(tableToText(anovaData));
    lines.push("");
  }

  // Means and Tukey Groupings
  const meansData = tables?.means;
  if (meansData) {
    lines.push("MEANS AND TUKEY GROUPINGS");
    lines.push("-".repeat(40));
    lines.push(tableToText(meansData));
    lines.push("");
  }

  // Assumption guidance
  const assumptionGuidance = tables?.assumption_guidance as Record<string, unknown> | undefined;
  if (assumptionGuidance?.overall) {
    lines.push("ASSUMPTION GUIDANCE");
    lines.push("-".repeat(40));
    lines.push(String(assumptionGuidance.overall));
    lines.push("");
  } else if (typeof trait.assumption_verdict === "string") {
    lines.push("ASSUMPTION GUIDANCE");
    lines.push("-".repeat(40));
    lines.push(trait.assumption_verdict);
    lines.push("");
  } else if (typeof trait.assumptions === "string") {
    lines.push("ASSUMPTION GUIDANCE");
    lines.push("-".repeat(40));
    lines.push(trait.assumptions);
    lines.push("");
  }

  return lines;
}

export function downloadResultsAsText(
  analysisType: string,
  results: { meta?: Record<string, unknown>; tables?: Record<string, unknown>; interpretation?: string; per_trait?: Record<string, unknown>; [key: string]: unknown },
  drInterpretation: string,
  chatMessages?: { role: "user" | "assistant"; content: string }[],
) {
  const lines: string[] = [];
  const date = new Date();
  const dateStr = date.toISOString().slice(0, 10);

  lines.push("=" .repeat(60));
  lines.push("  VivaSense — Statistical Analysis Report");
  lines.push("=" .repeat(60));
  lines.push("");
  lines.push(`Analysis Type : ${analysisType}`);
  lines.push(`Date          : ${date.toLocaleString()}`);
  lines.push("");

  // Meta / Summary
  if (results.meta && Object.keys(results.meta).length > 0) {
    lines.push("-".repeat(40));
    lines.push("ANALYSIS SUMMARY");
    lines.push("-".repeat(40));
    for (const [k, v] of Object.entries(results.meta)) {
      lines.push(`  ${k.replace(/_/g, " ")}: ${String(v)}`);
    }
    lines.push("");
  }

  // Tables (summary-level)
  if (results.tables) {
    for (const [name, data] of Object.entries(results.tables)) {
      if (!data) continue;
      lines.push("-".repeat(40));
      lines.push(name.replace(/_/g, " ").toUpperCase());
      lines.push("-".repeat(40));
      lines.push(tableToText(data));
      lines.push("");
    }
  }

  // Per-trait detailed results
  // Case 1: Multi-trait — nested under results.per_trait
  if (results.per_trait && typeof results.per_trait === "object" && Object.keys(results.per_trait).length > 0) {
    lines.push("=".repeat(60));
    lines.push("  PER-TRAIT DETAILED RESULTS");
    lines.push("=".repeat(60));
    lines.push("");

    for (const [traitName, traitData] of Object.entries(results.per_trait)) {
      const trait = traitData as Record<string, unknown> | undefined;
      if (!trait) continue;
      lines.push(...formatTraitSection(traitName, trait));
    }
  }
  // Case 2: Single-trait — anova/means at top-level results.tables
  else if (results.tables && (results.tables.anova || results.tables.means)) {
    const traitName = String(results.meta?.trait || results.meta?.analysis_type || analysisType);
    lines.push("=".repeat(60));
    lines.push("  PER-TRAIT DETAILED RESULTS");
    lines.push("=".repeat(60));
    lines.push("");
    lines.push(...formatTraitSection(traitName, { tables: results.tables, assumption_verdict: (results as any).assumption_verdict, assumptions: (results as any).assumptions }));
  }

  if (results.interpretation) {
    lines.push("-".repeat(40));
    lines.push("BACKEND INTERPRETATION");
    lines.push("-".repeat(40));
    lines.push(results.interpretation);
    lines.push("");
  }

  // Dr. Fayeun's interpretation
  if (drInterpretation) {
    lines.push("-".repeat(40));
    lines.push("DR. FAYEUN'S INTERPRETATION");
    lines.push("-".repeat(40));
    lines.push(drInterpretation);
    lines.push("");
  }

  // Chat History
  if (chatMessages && chatMessages.length > 0) {
    lines.push("-".repeat(40));
    lines.push("CHAT HISTORY — FOLLOW-UP QUESTIONS");
    lines.push("-".repeat(40));
    lines.push("");
    for (const msg of chatMessages) {
      const label = msg.role === "user" ? "Researcher" : "Dr. Fayeun";
      lines.push(`${label}:`);
      lines.push(msg.content);
      lines.push("");
    }
  }

  lines.push("=".repeat(60));
  lines.push("  Generated by VivaSense — https://fieldtoinsightacademy.com.ng/vivasense");
  lines.push("=".repeat(60));

  const trait = String(results.meta?.trait || results.meta?.analysis_type || analysisType)
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_]/g, "");

  const filename = `VivaSense_Results_${trait}_${dateStr}.txt`;
  let content: string;
  try {
    content = lines.join("\n");
  } catch {
    content = "Error generating report content.";
  }
  const blob = new Blob([content], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
