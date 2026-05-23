/**
 * Auto-generates publishable HTML tables from arbitrary JSON result data
 * when the backend doesn't return pre-built `html_tables`.
 */

// ── Correlation-specific logic ──────────────────────────────────────

interface CorrelationEntry {
  r?: number;
  r_g?: number;
  p_value?: number | null;
  significant?: boolean | null;
}

interface CorrelationMatrix {
  matrix: Record<string, Record<string, CorrelationEntry>>;
  n_observations?: number;
  n_genotypes?: number;
  per_location?: Record<string, { n_observations?: number; matrix: Record<string, Record<string, CorrelationEntry>> }>;
}

function fmt(v: number | undefined | null, decimals = 4): string {
  if (v == null) return "—";
  return Number(v).toFixed(decimals);
}

function sigMark(p: number | null | undefined, sig: boolean | null | undefined): string {
  if (sig === true || (p != null && p < 0.05)) {
    if (p != null && p < 0.01) return "**";
    return "*";
  }
  return "";
}

function correlationMatrixToHtml(
  matrix: Record<string, Record<string, CorrelationEntry>>,
  type: "phenotypic" | "genotypic",
  caption: string,
  n?: number
): string {
  const traits = Object.keys(matrix);
  const rKey = type === "genotypic" ? "r_g" : "r";

  let html = `<table>\n<caption>${caption}${n ? ` (n = ${n})` : ""}</caption>\n`;
  html += "<thead><tr><th>Trait</th>";
  traits.forEach((t) => { html += `<th>${t}</th>`; });
  html += "</tr></thead>\n<tbody>\n";

  traits.forEach((rowTrait) => {
    html += `<tr><td><strong>${rowTrait}</strong></td>`;
    traits.forEach((colTrait) => {
      const entry = matrix[rowTrait]?.[colTrait];
      if (!entry) { html += "<td>—</td>"; return; }
      const val = (entry as Record<string, unknown>)[rKey] as number ?? entry.r ?? entry.r_g;
      if (rowTrait === colTrait) {
        html += "<td>1.000</td>";
      } else {
        const sig = sigMark(entry.p_value, entry.significant);
        html += `<td>${fmt(val)}${sig}</td>`;
      }
    });
    html += "</tr>\n";
  });

  html += "</tbody></table>";
  if (type === "phenotypic") {
    html += "\n<p><em>* p &lt; 0.05; ** p &lt; 0.01</em></p>";
  }
  return html;
}

function extractCorrelationTables(results: Record<string, unknown>): Record<string, string> {
  const tables: Record<string, string> = {};
  const phenotypic = results.phenotypic as CorrelationMatrix | undefined;
  const genotypic = results.genotypic as CorrelationMatrix | undefined;

  if (phenotypic?.matrix) {
    tables["phenotypic_correlation_matrix"] = correlationMatrixToHtml(
      phenotypic.matrix, "phenotypic",
      "Table: Phenotypic Correlation Coefficients Among Traits",
      phenotypic.n_observations
    );
    if (phenotypic.per_location) {
      Object.entries(phenotypic.per_location).forEach(([loc, locData]) => {
        const key = `phenotypic_correlation_${loc.replace(/\s+/g, "_").toLowerCase()}`;
        tables[key] = correlationMatrixToHtml(
          locData.matrix, "phenotypic",
          `Table: Phenotypic Correlations at ${loc}`, locData.n_observations
        );
      });
    }
  }
  if (genotypic?.matrix) {
    tables["genotypic_correlation_matrix"] = correlationMatrixToHtml(
      genotypic.matrix, "genotypic",
      "Table: Genotypic Correlation Coefficients Among Traits",
      genotypic.n_genotypes
    );
  }
  return tables;
}

// ── Generic table conversion ────────────────────────────────────────

function formatTitle(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Convert array-of-arrays into an HTML table */
function arrayOfArraysToHtml(data: unknown[][], caption: string): string {
  if (data.length === 0) return "";
  const headers = data[0];
  const rows = data.slice(1);
  let html = `<table>\n<caption>${caption}</caption>\n<thead><tr>`;
  headers.forEach((h) => { html += `<th>${String(h)}</th>`; });
  html += "</tr></thead>\n<tbody>\n";
  rows.forEach((row) => {
    html += "<tr>";
    (row as unknown[]).forEach((cell) => { html += `<td>${String(cell ?? "")}</td>`; });
    html += "</tr>\n";
  });
  html += "</tbody></table>";
  return html;
}

/** Convert array-of-objects into an HTML table */
function arrayOfObjectsToHtml(data: Record<string, unknown>[], caption: string): string {
  if (data.length === 0) return "";
  const headers = Object.keys(data[0]);
  let html = `<table>\n<caption>${caption}</caption>\n<thead><tr>`;
  headers.forEach((h) => { html += `<th>${formatTitle(h)}</th>`; });
  html += "</tr></thead>\n<tbody>\n";
  data.forEach((row) => {
    html += "<tr>";
    headers.forEach((h) => { html += `<td>${row[h] != null ? String(row[h]) : ""}</td>`; });
    html += "</tr>\n";
  });
  html += "</tbody></table>";
  return html;
}

/** Convert a key-value object into a 2-column HTML table */
function kvObjectToHtml(data: Record<string, unknown>, caption: string): string {
  const entries = Object.entries(data).filter(([, v]) => v != null && typeof v !== "object");
  if (entries.length === 0) return "";
  let html = `<table>\n<caption>${caption}</caption>\n<thead><tr><th>Parameter</th><th>Value</th></tr></thead>\n<tbody>\n`;
  entries.forEach(([k, v]) => {
    html += `<tr><td>${formatTitle(k)}</td><td>${String(v)}</td></tr>\n`;
  });
  html += "</tbody></table>";
  return html;
}

/** Try to convert an unknown data blob into an HTML table string */
function dataToHtml(data: unknown, caption: string): string | null {
  if (!data) return null;

  // Array of arrays
  if (Array.isArray(data) && data.length > 0 && Array.isArray(data[0])) {
    return arrayOfArraysToHtml(data as unknown[][], caption);
  }
  // Array of objects
  if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object" && data[0] !== null) {
    return arrayOfObjectsToHtml(data as Record<string, unknown>[], caption);
  }
  // Plain object with scalar values
  if (typeof data === "object" && !Array.isArray(data)) {
    return kvObjectToHtml(data as Record<string, unknown>, caption);
  }
  return null;
}

// ── Keys we skip (not tabular data) ─────────────────────────────────

const SKIP_KEYS = new Set([
  "status", "design", "response", "treatment", "formula", "error",
  "plots", "html_tables", "interpretation", "meta", "per_trait",
  "trait_results", "significance_heatmap", "correlation_heatmap",
  "pca_biplot", "intelligence", "phenotypic", "genotypic",
]);

/**
 * Master function: given raw backend results, produce publishable
 * HTML tables from ALL structured data when html_tables is absent.
 */
/** Convert a dict-of-dicts (pandas-style) into an HTML table.
 *  e.g. { df: { Genotype: 4, Residual: 10 }, sum_sq: { Genotype: 200, Residual: 50 } }
 *  → rows keyed by inner keys (Genotype, Residual), columns by outer keys (df, sum_sq)
 */
function dictOfDictsToHtml(data: Record<string, Record<string, unknown>>, caption: string): string | null {
  const colKeys = Object.keys(data);
  if (colKeys.length === 0) return null;
  // Check that values are objects (not scalars)
  const firstVal = data[colKeys[0]];
  if (!firstVal || typeof firstVal !== "object" || Array.isArray(firstVal)) return null;

  const rowKeys = Object.keys(firstVal);
  if (rowKeys.length === 0) return null;

  let html = `<table>\n<caption>${caption}</caption>\n<thead><tr><th>Source</th>`;
  colKeys.forEach((c) => { html += `<th>${formatTitle(c)}</th>`; });
  html += `</tr></thead>\n<tbody>\n`;
  rowKeys.forEach((rk) => {
    html += `<tr><td>${rk.replace(/Q\('(.+?)'\)/g, "$1")}</td>`;
    colKeys.forEach((ck) => {
      const v = (data[ck] as Record<string, unknown>)[rk];
      if (v == null) { html += "<td>—</td>"; return; }
      if (typeof v === "number") {
        html += `<td>${Number.isInteger(v) ? String(v) : v.toFixed(4)}</td>`;
      } else {
        html += `<td>${String(v)}</td>`;
      }
    });
    html += `</tr>\n`;
  });
  html += `</tbody></table>`;
  return html;
}

/** Detect if an object is dict-of-dicts (all values are non-array objects) */
function isDictOfDicts(data: unknown): data is Record<string, Record<string, unknown>> {
  if (!data || typeof data !== "object" || Array.isArray(data)) return false;
  const vals = Object.values(data as Record<string, unknown>);
  return vals.length > 0 && vals.every((v) => v && typeof v === "object" && !Array.isArray(v));
}

/** Enhanced dataToHtml that also handles dict-of-dicts */
function dataToHtmlEnhanced(data: unknown, caption: string): string | null {
  // Try standard formats first
  const standard = dataToHtml(data, caption);
  if (standard) return standard;
  // Try dict-of-dicts
  if (isDictOfDicts(data)) {
    return dictOfDictsToHtml(data as Record<string, Record<string, unknown>>, caption);
  }
  return null;
}

/**
 * Master function: given raw backend results, produce publishable
 * HTML tables from ALL structured data when html_tables is absent.
 */
export function generatePublishableHtmlTables(
  results: Record<string, unknown>
): Record<string, string> {
  const tables: Record<string, string> = {};

  // 1. Correlation matrices (special handling)
  const corrTables = extractCorrelationTables(results);
  Object.assign(tables, corrTables);

  // 2. Named table keys (anova, means, letters, etc.)
  const TABLE_KEYS = ["anova", "means", "letters", "effect_sizes", "assumptions", "summary_table"];
  const TABLE_LABELS: Record<string, string> = {
    anova: "ANOVA Table",
    means: "Treatment Means",
    letters: "Mean Separation (Tukey Letters)",
    effect_sizes: "Effect Sizes",
    assumptions: "Assumption Tests",
    summary_table: "Trait Summary Table",
  };

  TABLE_KEYS.forEach((key) => {
    const data = results[key];
    if (data) {
      const caption = `Table: ${TABLE_LABELS[key] || formatTitle(key)}`;
      const html = dataToHtmlEnhanced(data, caption);
      if (html) tables[key] = html;
    }
  });

  // 2b. Descriptive statistics (special handling for nested overall + group structure)
  const descStats = results.descriptive_stats as Record<string, unknown> | undefined;
  if (descStats && typeof descStats === "object") {
    const overall = descStats.overall as Record<string, unknown> | undefined;
    const groupKey = Object.keys(descStats).find((k) => k !== "overall");
    const groups = groupKey ? descStats[groupKey] as Record<string, Record<string, unknown>> | undefined : null;

    if (overall && typeof overall === "object") {
      const statKeys = ["n", "mean", "std", "sem", "cv", "min", "max", "range", "q1", "median", "q3", "iqr"];
      const availableKeys = statKeys.filter((k) => overall[k] != null);
      if (availableKeys.length > 0) {
        let html = `<table>\n<caption>Table: Overall Descriptive Statistics</caption>\n<thead><tr>`;
        availableKeys.forEach((h) => { html += `<th>${formatTitle(h)}</th>`; });
        html += `</tr></thead>\n<tbody>\n<tr>`;
        availableKeys.forEach((h) => {
          const v = overall[h];
          html += `<td>${typeof v === "number" ? (Number.isInteger(v) ? String(v) : Number(v).toFixed(2)) : String(v ?? "")}</td>`;
        });
        html += `</tr>\n</tbody></table>`;
        tables["descriptive_overall"] = html;
      }
    }

    if (groups && typeof groups === "object") {
      const firstGroup = Object.values(groups)[0];
      if (firstGroup && typeof firstGroup === "object") {
        const statKeys = ["count", "n", "mean", "std", "sem", "min", "max", "cv"];
        const availableKeys = statKeys.filter((k) => firstGroup[k] != null);
        if (availableKeys.length > 0) {
          let html = `<table>\n<caption>Table: Descriptive Statistics by ${groupKey}</caption>\n<thead><tr><th>${groupKey}</th>`;
          availableKeys.forEach((h) => { html += `<th>${formatTitle(h)}</th>`; });
          html += `</tr></thead>\n<tbody>\n`;
          Object.entries(groups).forEach(([lvl, s]) => {
            html += `<tr><td>${lvl}</td>`;
            availableKeys.forEach((h) => {
              const v = (s as Record<string, unknown>)[h];
              html += `<td>${typeof v === "number" ? (Number.isInteger(v) ? String(v) : Number(v).toFixed(2)) : String(v ?? "")}</td>`;
            });
            html += `</tr>\n`;
          });
          html += `</tbody></table>`;
          tables["descriptive_by_group"] = html;
        }
      }
    }
  }

  // 2c. Means separation — flatten nested means+letters into a combined table
  if (!tables["means"] && results.means && results.letters) {
    const meansObj = results.means as Record<string, unknown>;
    const lettersObj = results.letters as Record<string, unknown>;
    const treatKey = Object.keys(meansObj)[0];
    const meansMap = (meansObj[treatKey] ?? meansObj) as Record<string, number>;
    const lettersMap = ((lettersObj as any)[treatKey] ?? lettersObj) as Record<string, string>;
    const levels = Object.keys(meansMap);
    if (levels.length > 0) {
      let html = `<table>\n<caption>Table: Treatment Means &amp; Tukey HSD Letters</caption>\n<thead><tr><th>${treatKey || "Treatment"}</th><th>Mean</th><th>Tukey Group</th></tr></thead>\n<tbody>\n`;
      levels
        .sort((a, b) => (meansMap[b] ?? 0) - (meansMap[a] ?? 0))
        .forEach((lvl) => {
          html += `<tr><td>${lvl}</td><td>${Number(meansMap[lvl]).toFixed(2)}</td><td>${lettersMap[lvl] || "—"}</td></tr>\n`;
        });
      html += `</tbody></table>`;
      tables["means"] = html;
    }
  }

  // 2d. Assumptions — flatten nested object format
  if (!tables["assumptions"] && results.assumptions && typeof results.assumptions === "object" && !Array.isArray(results.assumptions)) {
    const assObj = results.assumptions as Record<string, Record<string, unknown>>;
    const entries = Object.entries(assObj);
    if (entries.length > 0 && typeof entries[0][1] === "object") {
      let html = `<table>\n<caption>Table: Assumption Tests</caption>\n<thead><tr><th>Test</th><th>Method</th><th>Statistic</th><th>p-value</th><th>Result</th></tr></thead>\n<tbody>\n`;
      entries.forEach(([key, val]) => {
        const testName = (val.test_name as string) || formatTitle(key);
        const stat = val.statistic != null ? Number(val.statistic).toFixed(4) : "—";
        const pVal = val.p_value != null ? Number(val.p_value).toFixed(4) : "—";
        const passed = val.passed != null ? (val.passed ? "✓ Passed" : "✗ Failed") : "—";
        html += `<tr><td>${testName}</td><td>${formatTitle(key)}</td><td>${stat}</td><td>${pVal}</td><td>${passed}</td></tr>\n`;
      });
      html += `</tbody></table>`;
      tables["assumptions"] = html;
    }
  }

  // 2e. Effect sizes — flatten nested per-source format
  if (!tables["effect_sizes"] && results.effect_sizes && typeof results.effect_sizes === "object") {
    const esObj = results.effect_sizes as Record<string, Record<string, unknown>>;
    const entries = Object.entries(esObj).filter(([k]) => k !== "Residual");
    if (entries.length > 0 && typeof entries[0][1] === "object" && entries[0][1] !== null && "eta_squared" in entries[0][1]) {
      let html = `<table>\n<caption>Table: Effect Sizes</caption>\n<thead><tr><th>Source</th><th>η²</th><th>ω²</th><th>Cohen's f</th><th>Interpretation</th></tr></thead>\n<tbody>\n`;
      entries.forEach(([src, es]) => {
        html += `<tr><td>${src.replace(/Q\('(.+?)'\)/g, "$1")}</td><td>${fmt(es.eta_squared as number)}</td><td>${fmt(es.omega_squared as number)}</td><td>${fmt(es.cohens_f as number, 2)}</td><td>${(es.interpretation as string) || "—"}</td></tr>\n`;
      });
      html += `</tbody></table>`;
      tables["effect_sizes"] = html;
    }
  }

  // 3. Nested "tables" object (legacy backend shape)
  const nestedTables = results.tables as Record<string, unknown> | undefined;
  if (nestedTables && typeof nestedTables === "object") {
    Object.entries(nestedTables).forEach(([key, data]) => {
      if (!tables[key] && data) {
        const html = dataToHtmlEnhanced(data, `Table: ${formatTitle(key)}`);
        if (html) tables[key] = html;
      }
    });
  }

  // 4. Per-trait tables (from per_trait or trait_results)
  const perTrait = (results.per_trait || results.trait_results) as Record<string, Record<string, unknown>> | undefined;
  if (perTrait && typeof perTrait === "object") {
    Object.entries(perTrait).forEach(([traitName, traitData]) => {
      if (!traitData || typeof traitData !== "object") return;
      const traitTables = (traitData as Record<string, unknown>).tables as Record<string, unknown> | undefined;
      if (traitTables && typeof traitTables === "object") {
        Object.entries(traitTables).forEach(([tKey, tData]) => {
          if (tData && tKey !== "assumption_guidance") {
            const tableKey = `${traitName.replace(/\s+/g, "_").toLowerCase()}_${tKey}`;
            const html = dataToHtmlEnhanced(tData, `Table: ${traitName} — ${formatTitle(tKey)}`);
            if (html) tables[tableKey] = html;
          }
        });
      }
    });
  }

  // 5. Variance components (genetics)
  const vc = results.variance_components as Record<string, unknown> | undefined;
  if (vc && typeof vc === "object") {
    const html = kvObjectToHtml(vc, "Table: Variance Components");
    if (html) tables["variance_components"] = html;
  }

  return tables;
}
