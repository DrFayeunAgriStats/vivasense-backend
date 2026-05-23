/**
 * VivaSense — Shared design system tokens.
 *
 * Single source of truth for both the on-screen results UI and the
 * downloadable Word report. Anything that should look the same in both
 * places (section titles, ordering, terminology, typography classes,
 * accent colors) lives here.
 *
 * No backend / engine logic here — presentation only.
 */

// ────────────────────────────────────────────────────────────────────────────
// Section ordering (numbered) — used by Word report headings AND the live UI
// eyebrow labels so users perceive the same scientific structure on screen
// and in the exported document.
// ────────────────────────────────────────────────────────────────────────────
export const VS_REPORT_SECTIONS = {
  dataset: "1. Dataset & Variables",
  model: "2. Model Summary",
  results: "3. Results",
  interpretation: "4. Interpretation",
  caution: "5. Caution & Reliability",
  finalInsight: "6. Final Insight Summary",
  chat: "7. Chat History",
  note: "8. Interpretation Note",
} as const;

// Back-compat alias
export const VS_SECTIONS = VS_REPORT_SECTIONS;

// ────────────────────────────────────────────────────────────────────────────
// Brand colors
//
// `hex` (no leading #) is for docx; `hsl` is for Tailwind/CSS-variable use
// in the live UI. Keep these aligned with index.css tokens so the on-screen
// result cards visually match the Word report.
// ────────────────────────────────────────────────────────────────────────────
export const VS_COLORS = {
  ink: "1A1A1A",
  muted: "6B7280",
  brand: "0A7F5A", // VivaSense emerald
  good: "0A7F5A",
  warn: "B45309",
  bad: "B91C1C",
  cellBorder: "CCCCCC",
  headerFill: "F2F2F2",
} as const;

// ────────────────────────────────────────────────────────────────────────────
// Shared typography classes for the results UI.
//
// Tailwind class strings — kept in one place so all results surfaces
// (ANOVA, Genetics, Trait Relationships) render the same scholarly hierarchy
// and so it stays in sync with the heading sizes used in the Word export.
// ────────────────────────────────────────────────────────────────────────────
export const VS_TYPOGRAPHY = {
  /** Top-of-page report title (e.g. "Analysis Results"). */
  pageTitle: "font-serif text-3xl lg:text-4xl font-bold text-foreground tracking-tight",
  /** Page subtitle / context line under the page title. */
  pageSubtitle: "text-base text-muted-foreground",
  /** Section header inside a result card (mirrors Word "Heading 2"). */
  sectionTitle: "font-serif text-xl font-semibold text-foreground tracking-tight",
  /** Tiny uppercase eyebrow label above section titles ("Section 3 · Results"). */
  sectionEyebrow:
    "text-[11px] uppercase tracking-[0.14em] font-semibold text-primary/80",
  /** Sub-heading inside a section. */
  subHeading: "font-serif text-base font-semibold text-foreground",
  /** Caption / footnote text under tables and figures. */
  caption: "text-xs italic text-muted-foreground",
  /** Inline figure / table number labels (Table 1, Figure 2). */
  figureLabel: "text-xs font-semibold uppercase tracking-wider text-primary",
  /** Mono numerics inside tables. */
  mono: "font-mono text-xs",
} as const;

// ────────────────────────────────────────────────────────────────────────────
// Canonical terminology — used both in UI labels and Word export
// to prevent drift (e.g. "G×E Interaction" vs "GxE Interaction").
// ────────────────────────────────────────────────────────────────────────────
export const VS_TERMS = {
  anova: "ANOVA Table",
  groupMeans: "Group Means",
  postHoc: "Post-Hoc Comparisons",
  assumptions: "Assumption Diagnostics",
  interpretation: "Scientific Interpretation",
  caution: "Caution & Reliability",
  finalInsight: "Final Insight Summary",
  regression: "Regression Results",
  pathAnalysis: "Path Analysis",
  correlation: "Correlation Matrix",
  geneticParameters: "Genetic Parameters",
  ammi: "AMMI / GGE Analysis",
  rcode: "Reproducible R Code",
  reviewerCritique: "Reviewer-Style Critique",
  datasetQuality: "Dataset Quality",
  statisticalAudit: "Statistical Audit",
  downloads: "Download Results",
} as const;

// ────────────────────────────────────────────────────────────────────────────
// Shared significance / p-value formatter — same across UI and Word.
// ────────────────────────────────────────────────────────────────────────────
export function vsFormatP(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return "—";
  if (p < 0.001) return "< 0.001 ***";
  if (p < 0.01) return `${p.toFixed(3)} **`;
  if (p < 0.05) return `${p.toFixed(3)} *`;
  return p.toFixed(3);
}

export function vsFormatNumber(n: number | null | undefined, digits = 2): string {
  if (n == null || !Number.isFinite(n)) return "—";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}
