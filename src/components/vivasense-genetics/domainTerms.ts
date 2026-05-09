/**
 * domainTerms
 * ===========
 * Domain-specific terminology and metadata for VivaSense Genetics.
 *
 * Research domains drive:
 *  - UI label choices (genotype vs treatment, H² vs variation ratio, etc.)
 *  - Correlation mode visibility (which modes are scientifically appropriate)
 *  - Export/report language
 */

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type DomainKey = "plant_breeding" | "agronomy" | "general";

export interface ResearchDomainOption {
  value: DomainKey;
  icon: string;
  label: string;
  desc: string;
}

export interface DomainTerms {
  /** Singular form: "Genotype" or "Treatment" */
  treatment: string;
  /** Plural form: "Genotypes" or "Treatments" */
  treatments: string;
  /** Lowercase plural: "genotypes" or "treatments" */
  treatment_plural: string;
  /** Heritability / variation column header */
  h2_label: string;
  /** GCV column header */
  gcv_label: string;
  /** PCV column header */
  pcv_label: string;
  /** GAM column header */
  gam_label: string;
  /** Variance/parameters module heading */
  variance_module: string;
  /** Description used for top performers */
  top_performer: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Domain option list (shown in ColumnMappingConfirm)
// ─────────────────────────────────────────────────────────────────────────────

export const RESEARCH_DOMAINS: ResearchDomainOption[] = [
  {
    value: "plant_breeding",
    icon: "🧬",
    label: "Plant Breeding",
    desc: "Genotypes, cultivars, varieties, germplasm evaluation",
  },
  {
    value: "agronomy",
    icon: "🌱",
    label: "Agronomy",
    desc: "Fertilizer, irrigation, spacing, management trials",
  },
  {
    value: "general",
    icon: "📊",
    label: "General",
    desc: "Other or mixed experimental workflows",
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// Domain term lookup
// ─────────────────────────────────────────────────────────────────────────────

const DOMAIN_TERMS: Record<DomainKey, DomainTerms> = {
  plant_breeding: {
    treatment:       "Genotype",
    treatments:      "Genotypes",
    treatment_plural: "genotypes",
    h2_label:        "H²",
    gcv_label:       "GCV%",
    pcv_label:       "PCV%",
    gam_label:       "GAM%",
    variance_module: "Genetic Parameters",
    top_performer:   "elite genotype",
  },
  agronomy: {
    treatment:       "Treatment",
    treatments:      "Treatments",
    treatment_plural: "treatments",
    h2_label:        "Var. Ratio",
    gcv_label:       "GCV%",
    pcv_label:       "PCV%",
    gam_label:       "Resp.%",
    variance_module: "Variance Components",
    top_performer:   "top treatment",
  },
  general: {
    treatment:       "Treatment",
    treatments:      "Treatments",
    treatment_plural: "treatments",
    h2_label:        "Var. Ratio",
    gcv_label:       "GCV%",
    pcv_label:       "PCV%",
    gam_label:       "Resp.%",
    variance_module: "Variance Components",
    top_performer:   "top treatment",
  },
};

/**
 * Return domain-specific UI terminology.
 * Falls back to "general" when domain is undefined or unrecognised.
 */
export function getDomainTerms(domain?: DomainKey | string): DomainTerms {
  const key = (domain ?? "general") as DomainKey;
  return DOMAIN_TERMS[key] ?? DOMAIN_TERMS["general"];
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-detection from column names
// ─────────────────────────────────────────────────────────────────────────────

const BREEDING_KEYWORDS = [
  "genotype", "variety", "cultivar", "accession", "line", "cross",
  "hybrid", "clone", "germplasm", "breed",
];
const AGRONOMY_KEYWORDS = [
  "fertilizer", "fertiliser", "nitrogen", "irrigation",
  "tillage", "spacing", "density", "rate", "dose", "treatment",
];

/**
 * Infer the most likely research domain from a list of column names.
 * Returns "plant_breeding" when genotype-related columns are detected,
 * "agronomy" when agronomy-management columns are detected, and "general"
 * otherwise.
 */
export function detectDomainFromColumns(columns: string[]): DomainKey {
  const lower = columns.map((c) => c.toLowerCase());
  if (BREEDING_KEYWORDS.some((kw) => lower.some((c) => c.includes(kw)))) {
    return "plant_breeding";
  }
  if (AGRONOMY_KEYWORDS.some((kw) => lower.some((c) => c.includes(kw)))) {
    return "agronomy";
  }
  return "general";
}

// ─────────────────────────────────────────────────────────────────────────────
// Correlation mode governance helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Keywords that indicate a treatment/genotype column contains genotype semantics. */
const GENOTYPE_COLUMN_KEYWORDS = [
  "genotype", "variety", "cultivar", "accession", "line", "cross",
  "hybrid", "clone", "germplasm", "breed", "entry", "selection",
];

/**
 * Return which correlation modes are scientifically appropriate for the given
 * research domain and treatment-column semantics.
 *
 * Governance rules (see problem statement):
 *  - Phenotypic correlation: ALWAYS shown.
 *  - Between-genotype association: shown when treatments represent
 *    genotypes/varieties/cultivars/lines (domain === "plant_breeding" OR
 *    the genotype-column name contains a genotype-related keyword).
 *  - Genotypic correlation (VC-based): shown ONLY for plant_breeding domain.
 *    Hidden for agronomy, general, fertilizer trials, spacing experiments, etc.
 */
export function deriveAllowedCorrelationModes(
  domain: DomainKey | undefined,
  genotypeColumn: string
): { showBetweenGenotype: boolean; showGenotypic: boolean } {
  const d = domain ?? "general";
  const colLower = genotypeColumn.toLowerCase();

  // Column contains genotype-type semantics?
  const columnIsGenotypeType = GENOTYPE_COLUMN_KEYWORDS.some((kw) =>
    colLower.includes(kw)
  );

  const showBetweenGenotype = d === "plant_breeding" || columnIsGenotypeType;
  const showGenotypic = d === "plant_breeding";

  return { showBetweenGenotype, showGenotypic };
}
