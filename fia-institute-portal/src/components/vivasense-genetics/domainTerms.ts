export type DomainKey = "plant_breeding" | "agronomy" | "general";

export interface DomainTerms {
  treatment: string;
  treatments: string;
  treatment_plural: string;
  variance_module: string;
  variance_short: string;
  h2_label: string;
  gcv_label: string;
  pcv_label: string;
  gam_label: string;
  summary_title: string;
  top_performer: string;
  recommendation: string;
  scope_statement: string;
}

export const RESEARCH_DOMAINS: { value: DomainKey; label: string; icon: string; desc: string }[] = [
  {
    value: "plant_breeding",
    label: "Plant Breeding",
    icon: "🧬",
    desc: "Genotype evaluation, heritability, genetic advance, MET trials",
  },
  {
    value: "agronomy",
    label: "Agronomy",
    icon: "🌾",
    desc: "Fertiliser, irrigation, spacing, tillage, management trials",
  },
  {
    value: "general",
    label: "General",
    icon: "📊",
    desc: "Any experimental data — domain-neutral language throughout",
  },
];

export const DOMAIN_TERMS: Record<DomainKey, DomainTerms> = {
  plant_breeding: {
    treatment: "Genotype",
    treatments: "Genotypes",
    treatment_plural: "genotypes",
    variance_module: "Genetic Parameters",
    variance_short: "Heritability & GCV",
    h2_label: "Heritability (H²)",
    gcv_label: "GCV %",
    pcv_label: "PCV %",
    gam_label: "GAM %",
    summary_title: "Breeding Strategy Summary",
    top_performer: "promising candidate for selection",
    recommendation: "selection and crossing recommendation",
    scope_statement: "cannot support general breeding recommendations",
  },
  agronomy: {
    treatment: "Treatment",
    treatments: "Treatments",
    treatment_plural: "treatments",
    variance_module: "Treatment Variance",
    variance_short: "Variance Components",
    h2_label: "Treatment Repeatability",
    gcv_label: "Treatment CV %",
    pcv_label: "Total CV %",
    gam_label: "Response %",
    summary_title: "Management Recommendations",
    top_performer: "recommended management practice",
    recommendation: "management recommendation",
    scope_statement: "cannot support general management recommendations",
  },
  general: {
    treatment: "Treatment",
    treatments: "Treatments",
    treatment_plural: "treatments",
    variance_module: "Variance Components",
    variance_short: "Variance Analysis",
    h2_label: "Repeatability",
    gcv_label: "Treatment CV %",
    pcv_label: "Total CV %",
    gam_label: "Response %",
    summary_title: "Treatment Recommendations",
    top_performer: "top-performing treatment",
    recommendation: "recommendation",
    scope_statement: "cannot support general recommendations",
  },
};

const PLANT_BREEDING_KEYWORDS = ["genotype", "variety", "cultivar", "accession"];
const AGRONOMY_KEYWORDS = ["fertilizer", "fertiliser", "nitrogen", "irrigation", "spacing", "tillage"];

export function detectDomainFromColumns(columns: string[]): DomainKey {
  const lower = columns.map((c) => c.toLowerCase());
  if (lower.some((c) => PLANT_BREEDING_KEYWORDS.some((k) => c.includes(k)))) {
    return "plant_breeding";
  }
  if (lower.some((c) => AGRONOMY_KEYWORDS.some((k) => c.includes(k)))) {
    return "agronomy";
  }
  return "general";
}

export function getDomainTerms(domain: DomainKey | undefined): DomainTerms {
  return DOMAIN_TERMS[domain ?? "plant_breeding"];
}
