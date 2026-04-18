/**
 * Preview builder utilities for all VivaSense modules.
 * Shared logic for constructing consistent Word export preview layouts.
 */

import { PreviewSection } from "./normalizeModuleData";

/**
 * Build ANOVA analysis preview sections.
 * Normalizes ANOVA data into a consistent preview layout.
 */
export function buildAnovaPreview(data: {
  designType?: string;
  nTreatments?: number;
  nReplicates?: number;
  nEnvironments?: number;
  anovaTableSources?: string[];
  significantSources?: string[];
  interpretation?: string;
}): {
  sections: PreviewSection[];
  warnings: string[];
  notes: string[];
} {
  const sections: PreviewSection[] = [];
  const warnings: string[] = [];
  const notes: string[] = [];

  // Design overview
  sections.push({
    title: "Experimental Design",
    rows: [
      ...(data.designType ? [{ label: "Design", value: data.designType }] : []),
      ...(data.nTreatments ? [{ label: "Treatments", value: String(data.nTreatments) }] : []),
      ...(data.nReplicates ? [{ label: "Replicates", value: String(data.nReplicates) }] : []),
      ...(data.nEnvironments ? [{ label: "Environments", value: String(data.nEnvironments) }] : []),
    ],
  });

  // ANOVA summary
  if (data.anovaTableSources && data.anovaTableSources.length > 0) {
    sections.push({
      title: "ANOVA Summary",
      rows: [
        { label: "Sources of variation", value: data.anovaTableSources.join(", ") },
        ...(data.significantSources
          ? [{ label: "Significant sources (p < 0.05)", value: data.significantSources.join(", ") }]
          : []),
      ],
    });
  } else {
    warnings.push("No ANOVA data available for this trait.");
  }

  if (data.interpretation) {
    notes.push(data.interpretation);
  }

  return { sections, warnings, notes };
}

/**
 * Build Genetic Parameters preview sections.
 * Consolidates heritability, variability, and genetic advance data.
 */
export function buildGeneticParametersPreview(data: {
  nTraits?: number;
  heritability?: Array<{ trait: string; h2: number; class?: string }>;
  variability?: Array<{ label: string; value: number; unit?: string }>;
  geneticAdvance?: Array<{ trait: string; gam: number }>;
  interpretation?: string;
}): {
  sections: PreviewSection[];
  warnings: string[];
  notes: string[];
} {
  const sections: PreviewSection[] = [];
  const warnings: string[] = [];
  const notes: string[] = [];

  // Summary
  sections.push({
    title: "Dataset Summary",
    rows: [{ label: "Traits analyzed", value: String(data.nTraits ?? 0) }],
  });

  // Heritability
  if (data.heritability && data.heritability.length > 0) {
    sections.push({
      title: "Broad-sense Heritability (H²)",
      rows: data.heritability.map(({ trait, h2, class: cls }) => ({
        label: trait,
        value: `${h2.toFixed(3)} ${cls ? `(${cls})` : ""}`,
      })),
      note: "H² ranges from 0 (no genetic effect) to 1 (perfect heritability). Class: low (<0.3), moderate (0.3–0.6), high (>0.6).",
    });
  }

  // Variability
  if (data.variability && data.variability.length > 0) {
    sections.push({
      title: "Phenotypic Variability",
      rows: data.variability.map(({ label, value, unit }) => ({
        label,
        value: unit ? `${value.toFixed(1)}${unit}` : String(value.toFixed(1)),
      })),
      note: "GCV = genotypic coefficient of variation; PCV = phenotypic coefficient of variation.",
    });
  }

  // Genetic Advance
  if (data.geneticAdvance && data.geneticAdvance.length > 0) {
    sections.push({
      title: "Genetic Advance as % of Mean (GAM)",
      rows: data.geneticAdvance.map(({ trait, gam }) => ({
        label: trait,
        value: `${gam.toFixed(1)}%`,
      })),
      note: "GAM predicts the response to selection. Higher values indicate greater breeding potential.",
    });
  }

  if (data.interpretation) {
    notes.push(data.interpretation);
  }

  return { sections, warnings, notes };
}

/**
 * Build Correlation/Heatmap preview sections.
 * Constructs preview for dual-mode correlation matrices.
 */
export function buildCorrelationHeatmapPreview(data: {
  nTraits?: number;
  method?: "pearson" | "spearman";
  userObjective?: string;
  activeMode?: "phenotypic" | "between_genotype" | "genotypic";
  phenotypicN?: number;
  genotypicN?: number;
  significantPairs?: number;
  interpretation?: string;
}): {
  sections: PreviewSection[];
  warnings: string[];
  notes: string[];
} {
  const sections: PreviewSection[] = [];
  const warnings: string[] = [];
  const notes: string[] = [];

  // Analysis settings
  sections.push({
    title: "Analysis Settings",
    rows: [
      ...(data.nTraits ? [{ label: "Traits", value: String(data.nTraits) }] : []),
      ...(data.method ? [{ label: "Correlation method", value: data.method === "spearman" ? "Spearman ρ" : "Pearson r" }] : []),
      ...(data.userObjective ? [{ label: "User objective", value: data.userObjective }] : []),
      ...(data.activeMode ? [{ label: "Visualization mode", value: data.activeMode }] : []),
    ],
  });

  // Data availability
  sections.push({
    title: "Data Availability",
    rows: [
      ...(data.phenotypicN ? [{ label: "Phenotypic observations", value: String(data.phenotypicN) }] : []),
      ...(data.genotypicN ? [{ label: "Genotypic means", value: String(data.genotypicN) }] : []),
      ...(data.significantPairs ? [{ label: "Significant pairs (p < 0.05)", value: String(data.significantPairs) }] : []),
    ],
  });

  if (data.interpretation) {
    notes.push(data.interpretation);
  }

  // Warn if activeMode is between_genotype but user asked for breeding decision
  if (
    data.activeMode === "between_genotype" &&
    (data.userObjective === "Breeding decision" || data.userObjective === "Genotype comparison")
  ) {
    warnings.push(
      "Displaying between-genotype association as fallback. Full genotypic VC unavailable — check server configuration."
    );
  }

  return { sections, warnings, notes };
}

/**
 * Common utility: format a numeric value with unit.
 */
export function formatStat(value: number | null | undefined, decimals: number = 2, unit: string = ""): string {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(decimals)}${unit}`;
}
