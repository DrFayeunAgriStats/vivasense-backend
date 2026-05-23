/**
 * Design-Aware and Domain-Aware Label Generation
 * Dynamically adapts field labels based on experimental design and research domain
 */

import { DomainKey, DOMAIN_TERMS } from "./domainTerms";

type DesignType = "crd" | "rcbd" | "factorial" | "split_plot_rcbd";

export interface DesignAwareLabels {
  treatmentLabel: string;
  treatmentLabelPlural: string;
  blockingLabel: string;
  factorALabel: string;
  factorBLabel: string;
  mainPlotLabel: string;
  subplotLabel: string;
  responseLabel: string;
  designDescription: string;
}

/**
 * Generate design-aware and domain-aware labels for UI fields
 */
export function getDesignAwareLabels(
  design: DesignType,
  domain: DomainKey = "general"
): DesignAwareLabels {
  const terms = DOMAIN_TERMS[domain];

  // Base treatment terminology from domain
  const treatmentLabel = terms.treatment;
  const treatmentLabelPlural = terms.treatments;

  // Split-plot uses domain-neutral factor terminology
  if (design === "split_plot_rcbd") {
    return {
      treatmentLabel: "Factor", // Domain-neutral for split-plot
      treatmentLabelPlural: "Factors",
      blockingLabel: "Replication / Block",
      factorALabel: "Factor A",
      factorBLabel: "Factor B",
      mainPlotLabel: "Main-Plot Factor",
      subplotLabel: "Subplot Factor",
      responseLabel: "Response Variables",
      designDescription:
        "Split-plot design: main-plot factors assigned to whole plots, subplot factors nested within",
    };
  }

  // Factorial uses factor-specific terminology
  if (design === "factorial") {
    return {
      treatmentLabel: terms.treatment,
      treatmentLabelPlural: treatmentLabelPlural,
      blockingLabel: "Replication / Block",
      factorALabel: "Factor A",
      factorBLabel: "Factor B",
      mainPlotLabel: "Factor A",
      subplotLabel: "Factor B",
      responseLabel: "Response Variables",
      designDescription: "Factorial design: tests all combinations of Factor A and Factor B",
    };
  }

  // RCBD uses domain-specific treatment + blocking
  if (design === "rcbd") {
    return {
      treatmentLabel,
      treatmentLabelPlural,
      blockingLabel: "Replication / Block",
      factorALabel: "Factor A",
      factorBLabel: "Factor B",
      mainPlotLabel: "Main Factor",
      subplotLabel: "Subplot Factor",
      responseLabel: "Response Variables",
      designDescription: `RCBD: ${treatmentLabelPlural.toLowerCase()} randomized within each block`,
    };
  }

  // CRD uses domain-specific treatment only (no blocking)
  return {
    treatmentLabel,
    treatmentLabelPlural,
    blockingLabel: "Replication",
    factorALabel: "Factor A",
    factorBLabel: "Factor B",
    mainPlotLabel: "Main Factor",
    subplotLabel: "Subplot Factor",
    responseLabel: "Response Variables",
    designDescription: `CRD: ${treatmentLabelPlural.toLowerCase()} completely randomized (no blocking)`,
  };
}

/**
 * Get placeholder text for column selectors based on design and field
 */
export function getColumnPlaceholder(
  field: "treatment" | "blocking" | "mainPlot" | "subplot" | "factorA" | "factorB",
  design: DesignType,
  domain: DomainKey = "general"
): string {
  const labels = getDesignAwareLabels(design, domain);

  const placeholders = {
    treatment: `Select ${labels.treatmentLabel.toLowerCase()} column…`,
    blocking: `Select ${labels.blockingLabel.toLowerCase()} column…`,
    mainPlot: `Select ${labels.mainPlotLabel.toLowerCase()} column…`,
    subplot: `Select ${labels.subplotLabel.toLowerCase()} column…`,
    factorA: `Select ${labels.factorALabel} column…`,
    factorB: `Select ${labels.factorBLabel} column…`,
  };

  return placeholders[field];
}

/**
 * Get help text for fields based on design context
 */
export function getFieldHelpText(
  field: "treatment" | "blocking" | "mainPlot" | "subplot" | "factorA" | "factorB",
  design: DesignType
): string | null {
  if (design === "split_plot_rcbd") {
    const helpTexts = {
      blocking: "Blocks control spatial or temporal variability",
      mainPlot: "Coarse treatment factor applied to whole plots (e.g., irrigation, variety)",
      subplot: "Fine treatment factor applied within each whole plot (e.g., nitrogen rate, spacing)",
      treatment: null,
      factorA: null,
      factorB: null,
    };
    return helpTexts[field];
  }

  if (design === "factorial") {
    const helpTexts = {
      factorA: "First treatment factor (e.g., nitrogen level)",
      factorB: "Second treatment factor (e.g., spacing)",
      blocking: "Blocks control variability — each block contains all factor combinations",
      treatment: null,
      mainPlot: null,
      subplot: null,
    };
    return helpTexts[field];
  }

  if (design === "rcbd") {
    return field === "blocking"
      ? "Each block contains one replicate of each treatment"
      : null;
  }

  return null;
}
