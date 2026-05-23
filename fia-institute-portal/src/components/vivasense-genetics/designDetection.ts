/**
 * Design Detection Utilities
 * Auto-detect experimental design patterns from uploaded dataset structure
 */

export interface DesignDetectionResult {
  suggestedDesign: "crd" | "rcbd" | "factorial" | "split_plot_rcbd";
  confidence: "high" | "medium" | "low";
  reasons: string[];
  detectedFactors: {
    blocking?: string[];
    categorical?: string[];
    possibleMainPlot?: string[];
    possibleSubplot?: string[];
  };
}

export interface ColumnAnalysis {
  name: string;
  uniqueValues: number;
  isNumeric: boolean;
  sampleValues: any[];
}

// Keywords that suggest blocking/replication structure
const BLOCKING_KEYWORDS = [
  "block", "blk", "rep", "replication", "replicate",
];

// Keywords that suggest factorial structure
const FACTORIAL_KEYWORDS = [
  "factor", "treatment", "level", "dose",
];

// Keywords that suggest main-plot factors (coarse treatments)
const MAIN_PLOT_KEYWORDS = [
  "irrigation", "tillage", "variety", "genotype", "cultivar",
  "fertilizer", "fertiliser", "management", "system",
];

// Keywords that suggest subplot factors (fine treatments)
const SUBPLOT_KEYWORDS = [
  "nitrogen", "n_rate", "n_level", "spacing", "density",
  "date", "time", "application", "rate", "dose",
];

/**
 * Check if a column name matches any keywords (case-insensitive partial match)
 */
function matchesKeywords(columnName: string, keywords: string[]): boolean {
  const lower = columnName.toLowerCase().trim();
  return keywords.some(keyword => lower.includes(keyword));
}

/**
 * Detect likely experimental design from column structure
 */
export function detectExperimentalDesign(
  columns: ColumnAnalysis[],
  columnNames: string[]
): DesignDetectionResult {
  const categorical = columns.filter(c => !c.isNumeric && c.uniqueValues > 1);
  const reasons: string[] = [];
  
  // Detect blocking structure
  const blockingCols = categorical.filter(c =>
    matchesKeywords(c.name, BLOCKING_KEYWORDS)
  );
  
  // Detect factorial structure (2+ non-blocking categorical factors)
  const factorialCandidates = categorical.filter(c =>
    !matchesKeywords(c.name, BLOCKING_KEYWORDS) &&
    !matchesKeywords(c.name, ["environment", "env", "location", "site"])
  );
  
  // Detect split-plot indicators
  const mainPlotCandidates = categorical.filter(c =>
    matchesKeywords(c.name, MAIN_PLOT_KEYWORDS) &&
    !matchesKeywords(c.name, BLOCKING_KEYWORDS)
  );
  
  const subplotCandidates = categorical.filter(c =>
    matchesKeywords(c.name, SUBPLOT_KEYWORDS) &&
    !matchesKeywords(c.name, BLOCKING_KEYWORDS)
  );
  
  // Decision tree for design detection
  
  // Split-plot RCBD: blocking + main-plot + subplot indicators
  if (
    blockingCols.length >= 1 &&
    mainPlotCandidates.length >= 1 &&
    subplotCandidates.length >= 1
  ) {
    reasons.push(`Detected blocking column: ${blockingCols[0].name}`);
    reasons.push(`Detected main-plot factor: ${mainPlotCandidates[0].name}`);
    reasons.push(`Detected subplot factor: ${subplotCandidates[0].name}`);
    reasons.push("Split-plot designs apply two-level randomization");
    
    return {
      suggestedDesign: "split_plot_rcbd",
      confidence: "high",
      reasons,
      detectedFactors: {
        blocking: blockingCols.map(c => c.name),
        possibleMainPlot: mainPlotCandidates.map(c => c.name),
        possibleSubplot: subplotCandidates.map(c => c.name),
      },
    };
  }
  
  // Factorial RCBD: blocking + 2+ treatment factors
  if (blockingCols.length >= 1 && factorialCandidates.length >= 2) {
    reasons.push(`Detected blocking column: ${blockingCols[0].name}`);
    reasons.push(`Detected ${factorialCandidates.length} treatment factors`);
    reasons.push("Factorial designs test factor interactions");
    
    return {
      suggestedDesign: "factorial",
      confidence: "medium",
      reasons,
      detectedFactors: {
        blocking: blockingCols.map(c => c.name),
        categorical: factorialCandidates.map(c => c.name),
      },
    };
  }
  
  // RCBD: blocking present
  if (blockingCols.length >= 1) {
    reasons.push(`Detected blocking column: ${blockingCols[0].name}`);
    reasons.push("Blocking controls spatial/temporal variability");
    
    return {
      suggestedDesign: "rcbd",
      confidence: "high",
      reasons,
      detectedFactors: {
        blocking: blockingCols.map(c => c.name),
      },
    };
  }
  
  // CRD: no blocking
  reasons.push("No blocking structure detected");
  reasons.push("Completely randomized design (no systematic control)");
  
  return {
    suggestedDesign: "crd",
    confidence: "medium",
    reasons,
    detectedFactors: {
      categorical: factorialCandidates.map(c => c.name),
    },
  };
}

/**
 * Generate human-readable design recommendation message
 */
export function formatDesignRecommendation(result: DesignDetectionResult): string {
  const designNames = {
    crd: "Completely Randomized Design (CRD)",
    rcbd: "Randomized Complete Block Design (RCBD)",
    factorial: "Factorial Design",
    split_plot_rcbd: "Split-Plot RCBD",
  };
  
  return `${designNames[result.suggestedDesign]} detected with ${result.confidence} confidence`;
}
