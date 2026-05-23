/**
 * VivaSense Free vs Pro gating
 * -----------------------------------------------------------------
 * Centralizes the rules that decide whether a given analysis request
 * can run in Free Mode or requires the Pro tier.
 *
 * Free tier allowances:
 *   - Descriptive statistics (single trait or multi-column)
 *   - Basic single-environment ANOVA (oneway, oneway_rcbd, twoway,
 *     rcbd_factorial, splitplot, multitrait variants of the same)
 *
 * Pro-only:
 *   - Combined / multi-environment ANOVA (any environment / location /
 *     year factor, or G×E)
 *   - Genetic parameters (variance components, heritability, GA…)
 *   - PCA / multivariate
 *   - Path analysis (correlations module covers this)
 *   - Clustering (multivariate)
 *   - Selection index / MGIDI
 *   - Stability / AMMI / GGE
 *   - Molecular markers
 *   - Multiple regression
 *   - Word export
 *   - Advanced AI interpretation
 *
 * Mode source: hardcoded for now; swap to a hook reading the user's
 * subscription tier when payments land.
 */

export type VivaSenseMode = "free" | "pro";

const STORAGE_KEY = "vivasense_mode";
const MODE_CHANGE_EVENT = "vivasense:mode-change";

const isValidMode = (v: unknown): v is VivaSenseMode => v === "free" || v === "pro";

/**
 * Resolve the current VivaSense mode.
 *
 * Precedence:
 *   1. ?mode=pro|free in the URL — persists to localStorage and wins.
 *   2. Existing localStorage value (preserved across refreshes).
 *   3. Default: "free".
 */
export function getVivaSenseMode(): VivaSenseMode {
  if (typeof window === "undefined") return "free";

  try {
    const params = new URLSearchParams(window.location.search);
    const urlMode = params.get("mode");
    if (isValidMode(urlMode)) {
      const current = window.localStorage.getItem(STORAGE_KEY);
      if (current !== urlMode) {
        window.localStorage.setItem(STORAGE_KEY, urlMode);
        window.dispatchEvent(new CustomEvent(MODE_CHANGE_EVENT, { detail: urlMode }));
      }
      return urlMode;
    }

    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (isValidMode(stored)) return stored;
  } catch {
    // localStorage may be unavailable (SSR, privacy mode) — fall through.
  }

  return "free";
}

/** Manually set the mode (e.g. from a debug toggle). */
export function setVivaSenseMode(mode: VivaSenseMode): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, mode);
    window.dispatchEvent(new CustomEvent(MODE_CHANGE_EVENT, { detail: mode }));
  } catch {
    // ignore
  }
}

/** Subscribe to mode changes (URL flip, manual setVivaSenseMode, or other tabs). */
export function subscribeVivaSenseMode(cb: (mode: VivaSenseMode) => void): () => void {
  if (typeof window === "undefined") return () => {};
  const onCustom = (e: Event) => {
    const detail = (e as CustomEvent).detail;
    if (isValidMode(detail)) cb(detail);
  };
  const onStorage = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY && isValidMode(e.newValue)) cb(e.newValue);
  };
  window.addEventListener(MODE_CHANGE_EVENT, onCustom);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(MODE_CHANGE_EVENT, onCustom);
    window.removeEventListener("storage", onStorage);
  };
}

export type ProGuardKey =
  | "anova_combined"
  | "anova_multi_environment"
  | "anova_gxe"
  | "genetic_parameters"
  | "pca"
  | "path_analysis"
  | "clustering"
  | "selection_index"
  | "stability"
  | "ammi"
  | "gge"
  | "molecular"
  | "regression"
  | "export_word"
  | "advanced_interpretation";

export interface ProGuardInfo {
  key: ProGuardKey;
  title: string;
  description: string;
}

const PRO_GUARDS: Record<ProGuardKey, ProGuardInfo> = {
  anova_combined: {
    key: "anova_combined",
    title: "Combined ANOVA",
    description:
      "Combined ANOVA across environments, locations, or years is a Pro feature. Free Mode supports basic single-environment ANOVA.",
  },
  anova_multi_environment: {
    key: "anova_multi_environment",
    title: "Multi-environment ANOVA",
    description:
      "Multi-environment analysis (year / location / environment factor) is a Pro feature.",
  },
  anova_gxe: {
    key: "anova_gxe",
    title: "Genotype × Environment Analysis",
    description:
      "G×E interaction analysis is a Pro feature. Upgrade to access stability and adaptability tools.",
  },
  genetic_parameters: {
    key: "genetic_parameters",
    title: "Genetic Parameters & Heritability",
    description:
      "Variance components, heritability, GCV/PCV, and genetic advance estimation are Pro features.",
  },
  pca: {
    key: "pca",
    title: "Principal Component Analysis",
    description: "PCA and dimensionality reduction are Pro features.",
  },
  path_analysis: {
    key: "path_analysis",
    title: "Path Analysis",
    description:
      "Path coefficient analysis and correlation decomposition are Pro features.",
  },
  clustering: {
    key: "clustering",
    title: "Cluster Analysis",
    description:
      "Hierarchical clustering and dendrograms are Pro features.",
  },
  selection_index: {
    key: "selection_index",
    title: "Selection Index / MGIDI",
    description:
      "Selection index and MGIDI multi-trait selection are Pro features.",
  },
  stability: {
    key: "stability",
    title: "Stability Analysis",
    description:
      "Eberhart–Russell stability analysis is a Pro feature.",
  },
  ammi: {
    key: "ammi",
    title: "AMMI Analysis",
    description: "AMMI model fitting and biplots are Pro features.",
  },
  gge: {
    key: "gge",
    title: "GGE Biplot",
    description: "GGE biplot and which-won-where analysis are Pro features.",
  },
  molecular: {
    key: "molecular",
    title: "Molecular Marker Analysis",
    description: "SSR / SNP marker diversity analysis is a Pro feature.",
  },
  regression: {
    key: "regression",
    title: "Multiple Regression",
    description:
      "Stepwise multiple regression with VIF diagnostics is a Pro feature.",
  },
  export_word: {
    key: "export_word",
    title: "Word Export",
    description:
      "Exporting publication-ready Word reports is a Pro feature.",
  },
  advanced_interpretation: {
    key: "advanced_interpretation",
    title: "Advanced AI Interpretation",
    description:
      "AI-generated extended interpretation and breeding implications are Pro features.",
  },
};

export const getProGuardInfo = (key: ProGuardKey): ProGuardInfo => PRO_GUARDS[key];

/** Names that strongly suggest a multi-environment / G×E factor. */
const ENV_FACTOR_PATTERN =
  /\b(env|environment|envt|loc|location|site|year|yr|season|trial|ge|gxe|g_x_e|g x e)\b/i;

const looksLikeEnvName = (name?: string | null): boolean => {
  if (!name) return false;
  return ENV_FACTOR_PATTERN.test(name.trim());
};

/**
 * Inspect an ANOVA FormData payload and decide whether it's a basic
 * single-environment design (Free) or combined / multi-env / G×E (Pro).
 *
 * Returns null when the request is allowed in Free Mode, or a
 * ProGuardInfo describing which Pro feature is required.
 */
export function classifyAnovaRequest(
  analysisType: string,
  formData: FormData
): ProGuardInfo | null {
  // Fields that, if present, may carry an environment factor name.
  const candidateFields = [
    "factor",
    "factor_a",
    "factor_b",
    "main_plot",
    "sub_plot",
    "treatment",
    "by",
  ];

  const triggered = candidateFields.some((field) =>
    looksLikeEnvName(formData.get(field) as string | null)
  );

  if (triggered) {
    // Pick the most specific guard message we can.
    const a = (formData.get("factor_a") as string) || "";
    const b = (formData.get("factor_b") as string) || "";
    if (looksLikeEnvName(a) && looksLikeEnvName(b)) {
      return PRO_GUARDS.anova_gxe;
    }
    if (a || b) return PRO_GUARDS.anova_gxe;
    return PRO_GUARDS.anova_multi_environment;
  }

  // Some backends accept an explicit `environment` field — treat as combined.
  if (formData.get("environment") || formData.get("location") || formData.get("year")) {
    return PRO_GUARDS.anova_combined;
  }

  // Basic single-environment ANOVA → allowed in Free Mode.
  return null;
}

/** Map of genetics analysis types → Pro guard key. */
const GENETICS_PRO_MAP: Record<string, ProGuardKey> = {
  variance_components: "genetic_parameters",
  stability: "stability",
  ammi: "ammi",
  gge: "gge",
  correlations: "path_analysis",
  multivariate: "pca",
  molecular: "molecular",
  regression: "regression",
};

export function classifyGeneticsRequest(
  analysisType: string
): ProGuardInfo | null {
  const key = GENETICS_PRO_MAP[analysisType];
  return key ? PRO_GUARDS[key] : null;
}

/** True when the current mode permits Pro features. */
export const isProMode = (): boolean => getVivaSenseMode() === "pro";
