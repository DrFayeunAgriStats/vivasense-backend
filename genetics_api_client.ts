/**
 * VivaSense Genetics API Client
 * ==============================
 * Add this file to your Lovable project root as: src/genetics_api_client.ts
 *
 * In Lovable → Settings → Environment Variables, add:
 *   VITE_GENETICS_ENGINE_BASE = https://vivasense-genetics.onrender.com
 *
 * Then import in your components:
 *   import { runVarianceComponents } from "@/genetics_api_client";
 */

const GENETICS_BASE: string =
  import.meta.env.VITE_GENETICS_ENGINE_BASE ||
  import.meta.env.VITE_GENETICS_API_BASE ||
  "https://vivasense-genetics.onrender.com";

// ─────────────────────────────────────────────────────────────────────────────
// TRAIT NAME MAPPING — display labels → CSV column names
// ─────────────────────────────────────────────────────────────────────────────

const TRAIT_NAME_MAP: Record<string, string> = {
  "Days to Flowering":    "Days_to_Flowering",
  "Plant Height (cm)":   "Plant_Height_cm",
  "Plant Height":        "Plant_Height_cm",
  "Pod Number":          "Pod_Number",
  "Pod Count":           "Pod_Number",
  "Seed Yield (kg/ha)":  "Seed_Yield_kg_ha",
  "Seed Yield":          "Seed_Yield_kg_ha",
};

/**
 * Convert a display-friendly trait name to the exact CSV column name.
 * Explicit entries in TRAIT_NAME_MAP take priority.
 * Fallback: spaces → underscores, parentheses stripped.
 */
function traitDisplayToColumn(displayName: string): string {
  if (TRAIT_NAME_MAP[displayName]) {
    return TRAIT_NAME_MAP[displayName];
  }
  return displayName
    .replace(/\s+/g, "_")
    .replace(/[()]/g, "");
}

// ─────────────────────────────────────────────────────────────────────────────
// TYPES — matches the V2.2 response envelope from the backend
// ─────────────────────────────────────────────────────────────────────────────

export interface GeneticsRequestOpts {
  /** Column name for genotype/variety. Default: "Genotype" */
  genotypeCol?: string;
  /** Column name for location/environment. Default: "Location" */
  locationCol?: string;
  /** Column name for replicate/block. Default: "Rep" */
  repCol?: string;
  /** Significance level. Default: 0.05 */
  alpha?: number;
  /** Number of AMMI axes to retain. Default: 2 */
  nAmmiAxes?: number;
  /** Which trait to feature in the main envelope (blank = first detected) */
  primaryTrait?: string;
}

/** Alias — import whichever name your component uses */
export type GeneticsOptions = GeneticsRequestOpts;

export interface VarianceComponentRecord {
  grand_mean?: number;
  n_genotypes?: number;
  n_locations?: number;
  sigma2_g?: number;
  sigma2_e?: number;
  sigma2_gl?: number;
  sigma2_p?: number;
  H2_broad?: number;
  H2_broad_pct?: number;
  GA?: number;
  GA_percent?: number;
  GCV?: number;
  PCV?: number;
  ECV?: number;
}

export interface GenotypeMeanRecord {
  [genotypeCol: string]: string;
  mean?: number;
  letter?: string;
  tukey_letter?: string;
}

export interface StabilityRecord {
  genotype?: string;
  grand_mean?: number;
  bi?: number | { value: number; interpretation: string };
  S2di?: number | { value: number; significant: boolean };
  ASV?: number | { value: number; rank: number };
  classification?: string;
}

export interface HtmlTable {
  name: string;
  html: string;
}

export interface PublicationFigure {
  name: string;
  caption: string;
  image_base64: string;
}

export interface GeneticsMeta {
  design: string;
  trait: string;
  n_genotypes: number;
  n_locations: number;
  analysis_id: string;
  timestamp: string;
  filename: string;
  genotype_col: string;
  location_col: string;
  rep_col: string;
  mode: string;
  warnings: string[];
}

export interface GeneticsEnvelope {
  meta: GeneticsMeta;
  tables: {
    combined_anova?: Record<string, unknown>[];
    variance_components?: VarianceComponentRecord[];
    genotype_means?: GenotypeMeanRecord[];
    stability?: StabilityRecord[];
    assumptions?: Record<string, unknown>[];
    assumption_guidance?: Record<string, unknown>;
    ammi_ipca?: Record<string, unknown>[];
    ammi_explained_variance?: Record<string, unknown>[];
    gge_which_won_where?: Record<string, unknown>[];
    correlations?: Record<string, unknown>;
    path_analysis?: Record<string, unknown>[] | Record<string, unknown>;
    selection_index?: Record<string, unknown>[] | Record<string, unknown>;
    multivariate?: Record<string, unknown>;
    [key: string]: unknown;
  };
  plots: Record<string, string>;                // key → base64 PNG
  html_tables: HtmlTable[];                     // copy-paste ready HTML
  publication_figures: PublicationFigure[];     // 300 DPI base64 figures
  publication_tables: Record<string, unknown>;  // structured JSON tables
  interpretation: string;
  strict_template: Record<string, unknown>;
  intelligence: {
    executive_insight: string;
    reviewer_radar: Record<string, unknown>;
    decision_rules: Record<string, unknown>[];
    assumptions_verdict: string;
  };
  per_trait?: Record<string, GeneticsEnvelope>; // when multiple traits selected
}

// ─────────────────────────────────────────────────────────────────────────────
// SHARED UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a FormData object for any genetics analysis endpoint.
 *
 * IMPORTANT: traits are sent as a single comma-separated string —
 * e.g., "Days_to_Flowering,Plant_Height_cm,Pod_Number,Seed_Yield_kg_ha"
 * Column names must match the CSV header EXACTLY (case-sensitive).
 */
function buildGeneticsFormData(
  file: File,
  selectedTraits: string[] = [],
  opts: GeneticsRequestOpts = {}
): FormData {
  const fd = new FormData();
  fd.append("file", file);

  // ✅ CORRECT: map display names → CSV column names, then join
  if (selectedTraits.length > 0) {
    const csvTraitNames = selectedTraits.map(traitDisplayToColumn);
    console.log("[Genetics] Trait name conversion:");
    selectedTraits.forEach((display, i) =>
      console.log(`  "${display}" → "${csvTraitNames[i]}"`)
    );
    fd.append("traits", csvTraitNames.join(","));
  }

  fd.append("genotype_col", opts.genotypeCol ?? "Genotype");
  fd.append("location_col", opts.locationCol ?? "Location");
  fd.append("rep_col",      opts.repCol      ?? "Rep");
  fd.append("alpha",        String(opts.alpha     ?? 0.05));
  fd.append("n_ammi_axes",  String(opts.nAmmiAxes ?? 2));

  if (opts.primaryTrait) {
    fd.append("primary_trait_col", opts.primaryTrait);
  }

  return fd;
}

/**
 * POST to a genetics endpoint and parse the JSON response.
 * Throws a descriptive error (never silently fails to "Failed to fetch").
 */
async function postGeneticsEndpoint(
  endpoint: string,
  formData: FormData
): Promise<GeneticsEnvelope> {
  const url = `${GENETICS_BASE}${endpoint}`;

  let response: Response;
  try {
    response = await fetch(url, { method: "POST", body: formData });
  } catch (networkErr: unknown) {
    const msg = networkErr instanceof Error ? networkErr.message : String(networkErr);
    throw new Error(
      `Network error reaching ${url}: ${msg}. ` +
      "Verify the backend is running and VITE_GENETICS_API_BASE is set correctly."
    );
  }

  if (!response.ok) {
    let detail = `HTTP ${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail ?? body);
    } catch {
      try { detail = await response.text(); } catch { /* ignore */ }
    }
    throw new Error(`Analysis failed — ${detail}`);
  }

  return response.json() as Promise<GeneticsEnvelope>;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. FULL TRIAL ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Run the complete genetics pipeline in one request:
 * variance components, AMMI, GGE, stability, correlations,
 * path analysis, selection index, PCA, and clustering.
 *
 * Use this when you want everything at once.
 * Use the individual functions below for targeted analyses.
 *
 * @param file            CSV or Excel breeding trial file
 * @param traits          e.g. ["Yield", "Height", "DaysToFlowering"]
 * @param opts            column names, alpha, etc.
 */
export async function runFullTrialAnalysis(
  file: File,
  traits: string[],
  opts: GeneticsRequestOpts = {}
): Promise<GeneticsEnvelope> {
  return postGeneticsEndpoint("/analyze/genetics/trial", buildGeneticsFormData(file, traits, opts));
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. VARIANCE COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Compute σ²g, σ²e, σ²gl, σ²p, H², GA, GCV, PCV, ECV.
 * Requires ≥2 locations for G×L interaction estimates.
 */
export async function runVarianceComponents(
  file: File,
  traits: string[],
  opts: GeneticsRequestOpts = {}
): Promise<GeneticsEnvelope> {
  return postGeneticsEndpoint(
    "/analyze/genetics/variance-components",
    buildGeneticsFormData(file, traits, opts)
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. STABILITY ANALYSIS  (Eberhart & Russell 1966)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Compute bᵢ, S²di, ASV and classify genotypes as stable / unstable.
 * Requires ≥2 locations.
 */
export async function runStabilityAnalysis(
  file: File,
  traits: string[],
  opts: GeneticsRequestOpts = {}
): Promise<GeneticsEnvelope> {
  return postGeneticsEndpoint(
    "/analyze/genetics/stability",
    buildGeneticsFormData(file, traits, opts)
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. AMMI ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────
/**
 * AMMI model: combined ANOVA partitioned into G, E, G×E, IPCA scores,
 * and biplot-ready coordinates. Requires ≥3 locations.
 */
export async function runAmmiAnalysis(
  file: File,
  traits: string[],
  opts: GeneticsRequestOpts = {}
): Promise<GeneticsEnvelope> {
  return postGeneticsEndpoint(
    "/analyze/genetics/ammi",
    buildGeneticsFormData(file, traits, opts)
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. GGE BIPLOT
// ─────────────────────────────────────────────────────────────────────────────
/**
 * GGE biplot: which-won-where sectors, ideal genotype, mega-environment
 * delineation. Requires ≥2 locations.
 */
export async function runGgeBiplot(
  file: File,
  traits: string[],
  opts: GeneticsRequestOpts = {}
): Promise<GeneticsEnvelope> {
  return postGeneticsEndpoint(
    "/analyze/genetics/gge",
    buildGeneticsFormData(file, traits, opts)
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. CORRELATIONS, PATH ANALYSIS & SELECTION INDEX
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Phenotypic + genotypic correlation matrices, path analysis coefficients,
 * and Smith-Hazel selection index. Requires ≥2 trait columns.
 */
export async function runCorrelations(
  file: File,
  traits: string[],
  opts: GeneticsRequestOpts = {}
): Promise<GeneticsEnvelope> {
  return postGeneticsEndpoint(
    "/analyze/genetics/correlations",
    buildGeneticsFormData(file, traits, opts)
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. MULTIVARIATE  (PCA + Clustering)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * PCA biplot, hierarchical (UPGMA) dendrogram, and k-means cluster
 * assignments based on genotype means. Requires ≥2 trait columns.
 */
export async function runMultivariateAnalysis(
  file: File,
  traits: string[],
  opts: GeneticsRequestOpts = {}
): Promise<GeneticsEnvelope> {
  return postGeneticsEndpoint(
    "/analyze/genetics/multivariate",
    buildGeneticsFormData(file, traits, opts)
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. MOLECULAR MARKERS
// ─────────────────────────────────────────────────────────────────────────────
export interface MarkerRequestOpts {
  accessionCol?: string;
  markerPrefix?: string;
  nClusters?: number;
  similarityMetric?: "jaccard" | "dice" | "both";
}

/**
 * Jaccard / Dice similarity, PIC values, and UPGMA dendrogram for
 * SSR / RAPD / AFLP molecular marker data.
 */
export async function runMarkerAnalysis(
  file: File,
  opts: MarkerRequestOpts = {}
): Promise<GeneticsEnvelope> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("accession_col",     opts.accessionCol     ?? "Accession");
  fd.append("n_clusters",        String(opts.nClusters        ?? 3));
  fd.append("similarity_metric", opts.similarityMetric ?? "both");
  if (opts.markerPrefix) fd.append("marker_prefix", opts.markerPrefix);

  return postGeneticsEndpoint("/analyze/genetics/markers", fd);
}

// ─────────────────────────────────────────────────────────────────────────────
// HEALTH CHECK
// ─────────────────────────────────────────────────────────────────────────────
export async function checkGeneticsHealth(): Promise<{ status: string; [k: string]: unknown }> {
  try {
    const res = await fetch(`${GENETICS_BASE}/analyze/genetics/health`);
    return res.json();
  } catch {
    return { status: "unreachable" };
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   USAGE EXAMPLES FOR LOVABLE COMPONENTS
   ═══════════════════════════════════════════════════════════════════════════

   ── In your Genetics page component ─────────────────────────────────────────

   import { runVarianceComponents, GeneticsEnvelope } from "@/genetics_api_client";

   const [results, setResults] = useState<GeneticsEnvelope | null>(null);

   const handleRunAnalysis = async () => {
     if (!file) return;
     setLoading(true);
     setError(null);
     try {
       const data = await runVarianceComponents(
         file,
         ["Days_to_Flowering", "Plant_Height_cm", "Pod_Number", "Seed_Yield_kg_ha"],
         { genotypeCol: "Genotype", locationCol: "Location", repCol: "Rep" }
       );
       setResults(data);
     } catch (err) {
       setError(err instanceof Error ? err.message : "Analysis failed");
     } finally {
       setLoading(false);
     }
   };

   ── Read key results ─────────────────────────────────────────────────────────

   results.meta.trait                          // "Seed_Yield_kg_ha"
   results.meta.n_genotypes                    // 12
   results.meta.n_locations                    // 4
   results.tables.variance_components?.[0]     // { sigma2_g, H2_broad, GA, ... }
   results.tables.genotype_means               // sorted by mean desc
   results.tables.stability                    // bi, S²di, ASV, classification
   results.interpretation                      // plain-English paragraph
   results.intelligence.executive_insight      // 3-bullet dashboard summary

   ── Display HTML tables (copy-paste ready) ───────────────────────────────────

   {results.html_tables.map((t) => (
     <div key={t.name} className="mb-6">
       <h3 className="font-bold mb-2">{t.name}</h3>
       <div
         className="overflow-x-auto"
         dangerouslySetInnerHTML={{ __html: t.html }}
       />
     </div>
   ))}

   ── Display 300 DPI publication figures ──────────────────────────────────────

   {results.publication_figures.map((fig) => (
     <figure key={fig.name} className="mb-8">
       <img
         src={`data:image/png;base64,${fig.image_base64}`}
         alt={fig.name}
         className="w-full rounded shadow"
       />
       <figcaption className="text-sm text-gray-600 mt-2 italic">
         {fig.caption}
       </figcaption>
     </figure>
   ))}

   ── Switch between analysis types ────────────────────────────────────────────

   const ANALYSIS_MAP = {
     "Variance Components":  runVarianceComponents,
     "Stability Analysis":   runStabilityAnalysis,
     "AMMI Analysis":        runAmmiAnalysis,
     "GGE Biplot":           runGgeBiplot,
     "Correlations":         runCorrelations,
     "Multivariate (PCA)":   runMultivariateAnalysis,
   } as const;

   const runFn = ANALYSIS_MAP[selectedAnalysis];
   const data = await runFn(file, selectedTraits, { genotypeCol, locationCol, repCol });

   ═══════════════════════════════════════════════════════════════════════════ */
