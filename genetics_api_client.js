/**
 * VivaSense Genetics API Client — Lovable Frontend Integration
 * ============================================================
 * Paste this file into your Lovable project (or import from fia_api_client.js).
 *
 * Set in Lovable → Settings → Environment Variables:
 *   VITE_GENETICS_API_BASE = https://vivasense-backend.onrender.com
 *
 * CORRECT TRAIT FORMAT:
 *   formData.append('traits', selectedTraits.join(','));
 *   // e.g. "Days_to_Flowering,Plant_Height_cm,Pod_Number,Seed_Yield_kg_ha"
 *   // Column names must match CSV header EXACTLY (case-sensitive)
 */

const GENETICS_BASE =
  (typeof import.meta !== "undefined" && import.meta.env?.VITE_GENETICS_API_BASE) ||
  (typeof process !== "undefined" && process.env?.VITE_GENETICS_API_BASE) ||
  "https://vivasense-backend.onrender.com";

// ─────────────────────────────────────────────────────────────────────────────
// TRAIT NAME MAPPING — display labels → CSV column names
// ─────────────────────────────────────────────────────────────────────────────

const TRAIT_NAME_MAP = {
  "Days to Flowering":   "Days_to_Flowering",
  "Plant Height (cm)":  "Plant_Height_cm",
  "Plant Height":       "Plant_Height_cm",
  "Pod Number":         "Pod_Number",
  "Pod Count":          "Pod_Number",
  "Seed Yield (kg/ha)": "Seed_Yield_kg_ha",
  "Seed Yield":         "Seed_Yield_kg_ha",
};

/**
 * Convert a display-friendly trait name to the exact CSV column name.
 * Explicit entries in TRAIT_NAME_MAP take priority.
 * Fallback: spaces → underscores, parentheses stripped.
 * @param {string} displayName
 * @returns {string}
 */
function traitDisplayToColumn(displayName) {
  if (TRAIT_NAME_MAP[displayName]) {
    return TRAIT_NAME_MAP[displayName];
  }
  return displayName
    .replace(/\s+/g, "_")
    .replace(/[()]/g, "");
}

// ─────────────────────────────────────────────────────────────────────────────
// SHARED UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a FormData object from a genetics analysis request.
 * All 7 endpoints share this parameter structure.
 *
 * @param {File}     file            - the uploaded CSV / Excel file
 * @param {string[]} selectedTraits  - trait column names, e.g. ["Yield", "Height"]
 * @param {object}   opts            - optional overrides
 * @param {string}   opts.genotypeCol  - default "Genotype"
 * @param {string}   opts.locationCol  - default "Location"
 * @param {string}   opts.repCol       - default "Rep"
 * @param {string}   opts.primaryTrait - which trait to feature in the main envelope
 * @param {number}   opts.alpha        - significance level, default 0.05
 * @param {number}   opts.nAmmiAxes   - AMMI axes to retain, default 2
 */
function buildGeneticsFormData(file, selectedTraits = [], opts = {}) {
  const fd = new FormData();
  fd.append("file", file);

  // ✅ CORRECT: map display names → CSV column names, then join
  if (selectedTraits.length > 0) {
    const csvTraitNames = selectedTraits.map(traitDisplayToColumn);
    fd.append("traits", csvTraitNames.join(","));
  }

  fd.append("genotype_col",    opts.genotypeCol  || "Genotype");
  fd.append("location_col",    opts.locationCol  || "Location");
  fd.append("rep_col",         opts.repCol       || "Rep");
  fd.append("alpha",           String(opts.alpha     ?? 0.05));
  fd.append("n_ammi_axes",     String(opts.nAmmiAxes ?? 2));

  if (opts.primaryTrait) {
    fd.append("primary_trait_col", opts.primaryTrait);
  }

  return fd;
}

/**
 * Generic POST helper — parses JSON response, throws on HTTP error.
 * @param {string}   endpoint  - path after the base URL, e.g. "/analyze/genetics/trial"
 * @param {FormData} formData
 * @returns {Promise<object>}  the parsed JSON envelope
 */
async function postGeneticsEndpoint(endpoint, formData) {
  const url = `${GENETICS_BASE}${endpoint}`;

  let response;
  try {
    response = await fetch(url, { method: "POST", body: formData });
  } catch (networkErr) {
    // "Failed to fetch" — likely CORS, offline, or wrong URL
    throw new Error(
      `Network error contacting ${url}: ${networkErr.message}. ` +
      "Check that the backend is live and VITE_GENETICS_API_BASE is correct."
    );
  }

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail += `: ${body.detail || JSON.stringify(body)}`;
    } catch {
      detail += `: ${await response.text()}`;
    }
    throw new Error(`Analysis failed — ${detail}`);
  }

  return response.json();
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. FULL TRIAL ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Run the complete genetics pipeline: variance components, stability,
 * AMMI, GGE, correlations, path analysis, selection index, PCA, clustering.
 *
 * @param {File}     file
 * @param {string[]} traits   - e.g. ["Yield", "Height", "DaysToFlowering"]
 * @param {object}   opts     - see buildGeneticsFormData
 * @returns {Promise<object>} full V2.2 envelope
 */
export async function runFullTrialAnalysis(file, traits, opts = {}) {
  const fd = buildGeneticsFormData(file, traits, opts);
  return postGeneticsEndpoint("/analyze/genetics/trial", fd);
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. VARIANCE COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Compute σ²g, σ²e, σ²gl, σ²p, H², GA, GCV, PCV, ECV.
 * Requires ≥2 locations for G×L interaction estimates.
 *
 * @param {File}     file
 * @param {string[]} traits
 * @param {object}   opts
 * @returns {Promise<object>}
 */
export async function runVarianceComponents(file, traits, opts = {}) {
  const fd = buildGeneticsFormData(file, traits, opts);
  return postGeneticsEndpoint("/analyze/genetics/variance-components", fd);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. STABILITY ANALYSIS (Eberhart & Russell 1966)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Compute bᵢ, S²di, ASV, and classify genotypes as stable / unstable.
 * Requires ≥2 locations.
 *
 * @param {File}     file
 * @param {string[]} traits
 * @param {object}   opts
 * @returns {Promise<object>}
 */
export async function runStabilityAnalysis(file, traits, opts = {}) {
  const fd = buildGeneticsFormData(file, traits, opts);
  return postGeneticsEndpoint("/analyze/genetics/stability", fd);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. AMMI ANALYSIS
// ─────────────────────────────────────────────────────────────────────────────
/**
 * AMMI model: ANOVA partition + IPCA scores + biplot data.
 * Requires ≥3 locations.
 *
 * @param {File}     file
 * @param {string[]} traits
 * @param {object}   opts   - include opts.nAmmiAxes (default 2)
 * @returns {Promise<object>}
 */
export async function runAmmiAnalysis(file, traits, opts = {}) {
  const fd = buildGeneticsFormData(file, traits, opts);
  return postGeneticsEndpoint("/analyze/genetics/ammi", fd);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. GGE BIPLOT
// ─────────────────────────────────────────────────────────────────────────────
/**
 * GGE biplot: which-won-where, ideal genotype, mega-environment delineation.
 * Requires ≥2 locations.
 *
 * @param {File}     file
 * @param {string[]} traits
 * @param {object}   opts
 * @returns {Promise<object>}
 */
export async function runGgeBiplot(file, traits, opts = {}) {
  const fd = buildGeneticsFormData(file, traits, opts);
  return postGeneticsEndpoint("/analyze/genetics/gge", fd);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. CORRELATIONS, PATH ANALYSIS, SELECTION INDEX
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Phenotypic + genotypic correlations, path analysis coefficients, and
 * Smith-Hazel selection index. Requires ≥2 trait columns.
 *
 * @param {File}     file
 * @param {string[]} traits   - must include ≥2 trait columns
 * @param {object}   opts
 * @returns {Promise<object>}
 */
export async function runCorrelations(file, traits, opts = {}) {
  const fd = buildGeneticsFormData(file, traits, opts);
  return postGeneticsEndpoint("/analyze/genetics/correlations", fd);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. MULTIVARIATE (PCA + Clustering)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * PCA biplot, hierarchical clustering dendrogram, k-means cluster assignments.
 * Requires ≥2 trait columns.
 *
 * @param {File}     file
 * @param {string[]} traits
 * @param {object}   opts
 * @returns {Promise<object>}
 */
export async function runMultivariateAnalysis(file, traits, opts = {}) {
  const fd = buildGeneticsFormData(file, traits, opts);
  return postGeneticsEndpoint("/analyze/genetics/multivariate", fd);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. MOLECULAR MARKERS
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Jaccard/Dice similarity, PIC, UPGMA dendrogram for molecular marker data.
 * Different column structure — uses accession_col + marker columns.
 *
 * @param {File}   file
 * @param {object} opts
 * @param {string} opts.accessionCol     - default "Accession"
 * @param {string} opts.markerPrefix     - filter marker columns by prefix (optional)
 * @param {number} opts.nClusters        - k for clustering, default 3
 * @param {string} opts.similarityMetric - "jaccard" | "dice" | "both"
 * @returns {Promise<object>}
 */
export async function runMarkerAnalysis(file, opts = {}) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("accession_col",     opts.accessionCol     || "Accession");
  fd.append("n_clusters",        String(opts.nClusters        ?? 3));
  fd.append("similarity_metric", opts.similarityMetric || "both");
  if (opts.markerPrefix) fd.append("marker_prefix", opts.markerPrefix);

  return postGeneticsEndpoint("/analyze/genetics/markers", fd);
}

// ─────────────────────────────────────────────────────────────────────────────
// HEALTH CHECK
// ─────────────────────────────────────────────────────────────────────────────
export async function checkGeneticsHealth() {
  try {
    const res = await fetch(`${GENETICS_BASE}/analyze/genetics/health`);
    return await res.json();
  } catch {
    return { status: "unreachable" };
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   LOVABLE USAGE EXAMPLES
   ═══════════════════════════════════════════════════════════════════════════

   ── Variance Components ──────────────────────────────────────────────────

   import { runVarianceComponents } from './genetics_api_client';

   const handleRunAnalysis = async () => {
     setLoading(true);
     setError(null);
     try {
       const data = await runVarianceComponents(
         selectedFile,
         ['Days_to_Flowering', 'Plant_Height_cm', 'Pod_Number', 'Seed_Yield_kg_ha'],
         { genotypeCol: 'Genotype', locationCol: 'Location', repCol: 'Rep' }
       );
       setResults(data);
     } catch (err) {
       setError(err.message);   // shows the real error, not "Failed to fetch"
     } finally {
       setLoading(false);
     }
   };

   ── Stability Analysis ───────────────────────────────────────────────────

   const data = await runStabilityAnalysis(
     selectedFile,
     ['Seed_Yield_kg_ha'],
     { genotypeCol: 'Genotype', locationCol: 'Location', repCol: 'Rep' }
   );

   ── Full Trial (all analyses at once) ────────────────────────────────────

   const data = await runFullTrialAnalysis(
     selectedFile,
     ['Days_to_Flowering', 'Plant_Height_cm', 'Pod_Number', 'Seed_Yield_kg_ha'],
     {
       genotypeCol:  'Genotype',
       locationCol:  'Location',
       repCol:       'Rep',
       primaryTrait: 'Seed_Yield_kg_ha',
       alpha:        0.05,
       nAmmiAxes:    2,
     }
   );

   ── Reading the response ─────────────────────────────────────────────────

   data.meta                    // { trait, n_genotypes, n_locations, ... }
   data.tables.variance_components  // [{ sigma2_g, sigma2_e, H2_broad, ... }]
   data.tables.genotype_means       // [{ Genotype, mean, letter, ... }]
   data.tables.stability            // [{ genotype, bi, S2di, ASV, classification }]
   data.html_tables                 // [{ name, html }]  ← copy-paste ready
   data.publication_figures         // [{ name, caption, image_base64 }]
   data.interpretation              // plain-English paragraph
   data.intelligence.executive_insight  // 3-bullet summary for the dashboard

   ── Display HTML table in React ──────────────────────────────────────────

   {data.html_tables?.map(t => (
     <div key={t.name}>
       <div dangerouslySetInnerHTML={{ __html: t.html }} />
     </div>
   ))}

   ── Display base64 figure ────────────────────────────────────────────────

   {data.publication_figures?.map(f => (
     <figure key={f.name}>
       <img src={`data:image/png;base64,${f.image_base64}`} alt={f.name} />
       <figcaption>{f.caption}</figcaption>
     </figure>
   ))}

   ═══════════════════════════════════════════════════════════════════════════ */
