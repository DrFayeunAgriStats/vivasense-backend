# VivaSense Genetics Analysis Package — Comprehensive Design

**Version**: 1.0 (Design Only — Pre-Implementation)
**Target users**: Nigerian agricultural researchers, postgraduate students, plant breeders
**Scope**: Multilocational and single-environment factorial trials; molecular marker analysis

---

## Table of Contents

1. [File & Folder Structure](#1-file--folder-structure)
2. [Data Input Schemas](#2-data-input-schemas)
3. [API Endpoint Design](#3-api-endpoint-design)
4. [Formula Display in JSON](#4-formula-display-in-json)
5. [Output Structures (per analysis block)](#5-output-structures)
6. [G×L Interaction & Variance Partitioning](#6-gl-interaction--variance-partitioning)
7. [Integration with Existing VivaSense](#7-integration-with-existing-vivasense)
8. [Visualization Requirements](#8-visualization-requirements)
9. [New Dependencies](#9-new-dependencies)

---

## 1. File & Folder Structure

```
app/
├── main.py                          ← EXISTING (add 3 lines only — see §7)
├── requirements.txt                 ← EXISTING (append 1 new package — see §9)
├── packages.txt                     ← EXISTING (unchanged)
│
└── genetics/
    ├── __init__.py                  ← exports genetics_router and GeneticsPipeline
    ├── router.py                    ← APIRouter(prefix="/genetics"); all endpoints
    ├── pipeline.py                  ← GeneticsPipeline (orchestration, mirrors VivaSenseBackend)
    ├── config.py                    ← GeneticsConfig, TrialDesign, MarkerConfig (dataclasses)
    ├── models.py                    ← all result dataclasses (VarianceComponents, AMMIResult…)
    ├── validators.py                ← GeneticsDataValidator
    ├── serializers.py               ← numpy/pandas → JSON-safe converters (fixes known CLAUDE.md gap)
    ├── plotting.py                  ← GeneticsPlotter (all base64-PNG generators)
    │
    └── engines/
        ├── __init__.py
        ├── multilocational.py       ← MultilocationEngine  (combined ANOVA, AMMI, GGE)
        ├── variance_components.py   ← VarianceComponentEngine (σ²g, H², GA, GCV, PCV)
        ├── stability.py             ← StabilityEngine (Eberhart-Russell, ASV)
        ├── correlations.py          ← CorrelationEngine (phenotypic/genotypic r, path, index)
        ├── multivariate.py          ← MultivariateEngine (PCA, Ward, k-means)
        └── markers.py               ← MarkerEngine (Jaccard, Dice, Shannon, Simpson, UPGMA)
```

**Integration with `main.py`** — only three lines added after the CORS block:

```python
from genetics import genetics_router
app.include_router(genetics_router)
```

Zero changes to any existing class, endpoint, or configuration.

---

## 2. Data Input Schemas

### 2.1 Multilocational Trial — CSV Format

Column order is flexible. All columns are case-insensitive in the parser.

```csv
Genotype,Location,Season,Rep,Block,GrainYield_t_ha,PlantHeight_cm,DaysToFlowering,BiomassYield
G001,Kano,WS2023,1,1,4.52,112.3,58,14.21
G001,Kano,WS2023,2,1,4.71,114.1,57,14.89
G001,Kano,WS2023,3,2,4.38,110.9,59,13.97
G001,Ibadan,WS2023,1,1,5.10,118.4,55,15.44
G001,Ibadan,WS2023,2,2,5.24,119.7,54,15.88
...
G020,Zaria,WS2023,3,2,3.91,105.2,62,12.56
```

**Required columns** (declared in `TrialDesign`):

| Column | Type | Notes |
|--------|------|-------|
| `Genotype` | string | Genotype identifier (G001, Var_A, etc.) |
| `Location` | string | Site/environment name |
| `Rep` | integer | Replicate number (1, 2, 3…) |

**Optional structural columns**:

| Column | Type | Notes |
|--------|------|-------|
| `Season` | string | Growing season (WS2023, DS2024…) — can be merged with Location |
| `Block` | integer | Incomplete block number (Alpha designs) |
| `Row` / `Column` | integer | Field coordinates (row-column designs) |

**Trait columns**: all remaining numeric columns. Detected automatically.

**Minimum valid dataset**:
- ≥ 3 genotypes
- ≥ 2 locations (for G×L analysis; 1 location triggers single-environment mode)
- ≥ 2 reps per genotype-location cell

### 2.2 Multilocational Trial — JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "GeneticsTrialRequest",
  "type": "object",
  "required": ["data", "design"],
  "properties": {

    "data": {
      "type": "array",
      "description": "Array of row objects. Each key is a column name.",
      "minItems": 6,
      "items": {
        "type": "object",
        "example": {
          "Genotype": "G001",
          "Location": "Kano",
          "Season": "WS2023",
          "Rep": 1,
          "Block": 1,
          "GrainYield_t_ha": 4.52,
          "PlantHeight_cm": 112.3
        }
      }
    },

    "design": {
      "type": "object",
      "description": "Declares the experimental structure column mappings.",
      "required": ["genotype_col", "rep_col"],
      "properties": {
        "genotype_col":  { "type": "string", "default": "Genotype" },
        "location_col":  { "type": "string", "default": "Location",
                           "description": "Omit or set null for single-environment data." },
        "season_col":    { "type": "string", "default": null,
                           "description": "If provided, Location + Season are concatenated as the environment." },
        "rep_col":       { "type": "string", "default": "Rep" },
        "block_col":     { "type": "string", "default": null,
                           "description": "Required only for Alpha lattice designs." },
        "trait_cols":    { "type": "array", "items": {"type": "string"},
                           "default": null,
                           "description": "If null, all numeric columns not in structural columns are used." },
        "design_type":   { "type": "string", "enum": ["CRD", "RCBD", "Alpha"],
                           "default": "RCBD" }
      }
    },

    "config": {
      "type": "object",
      "description": "Analysis configuration overrides. All fields optional.",
      "properties": {
        "alpha":                   { "type": "number", "default": 0.05 },
        "n_ammi_axes":             { "type": "integer", "default": 2,
                                     "description": "Number of IPCA axes to retain in AMMI model." },
        "stability_method":        { "type": "string",
                                     "enum": ["eberhart_russell", "ammi", "both"],
                                     "default": "both" },
        "selection_intensity":     { "type": "number", "default": 2.063,
                                     "description": "k = 2.063 for top 5% selection." },
        "economic_weights":        { "type": "object",
                                     "description": "Trait name → weight for selection index. Default: equal weights.",
                                     "example": {"GrainYield_t_ha": 1.0, "PlantHeight_cm": 0.0} },
        "path_target_trait":       { "type": "string",
                                     "description": "Dependent variable for path analysis. Default: first trait." },
        "n_clusters":              { "type": "integer", "default": 3 },
        "figure_dpi":              { "type": "integer", "default": 300 },
        "p_value_correction":      { "type": "string",
                                     "enum": ["bonferroni", "fdr_bh", "none"],
                                     "default": "bonferroni" }
      }
    }
  }
}
```

### 2.3 Molecular Marker — CSV Format

```csv
Accession,SSR_01,SSR_02,SSR_03,SSR_04,SSR_05,SSR_06,SSR_07,SSR_08
G001,1,0,1,1,0,1,0,1
G002,1,1,1,0,0,1,1,0
G003,0,1,0,1,1,0,1,1
G004,1,1,0,0,1,1,0,0
```

Rules:
- First column = accession identifier
- All remaining columns = markers; values must be `0` (absent), `1` (present), or blank (missing)
- Minimum: ≥ 4 accessions, ≥ 5 markers

### 2.4 Molecular Marker — JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "MarkersRequest",
  "type": "object",
  "required": ["data"],
  "properties": {
    "data": {
      "type": "array",
      "items": { "type": "object" },
      "description": "Each row: {accession_col: name, marker1: 0|1, marker2: 0|1, ...}"
    },
    "marker_config": {
      "type": "object",
      "properties": {
        "accession_col":       { "type": "string", "default": "Accession" },
        "marker_prefix":       { "type": "string", "default": null,
                                 "description": "Filter marker columns by this prefix. Null = all non-accession columns." },
        "similarity_metric":   { "type": "string", "enum": ["jaccard", "dice", "both"], "default": "both" },
        "n_clusters":          { "type": "integer", "default": 3 }
      }
    }
  }
}
```

---

## 3. API Endpoint Design

All routes are under the prefix `/genetics`. All responses are `application/json`.

### 3.1 Single-Environment Endpoints

These are convenience wrappers. They accept the same schema as multilocational endpoints but with `design.location_col` omitted or null. The engine automatically detects single-location mode and skips G×L, AMMI, and GGE analyses.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/genetics/trial/analyze/` | Upload CSV/Excel file |
| `POST` | `/genetics/trial/json/` | Submit trial data as JSON body |
| `GET`  | `/genetics/results/{analysis_id}` | Retrieve cached result |
| `GET`  | `/genetics/health` | Module health check |

### 3.2 Multilocational Endpoints

Same paths as above. The engine activates multilocational analyses automatically when ≥ 2 locations are detected. You may also request sub-analyses individually:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/genetics/trial/analyze/` | Full analysis (file upload) |
| `POST` | `/genetics/trial/json/` | Full analysis (JSON body) |
| `POST` | `/genetics/stability/` | Stability only (Eberhart-Russell + ASV) |
| `POST` | `/genetics/ammi/` | AMMI model + biplot data only |
| `POST` | `/genetics/gge/` | GGE biplot data only |
| `POST` | `/genetics/variance-components/` | Variance components + heritability only |
| `POST` | `/genetics/correlations/` | Correlation matrix + path + selection index |
| `POST` | `/genetics/multivariate/` | PCA + clustering only |

### 3.3 Molecular Marker Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/genetics/markers/analyze/` | Full marker analysis (file upload) |
| `POST` | `/genetics/markers/json/` | Full marker analysis (JSON body) |

### 3.4 Endpoint Specification: Full Trial Analysis

```
POST /genetics/trial/analyze/
Content-Type: multipart/form-data

Parameters:
  file        (required)  CSV or Excel file
  design      (optional)  JSON string of TrialDesign fields
  config      (optional)  JSON string of GeneticsConfig fields

Success Response 200:
  Content-Type: application/json
  Body: GeneticsAnalysisResult (see §5)

Error Response 422 (validation failure):
{
  "status": "validation_error",
  "errors": ["Location column 'Location' not found in data",
             "Fewer than 2 replicates found for Genotype G005 in Location Kano"],
  "warnings": ["Unbalanced design detected: 3 missing cells"]
}

Error Response 500 (analysis failure):
{
  "status": "error",
  "detail": "AMMI SVD failed: matrix is singular",
  "analysis_id": null
}
```

### 3.5 Endpoint Specification: AMMI (standalone)

```
POST /genetics/ammi/
Content-Type: multipart/form-data

Parameters:
  file          (required)  CSV or Excel file
  trait         (optional)  Single trait name; default = first numeric trait
  n_axes        (optional)  int, default 2
  alpha         (optional)  float, default 0.05

Success Response 200:
{
  "status": "success",
  "trait": "GrainYield_t_ha",
  "ammi": { ... }   ← AMMIResult (see §5.B)
}
```

### 3.6 Endpoint Specification: Marker Analysis

```
POST /genetics/markers/analyze/
Content-Type: multipart/form-data

Parameters:
  file              (required)  CSV or Excel binary marker matrix
  marker_config     (optional)  JSON string of MarkerConfig fields

Success Response 200:
  Body: DiversityResult (see §5.F)
```

---

## 4. Formula Display in JSON Responses

Every computed statistic includes a `formula` field showing the mathematical expression used, and a `formula_description` field with plain-English explanation. This is consistently applied across all output objects and is especially important for postgraduate students.

### Convention

```json
{
  "sigma2_g": {
    "value": 1.24,
    "formula": "σ²g = (MSG - MSE) / r",
    "formula_description": "Genotypic variance: difference between genotype and error mean squares, divided by number of replicates",
    "components": {
      "MSG": 4.97,
      "MSE": 1.25,
      "r": 3
    }
  }
}
```

### All Formulas Used

**Variance Components:**

| Parameter | Formula field |
|-----------|---------------|
| σ²g | `"(MSG - MSE) / r"` |
| σ²e | `"MSE"` |
| σ²gl | `"(MSGL - MSE) / r"` where MSGL = G×L mean square |
| σ²p (single-loc) | `"σ²g + σ²e"` |
| σ²p (multi-loc) | `"σ²g + (σ²gl / l) + (σ²e / (r × l))"` |
| H² (broad-sense) | `"σ²g / σ²p"` |
| H² (multi-loc) | `"σ²g / (σ²g + σ²gl/l + σ²e/(r×l))"` |
| GA | `"k × √σ²p × H²"` |
| GA% | `"(GA / Grand Mean) × 100"` |
| GCV | `"(√σ²g / Grand Mean) × 100"` |
| PCV | `"(√σ²p / Grand Mean) × 100"` |

**Stability:**

| Parameter | Formula field |
|-----------|---------------|
| bi (E-R) | `"bi = Σ(Yij × Ij) / Σ(Ij²)"` |
| S²di (E-R) | `"[Σ(Yij - ȳi. - bI_j)² / (l-2)] - (MSE/r)"` |
| ASV | `"√[(SSIPCA1/SSIPCA2 × IPCA1_score)² + IPCA2_score²]"` |
| Env. Index | `"Ij = ȳ.j - ȳ.."` |

**Effect Sizes & Correlations:**

| Parameter | Formula field |
|-----------|---------------|
| Phenotypic r | `"r_p = Cov_p(xy) / √(σ²p_x × σ²p_y)"` |
| Genotypic r | `"r_g = σ_g(xy) / √(σ²g_x × σ²g_y)"` |
| Direct path p | `"p_ij = β_standardized"` |
| Indirect path | `"r_ij - p_ij"` |

**Diversity:**

| Parameter | Formula field |
|-----------|---------------|
| Jaccard | `"J(A,B) = |A∩B| / |A∪B|"` |
| Dice | `"D(A,B) = 2|A∩B| / (|A| + |B|)"` |
| Shannon H' | `"H' = -Σ(pᵢ × ln(pᵢ))"` |
| Simpson D | `"D = 1 - Σ(pᵢ²)"` |

---

## 5. Output Structures

All responses follow this envelope:

```json
{
  "status": "success | partial | error",
  "analysis_id": "uuid-v4",
  "timestamp": "2024-03-02T14:30:00Z",
  "metadata": { ... },
  "warnings": [],
  "...": "analysis-specific fields below"
}
```

### 5.A Variance Components Output

```json
{
  "variance_components": {
    "GrainYield_t_ha": {
      "status": "success",
      "grand_mean": 4.87,
      "n_genotypes": 20,
      "n_locations": 3,
      "n_reps": 3,
      "mode": "multilocational",

      "components": {
        "sigma2_g": {
          "value": 1.24,
          "formula": "σ²g = (MSG - MSE) / r",
          "formula_description": "Genotypic variance estimated from mean squares",
          "components": { "MSG": 4.97, "MSE": 1.25, "r": 3 }
        },
        "sigma2_e": {
          "value": 1.25,
          "formula": "σ²e = MSE",
          "formula_description": "Environmental (error) variance"
        },
        "sigma2_gl": {
          "value": 0.63,
          "formula": "σ²gl = (MSGL - MSE) / r",
          "formula_description": "Genotype × Location interaction variance",
          "components": { "MSGL": 3.14, "MSE": 1.25, "r": 3 }
        },
        "sigma2_p": {
          "value": 2.28,
          "formula": "σ²p = σ²g + (σ²gl / l) + (σ²e / (r × l))",
          "formula_description": "Phenotypic variance of a genotype mean over locations and reps",
          "components": { "sigma2_g": 1.24, "sigma2_gl": 0.63, "sigma2_e": 1.25, "l": 3, "r": 3 }
        }
      },

      "heritability": {
        "H2_broad": {
          "value": 0.54,
          "percentage": 54.2,
          "formula": "H² = σ²g / σ²p",
          "formula_description": "Broad-sense heritability on a mean basis across locations",
          "interpretation": "Moderate heritability — genetic selection will be moderately effective"
        },
        "H2_per_location": {
          "Kano":   { "value": 0.61, "formula": "H²_loc = σ²g / (σ²g + σ²e/r)" },
          "Ibadan": { "value": 0.58, "formula": "H²_loc = σ²g / (σ²g + σ²e/r)" },
          "Zaria":  { "value": 0.49, "formula": "H²_loc = σ²g / (σ²g + σ²e/r)" }
        }
      },

      "genetic_advance": {
        "GA": {
          "value": 1.47,
          "formula": "GA = k × √σ²p × H²",
          "formula_description": "Expected genetic advance under 5% selection intensity",
          "components": { "k": 2.063, "sqrt_sigma2p": 1.51, "H2": 0.54 }
        },
        "GA_percent": {
          "value": 30.1,
          "formula": "GA% = (GA / Grand Mean) × 100"
        },
        "GA_per_location": {
          "Kano":   { "GA": 1.53, "GA_percent": 31.4 },
          "Ibadan": { "GA": 1.61, "GA_percent": 29.8 },
          "Zaria":  { "GA": 1.29, "GA_percent": 28.7 }
        },
        "interpretation": "High genetic advance (>20%) — effective mass selection expected"
      },

      "coefficients_of_variation": {
        "GCV": {
          "value": 22.8,
          "formula": "GCV = (√σ²g / Grand Mean) × 100",
          "interpretation": "High GCV — substantial genotypic variation"
        },
        "PCV": {
          "value": 30.9,
          "formula": "PCV = (√σ²p / Grand Mean) × 100"
        },
        "ECV": {
          "value": 22.9,
          "formula": "ECV = (√σ²e / Grand Mean) × 100"
        },
        "GCV_per_location": {
          "Kano":   22.1,
          "Ibadan": 24.3,
          "Zaria":  19.8
        }
      },

      "anova_table": {
        "columns": ["Source", "df", "SS", "MS", "F", "p_value"],
        "rows": [
          ["Genotype",           19, 94.43, 4.97, 3.97, 0.0001],
          ["Location",            2, 48.21, 24.1, 19.3, 0.0000],
          ["Genotype × Location",38, 47.32, 1.25, 1.86, 0.0024],
          ["Error",             120, 75.00, 0.63, null, null]
        ]
      }
    }
  }
}
```

### 5.B AMMI Analysis Output

```json
{
  "ammi": {
    "GrainYield_t_ha": {
      "status": "success",
      "n_axes_retained": 2,
      "grand_mean": 4.87,

      "anova_table": {
        "columns": ["Source", "df", "SS", "MS", "F", "p_value"],
        "rows": [
          ["Genotype",           19, 94.43, 4.97, 3.97, 0.0001],
          ["Environment",         2, 48.21, 24.1, 19.3, 0.0000],
          ["G×E Interaction",    38, 47.32, 1.25, 1.86, 0.0024],
          ["  IPCA1",            20, 29.11, null, null, null],
          ["  IPCA2",            18, 12.73, null, null, null],
          ["  Residuals",        null,5.48, null, null, null],
          ["Error",             120, 75.00, 0.63, null, null]
        ]
      },

      "interaction_matrix": {
        "description": "Genotype × Environment cell means minus G main effect minus E main effect",
        "rows_are": "genotypes",
        "cols_are": "environments",
        "data": {
          "G001": { "Kano": 0.23, "Ibadan": -0.11, "Zaria": -0.12 },
          "G002": { "Kano": -0.41, "Ibadan": 0.38, "Zaria": 0.03 }
        }
      },

      "ipca_scores": {
        "genotypes": {
          "G001": { "IPCA1": 0.412, "IPCA2": -0.183 },
          "G002": { "IPCA1": -0.631, "IPCA2": 0.271 }
        },
        "environments": {
          "Kano":   { "IPCA1": 0.844, "IPCA2": 0.321 },
          "Ibadan": { "IPCA1": -0.512, "IPCA2": 0.447 },
          "Zaria":  { "IPCA1": -0.332, "IPCA2": -0.768 }
        }
      },

      "explained_variance": [
        { "axis": "IPCA1", "ss": 29.11, "percent": 61.5, "cumulative_percent": 61.5 },
        { "axis": "IPCA2", "ss": 12.73, "percent": 26.9, "cumulative_percent": 88.4 }
      ],

      "biplot_data": {
        "description": "Points for frontend AMMI biplot. Genotypes = circles, Environments = triangles.",
        "x_axis_label": "IPCA1 (61.5%)",
        "y_axis_label": "IPCA2 (26.9%)",
        "genotype_points": [
          { "id": "G001", "x": 0.412, "y": -0.183, "mean": 5.21, "type": "genotype" },
          { "id": "G002", "x": -0.631, "y": 0.271, "mean": 4.63, "type": "genotype" }
        ],
        "environment_points": [
          { "id": "Kano",   "x": 0.844, "y": 0.321, "mean": 5.04, "type": "environment" },
          { "id": "Ibadan", "x": -0.512, "y": 0.447, "mean": 4.97, "type": "environment" },
          { "id": "Zaria",  "x": -0.332, "y": -0.768, "mean": 4.61, "type": "environment" }
        ]
      },

      "plots": {
        "ammi_biplot": "<base64-encoded PNG>"
      }
    }
  }
}
```

### 5.C GGE Biplot Output

```json
{
  "gge": {
    "GrainYield_t_ha": {
      "status": "success",

      "method": "GGE biplot (environment-centered, singular value partitioning)",
      "explained_variance": [
        { "axis": "PC1", "percent": 68.3, "cumulative_percent": 68.3 },
        { "axis": "PC2", "percent": 19.7, "cumulative_percent": 88.0 }
      ],

      "genotype_scores": {
        "G001": { "PC1": 1.23, "PC2": 0.41, "mean": 5.21 },
        "G002": { "PC1": -0.87, "PC2": 0.92, "mean": 4.63 }
      },

      "environment_scores": {
        "Kano":   { "PC1": 1.44, "PC2": 0.21 },
        "Ibadan": { "PC1": -0.83, "PC2": 0.67 },
        "Zaria":  { "PC1": -0.61, "PC2": -0.88 }
      },

      "which_won_where": {
        "description": "Winning genotype in each environment (highest performance in that sector)",
        "Kano":   { "winner": "G001", "runner_up": "G003" },
        "Ibadan": { "winner": "G007", "runner_up": "G002" },
        "Zaria":  { "winner": "G014", "runner_up": "G007" }
      },

      "ideal_genotype": {
        "identified": "G003",
        "basis": "Closest to the ideal marker (mean of environment vectors)",
        "mean_yield": 5.08,
        "stability_rank": 2
      },

      "mega_environments": [
        { "id": 1, "environments": ["Kano"], "best_genotypes": ["G001", "G003"] },
        { "id": 2, "environments": ["Ibadan", "Zaria"], "best_genotypes": ["G007", "G014"] }
      ],

      "discriminating_environments": {
        "most_discriminating": "Kano",
        "least_discriminating": "Zaria",
        "basis": "Vector length from biplot origin"
      },

      "biplot_data": {
        "x_axis_label": "PC1 (68.3%)",
        "y_axis_label": "PC2 (19.7%)",
        "genotype_points": [
          { "id": "G001", "x": 1.23, "y": 0.41, "mean": 5.21, "type": "genotype" }
        ],
        "environment_vectors": [
          { "id": "Kano", "x": 1.44, "y": 0.21, "type": "environment" }
        ],
        "polygon_vertices": [
          { "id": "G001", "x": 1.23, "y": 0.41 },
          { "id": "G007", "x": -0.44, "y": 1.31 }
        ]
      },

      "plots": {
        "gge_biplot": "<base64-encoded PNG>"
      }
    }
  }
}
```

### 5.D Stability Analysis Output

```json
{
  "stability": {
    "GrainYield_t_ha": {
      "method": "Eberhart & Russell (1966) + AMMI Stability Value",
      "grand_mean": 4.87,
      "environmental_indices": {
        "Kano":   { "index": 0.17, "formula": "Ij = ȳ.j - ȳ.." },
        "Ibadan": { "index": 0.10, "formula": "Ij = ȳ.j - ȳ.." },
        "Zaria":  { "index": -0.26, "formula": "Ij = ȳ.j - ȳ.." }
      },
      "genotype_stability": [
        {
          "genotype": "G001",
          "mean": 5.21,
          "bi": {
            "value": 1.23,
            "formula": "bi = Σ(Yij × Ij) / Σ(Ij²)",
            "formula_description": "Regression coefficient of genotype on environmental index",
            "se": 0.14,
            "interpretation": "Above-average response to improving environments"
          },
          "S2di": {
            "value": 0.08,
            "formula": "S²di = [Σ(Yij - ȳi. - bi×Ij)² / (l-2)] - (MSE/r)",
            "formula_description": "Deviation from regression — lower = more predictable",
            "p_value": 0.312,
            "significant": false
          },
          "ASV": {
            "value": 0.71,
            "formula": "ASV = √[(SSIPCA1/SSIPCA2 × IPCA1)² + IPCA2²]",
            "rank": 3
          },
          "classification": "above_average_stable",
          "recommendation": "Suitable for high-input, high-yield environments"
        },
        {
          "genotype": "G007",
          "mean": 4.93,
          "bi": { "value": 0.91, "se": 0.11, "interpretation": "Average response — broadly adapted" },
          "S2di": { "value": 0.02, "p_value": 0.788, "significant": false },
          "ASV": { "value": 0.22, "rank": 1 },
          "classification": "stable_average",
          "recommendation": "Widely adapted — suitable for diverse environments"
        }
      ],
      "stability_classification_rules": {
        "stable_high_yielding":     "mean > grand_mean AND bi ≈ 1 AND S²di not significant",
        "above_average_responsive": "mean > grand_mean AND bi > 1",
        "stable_below_average":     "mean < grand_mean AND bi < 1 AND S²di not significant",
        "unstable":                 "S²di significant (p < alpha)"
      },
      "plots": {
        "stability_regression": "<base64-encoded PNG>",
        "mean_vs_bi": "<base64-encoded PNG>"
      }
    }
  }
}
```

### 5.E Correlation & Path Analysis Output

```json
{
  "correlations": {
    "n_genotypes": 20,
    "n_observations": 180,

    "phenotypic": {
      "matrix": {
        "GrainYield_t_ha": {
          "GrainYield_t_ha":   { "r": 1.000, "p_value": null },
          "PlantHeight_cm":    { "r": 0.623, "p_value": 0.003, "significant": true },
          "DaysToFlowering":   { "r": -0.441, "p_value": 0.021, "significant": true },
          "BiomassYield":      { "r": 0.812, "p_value": 0.000, "significant": true }
        }
      },
      "formula": "r_p = Cov_p(xy) / √(σ²p_x × σ²p_y)",
      "per_location": {
        "Kano":   { "GrainYield_t_ha": { "PlantHeight_cm": { "r": 0.68, "p_value": 0.009 } } },
        "Ibadan": { "GrainYield_t_ha": { "PlantHeight_cm": { "r": 0.54, "p_value": 0.031 } } }
      },
      "plots": { "heatmap": "<base64-encoded PNG>" }
    },

    "genotypic": {
      "matrix": {
        "GrainYield_t_ha": {
          "PlantHeight_cm":  { "r_g": 0.741, "formula": "r_g = σ_g(xy) / √(σ²g_x × σ²g_y)" },
          "DaysToFlowering": { "r_g": -0.512 },
          "BiomassYield":    { "r_g": 0.889 }
        }
      },
      "plots": { "heatmap": "<base64-encoded PNG>" }
    }
  },

  "path_analysis": {
    "target_trait": "GrainYield_t_ha",
    "predictor_traits": ["PlantHeight_cm", "DaysToFlowering", "BiomassYield"],
    "formula_system": "Solve: r_ij = Σ(p_ij × r_jk) for direct paths p_ij",

    "direct_effects": {
      "PlantHeight_cm":  { "value": 0.312, "formula": "p_ij = standardized β" },
      "DaysToFlowering": { "value": -0.188 },
      "BiomassYield":    { "value": 0.541 }
    },

    "indirect_effects": {
      "PlantHeight_cm_via_BiomassYield":    { "value": 0.198 },
      "DaysToFlowering_via_PlantHeight_cm": { "value": -0.067 }
    },

    "residual_effect": { "value": 0.341, "formula": "√(1 - R²)" },
    "R_squared": 0.883,

    "per_location": {
      "Kano":   { "direct_effects": { "PlantHeight_cm": 0.344 } },
      "Ibadan": { "direct_effects": { "PlantHeight_cm": 0.279 } }
    },

    "plots": { "path_diagram": "<base64-encoded PNG>" }
  },

  "selection_index": {
    "method": "Smith-Hazel Index",
    "formula": "I = b'x where b = P⁻¹Ga",
    "formula_description": "Optimal weights from phenotypic (P) and genotypic (G) covariance matrices with economic weights (a)",

    "economic_weights": {
      "GrainYield_t_ha":  1.0,
      "PlantHeight_cm":   0.0,
      "DaysToFlowering":  -0.5,
      "BiomassYield":     0.3
    },

    "index_weights": {
      "GrainYield_t_ha":  0.812,
      "PlantHeight_cm":   0.041,
      "DaysToFlowering": -0.234,
      "BiomassYield":     0.391
    },

    "discriminant_function": "I = 0.812×GrainYield - 0.234×DaysToFlowering + 0.391×BiomassYield",

    "genotype_index_scores": {
      "G001": 4.83,
      "G007": 4.71,
      "G003": 4.68
    },

    "top_selections": ["G001", "G007", "G003"],

    "expected_genetic_gain": {
      "GrainYield_t_ha":  { "absolute": 0.82, "percent": 16.8 },
      "DaysToFlowering":  { "absolute": -1.4, "percent": -2.3 },
      "BiomassYield":     { "absolute": 0.61, "percent": 4.1 }
    },

    "per_location": {
      "Kano":   { "top_selections": ["G001", "G003"], "expected_gain": { "GrainYield_t_ha": 0.91 } },
      "Ibadan": { "top_selections": ["G007", "G001"], "expected_gain": { "GrainYield_t_ha": 0.78 } }
    }
  }
}
```

### 5.F Multivariate Analysis Output

```json
{
  "multivariate": {
    "input_matrix": {
      "description": "Genotype × trait means used as input",
      "n_genotypes": 20,
      "n_traits": 4
    },

    "pca": {
      "method": "SVD on Z-score standardized genotype × mean-trait matrix",
      "n_components": 4,
      "explained_variance_ratio": [0.512, 0.241, 0.148, 0.099],
      "cumulative_explained_variance": [0.512, 0.753, 0.901, 1.000],

      "loadings": {
        "PC1": {
          "GrainYield_t_ha": 0.612,
          "PlantHeight_cm":  0.481,
          "DaysToFlowering": -0.521,
          "BiomassYield":    0.698
        },
        "PC2": {
          "GrainYield_t_ha": -0.321,
          "PlantHeight_cm":  0.712,
          "DaysToFlowering": 0.441,
          "BiomassYield":    -0.218
        }
      },

      "scores": {
        "G001": { "PC1": 1.84, "PC2": 0.42 },
        "G002": { "PC1": -0.92, "PC2": 1.11 }
      },

      "per_location": {
        "Kano":   { "explained_variance_ratio": [0.53, 0.24], "loadings": { "PC1": { "GrainYield_t_ha": 0.64 } } },
        "Ibadan": { "explained_variance_ratio": [0.49, 0.27], "loadings": { "PC1": { "GrainYield_t_ha": 0.58 } } }
      },

      "plots": {
        "biplot":     "<base64-encoded PNG>",
        "scree_plot": "<base64-encoded PNG>"
      }
    },

    "hierarchical_clustering": {
      "method": "Ward linkage on Euclidean distance of PC scores",
      "n_clusters_suggested": 3,
      "cluster_labels": {
        "G001": 1, "G003": 1, "G005": 1,
        "G002": 2, "G007": 2,
        "G004": 3, "G008": 3
      },
      "dendrogram_data": {
        "description": "Nested tree for frontend rendering",
        "format": "scipy linkage matrix as nested dict",
        "tree": {
          "id": "root",
          "height": 4.82,
          "children": [
            {
              "id": "cluster_1",
              "height": 2.11,
              "children": [
                { "id": "G001", "height": 0 },
                { "id": "G003", "height": 0 }
              ]
            }
          ]
        }
      },
      "plots": {
        "dendrogram": "<base64-encoded PNG>",
        "cluster_pca": "<base64-encoded PNG>"
      }
    },

    "kmeans_clustering": {
      "method": "k-means on PC scores",
      "k": 3,
      "cluster_labels": {
        "G001": 0, "G002": 1, "G004": 2
      },
      "cluster_centroids": {
        "0": { "PC1": 1.42, "PC2": 0.38 },
        "1": { "PC1": -0.71, "PC2": 0.91 },
        "2": { "PC1": -0.61, "PC2": -0.97 }
      },
      "within_cluster_ss": 12.4,
      "plots": { "cluster_scatter": "<base64-encoded PNG>" }
    }
  }
}
```

### 5.G Molecular Marker Analysis Output

```json
{
  "diversity": {
    "n_accessions": 30,
    "n_markers": 48,
    "percent_polymorphic": 87.5,
    "mean_missing_data_percent": 2.1,

    "per_locus_diversity": {
      "SSR_01": {
        "allele_freq_1": 0.63,
        "allele_freq_0": 0.37,
        "shannon_H":    { "value": 0.649, "formula": "H' = -Σ(pᵢ × ln(pᵢ))" },
        "simpson_D":    { "value": 0.466, "formula": "D = 1 - Σ(pᵢ²)" },
        "PIC":          { "value": 0.381, "formula": "PIC = 1 - Σ(pᵢ²)" }
      }
    },

    "summary_diversity": {
      "mean_shannon_H":   0.531,
      "mean_simpson_D":   0.387,
      "mean_PIC":         0.361,
      "gene_diversity_Nei": { "value": 0.412, "formula": "Ĥ = n/(n-1) × (1 - Σpᵢ²)" }
    },

    "similarity_matrix": {
      "metric": "jaccard",
      "formula": "J(A,B) = |A∩B| / |A∪B|",
      "matrix": {
        "G001": { "G001": 1.000, "G002": 0.612, "G003": 0.441 },
        "G002": { "G001": 0.612, "G002": 1.000, "G003": 0.523 }
      },
      "range": { "min": 0.287, "max": 0.831 }
    },

    "dice_matrix": {
      "metric": "dice",
      "formula": "D(A,B) = 2|A∩B| / (|A| + |B|)",
      "matrix": { "G001": { "G002": 0.758 } }
    },

    "dendrogram_data": {
      "linkage_method": "UPGMA (Unweighted Pair Group Method with Arithmetic Mean)",
      "distance_metric": "1 - Jaccard similarity",
      "tree": {
        "id": "root",
        "height": 0.713,
        "children": [
          {
            "id": "cluster_A",
            "height": 0.221,
            "children": [
              { "id": "G001", "height": 0, "mean_similarity": 0.718 },
              { "id": "G003", "height": 0, "mean_similarity": 0.718 }
            ]
          }
        ]
      }
    },

    "cluster_groups": {
      "method": "UPGMA clusters at 0.5 similarity threshold",
      "n_clusters": 4,
      "assignments": {
        "G001": "Cluster_I",
        "G002": "Cluster_II",
        "G003": "Cluster_I"
      }
    },

    "pca_molecular": {
      "description": "PCA on binary marker matrix",
      "explained_variance_ratio": [0.312, 0.187, 0.114],
      "scores": {
        "G001": { "PC1": 1.41, "PC2": -0.33 },
        "G002": { "PC1": -0.87, "PC2": 0.72 }
      }
    },

    "plots": {
      "similarity_heatmap": "<base64-encoded PNG>",
      "dendrogram":         "<base64-encoded PNG>",
      "pca_biplot":         "<base64-encoded PNG>"
    }
  }
}
```

---

## 6. G×L Interaction Handling & Variance Partitioning

### 6.1 Detection

The engine automatically determines analysis mode:

| Condition | Mode activated |
|-----------|----------------|
| 1 location or `location_col` is null | Single-environment ANOVA (no G×L) |
| 2 locations | G×L computed; AMMI skipped (needs ≥ 3 environments for meaningful SVD) |
| ≥ 3 locations | Full multilocational: G×L + AMMI + GGE + Stability |

### 6.2 Combined ANOVA Model

For RCBD across locations, the linear model fitted by `MultilocationEngine` is:

```
Yijkl = μ + Gi + Lj + (GL)ij + Rk(j) + εijkl
```

Where:
- `Yijkl` = trait value for genotype i, location j, rep k
- `μ` = grand mean
- `Gi` = genotype main effect
- `Lj` = location main effect
- `(GL)ij` = genotype × location interaction
- `Rk(j)` = rep within location (random)
- `εijkl` = residual error

This is fitted as a **fixed-effects OLS** using `statsmodels` (matching the existing ANOVA pattern in `StatisticalAnalyzer.run_anova()`), treating reps as nested within locations in the formula:

```
GrainYield ~ C(Genotype) + C(Location) + C(Genotype):C(Location) + C(Rep):C(Location)
```

Type II SS is used (matching the existing implementation).

### 6.3 Mean Square Expectations and Component Extraction

For balanced RCBD (g genotypes, l locations, r reps):

| Source | df | EMS |
|--------|-----|-----|
| Genotype | g−1 | σ²e + r·σ²gl + rl·σ²g |
| Location | l−1 | σ²e + r·σ²gl + rg·σ²l |
| G×L | (g−1)(l−1) | σ²e + r·σ²gl |
| Rep/Location | l(r−1) | σ²e |
| Error (residual) | (g−1)(r−1)l | σ²e |

Solving the EMS equations:

```python
sigma2_e  = MSE
sigma2_gl = (MSGL - MSE) / r
sigma2_g  = (MSG - MSGL) / (r * l)   # preferred; uses MSGL as denominator
```

**Note on negative variance estimates**: If `MSG < MSGL` (which can happen), `σ²g` is set to 0 and a warning is added to `result.warnings`. This is standard practice (Falconer & Mackay convention).

### 6.4 Phenotypic Variance on a Mean Basis

The denominator of H² depends on the inference basis:

```
σ²p_mean = σ²g + σ²gl/l + σ²e/(rl)
```

This is the variance of a genotype mean estimated over all locations and all reps. Heritability on this basis answers: "How reliably can we rank genotypes using their mean performance across our test network?"

### 6.5 Location-Specific Heritability

For a single location j with r reps:

```
σ²p_loc = σ²g + σ²e/r
H²_loc  = σ²g / (σ²g + σ²e/r)
```

Here `σ²e` is taken from the pooled Error MS (assumed homogeneous across locations; a Bartlett test is run and a warning issued if heterogeneous).

### 6.6 AMMI Partitioning of G×L

After obtaining the G×L interaction matrix (genotypes × locations cell means minus G main effect minus L main effect), AMMI decomposes it via SVD:

```
(GL)_ij = Σ_k [ λk × αik × γjk ] + ρij
```

Where:
- `λk` = k-th singular value (√eigenvalue)
- `αik` = genotype i score on IPCA axis k
- `γjk` = location j score on IPCA axis k
- `ρij` = AMMI residuals (noise)

The number of retained axes (`n_ammi_axes`) is determined by:
1. User config (`GeneticsConfig.n_ammi_axes`)
2. Or by FR test (Gollob 1968): retain axes with F > Fcrit

SS attribution: `SS_IPCAk = λk² × (g + l - 1 - 2k)` (Gollob's formula for AMMI degrees of freedom).

### 6.7 GGE Model

GGE does not remove the G main effect before SVD — only the environment (location) mean is removed:

```
Yij - ȳ.j = λ1 × ξi1 × η j1 + λ2 × ξi2 × ηj2 + ε'ij
```

This preserves genotype differences within the biplot, making "which-won-where" patterns directly readable from the genotype scores.

---

## 7. Integration with Existing VivaSense

### 7.1 Code Changes to `main.py` (minimal)

Add exactly three lines after the CORS middleware block (approximately line 1110 in the current file):

```python
# Genetics module
from genetics import genetics_router
app.include_router(genetics_router)
```

No other changes. All existing endpoints, classes, and the `backend` object are untouched.

### 7.2 Shared Infrastructure Reuse

| Existing component | How genetics package uses it |
|--------------------|------------------------------|
| `CacheManager` | Imported directly: `from main import CacheManager`. `GeneticsPipeline` instantiates it with the same `./cache/` directory. |
| `VivaSenseBackend.save_results()` | Imported and called by `GeneticsPipeline` to write `./results/{id}.json`. Gives genetics results the same persistence and retrieval as existing analyses. |
| `AnalysisConfig` | **Not shared** — genetics has its own `GeneticsConfig` to avoid coupling. But `GeneticsConfig.figure_dpi` and `figure_format` default to the same values (300, `'png'`) for visual consistency. |
| Plot encoding convention | All genetics plots are base64-encoded PNG at 300 DPI — identical to `StatisticalAnalyzer.generate_plots()` output. The frontend needs zero changes to render them. |
| NumPy serialization pattern | `serializers.numpy_to_python()` addresses the known gap documented in CLAUDE.md. All genetics endpoints call it before returning. |

### 7.3 Unified `/analyze/` vs. `/genetics/trial/analyze/`

The existing `POST /analyze/` endpoint performs basic ANOVA. The new `POST /genetics/trial/analyze/` performs the full breeding analysis. They are **independent pipelines** — a request to `/analyze/` still returns the existing ANOVA result; `/genetics/trial/analyze/` returns the genetics result.

Optionally, in a future iteration, `VivaSenseBackend.process_dataframe()` can call `GeneticsPipeline.run_trial_analysis()` to append genetics results to the existing ANOVA response. This is not part of the current design (out of scope for v1).

### 7.4 `AIInterpreter` Integration

The existing `AIInterpreter` class provides template-based plain-English summaries. The genetics package adds an `interpret()` call in `GeneticsPipeline` that generates breeding-specific interpretation text using the same template approach:

```python
{
  "interpretation": {
    "overall": "20 genotypes were evaluated across 3 locations (Kano, Ibadan, Zaria) ...",
    "heritability": "Grain yield showed moderate broad-sense heritability (H² = 0.54), ...",
    "stability": "G001 (mean 5.21 t/ha) showed above-average but unstable performance ...",
    "recommendations": [
      "Select G001 and G003 for high-input environments (Kano)",
      "G007 is broadly adapted and recommended for release across all three locations",
      "G×L interaction was significant — location-specific recommendations are warranted"
    ]
  }
}
```

This interpretation block is appended to the top-level `GeneticsAnalysisResult` alongside the numerical outputs.

---

## 8. Visualization Requirements

All plots are returned as `base64`-encoded PNG strings at 300 DPI. The frontend renders them as `<img src="data:image/png;base64,...">` — identical to how existing ANOVA plots are consumed.

### 8.1 Plot Inventory

| Plot key | Engine method | Description |
|----------|--------------|-------------|
| `ammi_biplot` | `GeneticsPlotter.plot_ammi_biplot()` | Genotype scores (circles) + environment scores (triangles) on IPCA1 × IPCA2 axes. Axes labeled with % variance explained. |
| `gge_biplot` | `GeneticsPlotter.plot_gge_biplot()` | GGE scores with which-won-where polygon, environment vectors from origin, ideal genotype marker. |
| `stability_regression` | `GeneticsPlotter.plot_stability_regression()` | Eberhart-Russell: each genotype plotted as a regression line of Yij on Ij. Reference line at bi=1. |
| `mean_vs_bi` | `GeneticsPlotter.plot_mean_vs_bi()` | Scatter of genotype mean (y) vs. bi regression coefficient (x). Quadrant lines at grand mean and bi=1. |
| `phenotypic_heatmap` | `GeneticsPlotter.plot_correlation_heatmap()` | Seaborn heatmap of lower-triangle phenotypic correlation matrix. Color scale: −1 (blue) → +1 (red). |
| `genotypic_heatmap` | `GeneticsPlotter.plot_correlation_heatmap()` | Same as phenotypic but for genotypic correlations. |
| `path_diagram` | `GeneticsPlotter.plot_path_diagram()` | Horizontal bar chart of direct effects. Arrows for top indirect effects. |
| `pca_biplot` | `GeneticsPlotter.plot_pca_biplot()` | PCA: genotype scores (PC1 × PC2) colored by cluster; trait loading vectors overlaid. |
| `scree_plot` | `GeneticsPlotter.plot_scree()` | Bar + line chart of % variance explained per PC, with cumulative overlay. |
| `dendrogram_phenotypic` | `GeneticsPlotter.plot_dendrogram()` | Scipy dendrogram of Ward hierarchical clustering on genotype × trait matrix. |
| `cluster_scatter` | `GeneticsPlotter.plot_cluster_scatter()` | PCA scores colored by k-means cluster assignment. |
| `similarity_heatmap` | `GeneticsPlotter.plot_similarity_heatmap()` | Seaborn heatmap of accession × accession Jaccard/Dice similarity matrix. |
| `dendrogram_markers` | `GeneticsPlotter.plot_dendrogram()` | UPGMA dendrogram on molecular distance matrix. Bootstrap values (if n_bootstrap > 0 in config). |
| `marker_pca` | `GeneticsPlotter.plot_pca_biplot()` | PCA on binary marker matrix — accession scatter colored by UPGMA cluster. |

### 8.2 AMMI Biplot Specification

```
Canvas:     12 × 10 inches, DPI 300
X axis:     IPCA1 (label: "IPCA1 (61.5% of G×E SS)")
Y axis:     IPCA2 (label: "IPCA2 (26.9% of G×E SS)")
Origin:     Dashed crosshair at (0, 0)
Genotypes:  Blue filled circles, labeled with genotype ID
Locations:  Red filled triangles, labeled with location name
Annotations: Stability circle (optional): ring around origin; genotypes inside = more stable
Legend:     "Genotypes" / "Environments" in upper corner
```

### 8.3 GGE Biplot Specification

```
Canvas:     12 × 10 inches, DPI 300
X axis:     PC1 (label: "PC1 (68.3%)")
Y axis:     PC2 (label: "PC2 (19.7%)")
Genotypes:  Blue circles
Environments: Red vectors from origin (arrows), labeled at tips
Polygon:    Convex hull of outermost genotypes; sectors show which-won-where
Ideal marker: Gold star at end of mean-environment vector
Mega-environments: Perpendicular lines from origin through polygon vertex edges
```

### 8.4 Dendrogram Specification

```
Canvas:     10 × max(8, n_accessions × 0.25) inches
Orientation: Left-to-right (leaf labels on right)
Color threshold: 70% of max linkage height (scipy default)
Labels:     Accession IDs, 8pt font
Scale bar:  Distance scale at bottom
```

---

## 9. New Dependencies

Append to `app/requirements.txt`:

```
scikit-learn==1.3.2
```

This provides:
- `sklearan.decomposition.PCA` — numerically stable PCA with built-in standardization
- `sklearn.cluster.KMeans` — more robust than `scipy.cluster.vq.kmeans2`
- `sklearn.preprocessing.StandardScaler` — used before PCA

All other required computations (Eberhart-Russell regression, SVD for AMMI/GGE, Ward linkage, UPGMA, Pearson/Shapiro, all plot types) are fully covered by the existing stack (`numpy`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`).

No changes to `packages.txt` — `scikit-learn` is a pure Python wheel on PyPI and does not require APT compilation dependencies.

---

## Summary of What Each Block Computes and Returns

| Block | Key outputs | Requires multilocational? |
|-------|-------------|--------------------------|
| Variance Components | σ²g, σ²e, σ²gl, σ²p, H², GA, GA%, GCV, PCV (overall + per-location) | Optimal (σ²gl=0 if single-location) |
| AMMI | ANOVA partition, IPCA scores, % variance by axis, biplot data | Yes (≥ 3 locations) |
| GGE | PC1/PC2 scores, which-won-where, ideal genotype, mega-environments | Yes (≥ 2 locations) |
| Stability | bi, S²di, ASV, stability classification per genotype per trait | Yes (≥ 2 locations) |
| Correlations | Phenotypic r matrix (overall + per-location), genotypic r matrix | Optimal |
| Path Analysis | Direct + indirect effects, R², per-location paths | No |
| Selection Index | Smith-Hazel weights, top selections, expected genetic gain | No |
| PCA | PC loadings, scores, scree, per-location PCA | No |
| Clustering | Ward dendrogram tree, k-means labels | No |
| Markers | Jaccard/Dice matrix, Shannon H', Simpson D, UPGMA dendrogram | No |
