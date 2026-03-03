# VivaSense Backend — CLAUDE.md

Developer reference for the VivaSense Statistical Engine (v2.0.0). Read this before making changes.

---

## Project Overview

VivaSense is a **FastAPI-based statistical analysis microservice** for journal-grade ANOVA analysis. It is designed for agricultural/biological researchers who need automated factorial ANOVA, assumption testing, post-hoc comparisons, effect sizes, and publication-ready plots — all via a REST API.

- **Deployment**: Heroku
- **Python**: 3.11.8 (`runtime.txt`)
- **Entry point**: `app/main.py` (single monolithic file, ~1331 lines)

---

## Project Structure

```
vivasense-backend/
├── runtime.txt              # Python 3.11.8 (Heroku)
├── requirements.txt         # Mirrors app/requirements.txt
└── app/
    ├── main.py              # Entire application (~1331 lines)
    ├── requirements.txt     # Python dependencies
    ├── packages.txt         # APT system dependencies (Heroku buildpack)
    └── README.md
```

Everything lives in `app/main.py`. There are no sub-modules, blueprints, or separate routers.

---

## Architecture

### Class Hierarchy

```
AnalysisConfig       # Dataclass — global analysis settings
AssumptionTest       # Dataclass — single assumption test result
EffectSize           # Dataclass — eta², omega², Cohen's f
AnalysisResult       # Dataclass — full result container

StatisticalAnalyzer  # Core computation engine
AIInterpreter        # Plain-English result summaries
CacheManager         # Dual-layer (memory + disk) caching
VivaSenseBackend     # Orchestration pipeline

FastAPI app          # REST API layer (endpoints at module level)
```

### Request Flow

```
HTTP Request
    → FastAPI endpoint
    → VivaSenseBackend.process_dataframe()
        → CacheManager.get()                  (cache hit? return early)
        → StatisticalAnalyzer.validate_data()
        → StatisticalAnalyzer.detect_variable_types()
        → for each continuous trait:
            → StatisticalAnalyzer.build_formula()
            → StatisticalAnalyzer.run_anova()
            → StatisticalAnalyzer.check_assumptions()
            → StatisticalAnalyzer.calculate_effect_sizes()
            → StatisticalAnalyzer.descriptive_statistics()
            → StatisticalAnalyzer.generate_plots()
            → AIInterpreter.interpret()
        → CacheManager.set()
    → JSON Response
```

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Service info / version |
| `GET` | `/health` | Health check + timestamp |
| `POST` | `/analyze/` | Upload CSV or Excel file, run full analysis |
| `POST` | `/analyze/json/` | Submit data as JSON, run full analysis |
| `GET` | `/results/{analysis_id}` | Retrieve cached result by UUID |
| `DELETE` | `/cache/` | Clear all cached results |
| `GET` | `/config/` | Get current `AnalysisConfig` |
| `POST` | `/config/` | Update `AnalysisConfig` fields |

**Main endpoint** (`POST /analyze/`):
- Accepts: `multipart/form-data` with a CSV, XLSX, or XLS file
- Returns: Full `AnalysisResult` JSON (see Data Models section)

---

## Data Models

### Input

A Pandas DataFrame where:
- **Categorical columns** → treatment groups, blocks, experimental factors
- **Continuous columns** → response variables (traits) to analyze

### AnalysisConfig (dataclass)

```python
alpha: float = 0.05
figure_dpi: int = 300
figure_format: str = 'png'
include_interactions: bool = True
max_interaction_level: int = 2          # 2-way or 3-way interactions
p_value_correction: str = 'bonferroni'  # or 'fdr_bh'
effect_size_threshold_small: float = 0.01
effect_size_threshold_medium: float = 0.06
effect_size_threshold_large: float = 0.14
```

### AnalysisResult (JSON response shape)

```jsonc
{
  "status": "success",
  "metadata": {
    "filename": "string",
    "timestamp": "ISO-8601",
    "n_rows": 120,
    "n_cols": 6,
    "categorical_vars": ["Treatment", "Block"],
    "continuous_vars": ["Yield", "Height"],
    "blocks": ["Block"],
    "config": { /* AnalysisConfig fields */ },
    "analysis_id": "uuid-v4"
  },
  "warnings": [],
  "traits": {
    "Yield": {
      "status": "success",
      "statistical_results": {
        "formula": "Yield ~ C(Treatment) + C(Block)",
        "r_squared": 0.87,
        "adj_r_squared": 0.85,
        "f_value": 34.2,
        "f_pvalue": 0.0001,
        "anova": { /* ANOVA table rows */ },
        "means": { "TreatmentA": 45.3, "TreatmentB": 38.1 },
        "letters": { "TreatmentA": "a", "TreatmentB": "b" },
        "effect_sizes": { "eta_squared": 0.72, "omega_squared": 0.69, "cohens_f": 1.6 },
        "assumptions": { "normality": true, "homogeneity": true, "independence": true, "linearity": true },
        "descriptive_stats": { /* overall and per-group stats */ }
      },
      "plots": {
        "bar": "<base64-png>",
        "box": "<base64-png>",
        "interaction": "<base64-png>",
        "residuals": "<base64-png>"
      },
      "interpretation": "string — plain-English summary"
    }
  }
}
```

---

## Statistical Analysis Structure

### 1. Variable Detection (`detect_variable_types`)

Auto-classifies columns without user input:
- `object` dtype → categorical
- Integer with fewer than 10 unique values → categorical
- Blocking factors detected by keyword: `block`, `rep`, `replicate`, `batch`, `plot`, `field`

### 2. Data Validation (`validate_data`)

- Minimum 10 rows required
- Reports missing values (warns, does not abort)
- Detects constant columns (no variance)
- Flags extreme outliers (> 6 standard deviations)

### 3. Formula Building (`build_formula`)

Automatically generates `statsmodels` formula strings:
- Main effects for all predictors
- 2-way (or 3-way) interaction terms based on `max_interaction_level`
- Example: `"Yield ~ C(Treatment) * C(Block)"`

### 4. ANOVA (`run_anova`)

- Uses `statsmodels` OLS with **Type II ANOVA** (`anova_lm`)
- Computes R², adjusted R², F-statistic, p-value
- Returns full ANOVA table

### 5. Post-hoc Testing

- **Tukey HSD** (`statsmodels.stats.multicomp.pairwise_tukeyhsd`)
- **Compact Letter Display (CLD)**: groups treatments with shared letters into non-significant clusters

### 6. Assumption Testing (`check_assumptions`)

| Test | Method | Threshold |
|------|--------|-----------|
| Normality (≤5000 samples) | Shapiro-Wilk | α = 0.05 |
| Normality (>5000 samples) | Kolmogorov-Smirnov | α = 0.05 |
| Homogeneity of variance | Levene's test | α = 0.05 |
| Independence | Durbin-Watson statistic | — |
| Linearity | Rainbow test | α = 0.05 |

### 7. Effect Sizes (`calculate_effect_sizes`)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Eta-squared (η²) | SS_effect / SS_total | Negligible <0.01, Small <0.06, Medium <0.14, Large ≥0.14 |
| Omega-squared (ω²) | Unbiased η² variant | Same thresholds |
| Cohen's f | √(η² / (1 − η²)) | — |

### 8. Descriptive Statistics (`descriptive_statistics`)

- **Overall**: n, mean, std, SEM, CV, min, max, range, Q1, Q3, IQR
- **Per group**: count, mean, std, SEM, min, max, CV

### 9. Visualization (`generate_plots`)

All plots are 300 DPI PNG, base64-encoded in the JSON response:

| Plot key | Description |
|----------|-------------|
| `bar` | Group means with SEM error bars |
| `box` | Box plot by predictor |
| `interaction` | Interaction plot (multi-predictor) |
| `residuals` | Residuals diagnostic plot |

---

## Caching (`CacheManager`)

- **Memory layer**: Python dict (fast, ephemeral)
- **Disk layer**: Pickle files (survives process restart within dyno lifetime)
- **Cache key**: MD5 hash of (data + config)
- Results retrieved by UUID via `GET /results/{analysis_id}`
- Cache cleared via `DELETE /cache/`

---

## Coding Patterns

### Dataclasses for structured data

All configuration and result objects use `@dataclass`. Add fields here — do not pass raw dicts between major components.

```python
@dataclass
class AnalysisConfig:
    alpha: float = 0.05
    # ...
```

### Type hints everywhere

All functions use PEP 484 type hints. Keep this consistent when adding new functions.

### Logging convention

```python
logger = logging.getLogger(__name__)
logger.info("Starting analysis for trait: %s", trait_name)
logger.warning("Missing values detected: %d", n_missing)
logger.error("ANOVA failed: %s", str(e), exc_info=True)
```

### Error handling

- Statistical failures for a single trait do **not** abort the whole analysis — they set `"status": "failed"` on that trait's result and continue.
- API-level errors raise `HTTPException` with structured `detail`.

### JSON serialization

NumPy types (`np.float64`, `np.int64`, `np.bool_`) must be explicitly cast to Python natives before returning from endpoints. This has caused bugs in the past — check conversions when adding new result fields.

### Async endpoints

FastAPI endpoints are `async def`. Heavy computation runs synchronously inside them (no background tasks or thread pool). For large datasets this blocks the event loop — acceptable for the current single-user Heroku deployment.

---

## Dependencies

### Python (`app/requirements.txt`)

| Package | Version | Role |
|---------|---------|------|
| fastapi | 0.104.1 | Web framework |
| uvicorn[standard] | 0.24.0 | ASGI server |
| python-multipart | 0.0.6 | File upload parsing |
| pandas | 2.1.3 | DataFrame operations |
| numpy | 1.26.2 | Numerical computing |
| scipy | 1.11.4 | Statistical functions |
| statsmodels | 0.14.0 | OLS, ANOVA, Tukey HSD |
| matplotlib | 3.8.2 | Plot generation |
| seaborn | 0.13.0 | Statistical plot styling |
| openpyxl | 3.1.2 | Excel file I/O |

### System (`app/packages.txt`)

Required by Heroku buildpack before pip install:
- `build-essential` — C compiler
- `gfortran` — Fortran compiler (scipy)
- `libopenblas-dev` — BLAS (linear algebra)
- `liblapack-dev` — LAPACK (linear algebra)

---

## Running Locally

```bash
cd app
pip install -r requirements.txt

# Server mode
python main.py --host 0.0.0.0 --port 8000 --reload

# Batch file mode (no server)
python main.py --input-file data.csv --output-dir ./output

# With AI interpretation
python main.py --ai-key sk-...
```

Environment variables:
- `PORT` — overrides default port 8000
- `AI_KEY` — enables AI-powered interpretation (optional)

---

## Deployment (Heroku)

- `runtime.txt` pins Python 3.11.8
- `packages.txt` installs APT dependencies via the heroku-buildpack-apt buildpack
- CORS is currently set to `allow_origins=["*"]` — restrict this for production
- Heroku dyno restarts clear the in-memory cache; disk cache is also lost on dyno restart

---

## Known Gaps / Watch-outs

- **No test suite** — there are no pytest files. Test by uploading sample CSV files to `/analyze/`.
- **Monolithic file** — all 1331 lines are in `main.py`. When adding significant features consider splitting into modules.
- **NumPy serialization** — `np.float64` etc. will break `json.dumps`. Always convert to Python natives in response objects.
- **Synchronous computation in async endpoints** — large datasets will block the Uvicorn event loop.
- **AI interpretation is a stub** — `AIInterpreter` uses template-based strings, not an actual LLM call, unless `--ai-key` wires it up.
