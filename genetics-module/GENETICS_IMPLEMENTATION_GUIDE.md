# VivaSense Genetics Module - Implementation Guide
## R-Based Single & Multi-Environment Analysis

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Module Components](#module-components)
3. [Installation & Setup](#installation--setup)
4. [API Usage](#api-usage)
5. [Test Cases Summary](#test-cases-summary)
6. [Integration with FastAPI](#integration-with-fastapi)
7. [Deployment on Render](#deployment-on-render)

---

## Architecture Overview

The VivaSense Genetics Module is built on a **three-layer architecture**:

```
REQUEST (JSON)
    ↓
┌─────────────────────────────────────────┐
│ LAYER 1: COMPUTATION                    │
│ - compute_single_environment()          │
│ - compute_multi_environment()           │
│ Core ANOVA, variance partitioning       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ LAYER 2: VALIDATION                     │
│ - validate_input_data()                 │
│ - validate_variance_components()        │
│ Check data quality, flag edge cases     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ LAYER 3: INTERPRETATION                 │
│ - interpret_single_environment()        │
│ - interpret_multi_environment()         │
│ Generate publication-ready text         │
└─────────────────────────────────────────┘
    ↓
RESPONSE (JSON + Interpretation Text)
```

### Key Design Principles

1. **Strict Variance Partitioning**
   - Single-env: σ²ₚ = σ²ₘ + σ²ₑ/r
   - Multi-env (fixed): σ²ₚ = σ²ₘ + σ²ₘₑ/e + σ²ₑ/(re)
   - Multi-env (random): Includes σ²ₑ/e in denominator when requested

2. **Entry-Mean Heritability**
   - All h² estimates are on entry-mean basis
   - Fully respects replication structure
   - Clear distinction: single-env vs across-env

3. **Modular & Testable**
   - Each layer independent and testable
   - No hidden dependencies
   - Ready for unit testing

4. **FastAPI Ready**
   - All outputs JSON-serializable
   - Subprocess orchestration (R → Python)
   - Clean error handling and logging

---

## Module Components

### 1. Core R Engine (`vivasense_genetics.R`)

**Computation Layer**
- `compute_single_environment(data, trait_name)`
  - ANOVA-based variance partitioning
  - Single location, multiple genotypes, multiple reps
  - Returns: variance components, h², GCV, PCV, GAM

- `compute_multi_environment(data, trait_name, random_environment)`
  - Two-factor ANOVA with G×E interaction
  - Respects environment nesting
  - Flexible environment model (fixed or random)
  - Returns: G, E, G×E components + heritability

**Validation Layer**
- `validate_input_data(data, env_mode)`
  - Checks required columns
  - Detects missing values
  - Verifies minimum replication
  - Tests trait variation

- `validate_variance_components(result)`
  - Flags negative variance estimates
  - Identifies low heritability
  - Warns on weak genetic signal
  - Checks GCV/PCV validity

**Interpretation Layer**
- `interpret_single_environment(result, warnings)`
  - Structured interpretation text
  - Heritability classification
  - Genetic parameter summary
  - Actionable recommendations

- `interpret_multi_environment(result, warnings)`
  - Environment model explanation
  - G×E interaction interpretation
  - Multi-env specific warnings
  - Selection strategy guidance

**Orchestration**
- `genetics_analysis(data, mode, trait_name, random_environment)`
  - Main entry point
  - Coordinates all three layers
  - Returns unified output object

- `export_to_json(analysis_result)`
  - Converts R objects to JSON
  - Handles complex structures (ANOVA tables)
  - Ready for HTTP response

### 2. OpenAPI Schema (`vivasense_genetics_schema.json`)

Defines:
- Request payloads (data structure, mode selection)
- Response format (result + interpretation)
- All variance components
- Validation endpoints

### 3. FastAPI Backend (`app_genetics.py`)

- **Pydantic Models**: Validate requests, enforce schema
- **R Engine Orchestration**: Subprocess management, data serialization
- **HTTP Endpoints**:
  - `POST /genetics/analyze` - Full analysis
  - `POST /genetics/validate` - Data validation only
  - `GET /health` - Health check
- **Error Handling**: Proper HTTP status codes, detailed messages
- **Logging**: Track execution, debug failures

### 4. Test Cases (`test_cases_genetics.R`)

**Test 1: Single-Environment Valid**
- 10 genotypes, 3 reps
- Clear genetic signal (30 cm range)
- Environmental noise SD = 2
- Expected: h² ≈ 0.78–0.85, clean output

**Test 2: Multi-Environment Valid**
- 15 genotypes, 2 environments, 3 reps per G×E
- Genetic effect + environment effect + G×E interaction
- Demonstrates entry-mean heritability
- Expected: h² ≈ 0.60–0.70, moderate G×E

**Test 3: Edge Case - Negative Variance**
- 5 genotypes, 2 reps (minimal sample)
- Tiny genetic effect (2 cm range)
- High noise (SD = 5)
- Expected: h² ≈ 0, warnings raised

---

## Installation & Setup

### Prerequisites

- **R ≥ 4.0** with packages:
  - `jsonlite`
  - `agricolae`
  - `dplyr`
  - `tidyr`

- **Python ≥ 3.8** with:
  - `fastapi`
  - `pydantic`
  - `uvicorn`

### Step 1: Install R Packages

```r
# In R console
packages <- c("jsonlite", "agricolae", "dplyr", "tidyr")
install.packages(packages)

# Verify
library(jsonlite)
library(agricolae)
```

### Step 2: Install Python Dependencies

```bash
pip install fastapi pydantic uvicorn
```

### Step 3: Copy Files to Deployment Directory

```bash
# Project structure
project/
├── vivasense_genetics.R        # Core R engine
├── app_genetics.py             # FastAPI app
├── vivasense_genetics_schema.json  # OpenAPI spec
└── requirements.txt            # Python dependencies
```

### Step 4: Test Locally

```bash
# Run test cases
Rscript test_cases_genetics.R

# Start API server
python -m uvicorn app_genetics:app --reload --host 0.0.0.0 --port 8000

# In another terminal, test an endpoint
curl -X POST http://localhost:8000/genetics/analyze \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

---

## API Usage

### Example 1: Single-Environment Analysis

**Request:**
```json
{
  "data": [
    {"genotype": "G01", "rep": "R1", "trait_value": 45.2},
    {"genotype": "G01", "rep": "R2", "trait_value": 46.1},
    {"genotype": "G01", "rep": "R3", "trait_value": 44.8},
    {"genotype": "G02", "rep": "R1", "trait_value": 48.5},
    {"genotype": "G02", "rep": "R2", "trait_value": 49.2},
    {"genotype": "G02", "rep": "R3", "trait_value": 48.9}
  ],
  "mode": "single",
  "trait_name": "Plant Height (cm)"
}
```

**Response (abbreviated):**
```json
{
  "status": "SUCCESS",
  "mode": "single",
  "result": {
    "environment_mode": "single",
    "n_genotypes": 2,
    "n_reps": 3,
    "grand_mean": 47.12,
    "variance_components": {
      "sigma2_genotype": 3.45,
      "sigma2_error": 0.89,
      "sigma2_phenotypic": 3.75
    },
    "heritability": {
      "h2_broad_sense": 0.920,
      "interpretation_basis": "entry-mean (single environment)"
    },
    "genetic_parameters": {
      "GCV": 3.51,
      "PCV": 3.66,
      "GAM": 3.24,
      "GAM_percent": 6.87,
      "selection_intensity": 1.4
    }
  },
  "interpretation": "Single-Environment Genetic Analysis\n==================================================\n\n..."
}
```

### Example 2: Multi-Environment Analysis

**Request:**
```json
{
  "data": [
    {"genotype": "G01", "environment": "E1", "rep": "R1", "trait_value": 45.2},
    {"genotype": "G01", "environment": "E1", "rep": "R2", "trait_value": 46.1},
    {"genotype": "G01", "environment": "E2", "rep": "R1", "trait_value": 48.5},
    {"genotype": "G01", "environment": "E2", "rep": "R2", "trait_value": 49.2}
  ],
  "mode": "multi",
  "trait_name": "Plant Height (cm)",
  "random_environment": false
}
```

**Response (key differences):**
```json
{
  "status": "SUCCESS",
  "mode": "multi",
  "result": {
    "n_environments": 2,
    "variance_components": {
      "sigma2_genotype": 2.34,
      "sigma2_ge": 1.12,
      "sigma2_error": 0.95,
      "sigma2_phenotypic": 3.12,
      "heritability_basis": "fixed_environment_model"
    },
    "heritability": {
      "h2_broad_sense": 0.750,
      "interpretation_basis": "entry-mean across environments",
      "formula": "σ²p = σ²g + (σ²ge / e) + (σ²error / re)"
    }
  }
}
```

### Example 3: Validation Endpoint

```bash
curl -X POST http://localhost:8000/genetics/validate \
  -H "Content-Type: application/json" \
  -d '{...same as analyze request...}'
```

Response:
```json
{
  "is_valid": true,
  "warnings": {}
}
```

---

## Test Cases Summary

### Test 1: Single-Environment Valid Case
- **File**: `test_case_1_output.json`
- **Genotypes**: 10
- **Reps**: 3 per genotype
- **Total observations**: 30
- **Expected h²**: 0.78–0.85
- **Key finding**: Clear genetic signal, no warnings
- **Interpretation**: Strong genetic basis for selection

### Test 2: Multi-Environment Valid Case
- **File**: `test_case_2_output.json`
- **Genotypes**: 15
- **Environments**: 2
- **Reps**: 3 per G×E
- **Total observations**: 90
- **Expected h²**: 0.60–0.70
- **Key finding**: Moderate genetic effect, significant G×E
- **Interpretation**: Genotypes rank consistently but have environment-specific performance

### Test 3: Edge Case - Negative Variance
- **File**: `test_case_3_output.json`
- **Genotypes**: 5
- **Reps**: 2 per genotype
- **Total observations**: 10
- **Expected h²**: <0.05
- **Expected warnings**:
  - `negative_sigma2_genotype` (or clamped to 0)
  - `low_heritability`
  - `weak_genetic_signal`
- **Interpretation**: Weak genetic basis, environmental factors dominate

---

## Integration with FastAPI

### How It Works

1. **Request arrives** at FastAPI endpoint (`/genetics/analyze`)
2. **Pydantic validates** request against schema
3. **Python serializes** data as JSON
4. **Subprocess spawns** Rscript with temp R script
5. **R script** loads `vivasense_genetics.R`, reads JSON, runs analysis
6. **R outputs** JSON result to stdout
7. **Python parses** JSON, validates result schema
8. **Response returned** to client

### Code Flow

```python
# app_genetics.py

@app.post("/genetics/analyze", response_model=GeneticsResponse)
async def analyze_genetics(request: GeneticsRequest):
    # 1. Validate request (Pydantic)
    # 2. Call R engine
    result = r_engine.run_analysis(
        data=request.data,
        mode=request.mode,
        trait_name=request.trait_name,
        random_environment=request.random_environment
    )
    # 3. Return response
    return GeneticsResponse(**result)
```

### Subprocess Execution

```python
# R code dynamically generated, executed via Rscript

r_code = f'''
source("vivasense_genetics.R")
data <- fromJSON('{data_json}')
result <- genetics_analysis(data, mode="{mode}", ...)
cat(toJSON(result))
'''
```

---

## Deployment on Render

### Step 1: Prepare `requirements.txt`

```
fastapi==0.104.1
pydantic==2.4.2
uvicorn==0.24.0
python-multipart==0.0.6
```

### Step 2: Create `render.yaml`

```yaml
services:
  - type: web
    name: vivasense-genetics
    env: python
    buildCommand: |
      apt-get update && apt-get install -y r-base r-cran-jsonlite \
      r-cran-agricolae r-cran-dplyr r-cran-tidyr
      pip install -r requirements.txt
    startCommand: "uvicorn app_genetics:app --host 0.0.0.0 --port 8000"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11
```

### Step 3: Deploy

```bash
# Push to GitHub
git add .
git commit -m "VivaSense genetics module (R-based)"
git push origin main

# Connect to Render
# 1. Go to https://dashboard.render.com
# 2. New → Web Service
# 3. Connect GitHub repo
# 4. Select render.yaml
# 5. Deploy
```

### Step 4: Verify Deployment

```bash
# Health check
curl https://vivasense-r-api.onrender.com/health

# Test analysis
curl -X POST https://vivasense-r-api.onrender.com/genetics/analyze \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

---

## Performance & Scaling

### Computation Time

- Single-env (10–50 genotypes): **0.5–2 seconds**
- Multi-env (15–100 genotypes, 2–10 environments): **1–5 seconds**

### Memory Usage

- Small datasets (< 1000 observations): ~50 MB
- Large datasets (> 10,000 observations): ~200 MB

### Optimization Opportunities

1. **Caching**: Cache ANOVA results for identical datasets
2. **Parallel R**: Use `parallel` package for multi-environment subset analyses
3. **Pre-computation**: Pre-fit common designs, cache model objects

---

## Extending the Module

### Adding AMMI Analysis (Future)

```r
# ammi_analysis.R (to be added)
compute_ammi <- function(multi_env_result, coordinates = NULL) {
  # Takes multi-env result, runs AMMI decomposition
  # Returns PC1, PC2, biplots
}
```

### Adding GGE Biplot (Future)

```r
compute_gge <- function(multi_env_result) {
  # Takes G×E matrix
  # Returns GGE components
}
```

### Adding Stability Analysis (Future)

```r
compute_stability_statistics <- function(multi_env_result) {
  # Takes multi-env result
  # Returns: Shukla variance, ASV, W², etc.
}
```

---

## Troubleshooting

### Issue: "R execution failed"

**Solution**: Check that R packages are installed
```r
# In R console
install.packages(c("jsonlite", "agricolae", "dplyr", "tidyr"))
```

### Issue: "Negative σ²G in edge case"

**Expected behavior**: Engine clamps to 0, raises warning
```json
{
  "variance_warnings": {
    "negative_sigma2_genotype": {
      "value": -0.034,
      "message": "Genotypic variance is negative..."
    }
  }
}
```

### Issue: API timeout on large dataset

**Solution**: Increase timeout in `app_genetics.py`
```python
result = subprocess.run(..., timeout=300)  # 5 minutes
```

---

## Summary of Files

| File | Purpose |
|------|---------|
| `vivasense_genetics.R` | Core R engine (computation, validation, interpretation) |
| `app_genetics.py` | FastAPI wrapper with subprocess orchestration |
| `vivasense_genetics_schema.json` | OpenAPI specification |
| `test_cases_genetics.R` | Three comprehensive test cases |
| `test_case_[1-3]_output.json` | Test case JSON outputs |

---

## Next Steps

1. **Run test cases locally** to verify R engine
2. **Start FastAPI server** and test HTTP endpoints
3. **Deploy to Render** using `render.yaml`
4. **Integrate with VivaSense frontend** (React/Lovable)
5. **Add AMMI/GGE modules** (future phase)

---

**Author**: VivaSense Development Team  
**Version**: 1.0.0  
**Last Updated**: 2024-01-15
