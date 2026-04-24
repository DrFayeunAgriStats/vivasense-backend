"""
VivaSense Genetics API Backend
FastAPI wrapper for R genetics computation engine

Handles:
- Request validation against OpenAPI schema
- R engine orchestration via subprocess
- JSON serialization/deserialization
- Error handling and response formatting
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from genetics_schemas import GeneticsResult, GeneticsResponse
import subprocess
import json
import tempfile
import os
import sys
from pathlib import Path
import logging
from interpretation import InterpretationEngine

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS (Request/Response Schema)
# ============================================================================

class SingleEnvironmentObservation(BaseModel):
    """Single-environment observation record"""
    genotype: str
    rep: str
    trait_value: float

    class Config:
        example = {"genotype": "G01", "rep": "R1", "trait_value": 45.2}


class MultiEnvironmentObservation(BaseModel):
    """Multi-environment observation record"""
    genotype: str
    environment: str
    rep: str
    trait_value: float

    class Config:
        example = {"genotype": "G01", "environment": "E1", "rep": "R1", "trait_value": 45.2}


class GeneticsRequest(BaseModel):
    """Request payload for genetics analysis"""
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Array of observation records"
    )
    mode: str = Field(
        ...,
        description="Analysis mode: 'single' or 'multi'",
        pattern="^(single|multi)$"
    )
    trait_name: Optional[str] = Field(
        default="Trait",
        description="Name of the trait being analyzed"
    )
    random_environment: Optional[bool] = Field(
        default=False,
        description="Multi-mode only: treat environment as random effect"
    )

    @validator("data")
    def validate_data_not_empty(cls, v):
        if not v or len(v) < 6:
            raise ValueError("data must contain at least 6 observations")
        return v

    @validator("mode")
    def validate_mode(cls, v):
        if v not in ["single", "multi"]:
            raise ValueError("mode must be 'single' or 'multi'")
        return v


class VarianceComponents(BaseModel):
    """Variance components output"""
    sigma2_genotype: float
    sigma2_error: float
    sigma2_ge: Optional[float] = None
    sigma2_phenotypic: float
    heritability_basis: Optional[str] = None


class Heritability(BaseModel):
    """Heritability estimates"""
    h2_broad_sense: float
    interpretation_basis: str
    formula: Optional[str] = None


class GeneticParameters(BaseModel):
    """Genetic parameters (GCV, PCV, GAM)"""
    GCV: Optional[float] = None
    PCV: Optional[float] = None
    GAM: Optional[float] = None
    GAM_percent: Optional[float] = None
    selection_intensity: float = 1.4


class ValidationResponse(BaseModel):
    """Response from validation endpoint"""
    is_valid: bool
    warnings: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response format"""
    status: str = "ERROR"
    mode: Optional[str] = None
    errors: Dict[str, Any]
    result: None = None
    interpretation: None = None


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="VivaSense Genetics Engine",
    description="R-based genetics analysis with single and multi-environment support",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.fieldtoinsightacademy.com.ng",
        "https://fieldtoinsightacademy.com.ng",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Length", "Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log 422 validation errors with full detail so Render logs show the exact failing field."""
    import json as _json
    logger.error(
        "422 RequestValidationError on %s %s\nErrors: %s",
        request.method,
        request.url.path,
        _json.dumps(exc.errors(), default=str, indent=2),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


# Multi-trait upload endpoints
from multitrait_upload_routes import router as multitrait_router
app.include_router(multitrait_router)

# Trait relationships endpoints (Phase 2)
from trait_relationships_routes import router as tr_router
app.include_router(tr_router)

# Word export endpoint — wrapped in try/except so app starts even if
# python-docx is not yet installed (e.g. first Docker cold-start).
try:
    from genetics_export import router as export_router
    app.include_router(export_router)
    logger.info("genetics_export router loaded (/genetics/download-results, /genetics/export-word)")
except Exception as _export_err:
    logger.warning("genetics_export not loaded (%s) — /genetics/download-results will return 503", _export_err)

    # Fallback: register stub endpoints that return 503 so frontend gets a
    # meaningful error instead of a 404 / connection reset.
    from fastapi import Request as _Request
    from fastapi.responses import JSONResponse as _JSONResponse

    @app.post("/genetics/download-results", tags=["Export"], include_in_schema=True)
    @app.post("/genetics/export-word", tags=["Export"], include_in_schema=True)
    async def _export_unavailable(_request: _Request):
        return _JSONResponse(
            status_code=503,
            content={"detail": f"Word export unavailable: {_export_err}"},
        )

# ── Module-based pipeline (Steps 5–11) ───────────────────────────────────────

# Step 5 — shared upload router (/upload/preview, /upload/dataset)
try:
    from upload_routes import router as upload_router
    app.include_router(upload_router)
    logger.info("upload router loaded (/upload/preview, /upload/dataset)")
except Exception as _e:
    logger.warning("upload router not loaded — %s", _e)

# Step 5.5 — Descriptive Stats analysis module (/analysis/descriptive-stats)
try:
    from analysis_descriptive_stats_routes import router as desc_router
    app.include_router(desc_router)
    logger.info("analysis-descriptive-stats router loaded (/analysis/descriptive-stats)")
except Exception as _e:
    logger.warning("analysis-descriptive-stats router not loaded — %s", _e)

# Step 6 — ANOVA analysis module (/analysis/anova)
try:
    from analysis_anova_routes import router as anova_router
    app.include_router(anova_router)
    logger.info("analysis-anova router loaded (/analysis/anova)")
except Exception as _e:
    logger.warning("analysis-anova router not loaded — %s", _e)

# Step 7 — Genetic Parameters analysis module (/analysis/genetic-parameters)
try:
    from analysis_genetic_parameters_routes import router as gp_router
    app.include_router(gp_router)
    logger.info("analysis-genetic-parameters router loaded (/analysis/genetic-parameters)")
except Exception as _e:
    logger.warning("analysis-genetic-parameters router not loaded — %s", _e)

# Step 8 — Correlation analysis module (/analysis/correlation)
try:
    from analysis_correlation_routes import router as corr_router
    app.include_router(corr_router)
    logger.info("analysis-correlation router loaded (/analysis/correlation)")
except Exception as _e:
    logger.warning("analysis-correlation router not loaded — %s", _e)

# Step 9 — Heatmap analysis module (/analysis/heatmap)
try:
    from analysis_heatmap_routes import router as heatmap_router
    app.include_router(heatmap_router)
    logger.info("analysis-heatmap router loaded (/analysis/heatmap)")
except Exception as _e:
    logger.warning("analysis-heatmap router not loaded — %s", _e)

# Step 9.5 — Trait Association analysis module (/genetics/trait-association/analyze)
try:
    from analysis_trait_association_routes import router as trait_assoc_router
    app.include_router(trait_assoc_router)
    logger.info("analysis-trait-association router loaded (/genetics/trait-association/analyze)")
except Exception as _e:
    logger.warning("analysis-trait-association router not loaded — %s", _e)

# Step 9.6 — Regression analysis module (/analysis/regression)
try:
    from analysis_regression_routes import router as regression_router
    app.include_router(regression_router)
    logger.info("analysis-regression router loaded (/analysis/regression)")
except Exception as _e:
    logger.warning("analysis-regression router not loaded — %s", _e)

# Step 9.7 — Stability analysis module (/analysis/stability)
try:
    from analysis_stability_routes import router as stability_router
    app.include_router(stability_router)
    logger.info("analysis-stability router loaded (/analysis/stability)")
except Exception as _e:
    logger.warning("analysis-stability router not loaded — %s", _e)

# Step 9.8 — BLUP analysis module (/analysis/blup)
try:
    from analysis_blup_routes import router as blup_router
    app.include_router(blup_router)
    logger.info("analysis-blup router loaded (/analysis/blup)")
except Exception as _e:
    logger.warning("analysis-blup router not loaded — %s", _e)

# Step 9.9 — PCA analysis module (/analysis/pca)
try:
    from analysis_pca_routes import router as pca_router
    app.include_router(pca_router)
    logger.info("analysis-pca router loaded (/analysis/pca)")
except Exception as _e:
    logger.warning("analysis-pca router not loaded — %s", _e)

# Step 9.10 — Cluster analysis module (/analysis/cluster)
try:
    from analysis_cluster_routes import router as cluster_router
    app.include_router(cluster_router)
    logger.info("analysis-cluster router loaded (/analysis/cluster)")
except Exception as _e:
    logger.warning("analysis-cluster router not loaded — %s", _e)

# Step 9.11 — Path analysis module (/analysis/path-analysis)
try:
    from analysis_path_routes import router as path_analysis_router
    app.include_router(path_analysis_router)
    logger.info("analysis-path-analysis router loaded (/analysis/path-analysis)")
except Exception as _e:
    logger.warning("analysis-path-analysis router not loaded — %s", _e)

# Step 10 — Module-specific Word/report export (/export/*)
try:
    from export_module_routes import router as export_mod_router
    app.include_router(export_mod_router)
    logger.info("export-modules router loaded (/export/anova-word, /export/genetic-parameters-word, /export/correlation-word, /export/heatmap-report)")
except Exception as _e:
    logger.warning("export-modules router not loaded — %s", _e)

# Step 11 — Academic Mentor (/academic/interpret)
try:
    from academic_routes import router as academic_router
    app.include_router(academic_router)
    logger.info("academic-mentor router loaded (/academic/interpret)")
except Exception as _e:
    logger.warning("academic-mentor router not loaded — %s", _e)


# ============================================================================
# R ENGINE ORCHESTRATION
# ============================================================================

class RGeneticsEngine:
    """
    Wrapper to call R genetics engine via subprocess
    Handles data serialization, R script execution, and result parsing
    """

    def __init__(self, r_script_path: str = "vivasense_genetics.R"):
        self.r_script_path = r_script_path
        
        # Verify R script exists
        if not Path(r_script_path).exists():
            raise FileNotFoundError(f"R script not found: {r_script_path}")
        
        logger.info(f"R genetics engine initialized: {r_script_path}")

    def run_analysis(
        self,
        data: List[Dict[str, Any]],
        mode: str,
        trait_name: str = "Trait",
        random_environment: bool = False,
        crd_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute genetics analysis via R subprocess.

        Args:
            data: List of observation dicts.
            mode: 'single' or 'multi'.
            trait_name: Name of trait.
            random_environment: Include E in h² denominator (multi-mode).
            crd_mode: When True, use CRD model (trait ~ genotype or factorial)
                      instead of RCBD model (trait ~ rep + genotype).
                      Records must contain "rep" (synthetic) and optionally
                      "factor" for factorial CRD.
        Returns:
            Parsed JSON result from R.
        """

        try:
            # Serialize input data directly to a temporary JSON file (Memory Optimized)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_json:
                json.dump(data, tmp_json)
                tmp_json_path = tmp_json.name

            # Create temporary R script that loads data and runs analysis
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.R',
                delete=False
            ) as tmp_r:

                random_env_str = "TRUE" if random_environment else "FALSE"
                crd_mode_str   = "TRUE" if crd_mode else "FALSE"
                # Escape file path for R (converts Windows \ to /)
                r_json_path = tmp_json_path.replace('\\', '/')

                r_code = f'''
# VivaSense Genetics Analysis Execution
.libPaths(c("C:/Users/user/.gemini/antigravity/scratch/R_libs", .libPaths()))
source("{self.r_script_path}")

# Load data from JSON file
data_list <- jsonlite::fromJSON('{r_json_path}')
data <- as.data.frame(data_list)

# Convert to proper types
data$trait_value <- as.numeric(data$trait_value)
data$genotype <- as.character(data$genotype)
data$rep <- as.character(data$rep)
if ("{mode}" == "multi") {{
  data$environment <- as.character(data$environment)
}}
# Factorial CRD: factor column present when crd_mode and env supplied
if ("factor" %in% colnames(data)) {{
  data$factor <- as.character(data$factor)
}}

# Ensure Type III Sum of Squares is used for unbalanced data
if (requireNamespace("car", quietly = TRUE)) {{
  library(car)
  options(contrasts = c("contr.sum", "contr.poly"))
}} else {{
  warning("Package 'car' is not available. Standard ANOVA will be used, which may be biased for unbalanced data.")
}}

# Run analysis
result <- genetics_analysis(
    data = data,
    mode = "{mode}",
    trait_name = "{trait_name}",
    random_environment = {random_env_str},
    crd_mode = {crd_mode_str}
)

# Export to JSON and output
json_output <- export_to_json(result)
cat(json_output)
'''
                
                tmp_r.write(r_code)
                tmp_r_path = tmp_r.name
            
            # Execute R script via Rscript
            logger.info(f"Executing R analysis (mode={mode}, n_obs={len(data)})")
            
            result = subprocess.run(
                ["Rscript", tmp_r_path],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                stderr_preview = result.stderr[:800] if result.stderr else "(empty)"
                logger.error("R execution failed (rc=%d):\n%s", result.returncode, stderr_preview)
                raise RuntimeError(f"R error: {stderr_preview}")

            # Parse JSON output
            json_output = result.stdout.strip()
            if not json_output:
                logger.error("R produced no output. stderr: %s", result.stderr[:400])
                raise RuntimeError("R produced no JSON output. stderr: " + result.stderr[:400])

            analysis_result = json.loads(json_output)

            # InterpretationEngine (genetic parameter narrative) is intentionally
            # NOT called here. run_analysis() is module-agnostic — it services
            # ANOVA, Genetic Parameters, Correlation, and Heatmap requests.
            # Genetic parameter interpretation belongs in analysis_genetic_parameters_routes.py
            # and the Academic Mentor (/academic/interpret), not in this wrapper.

            logger.info("Analysis completed (status=%s, mode=%s)", analysis_result.get("status"), mode)

            return analysis_result

        except subprocess.TimeoutExpired:
            logger.error("R analysis timeout (120s exceeded)")
            raise RuntimeError("Analysis timeout: R execution exceeded 120 seconds")
        
        except json.JSONDecodeError as e:
            preview = json_output[:500] if json_output else "(empty)"
            logger.error("Failed to parse R JSON output: %s\nstdout preview: %s\nstderr: %s",
                         e, preview, result.stderr[:500] if result.stderr else "(empty)")
            raise RuntimeError(f"Invalid JSON from R: {str(e)}")
        
        except Exception as e:
            logger.error(f"Unexpected error in R execution: {e}")
            raise
        
        finally:
            # Clean up temporary files
            for tmp_file in ('tmp_r_path', 'tmp_json_path'):
                if tmp_file in locals():
                    try:
                        os.unlink(locals()[tmp_file])
                    except:
                        pass


# Initialize R engine (on app startup)
r_engine = None

@app.on_event("startup")
async def startup_event():
    global r_engine

    # Run R package installer to ensure all dependencies are present.
    # Runs at startup so missing packages are installed even after a hot restart.
    installer = Path(__file__).parent / "install_packages.R"
    if installer.exists():
        logger.info("Running install_packages.R …")
        result = subprocess.run(
            ["Rscript", str(installer)],
            capture_output=True, text=True
        )
        if result.stdout:
            logger.info("install_packages.R stdout: %s", result.stdout.strip())
        if result.returncode != 0:
            logger.warning("install_packages.R exited %d: %s", result.returncode, result.stderr.strip())

    try:
        r_engine = RGeneticsEngine("vivasense_genetics.R")
        logger.info("VivaSense R genetics engine ready")
    except Exception as e:
        logger.error(f"Failed to initialize R engine: {e}")
        raise

    # Trait relationships engine — non-fatal if R script missing
    from trait_relationships_routes import init_trait_relationships_engine
    init_trait_relationships_engine()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post(
    "/genetics/analyze",
    response_model=GeneticsResponse,
    summary="Run genetic analysis",
    tags=["Genetics"]
)
async def analyze_genetics(request: GeneticsRequest):
    """
    Execute genetics analysis on provided data.

    Uses ANOVA internally to partition variance and estimate heritability.
    ANOVA is the statistical foundation — not a separate tool.

    Supports:
    - Single-environment analysis (RCBD / CRD)
    - Multi-environment analysis with G×E variance partitioning

    Returns:
    - ANOVA table (source, df, SS, MS, F-value, p-value)
    - Mean separation (Tukey HSD grouping letters)
    - Variance components (σ²g, σ²e, σ²ge)
    - Heritability (broad-sense, entry-mean basis)
    - Genetic parameters (GCV, PCV, GAM)
    - Publication-ready interpretation text
    """
    
    if r_engine is None:
        raise HTTPException(status_code=503, detail="R engine not ready — check Render startup logs")

    try:
        result = r_engine.run_analysis(
            data=request.data,
            mode=request.mode,
            trait_name=request.trait_name,
            random_environment=request.random_environment,
        )
        return GeneticsResponse(**result)

    except ValueError as e:
        logger.warning("Validation error in /genetics/analyze: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        logger.error("R execution error in /genetics/analyze: %s", e)
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error("Unexpected error in /genetics/analyze: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/genetics/validate",
    response_model=ValidationResponse,
    summary="Validate data before analysis",
    tags=["Genetics"]
)
async def validate_data(request: GeneticsRequest):
    """
    Pre-flight validation of data without running full analysis
    
    Checks:
    - Required columns present
    - Missing values
    - Minimum replication
    - Trait variation
    """
    
    try:
        # Create minimal R validation script
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.R',
            delete=False
        ) as tmp_r:
            
            data_json = json.dumps(request.data)
            
            r_code = f'''
source("vivasense_genetics.R")

# Load data
data_list <- jsonlite::fromJSON('{data_json}')
data <- as.data.frame(data_list)
data$trait_value <- as.numeric(data$trait_value)

# Validate
validation <- validate_input_data(data, env_mode = "{request.mode}")

# Output
cat(jsonlite::toJSON(validation, pretty = TRUE))
'''
            
            tmp_r.write(r_code)
            tmp_r_path = tmp_r.name
        
        # Execute
        result = subprocess.run(
            [r"C:\Program Files\R\R-4.5.3\bin\x64\Rscript.exe", tmp_r_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"R error: {result.stderr}")
        
        validation_result = json.loads(result.stdout.strip())
        return ValidationResponse(**validation_result)
    
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    
    finally:
        if 'tmp_r_path' in locals():
            try:
                os.unlink(tmp_r_path)
            except:
                pass


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "VivaSense Genetics Engine",
        "version": "1.0.0",
        "r_engine_ready": r_engine is not None
    }


@app.get("/", tags=["Documentation"])
@app.head("/", tags=["Documentation"])
async def root():
    """Service information"""
    return {
        "name": "VivaSense Genetics Engine",
        "version": "1.0.0",
        "description": "R-based genetics analysis with single and multi-environment support",
        "endpoints": {
            "POST /genetics/analyze": "Run genetic analysis (ANOVA + heritability + Tukey HSD)",
            "POST /genetics/validate": "Validate data before analysis",
            "POST /genetics/upload-preview": "Preview uploaded file + detect columns",
            "POST /genetics/analyze-upload": "Analyze all traits in uploaded CSV/Excel",
            "POST /genetics/correlation": "Phenotypic correlation between trait pairs",
            "POST /genetics/download-results": "Download Word report (.docx)",
            "POST /genetics/export-word": "Download Word report (.docx) — alias",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation (Swagger UI)"
        },
        "documentation": "/docs"
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "status": "ERROR",
            "errors": {"validation": str(exc)}
        }
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "status": "ERROR",
            "errors": {"runtime": str(exc)}
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
