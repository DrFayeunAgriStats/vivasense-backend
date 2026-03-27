"""
VivaSense Genetics API Backend
FastAPI wrapper for R genetics computation engine

Handles:
- Request validation against OpenAPI schema
- R engine orchestration via subprocess
- JSON serialization/deserialization
- Error handling and response formatting
"""

from fastapi import FastAPI, HTTPException
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
        "https://fieldtoinsightacademy.com.ng",
        "https://www.fieldtoinsightacademy.com.ng",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Multi-trait upload endpoints
from multitrait_upload_routes import router as multitrait_router
app.include_router(multitrait_router)

# Trait relationships endpoints (Phase 2)
from trait_relationships_routes import router as tr_router
app.include_router(tr_router)


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
        random_environment: bool = False
    ) -> Dict[str, Any]:
        """
        Execute genetics analysis via R subprocess
        
        Args:
            data: List of observation dicts
            mode: 'single' or 'multi'
            trait_name: Name of trait
            random_environment: Include E in h² denominator (multi-mode)
        
        Returns:
            Parsed JSON result from R
        """
        
        try:
            # Create temporary R script that loads data and runs analysis
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.R',
                delete=False,
                dir='/tmp'
            ) as tmp_r:
                
                # Serialize input data as R code
                data_json = json.dumps(data)
                random_env_str = "TRUE" if random_environment else "FALSE"
                
                r_code = f'''
# VivaSense Genetics Analysis Execution
source("{self.r_script_path}")

# Load data from JSON
data_json <- '{data_json}'
data_list <- jsonlite::fromJSON(data_json)
data <- as.data.frame(data_list)

# Convert to proper types
data$trait_value <- as.numeric(data$trait_value)
data$genotype <- as.character(data$genotype)
data$rep <- as.character(data$rep)
if ("{mode}" == "multi") {{
  data$environment <- as.character(data$environment)
}}

# Run analysis
result <- genetics_analysis(
    data = data,
    mode = "{mode}",
    trait_name = "{trait_name}",
    random_environment = {random_env_str}
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
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"R execution failed: {result.stderr}")
                raise RuntimeError(f"R error: {result.stderr}")
            
            # Parse JSON output
            json_output = result.stdout.strip()
            analysis_result = json.loads(json_output)
            
            logger.info(f"Analysis completed successfully (status={analysis_result.get('status')})")
            
            return analysis_result
            
        except subprocess.TimeoutExpired:
            logger.error("R analysis timeout (60s exceeded)")
            raise RuntimeError("Analysis timeout: R execution exceeded 60 seconds")
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse R JSON output: {e}")
            raise RuntimeError(f"Invalid JSON from R: {str(e)}")
        
        except Exception as e:
            logger.error(f"Unexpected error in R execution: {e}")
            raise
        
        finally:
            # Clean up temporary R script
            if 'tmp_r_path' in locals():
                try:
                    os.unlink(tmp_r_path)
                except:
                    pass


# Initialize R engine (on app startup)
r_engine = None

@app.on_event("startup")
async def startup_event():
    global r_engine
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
    Execute genetics analysis on provided data
    
    Supports:
    - Single-environment ANOVA with heritability
    - Multi-environment analysis with G×E variance partitioning
    
    Returns:
    - Variance components
    - Heritability (broad-sense, entry-mean basis)
    - Genetic parameters (GCV, PCV, GAM)
    - Publication-ready interpretation text
    - Warnings for edge cases
    """
    
    try:
        # Call R engine
        result = r_engine.run_analysis(
            data=request.data,
            mode=request.mode,
            trait_name=request.trait_name,
            random_environment=request.random_environment
        )
        
        # Return as Pydantic model (validates schema)
        return GeneticsResponse(**result)
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except RuntimeError as e:
        logger.error(f"R execution error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error in /analyze endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


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
            delete=False,
            dir='/tmp'
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
            ["Rscript", tmp_r_path],
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
async def root():
    """Service information"""
    return {
        "name": "VivaSense Genetics Engine",
        "version": "1.0.0",
        "description": "R-based genetics analysis with single and multi-environment support",
        "endpoints": {
            "POST /genetics/analyze": "Run genetic analysis (manual input)",
            "POST /genetics/validate": "Validate data before analysis",
            "POST /genetics/upload-preview": "Preview uploaded file + detect columns",
            "POST /genetics/analyze-upload": "Analyze all traits in uploaded CSV/Excel",
            "POST /genetics/correlation": "Phenotypic correlation between trait pairs",
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
