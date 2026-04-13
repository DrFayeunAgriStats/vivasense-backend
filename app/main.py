"""
VivaSense Backend — main entry point for Render (uvicorn app.main:app).

Native-Python-safe: all genetics imports are guarded so the app always
starts and /health always returns 200, even if R or optional deps are absent.
"""
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

# Force stdout/stderr to be unbuffered so every print/log reaches Render immediately.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

print("=== app/main.py loading ===", flush=True)

app = FastAPI(title="VivaSense Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health probes — always 200, never depend on genetics state ────────────────

@app.get("/")
async def root():
    return {"message": "VivaSense backend running"}

@app.head("/")
async def root_head():
    return Response(status_code=200)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.head("/health")
async def health_head():
    return Response(status_code=200)

# ── Genetics module setup ─────────────────────────────────────────────────────
# chdir so relative R script paths ("vivasense_genetics.R" etc.) resolve.
_genetics_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "genetics-module")
)
os.chdir(_genetics_dir)
sys.path.insert(0, _genetics_dir)
print(f"genetics-module dir: {_genetics_dir}", flush=True)
print(f"CWD after chdir: {os.getcwd()}", flush=True)

# Step 1 — trait-relationships router (own R engine, only needs jsonlite)
_tr_ok = False
try:
    from trait_relationships_routes import (  # noqa: E402
        init_trait_relationships_engine,
        router as tr_router,
    )
    app.include_router(tr_router)
    _tr_ok = True
    print("Router registered: trait-relationships (/genetics/correlation)", flush=True)
except Exception as _e:
    print(f"WARN: trait-relationships router not loaded — {_e}", flush=True)
    init_trait_relationships_engine = None  # type: ignore[assignment]

# Step 2 — multitrait upload router (uses app_genetics.r_engine via lazy import)
_mt_ok = False
try:
    from multitrait_upload_routes import router as multitrait_router  # noqa: E402
    app.include_router(multitrait_router)
    _mt_ok = True
    print("Router registered: multitrait-upload (/genetics/upload-preview, /genetics/analyze-upload)", flush=True)
except Exception as _e:
    print(f"WARN: multitrait-upload router not loaded — {_e}", flush=True)

# Step 3 — Word export router (pure Python, no R dependency)
_ex_ok = False
try:
    from genetics_export import router as export_router  # noqa: E402
    app.include_router(export_router)
    _ex_ok = True
    print("Router registered: genetics-export (/genetics/download-results, /genetics/export-word)", flush=True)
except Exception as _e:
    print(f"WARN: genetics-export router not loaded — {_e}", flush=True)

# Step 4 — genetics analyze/validate routes from app_genetics
_ag_ok = False
try:
    from app_genetics import (  # noqa: E402
        GeneticsRequest,
        GeneticsResponse,
        ValidationResponse,
        analyze_genetics,
        validate_data,
    )
    from fastapi import HTTPException  # noqa: E402 (may already be imported)
    from fastapi.routing import APIRouter as _APIRouter  # noqa: E402

    _ag_router = _APIRouter(tags=["Genetics"])
    _ag_router.add_api_route("/genetics/analyze",  analyze_genetics, methods=["POST"], response_model=GeneticsResponse,   summary="Run genetic analysis")
    _ag_router.add_api_route("/genetics/validate", validate_data,    methods=["POST"], response_model=ValidationResponse, summary="Validate data before analysis")
    app.include_router(_ag_router)
    _ag_ok = True
    print("Router registered: genetics-analyze (/genetics/analyze, /genetics/validate)", flush=True)
except Exception as _e:
    print(f"WARN: genetics-analyze router not loaded — {_e}", flush=True)


# Step 5 — shared upload router (/upload/preview, /upload/dataset)
_up_ok = False
try:
    from upload_routes import router as upload_router  # noqa: E402
    app.include_router(upload_router)
    _up_ok = True
    print("Router registered: upload (/upload/preview, /upload/dataset)", flush=True)
except Exception as _e:
    print(f"WARN: upload router not loaded — {_e}", flush=True)

# Step 6 — ANOVA analysis module (/analysis/anova)
_an_anova_ok = False
try:
    from analysis_anova_routes import router as anova_router  # noqa: E402
    app.include_router(anova_router)
    _an_anova_ok = True
    print("Router registered: analysis-anova (/analysis/anova)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-anova router not loaded — {_e}", flush=True)

# Step 7 — Genetic Parameters analysis module (/analysis/genetic-parameters)
_an_gp_ok = False
try:
    from analysis_genetic_parameters_routes import router as gp_router  # noqa: E402
    app.include_router(gp_router)
    _an_gp_ok = True
    print("Router registered: analysis-genetic-parameters (/analysis/genetic-parameters)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-genetic-parameters router not loaded — {_e}", flush=True)

# Step 8 — Correlation analysis module (/analysis/correlation)
_an_corr_ok = False
try:
    from analysis_correlation_routes import router as corr_router  # noqa: E402
    app.include_router(corr_router)
    _an_corr_ok = True
    print("Router registered: analysis-correlation (/analysis/correlation)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-correlation router not loaded — {_e}", flush=True)

# Step 9 — Heatmap analysis module (/analysis/heatmap)
_an_hm_ok = False
try:
    from analysis_heatmap_routes import router as heatmap_router  # noqa: E402
    app.include_router(heatmap_router)
    _an_hm_ok = True
    print("Router registered: analysis-heatmap (/analysis/heatmap)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-heatmap router not loaded — {_e}", flush=True)

# Step 10 — Module-specific export endpoints (/export/*)
_ex_mod_ok = False
try:
    from export_module_routes import router as export_mod_router  # noqa: E402
    app.include_router(export_mod_router)
    _ex_mod_ok = True
    print("Router registered: export-modules (/export/anova-word, /export/genetic-parameters-word, /export/correlation-word, /export/heatmap-report)", flush=True)
except Exception as _e:
    print(f"WARN: export-modules router not loaded — {_e}", flush=True)


# ── Startup diagnostics + engine initialisation ───────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=== VivaSense startup ===")
    logger.info("CWD: %s", os.getcwd())
    logger.info("genetics-module path: %s", _genetics_dir)
    logger.info("trait-relationships router loaded: %s", _tr_ok)
    logger.info("multitrait-upload router loaded: %s", _mt_ok)
    logger.info("genetics-export router loaded: %s", _ex_ok)
    logger.info("genetics-analyze router loaded: %s", _ag_ok)
    logger.info("upload router loaded: %s", _up_ok)
    logger.info("analysis-anova router loaded: %s", _an_anova_ok)
    logger.info("analysis-genetic-parameters router loaded: %s", _an_gp_ok)
    logger.info("analysis-correlation router loaded: %s", _an_corr_ok)
    logger.info("analysis-heatmap router loaded: %s", _an_hm_ok)
    logger.info("export-modules router loaded: %s", _ex_mod_ok)

    rscript = shutil.which("Rscript")
    if rscript:
        logger.info("Rscript found: %s", rscript)
    else:
        logger.warning("Rscript NOT found — this is a native-Python deploy; R endpoints will return 503")

    # Run the CRAN package installer (only meaningful when R is present)
    installer = Path(_genetics_dir) / "install_packages.R"
    if rscript and installer.exists():
        logger.info("Running install_packages.R …")
        result = subprocess.run(
            ["Rscript", str(installer)], capture_output=True, text=True
        )
        logger.info("install_packages.R: %s", result.stdout.strip() or "(no output)")
        if result.returncode != 0:
            logger.warning("install_packages.R exited %d: %s", result.returncode, result.stderr.strip())

    # Initialise trait-relationships engine (non-fatal)
    if init_trait_relationships_engine is not None:
        init_trait_relationships_engine()

    # Initialise RGeneticsEngine so multitrait_upload_routes can use it (non-fatal)
    try:
        import app_genetics  # noqa: PLC0415
        from app_genetics import RGeneticsEngine  # noqa: PLC0415
        app_genetics.r_engine = RGeneticsEngine("vivasense_genetics.R")
        logger.info("RGeneticsEngine ready (vivasense_genetics.R)")
    except Exception as exc:
        logger.error("RGeneticsEngine init failed (503 on R endpoints): %s", exc)

    logger.info("=== VivaSense startup complete ===")
