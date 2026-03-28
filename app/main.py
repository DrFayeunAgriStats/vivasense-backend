import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VivaSense Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health probes (always return 200, independent of genetics module) ──────────

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

# Step 1 — trait-relationships (own engine, only needs jsonlite)
from trait_relationships_routes import (   # noqa: E402
    init_trait_relationships_engine,
    router as tr_router,
)
app.include_router(tr_router)
logger.info("Router registered: trait-relationships (/genetics/correlation)")

# Step 2 — multitrait upload (uses app_genetics.r_engine via lazy import)
from multitrait_upload_routes import router as multitrait_router  # noqa: E402
app.include_router(multitrait_router)
logger.info("Router registered: multitrait-upload (/genetics/upload-preview, /genetics/analyze-upload)")


# ── Startup diagnostics + engine initialisation ───────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=== VivaSense startup ===")
    logger.info("CWD: %s", os.getcwd())

    rscript = shutil.which("Rscript")
    if rscript:
        logger.info("Rscript found: %s", rscript)
    else:
        logger.warning("Rscript NOT found — R-backed endpoints will return 503")

    # Run the package installer so missing CRAN packages are fetched on first boot
    installer = Path(_genetics_dir) / "install_packages.R"
    if rscript and installer.exists():
        logger.info("Running install_packages.R …")
        result = subprocess.run(
            ["Rscript", str(installer)], capture_output=True, text=True
        )
        logger.info("install_packages.R: %s", result.stdout.strip() or "(no output)")
        if result.returncode != 0:
            logger.warning("install_packages.R exited %d: %s", result.returncode, result.stderr.strip())

    # Initialise the trait-relationships engine (non-fatal)
    init_trait_relationships_engine()

    # Initialise app_genetics.r_engine so multitrait_upload_routes can use it (non-fatal)
    try:
        import app_genetics  # noqa: PLC0415  – late import, genetics_dir already on path
        from app_genetics import RGeneticsEngine  # noqa: PLC0415
        app_genetics.r_engine = RGeneticsEngine("vivasense_genetics.R")
        logger.info("RGeneticsEngine ready (vivasense_genetics.R)")
    except Exception as exc:
        logger.error("RGeneticsEngine failed to initialise — multitrait endpoints will return 503: %s", exc)

    logger.info("=== VivaSense startup complete ===")
