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


# ── Startup diagnostics + engine initialisation ───────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=== VivaSense startup ===")
    logger.info("CWD: %s", os.getcwd())
    logger.info("genetics-module path: %s", _genetics_dir)
    logger.info("trait-relationships router loaded: %s", _tr_ok)
    logger.info("multitrait-upload router loaded: %s", _mt_ok)

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
