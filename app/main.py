"""
VivaSense Backend — main entry point for Render (uvicorn app.main:app).

Native-Python-safe: all genetics imports are guarded so the app always
starts and /health always returns 200, even if R or optional deps are absent.
"""
import logging
import json
import os
import sys

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from app.core.startup_checks import run_startup_checks
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

PRO_GATED_PATHS = {
    "/analysis/genetic-parameters",
    "/analysis/pca",
    "/analysis/cluster",
    "/analysis/selection-index",
    "/analysis/path-analysis",
    "/analysis/path-analysis/preflight",
    "/genetics/download-results",
    "/genetics/export-word",
    "/export/descriptive-stats-word",
    "/export/anova-word",
    "/export/genetic-parameters-word",
    "/export/correlation-word",
    "/export/heatmap-report",
    "/academic/interpret",
}


def _is_pro_gated_path(path: str) -> bool:
    return path in PRO_GATED_PATHS


async def _is_pro_analyze_upload_request(request: Request) -> bool:
    if request.url.path != "/genetics/analyze-upload" or request.method.upper() != "POST":
        return False

    module_query = (request.query_params.get("module") or "").strip().lower()
    body_module = ""
    body_mode = ""
    body_env_col = None

    try:
        raw = await request.body()
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                body_module = str(parsed.get("module") or "").strip().lower()
                body_mode = str(parsed.get("mode") or "").strip().lower()
                body_env_col = parsed.get("environment_column")
    except Exception:
        pass

    # Body module takes priority; fallback to query; endpoint default is genetic_parameters.
    actual_module = body_module or module_query or "genetic_parameters"

    if actual_module == "genetic_parameters":
        return True

    if actual_module == "anova":
        has_environment_factor = isinstance(body_env_col, str) and body_env_col.strip() != ""
        return body_mode == "multi" or has_environment_factor

    return False


@app.middleware("http")
async def vivasense_mode_gate(request: Request, call_next):
    requires_pro = _is_pro_gated_path(request.url.path)
    if not requires_pro:
        requires_pro = await _is_pro_analyze_upload_request(request)

    if requires_pro:
        mode = request.headers.get("X-VivaSense-Mode", "free").lower().strip()
        if mode != "pro":
            return JSONResponse(
                status_code=403,
                content={
                    "error": "PRO_FEATURE",
                    "message": "Upgrade to access this feature",
                },
            )
    return await call_next(request)

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

# Step 5.5 — Descriptive Stats analysis module (/analysis/descriptive-stats)
_an_desc_ok = False
try:
    from analysis_descriptive_stats_routes import router as desc_router  # noqa: E402
    app.include_router(desc_router)
    _an_desc_ok = True
    print("Router registered: analysis-descriptive-stats (/analysis/descriptive-stats)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-descriptive-stats router not loaded — {_e}", flush=True)

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

# Step 9.5 — Trait Association analysis module (/genetics/trait-association/analyze)
_an_ta_ok = False
try:
    from analysis_trait_association_routes import router as trait_assoc_router  # noqa: E402
    app.include_router(trait_assoc_router)
    _an_ta_ok = True
    print("Router registered: analysis-trait-association (/genetics/trait-association/analyze)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-trait-association router not loaded — {_e}", flush=True)

# Step 9.6 — Regression analysis module (/analysis/regression)
_an_reg_ok = False
try:
    from analysis_regression_routes import router as regression_router  # noqa: E402
    app.include_router(regression_router)
    _an_reg_ok = True
    print("Router registered: analysis-regression (/analysis/regression)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-regression router not loaded — {_e}", flush=True)

# Step 9.7 — Stability analysis module (/analysis/stability)
_an_stab_ok = False
try:
    from analysis_stability_routes import router as stability_router  # noqa: E402
    app.include_router(stability_router)
    _an_stab_ok = True
    print("Router registered: analysis-stability (/analysis/stability)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-stability router not loaded — {_e}", flush=True)

# Step 9.8 — BLUP analysis module (/analysis/blup)
_an_blup_ok = False
try:
    from analysis_blup_routes import router as blup_router  # noqa: E402
    app.include_router(blup_router)
    _an_blup_ok = True
    print("Router registered: analysis-blup (/analysis/blup)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-blup router not loaded — {_e}", flush=True)

# Step 9.9 — PCA analysis module (/analysis/pca)
_an_pca_ok = False
try:
    from analysis_pca_routes import router as pca_router  # noqa: E402
    app.include_router(pca_router)
    _an_pca_ok = True
    print("Router registered: analysis-pca (/analysis/pca)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-pca router not loaded — {_e}", flush=True)

# Step 9.10 — Cluster analysis module (/analysis/cluster)
_an_cluster_ok = False
try:
    from analysis_cluster_routes import router as cluster_router  # noqa: E402
    app.include_router(cluster_router)
    _an_cluster_ok = True
    print("Router registered: analysis-cluster (/analysis/cluster)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-cluster router not loaded — {_e}", flush=True)

# Step 9.11 — Non-parametric tests module (/analysis/nonparametric)
_an_np_ok = False
try:
    from analysis_nonparametric_routes import router as nonparametric_router  # noqa: E402
    app.include_router(nonparametric_router)
    _an_np_ok = True
    print("Router registered: analysis-nonparametric (/analysis/nonparametric)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-nonparametric router not loaded — {_e}", flush=True)

# Step 9.12 — MANOVA module (/analysis/manova)
_an_manova_ok = False
try:
    from analysis_manova_routes import router as manova_router  # noqa: E402
    app.include_router(manova_router)
    _an_manova_ok = True
    print("Router registered: analysis-manova (/analysis/manova)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-manova router not loaded — {_e}", flush=True)

# Step 9.13 — Path analysis module (/analysis/path-analysis)
_an_path_ok = False
try:
    from analysis_path_routes import router as path_router  # noqa: E402
    app.include_router(path_router)
    _an_path_ok = True
    print("Router registered: analysis-path-analysis (/analysis/path-analysis)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-path-analysis router not loaded — {_e}", flush=True)

# Step 9.14 — Selection Index module (/analysis/selection-index)
_an_si_ok = False
try:
    from analysis_selection_index_routes import router as selection_index_router  # noqa: E402
    app.include_router(selection_index_router)
    _an_si_ok = True
    print("Router registered: analysis-selection-index (/analysis/selection-index)", flush=True)
except Exception as _e:
    print(f"WARN: analysis-selection-index router not loaded — {_e}", flush=True)

# Step 10 — Module-specific export endpoints (/export/*)
_ex_mod_ok = False
try:
    from export_module_routes import router as export_mod_router  # noqa: E402
    app.include_router(export_mod_router)
    _ex_mod_ok = True
    print("Router registered: export-modules (/export/anova-word, /export/genetic-parameters-word, /export/correlation-word, /export/heatmap-report)", flush=True)
except Exception as _e:
    print(f"WARN: export-modules router not loaded — {_e}", flush=True)


# Step 11 — Academic Mentor (/academic/interpret)
_ac_ok = False
try:
    from academic_routes import router as academic_router  # noqa: E402
    app.include_router(academic_router)
    _ac_ok = True
    print("Router registered: academic-mentor (/academic/interpret)", flush=True)
except Exception as _e:
    print(f"WARN: academic-mentor router not loaded — {_e}", flush=True)


# ── Startup diagnostics + engine initialisation ───────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    logger.info("=== VivaSense startup ===")
    # Verify R is available and all required packages are installed.
    # This raises RuntimeError (and aborts startup) if the Docker build
    # omitted R or a package failed to install.
    try:
        run_startup_checks()
    except RuntimeError as _rsc_err:
        logger.critical("startup_checks FAILED — %s", _rsc_err)
        raise

    logger.info("CWD: %s", os.getcwd())
    logger.info("genetics-module path: %s", _genetics_dir)
    logger.info("trait-relationships router loaded: %s", _tr_ok)
    logger.info("multitrait-upload router loaded: %s", _mt_ok)
    logger.info("genetics-export router loaded: %s", _ex_ok)
    logger.info("genetics-analyze router loaded: %s", _ag_ok)
    logger.info("upload router loaded: %s", _up_ok)
    logger.info("analysis-descriptive-stats router loaded: %s", _an_desc_ok)
    logger.info("analysis-anova router loaded: %s", _an_anova_ok)
    logger.info("analysis-genetic-parameters router loaded: %s", _an_gp_ok)
    logger.info("analysis-correlation router loaded: %s", _an_corr_ok)
    logger.info("analysis-heatmap router loaded: %s", _an_hm_ok)
    logger.info("analysis-regression router loaded: %s", _an_reg_ok)
    logger.info("analysis-stability router loaded: %s", _an_stab_ok)
    logger.info("analysis-blup router loaded: %s", _an_blup_ok)
    logger.info("analysis-pca router loaded: %s", _an_pca_ok)
    logger.info("analysis-cluster router loaded: %s", _an_cluster_ok)
    logger.info("analysis-nonparametric router loaded: %s", _an_np_ok)
    logger.info("analysis-manova router loaded: %s", _an_manova_ok)
    logger.info("analysis-path-analysis router loaded: %s", _an_path_ok)
    logger.info("analysis-selection-index router loaded: %s", _an_si_ok)
    logger.info("export-modules router loaded: %s", _ex_mod_ok)
    logger.info("academic-mentor router loaded: %s", _ac_ok)

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

    # === DIAGNOSTIC: Dump all registered routes ===
    print("\n=== ROUTE DUMP (for debugging) ===", flush=True)
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        print(f"ROUTE: {methods} {path}", flush=True)
    print("=== END ROUTE DUMP ===\n", flush=True)

    logger.info("=== VivaSense startup complete ===")
