import os
import sys

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VivaSense Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health probes (must stay alive regardless of genetics module state) ────────

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

# ── Step 1: trait-relationships router ───────────────────────────────────────
# chdir so relative R script paths ("vivasense_trait_relationships.R") resolve.
_genetics_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "genetics-module")
)
os.chdir(_genetics_dir)
sys.path.insert(0, _genetics_dir)

from trait_relationships_routes import (   # noqa: E402
    init_trait_relationships_engine,
    router as tr_router,
)

app.include_router(tr_router)


@app.on_event("startup")
async def startup_event():
    init_trait_relationships_engine()   # non-fatal — 503 if R script missing
