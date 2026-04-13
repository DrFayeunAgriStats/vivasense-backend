"""
VivaSense – Dataset context cache for module-based analysis pipeline.

Two caches in one module:

  _dataset_store  — keyed by dataset_token, holds the parsed dataset context
                    (column mappings + base64 file bytes).  Populated by
                    POST /upload/dataset; read by every /analysis/* endpoint.

  _analysis_store — keyed by (dataset_token, trait_name), holds the full
                    GeneticsResponse returned by the R engine for that trait.
                    Populated by the first /analysis/* endpoint to touch a trait;
                    subsequent endpoints read from cache instead of re-running R.

Both caches use LRU eviction (oldest entry removed when the cap is exceeded)
and are protected by a single module-level lock — safe for single-worker deploy.
"""

import logging
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_DATASET_MAX  = 50    # max distinct upload sessions cached at once
_ANALYSIS_MAX = 500   # max (token, trait) pairs cached at once

_lock: threading.Lock = threading.Lock()

_dataset_store:  OrderedDict = OrderedDict()   # token → dict
_analysis_store: OrderedDict = OrderedDict()   # (token, trait) → GeneticsResponse


# ── token helpers ─────────────────────────────────────────────────────────────

def create_token() -> str:
    return str(uuid.uuid4())


# ── dataset context ───────────────────────────────────────────────────────────

def put_dataset(token: str, context: Dict[str, Any]) -> None:
    """
    Store a dataset context under *token*.

    context must contain at minimum:
      base64_content, file_type, genotype_column, rep_column,
      environment_column (or None), mode, random_environment, selection_intensity
    """
    with _lock:
        _dataset_store[token] = context
        _dataset_store.move_to_end(token)
        while len(_dataset_store) > _DATASET_MAX:
            evicted, _ = _dataset_store.popitem(last=False)
            logger.debug("dataset_cache: evicted dataset token %s", evicted)
    logger.info("dataset_cache: stored dataset token %s", token)


def get_dataset(token: str) -> Optional[Dict[str, Any]]:
    """Return the dataset context for *token*, or None if not found / evicted."""
    with _lock:
        ctx = _dataset_store.get(token)
    if ctx is None:
        logger.warning("dataset_cache: dataset miss for token %s", token)
    return ctx


# ── per-trait analysis cache ──────────────────────────────────────────────────

def put_analysis(token: str, trait: str, response: Any) -> None:
    """
    Cache the GeneticsResponse for (token, trait).

    Called by every /analysis/* endpoint after the R engine returns a result,
    so that a second module touching the same trait skips the R call.
    """
    key: Tuple[str, str] = (token, trait)
    with _lock:
        _analysis_store[key] = response
        _analysis_store.move_to_end(key)
        while len(_analysis_store) > _ANALYSIS_MAX:
            evicted_key, _ = _analysis_store.popitem(last=False)
            logger.debug("dataset_cache: evicted analysis key %s", evicted_key)
    logger.info(
        "dataset_cache: stored analysis for token=%s trait=%s", token, trait
    )


def get_analysis(token: str, trait: str) -> Optional[Any]:
    """Return the cached GeneticsResponse for (token, trait), or None."""
    key: Tuple[str, str] = (token, trait)
    with _lock:
        result = _analysis_store.get(key)
    if result is not None:
        logger.info(
            "dataset_cache: analysis hit for token=%s trait=%s", token, trait
        )
    return result
