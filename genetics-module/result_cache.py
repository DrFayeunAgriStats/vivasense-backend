"""
VivaSense Genetics – Server-side analysis result cache
======================================================

Stores the full UploadAnalysisResponse (including analysis_result for every
trait) in memory after /genetics/analyze-upload completes.

The export endpoint uses get_matching_traits() to recover the analysis_result
objects that the frontend does not re-send in its POST body.  It finds the
most-recently-cached response whose trait_results keys are a superset of the
traits in the export request — which is always the user's own analysis when
there is one worker (WEB_CONCURRENCY=1 on Render free tier).

Design notes
------------
- Thread-safe via a module-level Lock.
- LRU-style eviction: when the cache exceeds MAX_SIZE entries the oldest entry
  is removed.
- Intentionally in-process and in-memory: no Redis / DB dependency.
  If the Render instance restarts the cache is lost and the user must
  re-run the analysis (analysis and export are normally within the same
  browser session).
"""

import logging
import threading
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from multitrait_upload_schemas import UploadAnalysisResponse

logger = logging.getLogger(__name__)

_CACHE_MAX_SIZE = 200

_lock: threading.Lock = threading.Lock()
_store: OrderedDict = OrderedDict()   # token → UploadAnalysisResponse


def create_token() -> str:
    """Generate a new unique export token."""
    return str(uuid.uuid4())


def put(token: str, response: "UploadAnalysisResponse") -> None:
    """Cache an analysis response under *token*, evicting the oldest if needed."""
    with _lock:
        _store[token] = response
        _store.move_to_end(token)
        while len(_store) > _CACHE_MAX_SIZE:
            evicted_key, _ = _store.popitem(last=False)
            logger.debug("result_cache: evicted token %s", evicted_key)
    logger.info("result_cache: stored token %s (cache size=%d)", token, len(_store))


def get(token: str) -> "Optional[UploadAnalysisResponse]":
    """Return the cached response for *token*, or None if missing / expired."""
    with _lock:
        result = _store.get(token)
    if result is None:
        logger.warning("result_cache: cache miss for token %s", token)
    else:
        logger.info("result_cache: cache hit for token %s", token)
    return result


def get_matching_traits(
    trait_names: List[str],
) -> "Optional[UploadAnalysisResponse]":
    """
    Return the most-recently-cached response whose trait_results keys are a
    superset of *trait_names*, or None if no match is found.

    This is the primary lookup used by the export endpoint when the frontend
    does not echo back an export_token.  The most-recent match is virtually
    always the user's own analysis given single-worker deployment.
    """
    if not trait_names:
        return None

    needed = set(trait_names)
    with _lock:
        # Iterate newest → oldest
        entries = list(_store.values())

    for response in reversed(entries):
        cached_traits = set((response.trait_results or {}).keys())
        if needed.issubset(cached_traits):
            logger.info(
                "result_cache: matched by traits %s (cache size=%d)",
                trait_names,
                len(entries),
            )
            return response

    logger.warning(
        "result_cache: no cached entry contains traits %s", trait_names
    )
    return None
