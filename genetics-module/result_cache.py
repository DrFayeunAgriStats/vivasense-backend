"""
VivaSense Genetics – Server-side analysis result cache
======================================================

Stores the full UploadAnalysisResponse (including analysis_result for every
trait) in memory after /genetics/analyze-upload completes.

The export endpoint (POST /genetics/download-results) receives an export_token
in the request body and looks up the cached response to recover any
analysis_result objects the frontend did not include in its POST body.

Design notes
------------
- Thread-safe via a module-level Lock.
- LRU-style eviction: when the cache exceeds MAX_SIZE entries the oldest entry
  is removed.  At ~50 concurrent users this is effectively unlimited for a
  normal usage session.
- Intentionally in-process and in-memory: no Redis / DB dependency.
  If the Render instance restarts the cache is lost and the user must
  re-run the analysis (rare — analysis and export are normally within the same
  browser session).
"""

import logging
import threading
import uuid
from collections import OrderedDict
from typing import TYPE_CHECKING, Optional

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
        _store.move_to_end(token)          # mark as most-recently-used
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
