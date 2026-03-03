"""
NumPy / pandas → JSON-safe serialization helpers.
Addresses the known serialization gap documented in CLAUDE.md.
Every genetics endpoint must call numpy_to_python() before returning.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from dataclasses import asdict
from typing import Any


def numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serializable Python natives."""
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return numpy_to_python(obj.tolist())
    if isinstance(obj, pd.Series):
        return numpy_to_python(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return numpy_to_python(obj.to_dict(orient="records"))
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj


def dataclass_to_dict(obj: Any) -> Any:
    """Convert a genetics result dataclass to a fully JSON-safe nested dict."""
    try:
        return numpy_to_python(asdict(obj))
    except TypeError:
        return numpy_to_python(obj)


def genetics_json_default(obj: Any) -> Any:
    """Drop-in default= argument for json.dumps() on genetics results."""
    return numpy_to_python(obj)
