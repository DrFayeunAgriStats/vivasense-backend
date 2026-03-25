"""
VivaSense Genetics - Shared Pydantic schemas

GeneticsResult and GeneticsResponse are the authoritative types for a
single-trait analysis result.  They are defined here (not in app_genetics.py)
so that both app_genetics.py and multitrait_upload_schemas.py can import them
without creating a circular dependency.
"""

from pydantic import BaseModel
from typing import Any, Dict, Optional


class GeneticsResult(BaseModel):
    """Core analysis result"""
    environment_mode: str
    n_genotypes: int
    n_reps: int
    n_environments: Optional[int] = None
    grand_mean: float
    variance_components: Dict[str, Any]
    heritability: Dict[str, Any]
    genetic_parameters: Dict[str, Any]


class GeneticsResponse(BaseModel):
    """Response payload for genetics analysis"""
    status: str
    mode: str
    data_validation: Dict[str, Any] = {}
    variance_warnings: Dict[str, Any] = {}
    result: Optional[GeneticsResult] = None
    interpretation: Optional[str] = None
