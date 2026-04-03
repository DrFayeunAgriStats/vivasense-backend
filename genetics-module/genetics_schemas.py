"""
VivaSense Genetics - Shared Pydantic schemas

GeneticsResult and GeneticsResponse are the authoritative types for a
single-trait analysis result.  They are defined here (not in app_genetics.py)
so that both app_genetics.py and multitrait_upload_schemas.py can import them
without creating a circular dependency.
"""

from pydantic import BaseModel, field_validator
from typing import Any, Dict, List, Optional, Union


class AnovaTable(BaseModel):
    """
    ANOVA table serialised as parallel column-arrays.

    source   — row labels (e.g. "rep", "genotype", "Residuals")
    df       — degrees of freedom
    ss       — sum of squares
    ms       — mean squares
    f_value  — F statistic (null for the Residuals row)
    p_value  — p-value     (null for the Residuals row)
    """
    source: List[str]
    df: List[int]
    ss: List[Optional[float]]
    ms: List[Optional[float]]
    f_value: List[Optional[float]]
    p_value: List[Optional[float]]


class MeanSeparation(BaseModel):
    """
    Tukey HSD (or LSD) mean separation result for genotypes.

    genotype — genotype labels, sorted by mean descending
    mean     — observed means (same order as genotype)
    se       — standard errors  (null when not computable)
    group    — Tukey grouping letters ("a", "ab", "b", …)
    test     — name of the test used
    alpha    — significance level
    """
    genotype: List[str]
    mean: List[float]
    se: List[Optional[float]]
    group: List[str]
    test: str = "Tukey HSD"
    alpha: float = 0.05


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
    anova_table: Optional[AnovaTable] = None
    mean_separation: Optional[MeanSeparation] = None


class GeneticsResponse(BaseModel):
    """Response payload for genetics analysis"""
    status: str
    mode: str
    # R serialises empty lists as [] not {} — accept both and normalise to dict
    data_validation: Union[Dict[str, Any], List] = {}
    variance_warnings: Union[Dict[str, Any], List] = {}
    result: Optional[GeneticsResult] = None
    interpretation: Optional[str] = None

    @field_validator("data_validation", "variance_warnings", mode="before")
    @classmethod
    def coerce_empty_list_to_dict(cls, v: Any) -> Any:
        if isinstance(v, list):
            return {}
        return v
