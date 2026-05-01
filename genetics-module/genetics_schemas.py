"""
VivaSense Genetics - Shared Pydantic schemas

GeneticsResult and GeneticsResponse are the authoritative types for a
single-trait analysis result.  They are defined here (not in app_genetics.py)
so that both app_genetics.py and multitrait_upload_schemas.py can import them
without creating a circular dependency.
"""

from pydantic import BaseModel, field_validator, model_validator
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

    @field_validator("df", mode="before")
    @classmethod
    def coerce_df_to_int(cls, v: Any) -> List[int]:
        """Coerce df values to int — JSON round-tripping via JS can produce floats."""
        if isinstance(v, list):
            return [int(x) if x is not None else 0 for x in v]
        return v


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
    n_genotypes: Optional[int] = None   # None for generic split_plot_rcbd (no genotype column)
    n_reps: int
    n_environments: Optional[int] = None
    grand_mean: float
    variance_components: Dict[str, Any]
    heritability: Dict[str, Any]
    genetic_parameters: Dict[str, Any]
    anova_table: Optional[AnovaTable] = None
    mean_separation: Optional[MeanSeparation] = None
    # Optional fields returned by the R engine — preserved here so the export
    # can render them without schema changes each time the engine adds new fields.
    descriptive_stats: Optional[Dict[str, Any]] = None
    assumption_tests: Optional[Dict[str, Any]] = None
    breeding_implication: Optional[str] = None

    @field_validator("variance_components", "heritability", "genetic_parameters", mode="before")
    @classmethod
    def coerce_dict_field(cls, v: Any) -> Dict[str, Any]:
        """
        R serialises empty/failed dict fields as null or [] in some edge cases
        (singular models, insufficient degrees of freedom, etc.).
        Coerce both to {} so Pydantic validation never fails on these fields.
        """
        if v is None or isinstance(v, list):
            return {}
        return v


class GeneticsResponse(BaseModel):
    """Response payload for genetics analysis"""
    status: str
    mode: str
    # R serialises empty lists as [] not {} — accept both and normalise to dict
    data_validation: Union[Dict[str, Any], List] = {}
    variance_warnings: Union[Dict[str, Any], List] = {}
    result: Optional[GeneticsResult] = None
    interpretation: Optional[str] = None
    anova_type_warning: Optional[str] = None

    @field_validator("data_validation", "variance_warnings", mode="before")
    @classmethod
    def coerce_empty_list_to_dict(cls, v: Any) -> Any:
        if isinstance(v, list):
            return {}
        return v

    @field_validator("interpretation", mode="before")
    @classmethod
    def coerce_interpretation(cls, v: Any) -> Any:
        # R serialises the interpretation field as a named list ({}) when it
        # cannot generate text. Coerce any non-string to None so the field
        # stays Optional[str] without a ValidationError.
        if v is None or isinstance(v, (dict, list)):
            return None
        return v
