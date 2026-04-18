"""
VivaSense Genetics – Trait Relationships Schemas
Phase 2 / Phase 1 scope: phenotypic correlation only.

No imports from app_genetics or multitrait_upload_schemas — this file is
a leaf in the dependency graph.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class CorrelationStats(BaseModel):
    n_observations: int
    df: Optional[int] = None
    critical_r: Optional[float] = None
    r_matrix: List[List[Optional[float]]]
    p_matrix: List[List[Optional[float]]]
    p_adj_matrix: List[List[Optional[float]]]
    ci_lower_matrix: List[List[Optional[float]]]
    ci_upper_matrix: List[List[Optional[float]]]
    # True only for genotypic VC mode: p-values and CIs are Fisher-z approximations
    inference_approximate: bool = False
    inference_note: Optional[str] = None

class CorrelationRequest(BaseModel):
    base64_content: str = Field(..., description="Base64-encoded file content")
    file_type: str = Field(..., pattern="^(csv|xlsx|xls)$")
    genotype_column: str
    rep_column: str
    environment_column: Optional[str] = None
    trait_columns: List[str] = Field(..., min_length=2, max_length=20)
    mode: str = Field(default="single", pattern="^(single|multi)$")
    method: str = Field(default="pearson", pattern="^(pearson|spearman)$")
    user_objective: str = Field(default="Field understanding", pattern="^(Field understanding|Genotype comparison|Breeding decision)$")


class CorrelationResponse(BaseModel):
    """
    Response from POST /genetics/correlation.

    Three modes:
      phenotypic       — all observations (field-level co-variation)
      between_genotype — genotype means (between-genotype association; NOT a genetic parameter)
      genotypic        — variance-component based via bivariate REML; None if estimation failed
    """
    trait_names: List[str]
    method: str
    phenotypic: CorrelationStats
    between_genotype: CorrelationStats
    genotypic: Optional[CorrelationStats] = None
    interpretation: str
    warnings: List[str] = Field(default_factory=list)
    statistical_note: str = (
        "Three-mode analysis: phenotypic (all observations), "
        "between-genotype association (genotype means; not a true genetic parameter), "
        "and genotypic correlation (variance-component based via bivariate REML)."
    )
