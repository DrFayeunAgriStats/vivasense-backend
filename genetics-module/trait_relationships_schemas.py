"""
VivaSense Genetics – Trait Relationships Schemas
Phase 2 / Phase 1 scope: phenotypic correlation only.

No imports from app_genetics or multitrait_upload_schemas — this file is
a leaf in the dependency graph.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class CorrelationRequest(BaseModel):
    base64_content: str = Field(..., description="Base64-encoded file content")
    file_type: str = Field(..., pattern="^(csv|xlsx|xls)$")
    genotype_column: str
    rep_column: str
    environment_column: Optional[str] = None
    trait_columns: List[str] = Field(..., min_length=2, max_length=20)
    mode: str = Field(default="single", pattern="^(single|multi)$")
    method: str = Field(default="pearson", pattern="^(pearson|spearman)$")


class CorrelationResponse(BaseModel):
    """
    Response from POST /genetics/correlation.

    r_matrix  — n×n symmetric matrix; r_matrix[i][j] is the Pearson/Spearman r
                between trait_names[i] and trait_names[j]. Diagonal = 1.0.
    p_matrix  — n×n matrix of two-sided p-values from cor.test().
                Diagonal = 0.0. Null when a pair had < 3 complete observations.
    n_observations — number of unique genotype means used (not raw rows).
    """
    trait_names: List[str]
    n_observations: int
    method: str
    r_matrix: List[List[Optional[float]]]
    p_matrix: List[List[Optional[float]]]
    interpretation: str
    warnings: List[str] = Field(default_factory=list)
    statistical_note: str = (
        "Correlations computed using genotype-level means; "
        "significance based on number of genotypes."
    )
