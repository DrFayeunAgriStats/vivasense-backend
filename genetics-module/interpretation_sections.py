"""
VivaSense Interpretation Section Schemas

Strict, validated interpretation objects for ANOVA, Trait Association, and Genetics modules.
Each section corresponds to a required field in the final report, populated deterministically
from Python validator logic — never freeform generation.

These objects are the SOURCE OF TRUTH for final report content.
Export builders render ONLY these validated sections.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List


# ============================================================================
# ANOVA INTERPRETATION SECTIONS
# ============================================================================

class AnovaInterpretationSections(BaseModel):
    """
    Strict ANOVA interpretation object.
    All fields are populated from Python validator logic, never freeform.
    """
    
    overview: str = Field(
        ...,
        description="Overview of the ANOVA design, traits analyzed, and sample size"
    )
    
    descriptive_interpretation: str = Field(
        ...,
        description="Description of trait means, variation, and precision (CV)"
    )
    
    genotype_effect: str = Field(
        ...,
        description="Interpretation of genotype significance and effect magnitude"
    )
    
    environment_effect: str = Field(
        ...,
        description="Interpretation of environment significance (if multi-env) or statement that single-env"
    )
    
    gxe_interaction: str = Field(
        ...,
        description="Interpretation of genotype×environment interaction and ranking stability"
    )
    
    mean_performance_ranking: str = Field(
        ...,
        description="Ranking interpretation with confidence caveats if needed"
    )
    
    breeding_interpretation: str = Field(
        ...,
        description="Breeding feasibility and selection strategy recommendations"
    )
    
    risk_limitations: str = Field(
        ...,
        description="Data limitations, preliminary evidence wording, and confidence statements"
    )
    
    recommendation: str = Field(
        ...,
        description="Clear actionable recommendation for next steps"
    )

    @validator("*", pre=False, always=False)
    def no_template_placeholders(cls, v):
        """Ensure no literal {trait} or {value} templates survive."""
        if isinstance(v, str) and "{" in v and "}" in v:
            raise ValueError(f"Literal template placeholder found: {v}")
        return v


# ============================================================================
# TRAIT ASSOCIATION INTERPRETATION SECTIONS
# ============================================================================

class TraitAssociationInterpretationSections(BaseModel):
    """
    Strict Trait Association interpretation object.
    All fields populated from validator logic, significance-checking enforced.
    """
    
    overview: str = Field(
        ...,
        description="Overview of traits analyzed, sample size, and analysis basis"
    )
    
    key_associations: str = Field(
        ...,
        description="Description of significant associations (p < 0.05 only). "
                    "If no negative associations exist, explicitly state so. "
                    "Never describe non-significant correlations as meaningful."
    )
    
    breeding_interpretation: str = Field(
        ...,
        description="Implications for indirect selection and trait management; "
                    "no trade-off language unless significant negative correlation exists"
    )
    
    risk_limitations: str = Field(
        ...,
        description="Data limitations (sample size, pairwise_n tracking, GxE effects), "
                    "preliminary evidence wording if n < 10"
    )
    
    recommendation: str = Field(
        ...,
        description="Clear actionable next steps for breeding or further analysis"
    )

    @validator("*", pre=False, always=False)
    def no_template_placeholders(cls, v):
        """Ensure no literal {trait} or {value} templates survive."""
        if isinstance(v, str) and "{" in v and "}" in v:
            raise ValueError(f"Literal template placeholder found: {v}")
        return v


# ============================================================================
# GENETICS INTERPRETATION SECTIONS
# ============================================================================

class GeneticsInterpretationSections(BaseModel):
    """
    Strict Genetics (heritability/genetic parameters) interpretation object.
    All fields populated deterministically from h², GAM, GCV, PCV classifications.
    """
    
    overview: str = Field(
        ...,
        description="Overview of trait and data basis for genetic parameter estimation"
    )
    
    heritability_interpretation: str = Field(
        ...,
        description="Interpretation of h² estimate (high/moderate/low) with numeric values"
    )
    
    genetic_advance_interpretation: str = Field(
        ...,
        description="Interpretation of GAM (genetic advance as % of mean) "
                    "and response to selection potential"
    )
    
    variance_relationship_interpretation: str = Field(
        ...,
        description="GCV vs PCV comparison with environmental effect commentary"
    )
    
    breeding_interpretation: str = Field(
        ...,
        description="Breeding strategy recommendation based on h² and GAM classifications"
    )
    
    risk_limitations: str = Field(
        ...,
        description="Data limitations, GxE caution if significant, confidence wording"
    )
    
    recommendation: str = Field(
        ...,
        description="Clear actionable recommendation for selection and breeding decisions"
    )

    @validator("*", pre=False, always=False)
    def no_template_placeholders(cls, v):
        """Ensure no literal {trait} or {value} templates survive."""
        if isinstance(v, str) and "{" in v and "}" in v:
            raise ValueError(f"Literal template placeholder found: {v}")
        return v
