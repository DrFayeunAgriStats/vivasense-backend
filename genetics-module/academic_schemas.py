"""
VivaSense Academic Mentor — Pydantic Schemas
============================================

Request and response models for POST /academic/interpret.

Three-layer architecture:
  Layer A — ValidationResult       (academic_validator.py)
  Layer B — AcademicInterpretation (academic_interpretation.py)
  Layer C — GuidedWritingBlock     (guided_writing.py)

All three are bundled in AcademicInterpretationResponse.

Import note: this file is a leaf in the dependency graph.
It imports nothing from the genetics module — no circular deps.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ============================================================================
# LAYER A — VALIDATOR
# ============================================================================

class ValidationViolation(BaseModel):
    """One rule violation found in AI-generated text."""
    rule_id: str               # e.g. "FORBIDDEN_WORD", "MISSING_SCOPE"
    severity: Literal["block", "warn"]
    excerpt: str               # the offending passage (≤ 120 chars)
    message: str               # human-readable fix instruction


class ValidationResult(BaseModel):
    """Output from academic_validator.validate()."""
    passed: bool               # True when block_count == 0
    blocked: bool              # True when any block-severity violation exists
    violations: List[ValidationViolation] = Field(default_factory=list)
    warning_count: int = 0
    block_count: int = 0


# ============================================================================
# LAYER C — GUIDED WRITING
# ============================================================================

class SentenceStarter(BaseModel):
    """One partially-completed sentence scaffold for student writing practice."""
    purpose: str               # e.g. "Significance statement"
    template: str              # sentence with ___ blanks (never pre-filled)
    values_to_fill: List[str]  # describes what each blank should contain
    hint: Optional[str] = None # where to find the value ("ANOVA table, Genotype row")


class GuidedWritingBlock(BaseModel):
    """
    Layer C output — returned alongside every interpretation.

    The frontend should render this in a collapsible "Writing Support" panel.
    All blanks (___) must be filled by the student from their own analysis;
    pre-filling would defeat the academic purpose.
    """
    module_type: str
    trait: Optional[str] = None
    sentence_starters: List[SentenceStarter] = Field(default_factory=list)
    examiner_checkpoint: List[str] = Field(
        default_factory=list,
        description="5 ☐-items the student checks before submission",
    )
    scope_statement: str = (
        "These results apply to this experiment and should be interpreted "
        "within this context. Single-experiment results cannot support general "
        "management or breeding recommendations."
    )
    caution_note: Optional[str] = None   # low-rep, normality violation, etc.
    supervisor_prompt: str = (
        "Discuss these findings with your supervisor before finalising your "
        "write-up. — Dr. Fayeun, VivaSense Academic Mentor"
    )


# ============================================================================
# LAYER B — INTERPRETATION RESPONSE
# ============================================================================

class AcademicInterpretationResponse(BaseModel):
    """
    Full output from POST /academic/interpret.

    The frontend typically displays:
      - overall_finding + statistical_evidence in a summary card
      - module_sections in expandable sub-sections
      - guided_writing in a collapsible "Writing Support" drawer
      - examiner_checkpoint as ☐ checkboxes
      - closing + referral as a footer note
    """
    module_type: str
    trait: Optional[str] = None

    # ── Core interpretation (AI-generated or fallback) ──────────────────────
    overall_finding: str
    statistical_evidence: str
    module_sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Module-specific sections keyed by section name",
    )

    # ── Fixed-structure sections ─────────────────────────────────────────────
    scope_statement: str = (
        "These results apply to this experiment and should be interpreted "
        "within this context. Single-experiment results cannot support general "
        "management or breeding recommendations."
    )
    examiner_checkpoint: List[str] = Field(default_factory=list)
    closing: str = (
        "Discuss these findings with your supervisor before finalising your "
        "write-up. — Dr. Fayeun, VivaSense Academic Mentor"
    )
    research_writing_referral: str = (
        "For structured support with writing your results section, visit "
        "Field-to-Insight Academy: www.fieldtoinsightacademy.com.ng"
    )

    # ── Layer C ──────────────────────────────────────────────────────────────
    guided_writing: Optional[GuidedWritingBlock] = None

    # ── Layer A ──────────────────────────────────────────────────────────────
    validator_result: Optional[ValidationResult] = None

    # ── Provenance ───────────────────────────────────────────────────────────
    fallback_used: bool = False   # True when AI failed validation + repair
    ai_generated: bool = False
    raw_ai_text: Optional[str] = None   # full AI output before parsing; debug


# ============================================================================
# REQUEST MODEL
# ============================================================================

class AcademicInterpretRequest(BaseModel):
    """
    POST /academic/interpret request body.

    analysis_result can be:
      • A full module response dict (e.g. AnovaModuleResponse)
        → specify trait to select the per-trait result
      • A single per-trait result dict (e.g. AnovaTraitResult)
        → trait field must still be set for context
    """
    module_type: Literal["anova", "genetic_parameters", "correlation", "heatmap"]
    trait: Optional[str] = Field(
        default=None,
        description="Trait name for single-trait modules (anova, genetic_parameters)",
    )
    analysis_result: Dict[str, Any] = Field(
        ...,
        description="Dict from the matching /analysis/* endpoint",
    )
    crop_context: Optional[str] = Field(
        default=None,
        description="e.g. 'cowpea', 'cassava', 'maize + cowpea intercrop'",
    )
    include_writing_support: bool = True
