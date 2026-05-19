"""
VivaSense — Shared Report Terminology
======================================
Single source of truth for ALL section titles, labels, and
governance-controlled phrases used in Word exports.

Import from this module in ALL export builders.
Never inline section-title strings in export code.

Governance-protected phrases at the bottom MUST remain verbatim.
"""

from typing import Optional


# ============================================================================
# SECTION TITLES
# ============================================================================

class TERMS:
    # Level-1 sections
    executive_summary        = "Executive Summary"
    trait_summary            = "Trait Summary"
    breeding_strategy        = "Breeding Strategy Summary"
    trait_correlations       = "Trait Correlations"
    writing_support_guide    = "Writing Support Guide"

    # Level-2 analytical sections
    descriptive_stats        = "Descriptive Statistics"
    data_quality             = "Data Quality Overview"
    anova                    = "Analysis of Variance (ANOVA)"
    mean_separation          = "Mean Separation"
    treatment_variance       = "Treatment Variance"
    genetic_parameters       = "Genetic Parameters"
    variance_components      = "Variance Components"
    heritability             = "Heritability"
    coefficients_variation   = "Coefficients of Variation"
    genetic_advance          = "Genetic Advance"
    interpretation           = "Interpretation"
    statistical_interp       = "Statistical Interpretation"
    academic_interpretation  = "Academic Interpretation"
    breeding_implication     = "Breeding Implication"
    assumption_tests         = "Assumption Tests"
    writing_support          = "Academic Writing Support"
    presubmission_checklist  = "Pre-submission Checklist"
    sentence_starters        = "Sentence Starters"
    reliability_notes        = "Reliability Notes"
    data_warnings            = "Data Warnings"

    # Level-3 academic interpretation sub-sections
    overall_finding          = "Overall Finding"
    statistical_evidence     = "Statistical Evidence"
    treatment_interpretation = "Treatment Interpretation"
    assumption_check         = "Assumption Check"
    examiner_checkpoint      = "Examiner Checkpoint"

    # Miscellaneous
    correlation_interp       = "Correlation Interpretation"
    formulas                 = "Formulas Used"
    interpretation_note      = "Interpretation Note"

    # ── Dynamic heading builders ──────────────────────────────────────────────

    @staticmethod
    def mean_separation_header(test: str, alpha: float) -> str:
        return f"Mean Separation — {test} (α = {alpha})"

    @staticmethod
    def trait_header(trait: str) -> str:
        return f"Trait: {trait}"

    # ── Agronomy-aware overrides ──────────────────────────────────────────────

    @staticmethod
    def genetic_parameters_or_variance(is_agronomy: bool) -> str:
        return "Treatment Variance" if is_agronomy else "Genetic Parameters"

    @staticmethod
    def interpretation_and_recommendations(is_agronomy: bool) -> str:
        return (
            "Interpretation & Recommendations"
            if is_agronomy
            else "Interpretation & Breeding Recommendations"
        )


# ============================================================================
# GOVERNANCE-PROTECTED PHRASES — DO NOT MODIFY
# These phrases are scientifically important restraint controls.
# Reference them from here only. Never paraphrase, shorten, or rewrite.
# ============================================================================

class GOVERNANCE:
    MAY_WARRANT = "may warrant further evaluation"
    WITHIN_SCOPE = "within the scope of this experiment"
    COMPARATIVELY_HIGHER = "comparatively higher observed means"
    COMPARATIVELY_STRONG = (
        "comparatively strong performance under the conditions of this experiment"
    )
    RESULTS_SHOULD = (
        "results should be interpreted within the scope of this experiment, "
        "including the evaluated environment and replication structure"
    )

    # Standard scope note appended to every per-trait section
    SCOPE_NOTE = (
        "Note: These results apply to this experiment and should be interpreted "
        "within this context. Single-experiment results cannot support general "
        "management recommendations."
    )

    # Assumption absence notices (verbatim)
    SHAPIRO_NOT_REPORTED  = "Not reported in your input."
    LEVENE_NOT_REPORTED   = "Not reported in your input."

    SINGLE_ENVIRONMENT_DISCLAIMER = (
        "Cross-environment inference (e.g., GxE interaction, stability analysis) requires at least two environments."
    )


# ============================================================================
# DOMAIN LABELS
# ============================================================================

_DOMAIN_LABELS: dict = {
    "plant_breeding": "Genetics",
    "agronomy":       "Agronomy",
    "general":        "General",
}


def get_report_title(domain: Optional[str], module: str = "") -> str:
    """
    Domain-aware report title.

    BUG FIX: Word export previously hardcoded "Agronomy" regardless of
    active domain. This function derives the title from domain + module.
    """
    domain = (domain or "plant_breeding").strip().lower()
    label  = _DOMAIN_LABELS.get(domain, "General")

    module_map = {
        "anova":              f"VivaSense {label} ANOVA Report",
        "genetic_parameters": "VivaSense Genetic Parameters Report",
        "correlation":        "VivaSense Trait Correlation Report",
        "heatmap":            "VivaSense Correlation Heatmap Report",
        "regression":         "VivaSense Regression Analysis Report",
        "descriptive_stats":  "VivaSense Descriptive Statistics Report",
    }
    return module_map.get(module, f"VivaSense {label} Analysis Report")


# ============================================================================
# ANOVA SOURCE LABELS (shared between combined and module exports)
# ============================================================================

ANOVA_SOURCE_LABELS: dict = {
    "rep":                   "Replication",
    "genotype":              "Genotype",
    "environment":           "Environment",
    "environment:rep":       "Rep(Environment)",
    "genotype:environment":  "G×E Interaction",
    "Residuals":             "Error",
    "(intercept)":           "(Intercept) — not a treatment effect",
    "intercept":             "(Intercept) — not a treatment effect",
}

ANOVA_SOURCE_LABELS_AGRONOMY: dict = {
    **ANOVA_SOURCE_LABELS,
    "genotype":              "Treatment",
    "genotype:environment":  "T×E Interaction",
}


def get_anova_source_label(src: str, is_agronomy: bool) -> str:
    """Resolve a raw ANOVA source name to a human-readable label."""
    src_lower = src.strip().lower()
    lookup = ANOVA_SOURCE_LABELS_AGRONOMY if is_agronomy else ANOVA_SOURCE_LABELS
    # Try exact lower-case match first, then original-case
    return lookup.get(src_lower) or lookup.get(src, src)
