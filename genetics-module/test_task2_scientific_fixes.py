"""
TASK 2 Scientific Fixes — Regression Tests
============================================
Validates all TASK 2 requirements:
  1. Breeding language leakage removal from Agronomy/General paths
  2. Unsafe "top-performing / best / superior" language removal
  3. "Recommended for adoption" / adoption language removal
  4. η² (eta-squared) calculation correctness (no Intercept in SS total)
  5. CV-aware precision phrasing (no illogical "precision could not be assessed")
  6. Scope-aware limitation detection (no "No major experimental limitations")
  7. Improved failed-trait interpretation messaging
  8. Domain-aware recommendation engine for Agronomy
  9. Domain guard catches newly added forbidden terms
"""

import sys
sys.path.insert(0, ".")

import pytest
from domain_guard import find_forbidden_breeding_terms
from genetics_interpretation import generate_genetics_interpretation
from analysis_anova_routes import generate_anova_interpretation
from genetics_export import _find_breeding_governance_hits


# ============================================================================
# Part 1 — Breeding language leakage from Agronomy/General
# ============================================================================

def test_agronomy_generate_anova_no_selection_language():
    """Agronomy ANOVA interpretation must not use selection/breeding terms."""
    interp = generate_anova_interpretation(
        trait="yield",
        summary={"grand_mean": 3500.0, "cv_percent": 12.0, "min": 2900.0, "max": 4200.0, "range": 1300.0},
        precision_level="moderate",
        cv_interpretation_flag="cv_available",
        genotype_significant=True,
        environment_significant=False,
        gxe_significant=False,
        ranking_caution=False,
        selection_feasible=True,
        mean_separation=None,
        n_genotypes=5,
        n_environments=1,
        n_reps=3,
        domain="agronomy",
    )
    forbidden_hits = find_forbidden_breeding_terms(interp)
    assert not forbidden_hits, f"Breeding terms found in agronomy interpretation: {forbidden_hits}"


def test_general_generate_anova_no_selection_language():
    """General-domain ANOVA interpretation must not use selection/breeding terms."""
    interp = generate_anova_interpretation(
        trait="biomass",
        summary={"grand_mean": 150.0, "cv_percent": 18.0, "min": 100.0, "max": 200.0, "range": 100.0},
        precision_level="moderate",
        cv_interpretation_flag="cv_available",
        genotype_significant=True,
        environment_significant=True,
        gxe_significant=True,
        ranking_caution=True,
        selection_feasible=True,
        mean_separation=None,
        n_genotypes=4,
        n_environments=2,
        n_reps=3,
        domain="general",
        environment_mode="multi",
    )
    forbidden_hits = find_forbidden_breeding_terms(interp)
    assert not forbidden_hits, f"Breeding terms found in general interpretation: {forbidden_hits}"


def test_agronomy_no_advance_promising_genotypes():
    """'Consider advancing promising genotypes' must not appear in agronomy recommendations."""
    interp = generate_anova_interpretation(
        trait="grain_weight",
        summary={"grand_mean": 45.0, "cv_percent": 9.5},
        precision_level="good",
        cv_interpretation_flag="cv_available",
        genotype_significant=True,
        environment_significant=False,
        gxe_significant=False,
        ranking_caution=False,
        selection_feasible=True,
        mean_separation=None,
        n_genotypes=6,
        n_environments=1,
        n_reps=4,
        domain="agronomy",
    )
    assert "advance promising" not in interp.lower()
    assert "advancing promising" not in interp.lower()
    assert "genotype evaluation" not in interp.lower()


# ============================================================================
# Part 2 — Unsafe top-performing / best / superior language
# ============================================================================

def test_domain_guard_catches_recommended_for_adoption():
    """Domain guard must flag 'recommended for adoption' as forbidden."""
    text = "Treatment N150 is recommended for adoption under similar growing conditions."
    hits = find_forbidden_breeding_terms(text)
    assert any("recommended for adoption" in h for h in hits), (
        "Expected 'recommended for adoption' to be flagged as forbidden"
    )


def test_domain_guard_catches_top_performing_treatment():
    """Domain guard must flag 'top-performing treatment' as forbidden."""
    text = "The top-performing treatment in group 'a' was N150."
    hits = find_forbidden_breeding_terms(text)
    assert any("top-performing treatment" in h for h in hits), (
        "Expected 'top-performing treatment' to be flagged as forbidden"
    )


def test_domain_guard_catches_superior_treatment():
    """Domain guard must flag 'superior treatment' as forbidden."""
    text = "N200 was identified as the superior treatment."
    hits = find_forbidden_breeding_terms(text)
    assert any("superior treatment" in h for h in hits), (
        "Expected 'superior treatment' to be flagged as forbidden"
    )


def test_domain_guard_catches_best_treatment():
    """Domain guard must flag 'best treatment' as forbidden."""
    text = "The best treatment was T3 with the highest mean yield."
    hits = find_forbidden_breeding_terms(text)
    assert any("best treatment" in h for h in hits), (
        "Expected 'best treatment' to be flagged as forbidden"
    )


# ============================================================================
# Part 3 — No "recommended for adoption" in practical implication
# ============================================================================

def test_agronomy_genetics_interpretation_no_recommended_for_adoption():
    """Agronomy genetics interpretation must not contain adoption recommendation."""
    interpretation, implication = generate_genetics_interpretation(
        trait_name="yield",
        h2=0.92,
        gam=18.5,
        gcv=12.0,
        pcv=13.2,
        gxe_significant=False,
        environment_significant=False,
        domain="agronomy",
    )
    combined = f"{interpretation} {implication}".lower()
    assert "recommended for adoption" not in combined
    assert "should be adopted" not in combined
    assert "superior treatment" not in combined


# ============================================================================
# Part 4 — η² (eta-squared) calculation correctness
# ============================================================================

def test_eta_squared_excludes_intercept_from_ss_total():
    """η² must not include the Intercept row SS in the denominator."""
    from genetics_export import _eta_squared_for_source
    from genetics_schemas import AnovaTable

    # Simulate ANOVA table where Intercept has huge SS that would deflate η²
    # SS values: Intercept=10_000_000, rep=4_041, genotype=3_506_426, Residuals=6_426
    at = AnovaTable(
        source=["(Intercept)", "rep", "genotype", "Residuals"],
        df=[1, 2, 4, 8],
        ss=[10_000_000.0, 4_041.0, 3_506_426.0, 6_426.0],
        ms=[10_000_000.0, 2_020.5, 876_606.5, 803.25],
        f_value=[None, 2.5, 1091.4, None],
        p_value=[None, 0.14, 0.0001, None],
    )

    eta = _eta_squared_for_source(at, "genotype")
    # SS_total (excluding Intercept) = 4041 + 3506426 + 6426 = 3516893
    # η² = 3506426 / 3516893 ≈ 0.997
    assert eta is not None, "η² should not be None"
    assert eta > 0.99, (
        f"η² should be near 0.997 when treatment SS dominates, got {eta:.4f}. "
        "The Intercept row is likely inflating SS_total."
    )


def test_eta_squared_correct_with_no_intercept():
    """η² is computed correctly when Intercept row is absent."""
    from genetics_export import _eta_squared_for_source
    from genetics_schemas import AnovaTable

    at = AnovaTable(
        source=["rep", "genotype", "Residuals"],
        df=[2, 4, 8],
        ss=[4_041.0, 3_506_426.0, 6_426.0],
        ms=[2_020.5, 876_606.5, 803.25],
        f_value=[2.5, 1091.4, None],
        p_value=[0.14, 0.0001, None],
    )

    eta = _eta_squared_for_source(at, "genotype")
    # SS_total = 4041 + 3506426 + 6426 = 3516893
    # η² = 3506426 / 3516893 ≈ 0.997
    assert eta is not None
    assert abs(eta - 3_506_426 / 3_516_893) < 1e-6, f"Got η² = {eta}"


# ============================================================================
# Part 5 — CV-aware precision phrasing
# ============================================================================

def test_no_precision_could_not_be_assessed_when_cv_unavailable():
    """'Precision could not be fully assessed' must not appear in generated text."""
    interp = generate_anova_interpretation(
        trait="plant_height",
        summary={"grand_mean": 80.0},
        precision_level=None,
        cv_interpretation_flag="cv_unavailable",
        genotype_significant=True,
        environment_significant=False,
        gxe_significant=False,
        ranking_caution=False,
        selection_feasible=True,
        mean_separation=None,
        n_genotypes=5,
        n_environments=1,
        n_reps=3,
        domain="agronomy",
    )
    assert "precision could not be fully assessed" not in interp.lower()


def test_cv_available_produces_precision_statement():
    """When CV is available, a precision statement should appear."""
    interp = generate_anova_interpretation(
        trait="leaf_area",
        summary={"grand_mean": 200.0, "cv_percent": 8.5, "min": 160.0, "max": 240.0, "range": 80.0},
        precision_level="good",
        cv_interpretation_flag="cv_available",
        genotype_significant=True,
        environment_significant=False,
        gxe_significant=False,
        ranking_caution=False,
        selection_feasible=True,
        mean_separation=None,
        n_genotypes=4,
        n_environments=1,
        n_reps=3,
        domain="agronomy",
    )
    assert "8.5%" in interp or "precision" in interp.lower()


# ============================================================================
# Part 6 — Limitation detection (always acknowledge scope)
# ============================================================================

def test_no_major_limitations_phrase_absent_in_agronomy():
    """'No major experimental limitations' must not appear in agronomy output."""
    interp = generate_anova_interpretation(
        trait="tuber_weight",
        summary={"grand_mean": 500.0, "cv_percent": 11.0},
        precision_level="moderate",
        cv_interpretation_flag="cv_available",
        genotype_significant=False,
        environment_significant=False,
        gxe_significant=False,
        ranking_caution=False,
        selection_feasible=False,
        mean_separation=None,
        n_genotypes=3,
        n_environments=1,
        n_reps=3,
        domain="agronomy",
    )
    assert "no major experimental limitations" not in interp.lower()


def test_scope_statement_present_in_agronomy():
    """Scope statement about interpreting within experiment must appear."""
    interp = generate_anova_interpretation(
        trait="root_length",
        summary={"grand_mean": 30.0, "cv_percent": 14.0},
        precision_level="moderate",
        cv_interpretation_flag="cv_available",
        genotype_significant=True,
        environment_significant=False,
        gxe_significant=False,
        ranking_caution=False,
        selection_feasible=True,
        mean_separation=None,
        n_genotypes=4,
        n_environments=1,
        n_reps=4,
        domain="agronomy",
    )
    assert "scope of this experiment" in interp.lower() or "within the scope" in interp.lower()


# ============================================================================
# Part 7 — Domain guard catches newly added forbidden terms
# ============================================================================

def test_domain_guard_catches_advance_promising_genotypes():
    text = "Consider advancing promising genotypes to further evaluation."
    hits = find_forbidden_breeding_terms(text)
    assert any("advancing promising genotypes" in h or "advance promising genotypes" in h for h in hits)


def test_domain_guard_catches_genetic_improvement():
    text = "This approach supports genetic improvement of the trait."
    hits = find_forbidden_breeding_terms(text)
    assert any("genetic improvement" in h for h in hits)


def test_domain_guard_catches_should_be_adopted():
    text = "This treatment should be adopted by farmers in the region."
    hits = find_forbidden_breeding_terms(text)
    assert any("should be adopted" in h for h in hits)


def test_domain_guard_does_not_flag_plant_breeding_text_incorrectly():
    """Plant-breeding-specific text should still be flagged (guard is domain-agnostic scanner)."""
    text = "genotype evaluation and selection strategy for breeding."
    hits = find_forbidden_breeding_terms(text)
    # breeding, selection strategy, genotype evaluation are all forbidden terms
    assert len(hits) >= 2


# ============================================================================
# Part 8 — Further evaluation replaces adoption language in agronomy
# ============================================================================

def test_agronomy_recommendation_uses_further_evaluation_not_adoption():
    """Agronomy recommendations should say 'further evaluation' not 'adoption'."""
    interp = generate_anova_interpretation(
        trait="grain_yield",
        summary={"grand_mean": 4000.0, "cv_percent": 10.0},
        precision_level="good",
        cv_interpretation_flag="cv_available",
        genotype_significant=True,
        environment_significant=False,
        gxe_significant=False,
        ranking_caution=False,
        selection_feasible=True,
        mean_separation=None,
        n_genotypes=6,
        n_environments=1,
        n_reps=4,
        domain="agronomy",
    )
    interp_lower = interp.lower()
    assert "further evaluation" in interp_lower or "additional environments" in interp_lower
    assert "recommended for adoption" not in interp_lower
    assert "advance promising genotypes" not in interp_lower


def test_breeding_single_environment_omits_env_and_gxe_non_significant_claims():
    interpretation, implication = generate_genetics_interpretation(
        trait_name="grain_yield",
        h2=0.72,
        gam=11.5,
        gcv=14.0,
        pcv=14.2,
        anova_f_env=0.0,
        anova_p_env=None,
        anova_f_gxe=0.0,
        anova_p_gxe=None,
        analysis_type="single_environment",
        domain="plant_breeding",
    )
    text = f"{interpretation} {implication}".lower()
    assert "environmental effects were non-significant" not in text
    assert "gxe interaction was non-significant" not in text
    assert "consistent across environments" not in text
    assert "stability analysis" not in text


def test_breeding_replaces_gene_action_and_overconfident_phrasing():
    interpretation, implication = generate_genetics_interpretation(
        trait_name="plant_height",
        h2=0.81,
        gam=12.1,
        gcv=9.0,
        pcv=9.1,
        analysis_type="single_environment",
        domain="plant_breeding",
    )
    text = f"{interpretation} {implication}".lower()
    assert "additive gene effects" not in text
    assert "non-additive effects" not in text
    assert "gene action" not in text
    assert "clean genetic signal" not in text
    assert "closely track underlying genotypic value" not in text
    assert "may be effective under the conditions evaluated in this study" in text


def test_breeding_cv_precision_narrative_uses_governed_language():
    interpretation, _ = generate_genetics_interpretation(
        trait_name="seed_weight",
        h2=0.65,
        gam=7.0,
        gcv=6.2,
        pcv=7.1,
        cv_percent=0.8,
        analysis_type="single_environment",
        domain="plant_breeding",
    )
    lowered = interpretation.lower()
    assert "residual variability was extremely low relative to the trait mean" in lowered
    assert "verify raw data consistency" in lowered


def test_breeding_export_governance_scan_flags_single_env_forbidden_phrases():
    report_text = (
        "Environmental effects were non-significant. "
        "GxE interaction was non-significant. "
        "Top-performing genotype in group a."
    )
    hits = _find_breeding_governance_hits(report_text, analysis_type="single_environment")
    hits_text = " | ".join(hits).lower()
    assert "environmental effects non-significant" in hits_text
    assert "gxe non-significant" in hits_text
    assert "top-performing genotype" in hits_text
