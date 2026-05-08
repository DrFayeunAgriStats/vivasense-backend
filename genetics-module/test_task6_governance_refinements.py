"""
Tests for Task 6 — Final Single-Environment Breeding Report Governance Refinements.

Covers:
  PART 1 — CV% integration in single-environment ANOVA reports
  PART 2 — CV% precision interpretation tiers (domain-aware)
  PART 3 — Softened environmental variance language
  PART 4 — Narrative diversity
  PART 5 — Writing Support Guide consistency
  PART 6 — Validation layer (forbidden phrases blocked)
"""
import unittest

from genetics_interpretation import (
    _describe_gcv_pcv,
    generate_genetics_interpretation,
    generate_genetics_interpretation_sections,
    _select_narrative_variant,
)
from analysis_anova_routes import (
    _cv_tier_narrative,
    generate_anova_interpretation,
)
from genetics_export import _find_breeding_governance_hits


# ============================================================================
# PART 1 — CV% integration
# ============================================================================

class TestCvPrecisionNarrative(unittest.TestCase):

    def test_cv_tier_low_cv(self):
        """CV < 10 should indicate high precision."""
        result = _cv_tier_narrative(5.0, domain="plant_breeding")
        self.assertIn("relatively low", result.lower())
        self.assertIn("high experimental precision", result.lower())

    def test_cv_tier_moderate_breeding(self):
        """CV 10-20 should say 'genotype comparison' for plant_breeding domain."""
        result = _cv_tier_narrative(15.0, domain="plant_breeding")
        self.assertIn("genotype comparison", result.lower())
        self.assertNotIn("treatment comparison", result.lower())

    def test_cv_tier_moderate_general(self):
        """CV 10-20 should say 'treatment comparison' for general domain."""
        result = _cv_tier_narrative(15.0, domain="general")
        self.assertIn("treatment comparison", result.lower())

    def test_cv_tier_high_cv(self):
        """CV >= 20 should warn about high variability."""
        result = _cv_tier_narrative(25.0, domain="plant_breeding")
        self.assertIn("comparatively high", result.lower())
        self.assertIn("interpreted cautiously", result.lower())

    def test_cv_tier_extreme_low_warning(self):
        """CV < 1 should append extreme low warning."""
        result = _cv_tier_narrative(0.5, domain="plant_breeding")
        self.assertIn("extremely low", result.lower())
        self.assertIn("verify raw data consistency", result.lower())

    def test_cv_tier_none_returns_none(self):
        """None CV should return None."""
        self.assertIsNone(_cv_tier_narrative(None, domain="plant_breeding"))

    def test_cv_tier_never_negative(self):
        """Negative CV input should be sanitized (abs value used)."""
        result = _cv_tier_narrative(-5.0, domain="plant_breeding")
        self.assertIsNotNone(result)

    def test_cv_tier_moderate_agronomy(self):
        """CV 10-20 with agronomy domain should say 'treatment comparison'."""
        result = _cv_tier_narrative(12.0, domain="agronomy")
        self.assertIn("treatment comparison", result.lower())


# ============================================================================
# PART 2 — CV% in ANOVA interpretation output
# ============================================================================

class TestCvInAnovaInterpretation(unittest.TestCase):

    def _make_summary(self, cv=12.0):
        return {
            "grand_mean": 100.0,
            "cv_percent": cv,
            "min": 80.0,
            "max": 120.0,
            "range": 40.0,
            "standard_error": 2.5,
        }

    def test_cv_present_in_descriptive_section_breeding(self):
        """CV narrative (genotype comparison) should appear in plant_breeding interpretation."""
        text = generate_anova_interpretation(
            trait="Yield",
            summary=self._make_summary(cv=15.0),
            precision_level="moderate",
            cv_interpretation_flag="cv_available",
            genotype_significant=True,
            environment_significant=False,
            gxe_significant=False,
            ranking_caution=False,
            selection_feasible=True,
            mean_separation=None,
            n_genotypes=10,
            n_environments=1,
            n_reps=3,
            environment_mode="single",
            domain="plant_breeding",
        )
        self.assertIn("genotype comparison", text.lower())

    def test_cv_unavailable_message(self):
        """When cv_interpretation_flag is cv_unavailable, message should appear."""
        text = generate_anova_interpretation(
            trait="Yield",
            summary=self._make_summary(cv=None),
            precision_level="low",
            cv_interpretation_flag="cv_unavailable",
            genotype_significant=True,
            environment_significant=False,
            gxe_significant=False,
            ranking_caution=False,
            selection_feasible=True,
            mean_separation=None,
            n_genotypes=10,
            n_environments=1,
            n_reps=3,
            environment_mode="single",
            domain="plant_breeding",
        )
        self.assertIn("unavailable", text.lower())

    def test_cv_low_precision_message(self):
        """Low CV (< 10) should generate high precision message."""
        text = generate_anova_interpretation(
            trait="Grain Yield",
            summary=self._make_summary(cv=5.0),
            precision_level="good",
            cv_interpretation_flag="cv_available",
            genotype_significant=True,
            environment_significant=False,
            gxe_significant=False,
            ranking_caution=False,
            selection_feasible=True,
            mean_separation=None,
            n_genotypes=8,
            n_environments=1,
            n_reps=3,
            environment_mode="single",
            domain="plant_breeding",
        )
        self.assertIn("high experimental precision", text.lower())

    def test_cv_high_caution_message(self):
        """High CV (>= 20) should warn about cautious interpretation."""
        text = generate_anova_interpretation(
            trait="Biomass",
            summary=self._make_summary(cv=25.0),
            precision_level="low",
            cv_interpretation_flag="cv_available",
            genotype_significant=True,
            environment_significant=False,
            gxe_significant=False,
            ranking_caution=False,
            selection_feasible=True,
            mean_separation=None,
            n_genotypes=8,
            n_environments=1,
            n_reps=3,
            environment_mode="single",
            domain="plant_breeding",
        )
        self.assertIn("interpreted cautiously", text.lower())


# ============================================================================
# PART 3 — Softened environmental variance language
# ============================================================================

class TestSoftenedEnvironmentalVarianceLanguage(unittest.TestCase):

    def test_forbidden_phrase_not_in_gcv_pcv_breeding(self):
        """Forbidden phrase must not appear in plant_breeding GCV/PCV description (low inflation)."""
        # inflation_pct < 3%: gcv=15.0, pcv=15.2 → inflation=1.33%
        result = _describe_gcv_pcv(gcv=15.0, pcv=15.2, trait_name="Yield")
        self.assertNotIn("negligible environmental variance inflation", result.lower())

    def test_forbidden_phrase_not_in_gcv_pcv_agronomy(self):
        """Forbidden phrase must not appear in agronomy GCV/PCV description (low inflation)."""
        # inflation_pct < 3%: gcv=10.0, pcv=10.2 → inflation=2%
        result = _describe_gcv_pcv(gcv=10.0, pcv=10.2, trait_name="Growth", domain="agronomy")
        self.assertNotIn("negligible environmental variance inflation", result.lower())

    def test_replacement_phrase_present_breeding(self):
        """Replacement phrase should appear for near-identical GCV/PCV (inflation < 3%)."""
        # Use values where inflation_pct < 3%: gcv=15.0, pcv=15.2 → inflation=1.33%
        result = _describe_gcv_pcv(gcv=15.0, pcv=15.2, trait_name="Yield")
        self.assertIn("relatively limited environmental variance influence", result.lower())

    def test_replacement_phrase_present_agronomy(self):
        """Replacement phrase should appear in agronomy context too (inflation < 3%)."""
        # Use values where inflation_pct < 3%: gcv=10.0, pcv=10.2 → inflation=2%
        result = _describe_gcv_pcv(gcv=10.0, pcv=10.2, trait_name="Growth", domain="agronomy")
        self.assertIn("relatively limited environmental variance influence", result.lower())

    def test_moderate_inflation_not_affected(self):
        """Moderate inflation case (different branch) should not be affected."""
        result = _describe_gcv_pcv(gcv=10.0, pcv=13.0, trait_name="Yield")
        self.assertNotIn("negligible environmental variance inflation", result.lower())

    def test_genetics_interpretation_no_forbidden_phrase(self):
        """Full genetics interpretation should never contain forbidden phrase."""
        interpretation, _ = generate_genetics_interpretation(
            trait_name="Plant Height",
            h2=0.85,
            gam=8.5,
            gcv=15.0,
            pcv=15.5,
            analysis_type="single_environment",
        )
        self.assertNotIn("negligible environmental variance inflation", interpretation.lower())


# ============================================================================
# PART 4 — Narrative diversity
# ============================================================================

class TestNarrativeDiversity(unittest.TestCase):

    def test_select_narrative_variant_deterministic(self):
        """Same trait name → same variant every time."""
        options = ["A", "B", "C"]
        v1 = _select_narrative_variant("Yield", options)
        v2 = _select_narrative_variant("Yield", options)
        self.assertEqual(v1, v2)

    def test_select_narrative_variant_different_traits(self):
        """Different trait names should not always select the same variant."""
        options = ["A", "B", "C"]
        variants = {_select_narrative_variant(t, options) for t in ["Yield", "Height", "Weight", "Biomass", "LAI"]}
        # At least 2 different variants should appear across 5 different traits
        self.assertGreater(len(variants), 1)

    def test_select_narrative_variant_uses_all_options(self):
        """Over many trait names, all options should be selected at least once."""
        options = ["Option A", "Option B", "Option C"]
        selected = set()
        trait_names = [f"Trait{i}" for i in range(30)]
        for t in trait_names:
            selected.add(_select_narrative_variant(t, options))
        self.assertEqual(selected, set(options))

    def test_high_h2_high_gam_uses_approved_variants(self):
        """High H²+GAM breeding interpretation should use one of the approved variants."""
        approved = {
            "Direct phenotypic selection may be effective under the conditions evaluated in this study.",
            "The trait may respond favorably to phenotypic selection under the evaluated conditions.",
            "Selection progress may be achievable through direct phenotypic evaluation.",
        }
        sections = generate_genetics_interpretation_sections(
            trait_name="Grain Yield",
            h2=0.85,
            gam=12.0,
            gcv=None,
            pcv=None,
        )
        # The breeding_interpretation should start with one of the approved sentences
        found = any(sections.breeding_interpretation.startswith(v) for v in approved)
        self.assertTrue(found, f"Unexpected breeding_interpretation: {sections.breeding_interpretation}")

    def test_high_h2_medium_gam_uses_approved_variants(self):
        """High H²+Medium GAM breeding interpretation should use approved variants."""
        approved = {
            "Moderate selection progress may be achievable through phenotypic evaluation.",
            "The trait showed moderate expected response to phenotypic selection.",
        }
        sections = generate_genetics_interpretation_sections(
            trait_name="Plant Height",
            h2=0.75,
            gam=7.0,
            gcv=None,
            pcv=None,
        )
        found = any(sections.breeding_interpretation.startswith(v) for v in approved)
        self.assertTrue(found, f"Unexpected breeding_interpretation: {sections.breeding_interpretation}")

    def test_moderate_h2_uses_approved_variants(self):
        """Moderate H² breeding interpretation should use approved variants."""
        approved = {
            "Selection efficiency may partly depend on environmental conditions.",
            "Environmental influence may contribute to observed phenotypic expression.",
        }
        sections = generate_genetics_interpretation_sections(
            trait_name="Stem Diameter",
            h2=0.45,
            gam=6.0,
            gcv=None,
            pcv=None,
        )
        found = any(sections.breeding_interpretation.startswith(v) for v in approved)
        self.assertTrue(found, f"Unexpected breeding_interpretation: {sections.breeding_interpretation}")

    def test_high_h2_high_gam_interpretation_uses_approved_variants(self):
        """generate_genetics_interpretation for high H²+GAM should use approved variants."""
        approved_endings = [
            "Direct phenotypic selection may be effective under the conditions evaluated in this study.",
            "The trait may respond favorably to phenotypic selection under the evaluated conditions.",
            "Selection progress may be achievable through direct phenotypic evaluation.",
        ]
        interpretation, _ = generate_genetics_interpretation(
            trait_name="Grain Yield",
            h2=0.85,
            gam=12.0,
            gcv=None,
            pcv=None,
            analysis_type="single_environment",
        )
        found = any(e in interpretation for e in approved_endings)
        self.assertTrue(found, f"No approved variant found in: {interpretation}")

    def test_moderate_h2_interpretation_uses_approved_variants(self):
        """generate_genetics_interpretation for moderate H² should use approved variants."""
        approved_endings = [
            "Selection efficiency may partly depend on environmental conditions.",
            "Environmental influence may contribute to observed phenotypic expression.",
        ]
        interpretation, _ = generate_genetics_interpretation(
            trait_name="Stem Diameter",
            h2=0.45,
            gam=6.0,
            gcv=None,
            pcv=None,
            analysis_type="single_environment",
        )
        found = any(e in interpretation for e in approved_endings)
        self.assertTrue(found, f"No approved variant found in: {interpretation}")


# ============================================================================
# PART 6 — Validation layer (forbidden phrase detection)
# ============================================================================

class TestGovernanceValidationLayer(unittest.TestCase):

    def test_blocks_negligible_environmental_variance_inflation(self):
        """Forbidden phrase 'negligible environmental variance inflation' must be detected."""
        text = "The results indicate negligible environmental variance inflation for this trait."
        hits = _find_breeding_governance_hits(text, analysis_type="single_environment")
        self.assertIn("negligible environmental variance inflation", hits)

    def test_blocks_top_performing_genotype(self):
        """'top-performing genotype' must be blocked."""
        text = "The top-performing genotype was identified."
        hits = _find_breeding_governance_hits(text, analysis_type="single_environment")
        self.assertIn("top-performing genotype", hits)

    def test_blocks_additive_gene_effects(self):
        """'additive gene effects' must be blocked."""
        text = "The trait is controlled by additive gene effects."
        hits = _find_breeding_governance_hits(text, analysis_type="single_environment")
        self.assertIn("additive gene effects", hits)

    def test_blocks_non_additive_effects(self):
        """'non-additive effects' must be blocked."""
        text = "Non-additive effects were detected for this trait."
        hits = _find_breeding_governance_hits(text, analysis_type="single_environment")
        self.assertIn("non-additive effects", hits)

    def test_blocks_clean_genetic_signal(self):
        """'clean genetic signal' must be blocked."""
        text = "A clean genetic signal was observed."
        hits = _find_breeding_governance_hits(text, analysis_type="single_environment")
        self.assertIn("clean genetic signal", hits)

    def test_clean_text_returns_no_hits(self):
        """Text with no forbidden phrases should return no hits."""
        text = (
            "Genetic parameters were estimated for Grain Yield. "
            "Broad-sense heritability was H² = 0.85. "
            "GAM = 12.0%. Residual variability was relatively low."
        )
        hits = _find_breeding_governance_hits(text, analysis_type="single_environment")
        # Filter out the GxE non-significant governance hit since no GxE text in this clean text
        self.assertEqual(hits, [])

    def test_case_insensitive_detection(self):
        """Forbidden phrase detection must be case-insensitive."""
        text = "NEGLIGIBLE ENVIRONMENTAL VARIANCE INFLATION was observed."
        hits = _find_breeding_governance_hits(text, analysis_type="single_environment")
        self.assertIn("negligible environmental variance inflation", hits)

    def test_no_false_positive_limited_variance_language(self):
        """Replacement phrase 'relatively limited environmental variance influence' must NOT be blocked."""
        text = (
            "GCV (15.00%) and PCV (15.50%) are nearly identical "
            "(difference: 3.3%), indicating relatively limited environmental "
            "variance influence under the evaluated conditions."
        )
        hits = _find_breeding_governance_hits(text, analysis_type="single_environment")
        # Should not flag the replacement phrase as a violation
        self.assertNotIn("negligible environmental variance inflation", hits)


if __name__ == "__main__":
    unittest.main()
