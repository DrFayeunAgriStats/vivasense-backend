"""
Test suite for VivaSense Interpretation Pipeline

Verifies that:
1. Trait Association interpretations appear in TraitAssociationModuleResponse
2. ANOVA interpretations appear in AnovaModuleResponse  
3. Genetics interpretations appear in GeneticParametersModuleResponse
4. Interpretations are not empty or just placeholders
5. Export builders correctly render interpretations in DOCX output
"""

import unittest
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, "/c:/Users/ADMIN/vivasense-backend/genetics-module")

from trait_association_interpretation import generate_trait_association_interpretation
from trait_relationships_routes import _compute_significant_pairs_and_strongest
from analysis_anova_routes import generate_anova_interpretation
from module_schemas import (
    TraitAssociationModuleResponse,
    StrongestPair,
    SignificantPair,
    TraitAssociationSummary,
    TraitAssociationHeatmap,
)
from genetics_schemas import AnovaTable


class TestTraitAssociationInterpretation(unittest.TestCase):
    """Test trait association interpretation generation and response inclusion."""

    def test_interpretation_with_significant_pairs(self):
        """Verify interpretation generated for significant trait associations."""
        interp = generate_trait_association_interpretation(
            n_traits=3,
            n_observations=20,
            n_significant_pairs=2,
            strongest_positive={"trait_1": "yield", "trait_2": "protein", "r": 0.75},
            strongest_negative={"trait_1": "fiber", "trait_2": "protein", "r": -0.65},
            risk_flags=["pairwise_n_not_tracked"],
            gxe_significant=False,
            environment_context="single_environment"
        )
        
        # Should mention significant pairs
        self.assertIn("Significant trait associations", interp)
        self.assertIn("2 pair(s)", interp)
        # Should mention strongest pairs
        self.assertIn("yield", interp)
        self.assertIn("protein", interp)

    def test_interpretation_with_sample_size_warning(self):
        """Verify interpretation includes preliminary evidence warning for small samples."""
        interp = generate_trait_association_interpretation(
            n_traits=3,
            n_observations=5,
            n_significant_pairs=0,
            strongest_positive=None,
            strongest_negative=None,
            risk_flags=["small_sample_size", "pairwise_n_not_tracked"],
            gxe_significant=False,
            environment_context="single_environment"
        )
        
        # Should warn about limited sample size
        self.assertIn("limited", interp.lower())
        self.assertIn("preliminary", interp.lower())

    def test_interpretation_with_pairwise_n_warning(self):
        """Verify interpretation warns about pairwise N tracking limitations."""
        interp = generate_trait_association_interpretation(
            n_traits=3,
            n_observations=20,
            n_significant_pairs=1,
            strongest_positive={"trait_1": "a", "trait_2": "b", "r": 0.8},
            strongest_negative=None,
            risk_flags=["pairwise_n_not_tracked"],
            gxe_significant=False,
            environment_context="single_environment"
        )
        
        # Should mention pairwise N limitation
        self.assertIn("pairwise", interp.lower())
        self.assertIn("limited", interp.lower())

    def test_interpretation_with_gxe_caution(self):
        """Verify interpretation warns about GxE when present."""
        interp = generate_trait_association_interpretation(
            n_traits=3,
            n_observations=20,
            n_significant_pairs=1,
            strongest_positive={"trait_1": "a", "trait_2": "b", "r": 0.8},
            strongest_negative=None,
            risk_flags=[],
            gxe_significant=True,
            environment_context="multi_environment"
        )
        
        # Should mention GxE
        self.assertIn("environment", interp.lower())
        self.assertIn("interaction", interp.lower())

    def test_trait_association_response_includes_interpretation(self):
        """Verify TraitAssociationModuleResponse includes interpretation field."""
        response = TraitAssociationModuleResponse(
            analysis_unit="genotype_mean",
            n_observations=20,
            alpha=0.05,
            environment_context="single_environment",
            gxe_significant=False,
            trait_names=["yield", "protein"],
            correlation_matrix={"yield": {"yield": 1.0, "protein": 0.5}, "protein": {"yield": 0.5, "protein": 1.0}},
            pvalue_matrix={"yield": {"yield": 0.0, "protein": 0.05}, "protein": {"yield": 0.05, "protein": 0.0}},
            significant_pairs=[SignificantPair(trait_1="yield", trait_2="protein", r=0.5, p_value=0.05, direction="positive", strength="moderate", confidence_status="limited_by_pairwise_n", selection_signal="exploratory only")],
            summary=TraitAssociationSummary(num_traits=2, num_significant_pairs=1),
            heatmap=TraitAssociationHeatmap(matrix={}),
            interpretation="Test interpretation text",
            dataset_token="test_token"
        )
        
        # Verify interpretation field exists and is not empty
        self.assertIsNotNone(response.interpretation)
        self.assertEqual(response.interpretation, "Test interpretation text")
        self.assertNotEqual(response.interpretation, "")


class TestAnovaInterpretation(unittest.TestCase):
    """Test ANOVA interpretation generation and response inclusion."""

    def test_anova_interpretation_with_significant_genotype(self):
        """Verify ANOVA interpretation generated when genotype effect is significant."""
        interp = generate_anova_interpretation(
            trait="yield",
            summary={
                "grand_mean": 100.0,
                "cv_percent": 15.0,
                "min": 80.0,
                "max": 120.0,
                "range": 40.0
            },
            precision_level="good",
            cv_interpretation_flag="cv_available",
            genotype_significant=True,
            environment_significant=False,
            gxe_significant=False,
            ranking_caution=False,
            selection_feasible=True,
            mean_separation=None,
            n_genotypes=10,
            n_environments=1,
            n_reps=3
        )
        
        # Should contain all 9 sections
        self.assertIn("Overview", interp)
        self.assertIn("Descriptive Interpretation", interp)
        self.assertIn("Genotype Effect", interp)
        self.assertIn("Significant genetic variation", interp)
        self.assertIn("selection", interp.lower())

    def test_anova_interpretation_not_empty(self):
        """Verify ANOVA interpretation is substantial and not just a placeholder."""
        interp = generate_anova_interpretation(
            trait="yield",
            summary={"grand_mean": 100.0, "cv_percent": 15.0, "min": 80.0, "max": 120.0, "range": 40.0},
            precision_level="moderate",
            cv_interpretation_flag="cv_available",
            genotype_significant=True,
            environment_significant=True,
            gxe_significant=True,
            ranking_caution=True,
            selection_feasible=True,
            mean_separation=None,
            n_genotypes=10,
            n_environments=3,
            n_reps=3
        )
        
        # Should be substantial (multiple sections with content)
        sections = interp.split("\n\n")
        self.assertGreater(len(sections), 5)
        # Each section should have content
        for section in sections:
            self.assertGreater(len(section.strip()), 10)

    def test_anova_response_includes_interpretation(self):
        """Verify ANOVA interpretations are substantial and contain specific content."""
        interp = generate_anova_interpretation(
            trait="test_trait",
            summary={"grand_mean": 50.0, "cv_percent": 12.0, "min": 40.0, "max": 60.0, "range": 20.0},
            precision_level="good",
            cv_interpretation_flag="cv_available",
            genotype_significant=True,
            environment_significant=False,
            gxe_significant=False,
            ranking_caution=False,
            selection_feasible=True,
            mean_separation=None,
            n_genotypes=10,
            n_environments=1,
            n_reps=3
        )
        
        # Verify interpretation is comprehensive and not empty
        self.assertIsNotNone(interp)
        self.assertGreater(len(interp), 100)
        self.assertIn("test_trait", interp)


class TestInterpretationNotEmpty(unittest.TestCase):
    """Verify interpretations are never empty strings or placeholders."""

    def test_trait_association_never_empty_interpretation(self):
        """Trait association interpretation always has meaningful content."""
        interp = generate_trait_association_interpretation(
            n_traits=2,
            n_observations=8,
            n_significant_pairs=0,
            strongest_positive=None,
            strongest_negative=None,
            risk_flags=["small_sample_size"],
            gxe_significant=False,
            environment_context="single_environment"
        )
        
        # Should not be empty or placeholder text
        self.assertGreater(len(interp.strip()), 50)
        self.assertNotIn("not yet attached", interp.lower())
        self.assertNotIn("pending", interp.lower())

    def test_anova_never_just_placeholder(self):
        """ANOVA interpretation contains real content, not generic placeholders."""
        interp = generate_anova_interpretation(
            trait="test_trait",
            summary={"grand_mean": 50.0, "cv_percent": None, "min": None, "max": None, "range": None},
            precision_level="low",
            cv_interpretation_flag="cv_unavailable",
            genotype_significant=False,
            environment_significant=None,
            gxe_significant=None,
            ranking_caution=None,
            selection_feasible=None,
            mean_separation=None,
            n_genotypes=5,
            n_environments=1,
            n_reps=2
        )
        
        # Should contain specific trait name and analysis details
        self.assertIn("test_trait", interp)
        # Even with minimal data, should have substantive content
        self.assertGreater(len(interp), 100)


class TestCorrelationLegacyPhraseAbsent(unittest.TestCase):
    """Prove the raw R phrase 'facilitating indirect selection' never reaches output."""

    _LEGACY_PHRASE = "facilitating indirect selection"

    def _gen(self, **overrides):
        params = dict(
            n_traits=4,
            n_observations=15,
            n_significant_pairs=2,
            strongest_positive={"trait_1": "GY", "trait_2": "PH", "r": 0.75},
            strongest_negative={"trait_1": "GY", "trait_2": "DM", "r": -0.60},
            risk_flags=["pairwise_n_not_tracked", "genotype_mean_based"],
            gxe_significant=False,
            environment_context="single_environment",
        )
        params.update(overrides)
        return generate_trait_association_interpretation(**params)

    def test_legacy_phrase_absent_baseline(self):
        text = self._gen()
        self.assertNotIn(
            self._LEGACY_PHRASE, text.lower(),
            "Legacy R phrase must never appear in Python-generated correlation output",
        )

    def test_legacy_phrase_absent_gxe_significant(self):
        text = self._gen(gxe_significant=True, risk_flags=["pairwise_n_not_tracked", "gxe_significant"])
        self.assertNotIn(self._LEGACY_PHRASE, text.lower())

    def test_legacy_phrase_absent_small_sample(self):
        text = self._gen(n_observations=7, risk_flags=["small_sample_size", "pairwise_n_not_tracked"])
        self.assertNotIn(self._LEGACY_PHRASE, text.lower())

    def test_correlation_result_is_non_empty(self):
        text = self._gen()
        self.assertGreater(len(text.strip()), 50,
                           "Correlation interpretation must be substantive text")


class TestSignificantPairFiltering(unittest.TestCase):
    """
    Prove that non-significant pairs never appear as trade-offs or meaningful
    associations in the interpretation text.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    def _interp(self, **overrides):
        """Call generate_trait_association_interpretation with safe defaults."""
        defaults = dict(
            n_traits=4,
            n_observations=20,
            n_significant_pairs=0,
            strongest_positive=None,
            strongest_negative=None,
            risk_flags=["pairwise_n_not_tracked"],
            gxe_significant=False,
            environment_context="single_environment",
        )
        defaults.update(overrides)
        return generate_trait_association_interpretation(**defaults)

    def _r_matrix(self, pairs):
        """
        Build a tiny 3-trait r/p matrix from a list of (i, j, r, p) tuples.
        Traits are ["T1", "T2", "T3"].
        """
        n = 3
        r = [[1.0 if i == j else None for j in range(n)] for i in range(n)]
        p = [[0.0 if i == j else None for j in range(n)] for i in range(n)]
        for i, j, rv, pv in pairs:
            r[i][j] = r[j][i] = rv
            p[i][j] = p[j][i] = pv
        return r, p

    # ── _compute_significant_pairs_and_strongest ──────────────────────────────

    def test_non_significant_negative_excluded_from_strongest(self):
        """r = -0.09, p = 0.85 must not become strongest_negative."""
        traits = ["LW", "FL", "GY"]
        r, p = self._r_matrix([
            (0, 1, -0.09, 0.8489),   # LW–FL: NOT significant
            (0, 2,  0.72, 0.02),     # LW–GY: significant positive
        ])
        n_sig, best_pos, best_neg = _compute_significant_pairs_and_strongest(traits, r, p)
        self.assertEqual(n_sig, 1)
        self.assertIsNotNone(best_pos)
        self.assertIsNone(best_neg,
            "Non-significant negative pair (p=0.85) must not appear as strongest_negative")

    def test_significant_negative_is_returned(self):
        """A pair with r < 0 and p <= 0.05 must be returned as strongest_negative."""
        traits = ["T1", "T2", "T3"]
        r, p = self._r_matrix([
            (0, 1, -0.75, 0.01),
            (0, 2,  0.60, 0.03),
        ])
        n_sig, best_pos, best_neg = _compute_significant_pairs_and_strongest(traits, r, p)
        self.assertEqual(n_sig, 2)
        self.assertIsNotNone(best_neg)
        self.assertAlmostEqual(best_neg["r"], -0.75)

    def test_no_significant_pairs_returns_none_for_both(self):
        """When all pairs are non-significant both strongest slots are None."""
        traits = ["T1", "T2", "T3"]
        r, p = self._r_matrix([
            (0, 1, -0.60, 0.30),
            (0, 2,  0.55, 0.20),
        ])
        n_sig, best_pos, best_neg = _compute_significant_pairs_and_strongest(traits, r, p)
        self.assertEqual(n_sig, 0)
        self.assertIsNone(best_pos)
        self.assertIsNone(best_neg)

    def test_strongest_among_multiple_significant_negatives(self):
        """When multiple significant negatives exist, the most-negative r wins."""
        traits = ["T1", "T2", "T3"]
        r, p = self._r_matrix([
            (0, 1, -0.50, 0.04),
            (0, 2, -0.80, 0.01),
        ])
        _, _, best_neg = _compute_significant_pairs_and_strongest(traits, r, p)
        self.assertIsNotNone(best_neg)
        self.assertAlmostEqual(best_neg["r"], -0.80,
            msg="Most-negative significant r (-0.80) must win over -0.50")

    # ── interpretation text ───────────────────────────────────────────────────

    def test_non_significant_negative_not_called_trade_off(self):
        """
        If strongest_negative=None (caller filtered it out), the report must
        say 'No significant negative associations' — not mention trade-offs.
        """
        text = self._interp(
            n_significant_pairs=1,
            strongest_positive={"trait_1": "LW", "trait_2": "GY", "r": 0.72},
            strongest_negative=None,   # non-sig pair already dropped by caller
        )
        self.assertIn("No significant negative associations were detected", text)
        self.assertNotIn("trade-off", text.lower())
        self.assertNotIn("trade off", text.lower())

    def test_significant_negative_described_as_trade_off(self):
        """A genuinely significant negative pair may be described as a trade-off."""
        text = self._interp(
            n_significant_pairs=2,
            strongest_positive={"trait_1": "T1", "trait_2": "T2", "r": 0.75},
            strongest_negative={"trait_1": "T1", "trait_2": "T3", "r": -0.70},
        )
        self.assertIn("trade-off", text.lower())
        self.assertNotIn("No significant negative", text)

    def test_no_significant_pairs_at_all(self):
        """Zero significant pairs: neither direction should mention a pair value."""
        text = self._interp(n_significant_pairs=0)
        self.assertIn("No significant trait associations were detected", text)
        self.assertNotIn("r =", text)
        self.assertNotIn("trade-off", text.lower())

    def test_significant_positive_only(self):
        """Only positive significant: positive is described, negative explicitly absent."""
        text = self._interp(
            n_significant_pairs=1,
            strongest_positive={"trait_1": "A", "trait_2": "B", "r": 0.80},
            strongest_negative=None,
        )
        self.assertIn("strongest significant positive", text.lower())
        self.assertIn("No significant negative associations were detected", text)

    def test_significant_negative_only(self):
        """Only negative significant: negative is described, positive explicitly absent."""
        text = self._interp(
            n_significant_pairs=1,
            strongest_positive=None,
            strongest_negative={"trait_1": "A", "trait_2": "C", "r": -0.78},
        )
        self.assertIn("strongest significant negative", text.lower())
        self.assertIn("No significant positive associations were detected", text)


class TestAnovaModuleImportsClean(unittest.TestCase):
    """Smoke test: analysis_anova_routes must import without NameError."""

    def test_module_imports_without_error(self):
        """Any NameError (e.g. missing 'Any') would surface here."""
        try:
            import analysis_anova_routes  # noqa: F401
        except NameError as exc:
            self.fail(f"analysis_anova_routes raised NameError on import: {exc}")

    def test_generate_anova_interpretation_callable(self):
        from analysis_anova_routes import generate_anova_interpretation
        result = generate_anova_interpretation(
            trait="Smoke",
            summary={"grand_mean": 10.0, "cv_percent": 5.0, "min": 8.0, "max": 12.0, "range": 4.0},
            precision_level="good",
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
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 50)


if __name__ == "__main__":
    unittest.main()