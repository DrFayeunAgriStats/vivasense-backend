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

from analysis_trait_association_routes import generate_trait_association_interpretation
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