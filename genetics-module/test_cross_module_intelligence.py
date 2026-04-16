"""
Tests for Cross-Module Intelligence Layer

Tests cover:
- High heritability + significant genotype + significant GxE
- Strong correlation + pairwise_n_not_tracked
- Genotype not significant
- Low precision experiment
- Missing trait association or missing genetics result
"""

import unittest
from cross_module_intelligence import (
    build_cross_module_signals,
    determine_selection_feasibility,
    determine_adaptation_type,
    determine_stability_requirement,
    determine_indirect_selection_support,
    determine_experimental_reliability,
    determine_overall_confidence,
    collect_integrated_risk_flags,
    generate_integrated_interpretation,
    CrossModuleSignals,
    IntegratedSummary,
    DecisionSignals
)


class TestCrossModuleIntelligence(unittest.TestCase):

    def test_high_heritability_significant_genotype_gxe(self):
        """Test case: high heritability + significant genotype + significant GxE"""
        anova_result = {
            "trait": "yield",
            "status": "success",
            "genotype_significant": True,
            "gxe_significant": True,
            "precision_level": "good"
        }
        genetic_result = {
            "status": "success",
            "heritability": {"h2_broad_sense": 0.7}
        }

        signals = build_cross_module_signals(anova_result, None, genetic_result)

        self.assertEqual(signals.integrated_summary.selection_feasibility, "feasible_with_caution")
        self.assertEqual(signals.integrated_summary.adaptation_type, "specific_adaptation_likely")
        self.assertEqual(signals.integrated_summary.stability_requirement, "required")
        self.assertEqual(signals.integrated_summary.experimental_reliability, "high")
        self.assertEqual(signals.integrated_summary.overall_confidence, "medium")
        self.assertTrue(signals.decision_signals.advance_top_genotypes)
        self.assertTrue(signals.decision_signals.recommend_stability_analysis)
        self.assertTrue(signals.decision_signals.recommend_environment_specific_selection)

    def test_strong_correlation_pairwise_n_not_tracked(self):
        """Test case: strong correlation + pairwise_n_not_tracked"""
        anova_result = {
            "trait": "yield",
            "status": "success",
            "genotype_significant": True,
            "gxe_significant": False,
            "precision_level": "good"
        }
        trait_association_result = {
            "significant_pairs": [
                {
                    "trait_1": "yield",
                    "trait_2": "protein",
                    "strength": "strong",
                    "r": 0.8,
                    "p_value": 0.01
                }
            ],
            "risk_flags": ["pairwise_n_not_tracked"]
        }
        genetic_result = {
            "status": "success",
            "heritability": {"h2_broad_sense": 0.6}
        }

        signals = build_cross_module_signals(anova_result, trait_association_result, genetic_result)

        self.assertEqual(signals.integrated_summary.indirect_selection_support, "preliminary_only")
        self.assertFalse(signals.decision_signals.recommend_indirect_selection)

    def test_genotype_not_significant(self):
        """Test case: genotype not significant"""
        anova_result = {
            "trait": "yield",
            "status": "success",
            "genotype_significant": False,
            "gxe_significant": False,
            "precision_level": "good"
        }

        signals = build_cross_module_signals(anova_result)

        self.assertEqual(signals.integrated_summary.selection_feasibility, "not_supported")
        self.assertFalse(signals.decision_signals.advance_top_genotypes)

    def test_low_precision_experiment(self):
        """Test case: low precision experiment"""
        anova_result = {
            "trait": "yield",
            "status": "success",
            "genotype_significant": True,
            "gxe_significant": False,
            "precision_level": "low"
        }

        signals = build_cross_module_signals(anova_result)

        self.assertEqual(signals.integrated_summary.experimental_reliability, "limited")
        self.assertEqual(signals.integrated_summary.overall_confidence, "low")

    def test_missing_trait_association(self):
        """Test case: missing trait association result"""
        anova_result = {
            "trait": "yield",
            "status": "success",
            "genotype_significant": True,
            "gxe_significant": False,
            "precision_level": "good"
        }
        genetic_result = {
            "status": "success",
            "heritability": {"h2_broad_sense": 0.6}
        }

        signals = build_cross_module_signals(anova_result, None, genetic_result)

        self.assertEqual(signals.integrated_summary.indirect_selection_support, "not_supported")
        self.assertIn("missing_trait_association_data", signals.integrated_risk_flags)
        self.assertFalse(signals.decision_signals.recommend_indirect_selection)

    def test_missing_genetics_result(self):
        """Test case: missing genetics result"""
        anova_result = {
            "trait": "yield",
            "status": "success",
            "genotype_significant": True,
            "gxe_significant": False,
            "precision_level": "good"
        }
        trait_association_result = {
            "significant_pairs": [],
            "risk_flags": []
        }

        signals = build_cross_module_signals(anova_result, trait_association_result, None)

        self.assertEqual(signals.integrated_summary.selection_feasibility, "feasible")
        self.assertIn("missing_genetic_parameters_data", signals.integrated_risk_flags)

    def test_determine_selection_feasibility_logic(self):
        """Test selection feasibility determination logic"""
        # Genotype not significant
        self.assertEqual(
            determine_selection_feasibility({"genotype_significant": False}, None),
            "not_supported"
        )

        # Genotype significant, high heritability
        self.assertEqual(
            determine_selection_feasibility(
                {"genotype_significant": True},
                {"heritability": {"h2_broad_sense": 0.7}}
            ),
            "feasible"
        )

        # Genotype significant, GxE significant
        self.assertEqual(
            determine_selection_feasibility(
                {"genotype_significant": True, "gxe_significant": True},
                None
            ),
            "feasible_with_caution"
        )

    def test_determine_adaptation_type_logic(self):
        """Test adaptation type determination logic"""
        self.assertEqual(
            determine_adaptation_type({"gxe_significant": True}),
            "specific_adaptation_likely"
        )
        self.assertEqual(
            determine_adaptation_type({"gxe_significant": False}),
            "broad_adaptation_more_plausible"
        )

    def test_determine_stability_requirement_logic(self):
        """Test stability requirement determination logic"""
        self.assertEqual(
            determine_stability_requirement({"gxe_significant": True}),
            "required"
        )
        self.assertEqual(
            determine_stability_requirement({"gxe_significant": False}),
            "optional"
        )

    def test_determine_indirect_selection_support_logic(self):
        """Test indirect selection support determination logic"""
        # No trait association result
        self.assertEqual(
            determine_indirect_selection_support(None, None, {}),
            "not_supported"
        )

        # Strong correlation with pairwise_n_not_tracked
        ta_result = {
            "significant_pairs": [{"strength": "strong"}],
            "risk_flags": ["pairwise_n_not_tracked"]
        }
        self.assertEqual(
            determine_indirect_selection_support(ta_result, None, {}),
            "preliminary_only"
        )

        # Strong correlation with good genetics
        ta_result = {
            "significant_pairs": [{"strength": "strong"}],
            "risk_flags": []
        }
        genetic_result = {"heritability": {"h2_broad_sense": 0.5}}
        self.assertEqual(
            determine_indirect_selection_support(ta_result, genetic_result, {}),
            "supported"
        )

    def test_determine_experimental_reliability_logic(self):
        """Test experimental reliability determination logic"""
        self.assertEqual(
            determine_experimental_reliability({"precision_level": "low"}),
            "limited"
        )
        self.assertEqual(
            determine_experimental_reliability({"precision_level": "moderate"}),
            "acceptable"
        )
        self.assertEqual(
            determine_experimental_reliability({"precision_level": "good"}),
            "high"
        )

    def test_determine_overall_confidence_logic(self):
        """Test overall confidence determination logic"""
        # Failed ANOVA
        self.assertEqual(
            determine_overall_confidence({"status": "failed"}, None, None),
            "low"
        )

        # Low precision
        self.assertEqual(
            determine_overall_confidence({"status": "success", "precision_level": "low"}, None, None),
            "low"
        )

        # High confidence case
        anova = {"status": "success", "precision_level": "good"}
        genetic = {"status": "success", "heritability": {"h2_broad_sense": 0.7}}
        ta = {"significant_pairs": [{"strength": "strong"}]}
        self.assertEqual(
            determine_overall_confidence(anova, ta, genetic),
            "high"
        )

    def test_collect_integrated_risk_flags(self):
        """Test integrated risk flags collection"""
        anova = {
            "gxe_significant": True,
            "precision_level": "low",
            "data_warnings": ["unbalanced_design"]
        }
        ta = {"risk_flags": ["pairwise_n_not_tracked", "small_sample_size"]}
        genetic = {"data_warnings": ["missing_replications"]}

        flags = collect_integrated_risk_flags(anova, ta, genetic)

        expected_flags = {
            "gxe_interaction_detected",
            "low_experimental_precision",
            "unbalanced_design",
            "pairwise_n_not_tracked",
            "small_sample_size",
            "missing_replications"
        }
        self.assertEqual(set(flags), expected_flags)


class TestIntegratedInterpretation(unittest.TestCase):

    def test_interpretation_not_supported_selection(self):
        """Test interpretation when selection is not supported"""
        signals = CrossModuleSignals(
            trait="yield",
            integrated_summary=IntegratedSummary(
                selection_feasibility="not_supported",
                adaptation_type="broad_adaptation_more_plausible",
                stability_requirement="optional",
                indirect_selection_support="not_supported",
                experimental_reliability="high",
                overall_confidence="low"
            ),
            integrated_risk_flags=["missing_trait_association_data"],
            decision_signals=DecisionSignals(
                advance_top_genotypes=False,
                recommend_stability_analysis=False,
                recommend_indirect_selection=False,
                recommend_environment_specific_selection=False
            ),
            supporting_evidence={
                "anova": {"genotype_significant": False},
                "trait_association": {},
                "genetics": {}
            }
        )

        interpretation = generate_integrated_interpretation(signals)

        # Check that all required sections are present
        self.assertIn("Integrated Overview", interpretation)
        self.assertIn("Selection Implication", interpretation)
        self.assertIn("Adaptation and Stability", interpretation)
        self.assertIn("Trait Relationship Implication", interpretation)
        self.assertIn("Experimental Reliability", interpretation)
        self.assertIn("Risk and Caution", interpretation)
        self.assertIn("Final Recommendation", interpretation)

        # Check specific content
        self.assertIn("absence of significant genetic variation indicates that selection for this trait is not currently supported", interpretation)
        self.assertIn("Genotype advancement is not recommended", interpretation)

    def test_interpretation_feasible_with_gxe(self):
        """Test interpretation when selection is feasible but with GxE caution"""
        signals = CrossModuleSignals(
            trait="yield",
            integrated_summary=IntegratedSummary(
                selection_feasibility="feasible_with_caution",
                adaptation_type="specific_adaptation_likely",
                stability_requirement="required",
                indirect_selection_support="preliminary_only",
                experimental_reliability="acceptable",
                overall_confidence="medium"
            ),
            integrated_risk_flags=["gxe_interaction_detected", "pairwise_n_not_tracked"],
            decision_signals=DecisionSignals(
                advance_top_genotypes=True,
                recommend_stability_analysis=True,
                recommend_indirect_selection=False,
                recommend_environment_specific_selection=True
            ),
            supporting_evidence={
                "anova": {"genotype_significant": True, "gxe_significant": True},
                "trait_association": {"significant_pairs": [{"strength": "strong"}]},
                "genetics": {"heritability": {"h2_broad_sense": 0.6}}
            }
        )

        interpretation = generate_integrated_interpretation(signals)

        # Check GxE handling
        self.assertIn("genotype × environment interaction", interpretation)
        self.assertIn("stability analysis", interpretation)
        self.assertIn("interpreted cautiously", interpretation)
        self.assertIn("validation across multiple environments", interpretation)
        self.assertIn("preliminary and should not yet be used", interpretation)

    def test_interpretation_high_confidence_feasible(self):
        """Test interpretation with high confidence and feasible selection"""
        signals = CrossModuleSignals(
            trait="yield",
            integrated_summary=IntegratedSummary(
                selection_feasibility="feasible",
                adaptation_type="broad_adaptation_more_plausible",
                stability_requirement="optional",
                indirect_selection_support="supported",
                experimental_reliability="high",
                overall_confidence="high"
            ),
            integrated_risk_flags=[],
            decision_signals=DecisionSignals(
                advance_top_genotypes=True,
                recommend_stability_analysis=False,
                recommend_indirect_selection=True,
                recommend_environment_specific_selection=False
            ),
            supporting_evidence={
                "anova": {"genotype_significant": True, "gxe_significant": False},
                "trait_association": {"significant_pairs": [{"strength": "strong"}]},
                "genetics": {"heritability": {"h2_broad_sense": 0.7}}
            }
        )

        interpretation = generate_integrated_interpretation(signals)

        # Check positive outcomes
        self.assertIn("high heritability", interpretation)
        self.assertIn("broad adaptation", interpretation)
        self.assertIn("indirect selection strategies may be viable", interpretation)
        self.assertIn("strong evidence for informed decision-making", interpretation)

    def test_interpretation_low_reliability(self):
        """Test interpretation with low experimental reliability"""
        signals = CrossModuleSignals(
            trait="yield",
            integrated_summary=IntegratedSummary(
                selection_feasibility="feasible_with_caution",
                adaptation_type="specific_adaptation_likely",
                stability_requirement="required",
                indirect_selection_support="not_supported",
                experimental_reliability="limited",
                overall_confidence="low"
            ),
            integrated_risk_flags=["low_experimental_precision", "gxe_interaction_detected"],
            decision_signals=DecisionSignals(
                advance_top_genotypes=True,
                recommend_stability_analysis=True,
                recommend_indirect_selection=False,
                recommend_environment_specific_selection=True
            ),
            supporting_evidence={
                "anova": {"genotype_significant": True, "gxe_significant": True, "precision_level": "low"},
                "trait_association": {},
                "genetics": {"heritability": {"h2_broad_sense": 0.3}}
            }
        )

        interpretation = generate_integrated_interpretation(signals)

        # Check caution and limitations
        self.assertIn("experimental reliability is limited", interpretation)
        self.assertIn("high variability", interpretation)
        self.assertIn("cautious interpretation", interpretation)
        self.assertIn("validated through additional experimentation", interpretation)

    def test_interpretation_missing_data_flags(self):
        """Test interpretation with missing data risk flags"""
        signals = CrossModuleSignals(
            trait="yield",
            integrated_summary=IntegratedSummary(
                selection_feasibility="feasible",
                adaptation_type="broad_adaptation_more_plausible",
                stability_requirement="optional",
                indirect_selection_support="not_supported",
                experimental_reliability="high",
                overall_confidence="medium"
            ),
            integrated_risk_flags=["missing_trait_association_data", "missing_genetic_parameters_data"],
            decision_signals=DecisionSignals(
                advance_top_genotypes=True,
                recommend_stability_analysis=False,
                recommend_indirect_selection=False,
                recommend_environment_specific_selection=False
            ),
            supporting_evidence={
                "anova": {"genotype_significant": True, "gxe_significant": False},
                "trait_association": None,
                "genetics": None
            }
        )

        interpretation = generate_integrated_interpretation(signals)

        # Check missing data handling
        self.assertIn("absence of trait association data", interpretation)
        self.assertIn("lack of genetic parameters data", interpretation)


if __name__ == "__main__":
    unittest.main()