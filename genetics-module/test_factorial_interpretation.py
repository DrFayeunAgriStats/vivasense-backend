"""
Interaction-aware factorial interpretation (FAC-05 reporting layer).

The engine resolves the significance hierarchy (vivasense_genetics.R ->
mean_separation_basis). These tests pin that the REPORT narrates that decision
instead of falling back to marginal main effects, and that the generic
factorial pathway stays domain-neutral.

The defect being guarded: a factorial run was routed through the
genetics-specific narrative, which asked only whether the marginal "genotype"
main effect was significant. A factor whose effect is real but conditional on
another therefore produced "No significant genetic variation was detected"
alongside a significant interaction in the same ANOVA table.

Run from inside genetics-module/:
    python -m pytest test_factorial_interpretation.py -v
"""

import unittest

from analysis_anova_routes import generate_anova_interpretation

BANNED = ("genotype", "genetic", "environment", "Genotype", "Environment", "Genetic")
LABELS = {"genotype": "V", "factor": "P", "factor_c": "N"}


def interpret(selected, significant_terms, *, labels=LABELS,
              design="factorial_rcbd", domain="plant_breeding"):
    return generate_anova_interpretation(
        trait="DFT",
        summary={"grand_mean": 52.0, "cv_percent": 4.1},
        precision_level="good", cv_interpretation_flag=None,
        # Deliberately False: the marginal main effect of V is NOT significant.
        # The old path turned exactly this into "no significant variation".
        genotype_significant=False, environment_significant=None,
        gxe_significant=None, ranking_caution=None, selection_feasible=False,
        mean_separation=None, n_genotypes=2, n_environments=None, n_reps=3,
        design_type=design, domain=domain,
        mean_separation_basis={"selected": selected,
                               "significant_terms": significant_terms},
        factor_labels=labels,
        two_way_interaction_means=({significant_terms: {}}
                                   if selected == "two_way" else None),
    )


class TestSignificanceHierarchyDrivesInterpretation(unittest.TestCase):

    def test_a_three_way_significant_drives_interpretation(self):
        txt = interpret("three_way", "genotype:factor:factor_c")
        self.assertIn("V × P × N", txt)
        self.assertIn("Three-Way", txt)
        self.assertIn("depends on the levels of the other two", txt)
        self.assertIn("must not be read as unconditional", txt)

    def test_b_two_way_significant_drives_interpretation(self):
        """Three-way NS, A×B significant -> A×B is the primary result."""
        txt = interpret("two_way", "genotype:factor")
        self.assertIn("V × P", txt)
        self.assertIn("primary inferential result", txt)
        self.assertIn("three-way interaction was not significant", txt)
        # Marginal means are demoted, not deleted.
        self.assertIn("supplementary descriptive information", txt)

    def test_b_states_the_dependence_directionally(self):
        """'the effect of V is conditional on the level of P' — not a factor set."""
        txt = interpret("two_way", "genotype:factor")
        self.assertIn("the effect of V is conditional on the level of P", txt)

    def test_b_non_significant_marginal_is_not_called_absent(self):
        """The specific wrong conclusion must not reappear in any form."""
        txt = interpret("two_way", "genotype:factor")
        self.assertNotIn("No significant genetic variation", txt)
        self.assertIn("does not indicate the absence of an effect", txt)

    def test_b_interaction_not_marginal_p_establishes_variation(self):
        """The evidence for varying effect must be the interaction, not the p-value."""
        txt = interpret("two_way", "genotype:factor")
        self.assertIn(
            "significant interaction is what establishes that the effect varies", txt)
        self.assertIn("cannot establish this either way", txt)

    def test_b_multiple_two_ways_each_reported(self):
        txt = interpret("two_way", "genotype:factor, factor:factor_c")
        self.assertIn("V × P", txt)
        self.assertIn("P × N", txt)

    def test_c_no_interaction_allows_main_effects(self):
        txt = interpret("marginal", "")
        self.assertIn("No interaction among the treatment factors", txt)
        self.assertIn("main-effect mean separation is the appropriate", txt)
        self.assertNotIn("Governing Interaction", txt)

    def test_hierarchy_is_dynamic_not_hardcoded(self):
        """Same code path, different factor roles -> different named term."""
        other = {"genotype": "Irrigation", "factor": "Spacing", "factor_c": "Cultivar"}
        txt = interpret("two_way", "genotype:factor_c", labels=other)
        self.assertIn("Irrigation × Cultivar", txt)
        self.assertNotIn("V × P", txt)


class TestDomainNeutralTerminology(unittest.TestCase):
    """TEST G: no biological terminology in the generic factorial pathway."""

    def test_g_no_hardcoded_domain_terms_in_any_branch(self):
        for selected, terms in (("three_way", "genotype:factor:factor_c"),
                                ("two_way", "genotype:factor"),
                                ("marginal", "")):
            with self.subTest(branch=selected):
                txt = interpret(selected, terms)
                for word in BANNED:
                    self.assertNotIn(word, txt,
                                     f"{word!r} leaked in the {selected} branch")

    def test_g_holds_even_under_plant_breeding_domain(self):
        """Domain must not reintroduce genetics wording on a generic factorial."""
        txt = interpret("two_way", "genotype:factor", domain="plant_breeding")
        for word in BANNED:
            self.assertNotIn(word, txt)

    def test_g_unmapped_factors_fall_back_to_position_not_biology(self):
        txt = interpret("two_way", "genotype:factor", labels={})
        self.assertIn("Factor A", txt)
        self.assertIn("Factor B", txt)
        for word in BANNED:
            self.assertNotIn(word, txt)

    def test_g_factorial_crd_labelled_as_crd(self):
        txt = interpret("marginal", "", design="factorial_crd")
        self.assertIn("completely randomised design", txt)
        self.assertNotIn("randomised complete block", txt)


class TestNonFactorialDesignsUnaffected(unittest.TestCase):
    """The new dispatch must not capture any other design."""

    def test_split_plot_still_uses_its_own_path(self):
        txt = generate_anova_interpretation(
            trait="Yield", summary={"grand_mean": 10.0}, precision_level=None,
            cv_interpretation_flag=None, genotype_significant=None,
            environment_significant=None, gxe_significant=None,
            ranking_caution=None, selection_feasible=None, mean_separation=None,
            n_genotypes=None, n_environments=None, n_reps=3,
            design_type="split_plot_rcbd", main_plot_significant=True,
            subplot_significant=True, interaction_significant=False,
            mp_label="Tillage", sp_label="Fertiliser",
        )
        self.assertIn("split-plot", txt.lower())
        self.assertIn("Tillage", txt)

    def test_plain_rcbd_still_uses_the_genetics_path(self):
        txt = generate_anova_interpretation(
            trait="Yield", summary={"grand_mean": 10.0}, precision_level=None,
            cv_interpretation_flag=None, genotype_significant=True,
            environment_significant=None, gxe_significant=None,
            ranking_caution=None, selection_feasible=True, mean_separation=None,
            n_genotypes=5, n_environments=None, n_reps=3,
            design_type="rcbd", domain="plant_breeding",
        )
        self.assertNotIn("Governing Interaction", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
