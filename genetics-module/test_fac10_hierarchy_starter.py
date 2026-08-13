"""
FAC-10: the factorial starter sentence follows the interaction hierarchy.

Previously it named Factor A as "the treatment effect", asserting an
unconditional effect that a significant interaction contradicts — and for
2-factor designs mean_separation_basis was never populated, so no interpretation
section corrected it. Both layers now read the same basis and cannot diverge.

Statistics are untouched: this changes which result the sentence summarises,
not any number in it.
"""
import unittest
from genetics_export import _factorial_starter_sentence


class _MS:
    def __init__(self, label): self.treatment_label = label


class _AT:
    def __init__(self, sources, ps):
        self.source = sources
        self.p_value = ps
        self.f_value = [1.0] * len(sources)
        self.ss = [1.0] * len(sources)
        self.df = [1] * len(sources)


class _R:
    def __init__(self, design, basis, labels, at):
        self.design = design
        self.mean_separation_basis = basis
        self.mean_separation = _MS(labels[0]) if len(labels) > 0 else None
        self.mean_separation_b = _MS(labels[1]) if len(labels) > 1 else None
        self.mean_separation_c = _MS(labels[2]) if len(labels) > 2 else None
        self.anova_table = at


def two_factor(basis, p_a=0.001, p_b=0.001, p_ab=0.001, labels=("Irrigation", "Variety")):
    at = _AT([labels[0], labels[1], f"{labels[0]}×{labels[1]}", "Residuals"],
             [p_a, p_b, p_ab, None])
    return _R("factorial_rcbd", basis, labels, at)


def three_factor(basis, ps=(0.001, 0.001, 0.001), labels=("Irrigation", "Variety", "Spacing")):
    a, b, c = labels
    at = _AT([a, b, c, f"{a}×{b}", f"{a}×{c}", f"{b}×{c}", f"{a}×{b}×{c}", "Residuals"],
             [ps[0], ps[1], ps[2], 0.5, 0.5, 0.5, 0.5, None])
    return _R("factorial_rcbd", basis, labels, at)


BAN = ("genotype", "genetic", "environment", "Genotype", "Environment")


class TestTwoFactor(unittest.TestCase):
    def test_a_interaction_significant_is_governing(self):
        s = _factorial_starter_sentence(
            two_factor({"selected": "two_way", "significant_terms": "genotype:factor"}),
            two_factor({"selected": "two_way", "significant_terms": "genotype:factor"}).anova_table,
            "Yield")
        self.assertIn("Irrigation × Variety interaction was statistically significant", s)
        self.assertIn("conditional on the level of Variety", s)
        self.assertNotIn("treatment effect", s)

    def test_b_no_interaction_reports_each_factor(self):
        r = two_factor({"selected": "marginal", "significant_terms": ""}, p_a=0.001, p_b=0.9)
        s = _factorial_starter_sentence(r, r.anova_table, "Yield")
        self.assertIn("was not statistically significant", s)
        self.assertIn("Irrigation had a significant main effect", s)
        self.assertIn("Variety did not have a significant main effect", s)

    def test_c_both_main_effects_significant(self):
        r = two_factor({"selected": "marginal", "significant_terms": ""}, p_a=0.001, p_b=0.01)
        s = _factorial_starter_sentence(r, r.anova_table, "Yield")
        self.assertEqual(s.count("had a significant main effect"), 2)

    def test_d_no_significant_effects(self):
        r = two_factor({"selected": "marginal", "significant_terms": ""}, p_a=0.4, p_b=0.9)
        s = _factorial_starter_sentence(r, r.anova_table, "Yield")
        self.assertEqual(s.count("did not have a significant main effect"), 2)


class TestThreeFactor(unittest.TestCase):
    def test_e_three_way_governs(self):
        r = three_factor({"selected": "three_way", "significant_terms": "genotype:factor:factor_c"})
        s = _factorial_starter_sentence(r, r.anova_table, "Yield")
        self.assertIn("Irrigation × Variety × Spacing interaction was statistically significant", s)
        self.assertIn("cannot be interpreted independently", s)

    def test_f_single_two_way(self):
        r = three_factor({"selected": "two_way", "significant_terms": "genotype:factor"})
        s = _factorial_starter_sentence(r, r.anova_table, "Yield")
        self.assertIn("three-way interaction was not statistically significant", s)
        self.assertIn("Irrigation × Variety interaction was statistically significant", s)

    def test_g_multiple_two_ways_each_get_a_clause(self):
        r = three_factor({"selected": "two_way",
                          "significant_terms": "genotype:factor, factor:factor_c"})
        s = _factorial_starter_sentence(r, r.anova_table, "Yield")
        self.assertIn("Irrigation × Variety interaction", s)
        self.assertIn("Variety × Spacing interaction", s)
        self.assertEqual(s.count("conditional on the level of"), 2)

    def test_h_no_interactions_reports_all_three(self):
        r = three_factor({"selected": "marginal", "significant_terms": ""},
                         ps=(0.001, 0.9, 0.01))
        s = _factorial_starter_sentence(r, r.anova_table, "Yield")
        self.assertIn("No treatment interactions were statistically significant", s)
        self.assertIn("Irrigation had a significant main effect", s)
        self.assertIn("Variety did not have a significant main effect", s)
        self.assertIn("Spacing had a significant main effect", s)


class TestLabelsAndRegression(unittest.TestCase):
    def test_i_labels_dynamic_and_no_biological_terms(self):
        for basis in ({"selected": "two_way", "significant_terms": "genotype:factor"},
                      {"selected": "marginal", "significant_terms": ""}):
            r = two_factor(basis, labels=("Tillage", "Cultivar"))
            s = _factorial_starter_sentence(r, r.anova_table, "Yield")
            self.assertIn("Tillage", s)
            for w in BAN:
                self.assertNotIn(w, s)

    def test_j_non_factorial_returns_none(self):
        """Genuine genotype analysis keeps its existing sentence untouched."""
        r = two_factor({"selected": "marginal", "significant_terms": ""})
        r.design = "rcbd"
        self.assertIsNone(_factorial_starter_sentence(r, r.anova_table, "Yield"))

    def test_missing_basis_falls_back(self):
        r = two_factor(None)
        self.assertIsNone(_factorial_starter_sentence(r, r.anova_table, "Yield"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
