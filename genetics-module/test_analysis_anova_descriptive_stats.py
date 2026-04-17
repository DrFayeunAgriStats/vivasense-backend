import math
import unittest
import pandas as pd

from analysis_anova_routes import (
    compute_descriptive_stats,
    compute_cv_from_anova,
    compute_per_genotype_stats,
    classify_precision_level,
    get_cv_interpretation_flag,
    generate_anova_interpretation,
    is_genotype_effect_significant,
    is_environment_effect_significant,
    is_gxe_effect_significant,
)


class TestAnovaDescriptiveStatistics(unittest.TestCase):
    def test_compute_descriptive_stats_correct_cv(self):
        series = pd.Series([10.0, 12.0, 14.0])
        stats = compute_descriptive_stats(series)

        self.assertAlmostEqual(stats["grand_mean"], 12.0)
        self.assertAlmostEqual(stats["variance"], 4.0)
        self.assertAlmostEqual(stats["standard_deviation"], 2.0)
        self.assertAlmostEqual(stats["standard_error"], 2.0 / math.sqrt(3), places=8)
        self.assertAlmostEqual(stats["cv_percent"], (2.0 / 12.0) * 100, places=8)
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 14.0)
        self.assertEqual(stats["range"], 4.0)

    def test_compute_descriptive_stats_zero_mean(self):
        series = pd.Series([0.0, 0.0, 0.0])
        stats = compute_descriptive_stats(series)

        self.assertEqual(stats["grand_mean"], 0.0)
        self.assertEqual(stats["variance"], 0.0)
        self.assertEqual(stats["standard_deviation"], 0.0)
        self.assertEqual(stats["standard_error"], 0.0)
        self.assertIsNone(stats["cv_percent"])
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 0.0)
        self.assertEqual(stats["range"], 0.0)

    def test_compute_descriptive_stats_ignores_missing_values(self):
        series = pd.Series([1.0, float("nan"), 3.0, None, 7.0])
        stats = compute_descriptive_stats(series)

        self.assertAlmostEqual(stats["grand_mean"], 11.0 / 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 7.0)
        self.assertEqual(stats["range"], 6.0)
        self.assertIsNotNone(stats["standard_deviation"])
        self.assertIsNotNone(stats["variance"])
        self.assertIsNotNone(stats["standard_error"])

    def test_compute_descriptive_stats_small_dataset(self):
        series = pd.Series([5.0])
        stats = compute_descriptive_stats(series)

        self.assertEqual(stats["grand_mean"], 5.0)
        self.assertIsNone(stats["variance"])
        self.assertIsNone(stats["standard_deviation"])
        self.assertIsNone(stats["standard_error"])
        self.assertIsNone(stats["cv_percent"])
        self.assertEqual(stats["min"], 5.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertEqual(stats["range"], 0.0)

    def test_compute_per_genotype_stats(self):
        df = pd.DataFrame(
            {
                "genotype": ["G1", "G1", "G2", "G2", "G2"],
                "trait": [10.0, 12.0, 20.0, 22.0, 24.0],
            }
        )
        stats = compute_per_genotype_stats(df, "trait", "genotype")

        self.assertEqual(len(stats), 2)
        self.assertEqual(stats[0]["genotype"], "G1")
        self.assertEqual(stats[1]["genotype"], "G2")
        self.assertAlmostEqual(stats[0]["mean"], 11.0)
        self.assertAlmostEqual(stats[1]["mean"], 22.0)
        self.assertAlmostEqual(stats[0]["sd"], math.sqrt(2.0), places=8)
        self.assertAlmostEqual(stats[1]["sd"], math.sqrt(4.0), places=8)
        self.assertAlmostEqual(stats[0]["cv_percent"], (math.sqrt(2.0) / 11.0) * 100, places=8)
        self.assertAlmostEqual(stats[1]["cv_percent"], (math.sqrt(4.0) / 22.0) * 100, places=8)

    def test_classify_precision_level(self):
        self.assertEqual(classify_precision_level(5.0), "good")
        self.assertEqual(classify_precision_level(15.0), "moderate")
        self.assertEqual(classify_precision_level(25.0), "low")
        self.assertEqual(classify_precision_level(None), "low")

    def test_get_cv_interpretation_flag(self):
        self.assertEqual(get_cv_interpretation_flag(10.0), "cv_available")
        self.assertEqual(get_cv_interpretation_flag(None), "cv_unavailable")

    def test_is_genotype_effect_significant(self):
        # Mock anova_table
        class MockAnovaTable:
            def __init__(self, sources, p_values):
                self.source = sources
                self.p_value = p_values

        # Significant
        table_sig = MockAnovaTable(["genotype", "rep"], [0.01, 0.5])
        self.assertTrue(is_genotype_effect_significant(table_sig))

        # Not significant
        table_nonsig = MockAnovaTable(["genotype", "rep"], [0.1, 0.5])
        self.assertFalse(is_genotype_effect_significant(table_nonsig))

        # No genotype
        table_no_geno = MockAnovaTable(["rep"], [0.5])
        self.assertFalse(is_genotype_effect_significant(table_no_geno))

        # None table
        self.assertFalse(is_genotype_effect_significant(None))

    def test_is_environment_effect_significant(self):
        class MockAnovaTable:
            def __init__(self, sources, p_values):
                self.source = sources
                self.p_value = p_values

        # Significant
        table_sig = MockAnovaTable(["genotype", "environment"], [0.5, 0.01])
        self.assertTrue(is_environment_effect_significant(table_sig))

        # Not significant
        table_nonsig = MockAnovaTable(["genotype", "environment"], [0.5, 0.1])
        self.assertFalse(is_environment_effect_significant(table_nonsig))

        # No environment
        table_no_env = MockAnovaTable(["genotype"], [0.5])
        self.assertFalse(is_environment_effect_significant(table_no_env))

        # None table
        self.assertFalse(is_environment_effect_significant(None))

    def test_is_gxe_effect_significant(self):
        class MockAnovaTable:
            def __init__(self, sources, p_values):
                self.source = sources
                self.p_value = p_values

        # Significant with genotype:environment
        table_sig = MockAnovaTable(["genotype", "genotype:environment"], [0.5, 0.01])
        self.assertTrue(is_gxe_effect_significant(table_sig))

        # Significant with environment:genotype
        table_sig2 = MockAnovaTable(["genotype", "environment:genotype"], [0.5, 0.01])
        self.assertTrue(is_gxe_effect_significant(table_sig2))

        # Not significant
        table_nonsig = MockAnovaTable(["genotype", "genotype:environment"], [0.5, 0.1])
        self.assertFalse(is_gxe_effect_significant(table_nonsig))

        # No GxE
        table_no_gxe = MockAnovaTable(["genotype"], [0.5])
        self.assertFalse(is_gxe_effect_significant(table_no_gxe))

        # None table
        self.assertFalse(is_gxe_effect_significant(None))


class MockAnovaTable:
    """Minimal ANOVA table stub shared across test classes."""
    def __init__(self, sources, df_vals, ms_vals, p_vals):
        self.source = sources
        self.df = df_vals
        self.ms = ms_vals
        self.p_value = p_vals
        self.ss = [None] * len(sources)
        self.f_value = [None] * len(sources)


# ── Helpers to build minimal interpretation inputs ────────────────────────────

def _base_summary(grand_mean=100.0, cv_percent=12.0):
    return {
        "grand_mean": grand_mean,
        "cv_percent": cv_percent,
        "min": 80.0,
        "max": 120.0,
        "range": 40.0,
        "standard_error": 2.5,
    }


def _call_single(trait="Yield", genotype_significant=True, **kwargs):
    """Call generate_anova_interpretation in single-environment mode."""
    return generate_anova_interpretation(
        trait=trait,
        summary=_base_summary(),
        precision_level="moderate",
        cv_interpretation_flag="cv_available",
        genotype_significant=genotype_significant,
        environment_significant=False,
        gxe_significant=False,
        ranking_caution=False,
        selection_feasible=genotype_significant,
        mean_separation=None,
        n_genotypes=10,
        n_environments=1,
        n_reps=3,
        environment_mode="single",
        **kwargs,
    )


def _call_multi(trait="Yield", genotype_significant=True, gxe_significant=False,
                environment_significant=False):
    """Call generate_anova_interpretation in multi-environment mode."""
    return generate_anova_interpretation(
        trait=trait,
        summary=_base_summary(),
        precision_level="moderate",
        cv_interpretation_flag="cv_available",
        genotype_significant=genotype_significant,
        environment_significant=environment_significant,
        gxe_significant=gxe_significant,
        ranking_caution=gxe_significant,
        selection_feasible=genotype_significant,
        mean_separation=None,
        n_genotypes=10,
        n_environments=3,
        n_reps=3,
        environment_mode="multi",
    )


# ============================================================================
# CV from ANOVA MSE
# ============================================================================

class TestCvFromAnova(unittest.TestCase):
    def test_basic_residuals_row(self):
        """sqrt(MSE) / grand_mean * 100 with 'Residuals' source."""
        table = MockAnovaTable(
            sources=["genotype", "Residuals"],
            df_vals=[9, 20],
            ms_vals=[50.0, 25.0],   # MSE = 25 → sqrt = 5 → CV = 5/100*100 = 5%
            p_vals=[0.01, None],
        )
        cv = compute_cv_from_anova(table, grand_mean=100.0)
        self.assertAlmostEqual(cv, 5.0, places=6)

    def test_lowercase_residual_source(self):
        table = MockAnovaTable(
            sources=["genotype", "residual"],
            df_vals=[4, 10],
            ms_vals=[10.0, 4.0],    # CV = 2/50*100 = 4%
            p_vals=[0.03, None],
        )
        cv = compute_cv_from_anova(table, grand_mean=50.0)
        self.assertAlmostEqual(cv, 4.0, places=6)

    def test_error_source_name(self):
        table = MockAnovaTable(
            sources=["genotype", "Error"],
            df_vals=[3, 8],
            ms_vals=[20.0, 9.0],    # CV = 3/60*100 = 5%
            p_vals=[0.02, None],
        )
        cv = compute_cv_from_anova(table, grand_mean=60.0)
        self.assertAlmostEqual(cv, 5.0, places=6)

    def test_zero_grand_mean_returns_none(self):
        table = MockAnovaTable(
            sources=["genotype", "Residuals"],
            df_vals=[4, 8],
            ms_vals=[5.0, 2.0],
            p_vals=[0.01, None],
        )
        self.assertIsNone(compute_cv_from_anova(table, grand_mean=0.0))

    def test_none_grand_mean_returns_none(self):
        table = MockAnovaTable(["genotype", "Residuals"], [4, 8], [5.0, 2.0], [0.01, None])
        self.assertIsNone(compute_cv_from_anova(table, grand_mean=None))

    def test_none_table_returns_none(self):
        self.assertIsNone(compute_cv_from_anova(None, grand_mean=100.0))

    def test_no_error_term_returns_none(self):
        """Table with no recognisable error row → None."""
        table = MockAnovaTable(
            sources=["genotype", "rep"],
            df_vals=[4, 2],
            ms_vals=[10.0, 5.0],
            p_vals=[0.02, 0.3],
        )
        self.assertIsNone(compute_cv_from_anova(table, grand_mean=100.0))

    def test_negative_mse_returns_none(self):
        """Negative MSE (variance component issue) must return None."""
        table = MockAnovaTable(
            sources=["genotype", "Residuals"],
            df_vals=[4, 8],
            ms_vals=[5.0, -1.0],
            p_vals=[0.01, None],
        )
        self.assertIsNone(compute_cv_from_anova(table, grand_mean=100.0))


# ============================================================================
# Single-environment interpretation — section gating
# ============================================================================

class TestAnovaInterpretationSingleEnv(unittest.TestCase):

    def test_no_environment_effect_section(self):
        text = _call_single()
        self.assertNotIn("Environment Effect", text)

    def test_no_gxe_section(self):
        text = _call_single()
        self.assertNotIn("G\u00d7E Interaction", text)

    def test_no_broad_adaptation_claim(self):
        """'broad adaptation across environments' must not appear in single-env output."""
        text = _call_single()
        self.assertNotIn("broad adaptation", text.lower())

    def test_no_environment_stability_claim(self):
        """'environmental stability' must not appear in single-env output."""
        text = _call_single()
        self.assertNotIn("environmental stability", text.lower())

    def test_no_trait_placeholder_leak(self):
        """Literal '{trait}' must never appear in rendered output."""
        text = _call_single(trait="Plant Height")
        self.assertNotIn("{trait}", text)

    def test_trait_name_rendered(self):
        """Actual trait name must appear in output."""
        text = _call_single(trait="Grain Yield")
        self.assertIn("Grain Yield", text)

    def test_selection_feasible_text_present(self):
        text = _call_single(genotype_significant=True)
        self.assertIn("selection for improved Yield is feasible", text)

    def test_no_significant_genotype_text(self):
        text = _call_single(genotype_significant=False)
        self.assertIn("No significant genetic variation", text)

    def test_single_env_overview_no_multi_env_phrase(self):
        """Overview must not mention multiple environments for a single-env run."""
        text = _call_single()
        self.assertNotIn("environments with", text)


# ============================================================================
# Multi-environment interpretation — sections present
# ============================================================================

class TestAnovaInterpretationMultiEnv(unittest.TestCase):

    def test_has_environment_effect_section(self):
        text = _call_multi()
        self.assertIn("Environment Effect", text)

    def test_has_gxe_section(self):
        text = _call_multi()
        self.assertIn("G\u00d7E Interaction", text)

    def test_gxe_significant_triggers_stability_language(self):
        text = _call_multi(gxe_significant=True)
        self.assertIn("stability", text.lower())

    def test_no_trait_placeholder_leak_multi(self):
        text = _call_multi(trait="Biomass")
        self.assertNotIn("{trait}", text)

    def test_broad_adaptation_when_no_gxe(self):
        """'broad adaptation' phrase only appears in multi-env when GxE is absent."""
        text = _call_multi(gxe_significant=False, genotype_significant=True)
        self.assertIn("broad adaptation", text)

    def test_broad_adaptation_absent_when_gxe_significant(self):
        text = _call_multi(gxe_significant=True, genotype_significant=True)
        self.assertNotIn("broad adaptation", text)


# ============================================================================
# CV precision classification
# ============================================================================

class TestClassifyPrecisionLevelExtended(unittest.TestCase):
    def test_boundary_exactly_10(self):
        """CV == 10.0 → 'moderate' (boundary is < 10 for 'good')."""
        self.assertEqual(classify_precision_level(10.0), "moderate")

    def test_boundary_exactly_20(self):
        """CV == 20.0 → 'moderate' (boundary is < 20 for 'moderate')."""
        self.assertEqual(classify_precision_level(20.0), "moderate")

    def test_cv_just_above_20(self):
        self.assertEqual(classify_precision_level(20.1), "low")

    def test_cv_just_below_10(self):
        self.assertEqual(classify_precision_level(9.9), "good")


# ============================================================================
# Descriptive stats completeness
# ============================================================================

class TestDescriptiveStatsCompleteness(unittest.TestCase):
    def test_all_expected_keys_present(self):
        series = pd.Series([10.0, 20.0, 30.0, 40.0])
        stats = compute_descriptive_stats(series)
        for key in ("grand_mean", "standard_deviation", "variance",
                    "standard_error", "cv_percent", "min", "max", "range"):
            self.assertIn(key, stats, f"Key '{key}' missing from descriptive stats")

    def test_cv_is_float_not_none_for_valid_series(self):
        series = pd.Series([10.0, 20.0, 30.0])
        stats = compute_descriptive_stats(series)
        self.assertIsNotNone(stats["cv_percent"])
        self.assertIsInstance(stats["cv_percent"], float)

    def test_min_max_range_correct(self):
        series = pd.Series([5.0, 15.0, 25.0])
        stats = compute_descriptive_stats(series)
        self.assertEqual(stats["min"], 5.0)
        self.assertEqual(stats["max"], 25.0)
        self.assertEqual(stats["range"], 20.0)


if __name__ == "__main__":
    unittest.main()
