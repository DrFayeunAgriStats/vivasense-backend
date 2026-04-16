import math
import unittest
import pandas as pd

from analysis_anova_routes import (
    compute_descriptive_stats,
    compute_per_genotype_stats,
    classify_precision_level,
    get_cv_interpretation_flag,
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


if __name__ == "__main__":
    unittest.main()
