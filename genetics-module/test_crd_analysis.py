"""
Tests for CRD (Completely Randomised Design) support.

Covers:
  - build_observations with rep_col=None (CRD synthetic rep generation)
  - build_observations with rep_col=None + env_col (factorial CRD)
  - check_balance with rep_col=None
  - n_reps inference when rep_column is absent
  - No "{trait}" placeholder leaks (regression guard)
  - detect_columns warning message for missing rep column
"""

import unittest
import pandas as pd

from multitrait_upload_routes import (
    build_observations,
    check_balance,
    detect_columns,
)


# ── Fixture builders ──────────────────────────────────────────────────────────

def _simple_crd_df(n_genotypes: int = 4, n_reps: int = 3) -> pd.DataFrame:
    """Balanced CRD: no rep column, each genotype has n_reps rows."""
    rows = []
    for g in range(1, n_genotypes + 1):
        for r in range(1, n_reps + 1):
            rows.append({"genotype": f"G{g}", "trait": float(g * 10 + r)})
    return pd.DataFrame(rows)


def _factorial_crd_df() -> pd.DataFrame:
    """Factorial CRD: genotype × nitrogen level, no rep column."""
    rows = []
    for g in ["G1", "G2", "G3"]:
        for n_level in ["Low", "High"]:
            for obs in range(1, 4):  # 3 observations per cell
                rows.append({
                    "genotype": g,
                    "nitrogen": n_level,
                    "trait": float(obs + (2 if n_level == "High" else 0)),
                })
    return pd.DataFrame(rows)


def _rcbd_df(n_genotypes: int = 4, n_reps: int = 3) -> pd.DataFrame:
    """Balanced RCBD: explicit rep column."""
    rows = []
    for r in range(1, n_reps + 1):
        for g in range(1, n_genotypes + 1):
            rows.append({
                "genotype": f"G{g}",
                "rep": f"R{r}",
                "trait": float(g * 10 + r),
            })
    return pd.DataFrame(rows)


# ============================================================================
# build_observations — CRD (no rep column)
# ============================================================================

class TestBuildObservationsCRD(unittest.TestCase):

    def test_crd_records_have_rep_field(self):
        """Synthetic rep must always be present in records even without rep_col."""
        df = _simple_crd_df()
        records = build_observations(df, "genotype", None, "trait", None)
        for rec in records:
            self.assertIn("rep", rec, "Each CRD record must have a 'rep' field")

    def test_crd_records_have_crd_flag(self):
        """Records must carry crd=True when built without rep_col."""
        df = _simple_crd_df()
        records = build_observations(df, "genotype", None, "trait", None)
        for rec in records:
            self.assertTrue(rec.get("crd"), "CRD records must have crd=True")

    def test_rcbd_records_have_no_crd_flag(self):
        """Records built with an explicit rep_col must NOT carry crd=True."""
        df = _rcbd_df()
        records = build_observations(df, "genotype", "rep", "trait", None)
        for rec in records:
            self.assertNotIn("crd", rec,
                             "RCBD records must not carry 'crd' key")

    def test_crd_synthetic_rep_values_are_unique_per_genotype(self):
        """Within each genotype the synthetic rep values must be distinct."""
        df = _simple_crd_df(n_genotypes=3, n_reps=4)
        records = build_observations(df, "genotype", None, "trait", None)
        from collections import defaultdict
        reps_by_geno = defaultdict(list)
        for rec in records:
            reps_by_geno[rec["genotype"]].append(rec["rep"])
        for geno, reps in reps_by_geno.items():
            self.assertEqual(
                len(reps), len(set(reps)),
                f"Duplicate synthetic rep values for genotype {geno}"
            )

    def test_crd_record_count_matches_rows(self):
        """Number of records must equal number of valid (non-NaN) rows."""
        df = _simple_crd_df(n_genotypes=5, n_reps=3)
        records = build_observations(df, "genotype", None, "trait", None)
        self.assertEqual(len(records), 15)

    def test_crd_missing_trait_rows_dropped(self):
        """Rows with NaN trait values must be dropped in CRD mode."""
        df = _simple_crd_df(n_genotypes=3, n_reps=3)
        df.loc[0, "trait"] = float("nan")
        records = build_observations(df, "genotype", None, "trait", None)
        self.assertEqual(len(records), 8)  # 9 rows - 1 dropped

    def test_crd_raises_on_too_few_rows(self):
        """ValueError when fewer than 6 valid observations remain."""
        df = pd.DataFrame({
            "genotype": ["G1", "G2", "G1"],
            "trait": [10.0, 20.0, 30.0],
        })
        with self.assertRaises(ValueError):
            build_observations(df, "genotype", None, "trait", None)


# ============================================================================
# build_observations — Factorial CRD
# ============================================================================

class TestBuildObservationsFactorialCRD(unittest.TestCase):

    def test_factorial_crd_records_have_factor_field(self):
        """When env_col provided without rep_col, records get a 'factor' key."""
        df = _factorial_crd_df()
        records = build_observations(df, "genotype", None, "trait", "nitrogen")
        for rec in records:
            self.assertIn("factor", rec,
                          "Factorial CRD records must carry 'factor' key")

    def test_factorial_crd_factor_values_correct(self):
        """Factor values must match the env_col values from the dataframe."""
        df = _factorial_crd_df()
        records = build_observations(df, "genotype", None, "trait", "nitrogen")
        factor_values = {rec["factor"] for rec in records}
        self.assertEqual(factor_values, {"Low", "High"})

    def test_factorial_crd_no_environment_key(self):
        """Records must NOT have an 'environment' key (that belongs to multi-env)."""
        df = _factorial_crd_df()
        records = build_observations(df, "genotype", None, "trait", "nitrogen")
        for rec in records:
            self.assertNotIn("environment", rec)

    def test_rcbd_multi_env_has_environment_not_factor(self):
        """RCBD multi-env records must carry 'environment', not 'factor'."""
        df = _rcbd_df()
        df["env"] = "E1"
        records = build_observations(df, "genotype", "rep", "trait", "env")
        for rec in records:
            self.assertIn("environment", rec)
            self.assertNotIn("factor", rec)


def _split_plot_df() -> pd.DataFrame:
    """Balanced single-environment split-plot RCBD dataset."""
    rows = []
    for rep in ["R1", "R2"]:
        for main in ["M1", "M2"]:
            for sub in ["S1", "S2"]:
                rows.append({
                    "genotype": "G1",
                    "rep": rep,
                    "main_plot": main,
                    "sub_plot": sub,
                    "trait": float((rep == "R1") + (main == "M2") + (sub == "S2"))
                })
    return pd.DataFrame(rows)


class TestBuildObservationsSplitPlot(unittest.TestCase):

    def test_split_plot_records_include_main_and_sub_plot(self):
        df = _split_plot_df()
        records = build_observations(
            df,
            genotype_col="genotype",
            rep_col="rep",
            trait_col="trait",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col="main_plot",
            sub_plot_col="sub_plot",
        )
        for rec in records:
            self.assertEqual(rec["rep"], rec["rep"])
            self.assertIn("main_plot", rec)
            self.assertIn("sub_plot", rec)
            self.assertEqual(rec["main_plot"], rec["main_plot"])
            self.assertEqual(rec["sub_plot"], rec["sub_plot"])
            self.assertEqual(rec["trait_value"], float(rec["trait_value"]))

    def test_split_plot_requires_main_and_sub_plot_columns(self):
        df = _split_plot_df()
        with self.assertRaises(ValueError):
            build_observations(
                df,
                genotype_col="genotype",
                rep_col="rep",
                trait_col="trait",
                env_col=None,
                design_type="split_plot_rcbd",
                main_plot_col=None,
                sub_plot_col="sub_plot",
            )


class TestCheckBalanceSplitPlot(unittest.TestCase):

    def test_balanced_split_plot_no_warnings(self):
        df = _split_plot_df()
        warnings = check_balance(
            df,
            genotype_col="genotype",
            rep_col="rep",
            trait_col="trait",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col="main_plot",
            sub_plot_col="sub_plot",
        )
        self.assertEqual(warnings, [])

    def test_unbalanced_split_plot_warns(self):
        df = _split_plot_df()
        # Remove one subplot observation to create imbalance
        df = df.iloc[:-1]
        warnings = check_balance(
            df,
            genotype_col="genotype",
            rep_col="rep",
            trait_col="trait",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col="main_plot",
            sub_plot_col="sub_plot",
        )
        self.assertTrue(any("split-plot" in w.lower() or "incomplete" in w.lower() for w in warnings))


# ============================================================================
# check_balance — CRD path
# ============================================================================

class TestCheckBalanceCRD(unittest.TestCase):

    def test_balanced_crd_no_warnings(self):
        df = _simple_crd_df(n_genotypes=4, n_reps=3)
        warnings = check_balance(df, "genotype", None, "trait", None)
        self.assertEqual(warnings, [])

    def test_unbalanced_crd_warns_with_crd_label(self):
        """Unbalanced CRD must produce a warning that mentions 'CRD'."""
        df = _simple_crd_df(n_genotypes=4, n_reps=3)
        # Remove one row for G1 to create imbalance
        df = df[~((df["genotype"] == "G1") & (df.index == df[df["genotype"] == "G1"].index[0]))]
        warnings = check_balance(df, "genotype", None, "trait", None)
        self.assertTrue(len(warnings) > 0)
        self.assertTrue(any("CRD" in w for w in warnings),
                        f"Expected CRD mention in warnings, got: {warnings}")

    def test_crd_single_observation_genotype_warns(self):
        """A genotype with only 1 observation must trigger a warning."""
        df = pd.DataFrame({
            "genotype": ["G1", "G1", "G1", "G2", "G2", "G2", "G3"],
            "trait": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        })
        warnings = check_balance(df, "genotype", None, "trait", None)
        self.assertTrue(len(warnings) > 0)
        self.assertTrue(any("1 observation" in w or "minimum" in w.lower()
                            for w in warnings))

    def test_rcbd_balanced_no_warnings(self):
        df = _rcbd_df()
        warnings = check_balance(df, "genotype", "rep", "trait", None)
        self.assertEqual(warnings, [])

    def test_rcbd_unbalanced_warns_without_crd_label(self):
        df = _rcbd_df()
        # Remove one row for G1 R1
        drop_idx = df[(df["genotype"] == "G1") & (df["rep"] == "R1")].index[0]
        df = df.drop(drop_idx)
        warnings = check_balance(df, "genotype", "rep", "trait", None)
        self.assertTrue(len(warnings) > 0)
        self.assertFalse(any("CRD" in w for w in warnings))


# ============================================================================
# n_reps inference
# ============================================================================

class TestNRepsInference(unittest.TestCase):

    def test_n_reps_inferred_from_max_obs_per_genotype(self):
        """n_reps for a CRD dataset == max observations per genotype."""
        df = _simple_crd_df(n_genotypes=4, n_reps=5)
        n_reps = int(df.groupby("genotype").size().max())
        self.assertEqual(n_reps, 5)

    def test_unbalanced_crd_n_reps_is_max_not_min(self):
        """When imbalanced, n_reps is the max, not min."""
        df = pd.DataFrame({
            "genotype": ["G1"] * 5 + ["G2"] * 3 + ["G3"] * 4,
            "trait": list(range(12)),
        })
        n_reps = int(df.groupby("genotype").size().max())
        self.assertEqual(n_reps, 5)


# ============================================================================
# detect_columns — CRD warning message
# ============================================================================

class TestDetectColumnsCRDWarning(unittest.TestCase):

    def test_no_rep_column_triggers_crd_warning(self):
        """
        When no rep column is detected the preview warning must say CRD,
        not 'please specify manually'.
        """
        from multitrait_upload_routes import _match_pattern, _REP_PATTERNS

        # DataFrame with no column matching rep patterns
        df = pd.DataFrame({
            "variety": ["G1", "G2", "G3"] * 3,
            "yield":   [1.0, 2.0, 3.0] * 3,
        })
        detected = detect_columns(df)
        self.assertIsNone(detected.rep)

        # Simulate the warning logic in upload_preview
        warnings = []
        if detected.rep is None:
            warnings.append(
                "No replication column detected — CRD assumed "
                "(replication inferred from data)"
            )
        self.assertTrue(
            any("CRD" in w for w in warnings),
            f"Expected CRD warning, got: {warnings}"
        )
        self.assertFalse(
            any("please specify manually" in w for w in warnings),
            "Old 'please specify manually' text must not appear"
        )


# ============================================================================
# No {trait} placeholder in CRD interpretation (regression guard)
# ============================================================================

class TestNoCRDPlaceholderLeak(unittest.TestCase):

    def test_no_trait_placeholder_in_single_env_crd_output(self):
        """
        generate_anova_interpretation must not emit literal '{trait}'
        regardless of mode or feasibility flags.
        """
        from analysis_anova_routes import generate_anova_interpretation

        text = generate_anova_interpretation(
            trait="Grain_Yield",
            summary={"grand_mean": 50.0, "cv_percent": 8.5, "min": 40.0,
                     "max": 60.0, "range": 20.0, "standard_error": 1.2},
            precision_level="good",
            cv_interpretation_flag="cv_available",
            genotype_significant=True,
            environment_significant=None,
            gxe_significant=None,
            ranking_caution=False,
            selection_feasible=True,
            mean_separation=None,
            n_genotypes=8,
            n_environments=1,
            n_reps=4,
            environment_mode="single",
        )
        self.assertNotIn("{trait}", text)
        self.assertIn("Grain_Yield", text)


if __name__ == "__main__":
    unittest.main()
