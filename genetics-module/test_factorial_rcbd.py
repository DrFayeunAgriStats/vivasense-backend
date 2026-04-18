"""
Tests for Factorial RCBD support.

Covers:
  - build_observations with rep_col + factor_col (factorial RCBD)
  - Records carry "factor" key, not "environment" or "crd"
  - check_balance for balanced and unbalanced factorial RCBD
  - Simple RCBD still works when factor_col=None
  - Factorial CRD still works (factor via CRD path)
  - factor_col takes priority over env_col for RCBD datasets
"""

import unittest
import pandas as pd
from collections import defaultdict

from multitrait_upload_routes import (
    build_observations,
    check_balance,
)


# ── Fixture builders ──────────────────────────────────────────────────────────

def _factorial_rcbd_df(
    n_genotypes: int = 4,
    n_reps: int = 3,
    factor_levels: int = 2,
) -> pd.DataFrame:
    """Balanced factorial RCBD: genotype × factor, replicated n_reps times."""
    rows = []
    for r in range(1, n_reps + 1):
        for g in range(1, n_genotypes + 1):
            for f in range(1, factor_levels + 1):
                rows.append({
                    "genotype": f"G{g}",
                    "rep":      f"R{r}",
                    "nitrogen": f"N{f}",
                    "trait":    float(g * 10 + r + f),
                })
    return pd.DataFrame(rows)


def _simple_rcbd_df(n_genotypes: int = 4, n_reps: int = 3) -> pd.DataFrame:
    """Simple RCBD: no factor column."""
    rows = []
    for r in range(1, n_reps + 1):
        for g in range(1, n_genotypes + 1):
            rows.append({
                "genotype": f"G{g}",
                "rep":      f"R{r}",
                "trait":    float(g * 10 + r),
            })
    return pd.DataFrame(rows)


def _factorial_crd_df(
    n_genotypes: int = 3,
    n_obs: int = 3,
    factor_levels: int = 2,
) -> pd.DataFrame:
    """Factorial CRD: no rep column, genotype × factor."""
    rows = []
    for g in range(1, n_genotypes + 1):
        for f in range(1, factor_levels + 1):
            for _ in range(1, n_obs + 1):
                rows.append({
                    "genotype": f"G{g}",
                    "nitrogen": f"N{f}",
                    "trait":    float(g * 10 + f),
                })
    return pd.DataFrame(rows)


# ============================================================================
# build_observations — Factorial RCBD
# ============================================================================

class TestBuildObservationsFactorialRCBD(unittest.TestCase):

    def test_records_have_factor_key(self):
        """Factorial RCBD records must carry a 'factor' key."""
        df = _factorial_rcbd_df()
        records = build_observations(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        for rec in records:
            self.assertIn("factor", rec, "Factorial RCBD record must have 'factor' key")

    def test_records_have_rep_key(self):
        """Factorial RCBD records must carry the real rep, not a synthetic one."""
        df = _factorial_rcbd_df()
        records = build_observations(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        for rec in records:
            self.assertIn("rep", rec)
            # Synthetic rep values are "R1", "R2", etc.; real reps from data
            # should also match that pattern here but importantly no "crd" flag
            self.assertNotIn("crd", rec)

    def test_records_no_crd_flag(self):
        """Factorial RCBD records must NOT carry crd=True."""
        df = _factorial_rcbd_df()
        records = build_observations(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        for rec in records:
            self.assertNotIn("crd", rec)

    def test_records_no_environment_key(self):
        """Factorial RCBD records must NOT carry 'environment' key."""
        df = _factorial_rcbd_df()
        records = build_observations(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        for rec in records:
            self.assertNotIn("environment", rec)

    def test_factor_values_match_column(self):
        """Factor values in records must match values from the factor column."""
        df = _factorial_rcbd_df(factor_levels=3)
        records = build_observations(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        factor_vals = {rec["factor"] for rec in records}
        self.assertEqual(factor_vals, {"N1", "N2", "N3"})

    def test_record_count_correct(self):
        """Total records must equal n_genotypes × n_reps × n_factor_levels."""
        df = _factorial_rcbd_df(n_genotypes=3, n_reps=4, factor_levels=2)
        records = build_observations(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        self.assertEqual(len(records), 3 * 4 * 2)

    def test_factor_col_takes_priority_over_env_col(self):
        """When both env_col and factor_col provided for RCBD, factor_col wins."""
        df = _factorial_rcbd_df()
        df["env"] = "E1"  # add an env column too
        records = build_observations(df, "genotype", "rep", "trait", "env", factor_col="nitrogen")
        # Should have "factor", not "environment"
        for rec in records:
            self.assertIn("factor", rec)
            self.assertNotIn("environment", rec)

    def test_missing_trait_rows_dropped(self):
        """NaN trait rows must be dropped in factorial RCBD mode."""
        df = _factorial_rcbd_df(n_genotypes=2, n_reps=3, factor_levels=2)
        df.loc[0, "trait"] = float("nan")
        records = build_observations(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        self.assertEqual(len(records), 2 * 3 * 2 - 1)


# ============================================================================
# build_observations — Simple RCBD unchanged
# ============================================================================

class TestBuildObservationsSimpleRCBDUnchanged(unittest.TestCase):

    def test_simple_rcbd_no_factor_key(self):
        """Simple RCBD without factor_col must not have 'factor' key."""
        df = _simple_rcbd_df()
        records = build_observations(df, "genotype", "rep", "trait", None)
        for rec in records:
            self.assertNotIn("factor", rec)

    def test_simple_rcbd_has_rep_no_crd(self):
        """Simple RCBD records carry rep and no crd flag."""
        df = _simple_rcbd_df()
        records = build_observations(df, "genotype", "rep", "trait", None)
        for rec in records:
            self.assertIn("rep", rec)
            self.assertNotIn("crd", rec)

    def test_simple_rcbd_record_count(self):
        df = _simple_rcbd_df(n_genotypes=5, n_reps=3)
        records = build_observations(df, "genotype", "rep", "trait", None)
        self.assertEqual(len(records), 15)


# ============================================================================
# build_observations — Factorial CRD unchanged
# ============================================================================

class TestBuildObservationsFactorialCRDUnchanged(unittest.TestCase):

    def test_factorial_crd_has_factor_and_crd(self):
        """Factorial CRD (no rep_col, with factor via env_col) keeps existing behaviour."""
        df = _factorial_crd_df()
        records = build_observations(df, "genotype", None, "trait", "nitrogen")
        for rec in records:
            self.assertIn("factor", rec)
            self.assertTrue(rec.get("crd"), "CRD records must have crd=True")

    def test_factorial_crd_via_explicit_factor_col(self):
        """Factorial CRD via explicit factor_col (no rep_col) also works."""
        df = _factorial_crd_df()
        records = build_observations(df, "genotype", None, "trait", None, factor_col="nitrogen")
        for rec in records:
            self.assertIn("factor", rec)
            self.assertTrue(rec.get("crd"))


# ============================================================================
# check_balance — Factorial RCBD
# ============================================================================

class TestCheckBalanceFactorialRCBD(unittest.TestCase):

    def test_balanced_factorial_rcbd_no_warnings(self):
        df = _factorial_rcbd_df(n_genotypes=4, n_reps=3, factor_levels=2)
        warnings = check_balance(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        self.assertEqual(warnings, [])

    def test_unbalanced_cell_warns(self):
        """A missing genotype×factor combination must trigger a warning."""
        df = _factorial_rcbd_df(n_genotypes=3, n_reps=3, factor_levels=2)
        # Drop all rows for G1×N2
        df = df[~((df["genotype"] == "G1") & (df["nitrogen"] == "N2"))]
        warnings = check_balance(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        self.assertTrue(len(warnings) > 0)
        self.assertTrue(any("factorial" in w.lower() or "factor" in w.lower()
                            for w in warnings))

    def test_unequal_reps_per_cell_warns(self):
        """Unequal number of reps per genotype×factor cell must warn."""
        df = _factorial_rcbd_df(n_genotypes=3, n_reps=3, factor_levels=2)
        # Remove one rep for G1×N1 to create unequal cell sizes
        idx = df[(df["genotype"] == "G1") & (df["nitrogen"] == "N1")].index[0]
        df = df.drop(idx)
        warnings = check_balance(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        self.assertTrue(len(warnings) > 0)

    def test_simple_rcbd_warnings_unaffected(self):
        """Simple RCBD balance check must still work when factor_col=None."""
        df = _simple_rcbd_df()
        # Remove one row to create imbalance
        idx = df[(df["genotype"] == "G1") & (df["rep"] == "R1")].index[0]
        df = df.drop(idx)
        warnings = check_balance(df, "genotype", "rep", "trait", None)
        self.assertTrue(len(warnings) > 0)
        # Must NOT mention "factorial"
        self.assertFalse(any("factorial" in w.lower() for w in warnings))


# ============================================================================
# Design label in R flags (Python-level regression guard via records structure)
# ============================================================================

class TestFactorialRCBDRecordStructure(unittest.TestCase):

    def test_all_required_fields_present(self):
        """Every factorial RCBD record must have genotype, rep, factor, trait_value."""
        df = _factorial_rcbd_df()
        records = build_observations(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        for rec in records:
            for field in ("genotype", "rep", "factor", "trait_value"):
                self.assertIn(field, rec, f"Missing field '{field}' in factorial RCBD record")

    def test_trait_values_are_floats(self):
        """trait_value must be a float in every record."""
        df = _factorial_rcbd_df()
        records = build_observations(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        for rec in records:
            self.assertIsInstance(rec["trait_value"], float)

    def test_distinct_factor_levels(self):
        """All factor levels from the data must appear in the records."""
        df = _factorial_rcbd_df(factor_levels=3)
        records = build_observations(df, "genotype", "rep", "trait", None, factor_col="nitrogen")
        factor_vals = {rec["factor"] for rec in records}
        self.assertEqual(len(factor_vals), 3)


if __name__ == "__main__":
    unittest.main()
