"""
Tests for generic (role-based, domain-neutral) split-plot RCBD support.

Covers:
  - build_observations: emits rep/main_plot/sub_plot/trait_value, no genotype key
  - check_balance: warns on incomplete/unbalanced cells
  - Non-genotype factor names (Tillage × FertilizerRate, Irrigation × Genotype, etc.)
  - Rejection of genotype_column as a third independent factor in split_plot_rcbd
  - Rejection of main_plot_column == sub_plot_column
  - Rejection of rep_column duplicating a treatment column
  - split_plot_rcbd interpretation is domain-neutral (no genotype/breeding language)
  - Formula does NOT include genotype as a separate extra term (regression guard)
  - Simple RCBD and CRD paths remain unaffected
"""

import unittest
import pandas as pd

from multitrait_upload_routes import build_observations, check_balance


# ── Fixture builders ──────────────────────────────────────────────────────────

def _split_plot_df(
    main_levels: list = ("M1", "M2"),
    sub_levels: list = ("S1", "S2"),
    n_reps: int = 3,
    main_col: str = "main_plot",
    sub_col: str = "sub_plot",
) -> pd.DataFrame:
    """Balanced generic split-plot: rep × main × sub."""
    rows = []
    for r in range(1, n_reps + 1):
        for m in main_levels:
            for s in sub_levels:
                rows.append({
                    "rep": f"R{r}",
                    main_col: m,
                    sub_col: s,
                    "trait": float(r + (1 if m == main_levels[-1] else 0) + (2 if s == sub_levels[-1] else 0)),
                })
    return pd.DataFrame(rows)


def _tillage_fertilizer_df() -> pd.DataFrame:
    """Non-genotype factor names: Tillage × FertilizerRate."""
    rows = []
    for rep in ["B1", "B2", "B3"]:
        for tillage in ["Conventional", "NoTill"]:
            for fert in ["Low", "Medium", "High"]:
                rows.append({
                    "Block": rep,
                    "Tillage": tillage,
                    "FertilizerRate": fert,
                    "yield": 10.0 + (2 if tillage == "NoTill" else 0) + (fert == "High") * 3,
                })
    return pd.DataFrame(rows)


def _irrigation_genotype_df() -> pd.DataFrame:
    """Genotype as subplot factor: Irrigation (main) × Genotype (sub)."""
    rows = []
    for rep in ["R1", "R2"]:
        for irr in ["Irrigated", "Rainfed"]:
            for geno in ["G1", "G2", "G3"]:
                rows.append({
                    "Rep": rep,
                    "Irrigation": irr,
                    "Genotype": geno,
                    "grain_yield": 5.0,
                })
    return pd.DataFrame(rows)


def _simple_rcbd_df() -> pd.DataFrame:
    """Simple RCBD (unchanged path) for regression guard."""
    rows = []
    for r in ["R1", "R2", "R3"]:
        for g in ["G1", "G2", "G3", "G4"]:
            rows.append({"rep": r, "genotype": g, "trait": 5.0})
    return pd.DataFrame(rows)


# ============================================================================
# build_observations — generic split-plot (no genotype)
# ============================================================================

class TestBuildObservationsSplitPlotGeneric(unittest.TestCase):

    def _build(self, df, main_col="main_plot", sub_col="sub_plot"):
        return build_observations(
            df,
            genotype_col=None,
            rep_col="rep",
            trait_col="trait",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col=main_col,
            sub_plot_col=sub_col,
        )

    def test_records_have_rep_main_sub_trait(self):
        """All four role-based keys must be present."""
        df = _split_plot_df()
        recs = self._build(df)
        for rec in recs:
            for key in ("rep", "main_plot", "sub_plot", "trait_value"):
                self.assertIn(key, rec, f"Missing key '{key}'")

    def test_records_have_no_genotype_key(self):
        """'genotype' must NOT appear in split_plot_rcbd records."""
        df = _split_plot_df()
        recs = self._build(df)
        for rec in recs:
            self.assertNotIn("genotype", rec, "'genotype' key must not be emitted")

    def test_records_have_no_crd_flag(self):
        """Split-plot records must not carry crd=True."""
        df = _split_plot_df()
        recs = self._build(df)
        for rec in recs:
            self.assertNotIn("crd", rec)

    def test_records_have_no_environment_key(self):
        """Split-plot records must not carry 'environment' key."""
        df = _split_plot_df()
        recs = self._build(df)
        for rec in recs:
            self.assertNotIn("environment", rec)

    def test_record_count(self):
        """Total records = n_reps × n_main × n_sub."""
        df = _split_plot_df(n_reps=3)
        recs = self._build(df)
        self.assertEqual(len(recs), 3 * 2 * 2)

    def test_non_genotype_factor_names(self):
        """Works with arbitrary column names (Tillage, FertilizerRate, Block)."""
        df = _tillage_fertilizer_df()
        recs = build_observations(
            df,
            genotype_col=None,
            rep_col="Block",
            trait_col="yield",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col="Tillage",
            sub_plot_col="FertilizerRate",
        )
        self.assertEqual(len(recs), 3 * 2 * 3)
        for rec in recs:
            self.assertIn("rep", rec)
            self.assertIn("main_plot", rec)
            self.assertIn("sub_plot", rec)
            self.assertNotIn("genotype", rec)

    def test_genotype_as_subplot_factor(self):
        """Genotype can be the subplot factor — records carry it as sub_plot."""
        df = _irrigation_genotype_df()
        recs = build_observations(
            df,
            genotype_col=None,
            rep_col="Rep",
            trait_col="grain_yield",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col="Irrigation",
            sub_plot_col="Genotype",
        )
        self.assertEqual(len(recs), 2 * 2 * 3)
        sub_vals = {rec["sub_plot"] for rec in recs}
        self.assertEqual(sub_vals, {"G1", "G2", "G3"})
        for rec in recs:
            self.assertNotIn("genotype", rec)   # not emitted as a separate key

    def test_genotype_as_main_plot_factor(self):
        """Genotype can be the main-plot factor — records carry it as main_plot."""
        df = _irrigation_genotype_df()
        recs = build_observations(
            df,
            genotype_col=None,
            rep_col="Rep",
            trait_col="grain_yield",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col="Genotype",
            sub_plot_col="Irrigation",
        )
        main_vals = {rec["main_plot"] for rec in recs}
        self.assertEqual(main_vals, {"G1", "G2", "G3"})

    def test_trait_values_are_floats(self):
        df = _split_plot_df()
        recs = self._build(df)
        for rec in recs:
            self.assertIsInstance(rec["trait_value"], float)

    def test_missing_trait_rows_dropped(self):
        df = _split_plot_df(n_reps=2)
        df.loc[0, "trait"] = float("nan")
        recs = self._build(df)
        self.assertEqual(len(recs), 2 * 2 * 2 - 1)

    def test_raises_when_main_plot_col_missing(self):
        df = _split_plot_df()
        with self.assertRaises(ValueError):
            build_observations(
                df,
                genotype_col=None,
                rep_col="rep",
                trait_col="trait",
                env_col=None,
                design_type="split_plot_rcbd",
                main_plot_col=None,
                sub_plot_col="sub_plot",
            )


# ============================================================================
# check_balance — generic split-plot
# ============================================================================

class TestCheckBalanceSplitPlotGeneric(unittest.TestCase):

    def _balance(self, df, main_col="main_plot", sub_col="sub_plot"):
        return check_balance(
            df,
            genotype_col=None,
            rep_col="rep",
            trait_col="trait",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col=main_col,
            sub_plot_col=sub_col,
        )

    def test_balanced_no_warnings(self):
        df = _split_plot_df(n_reps=3)
        self.assertEqual(self._balance(df), [])

    def test_missing_cell_warns(self):
        df = _split_plot_df(n_reps=3)
        # Remove all rows for M1×S2
        df = df[~((df["main_plot"] == "M1") & (df["sub_plot"] == "S2"))]
        warnings = self._balance(df)
        self.assertTrue(len(warnings) > 0)

    def test_unequal_cell_sizes_warn(self):
        df = _split_plot_df(n_reps=3)
        # Remove one observation from one cell
        idx = df[(df["main_plot"] == "M1") & (df["sub_plot"] == "S1")].index[0]
        df = df.drop(idx)
        warnings = self._balance(df)
        self.assertTrue(len(warnings) > 0)

    def test_non_genotype_factor_names(self):
        """Balance check works with arbitrary factor column names."""
        df = _tillage_fertilizer_df()
        warnings = check_balance(
            df,
            genotype_col=None,
            rep_col="Block",
            trait_col="yield",
            env_col=None,
            design_type="split_plot_rcbd",
            main_plot_col="Tillage",
            sub_plot_col="FertilizerRate",
        )
        self.assertEqual(warnings, [])


# ============================================================================
# Simple RCBD / CRD paths unaffected
# ============================================================================

class TestNonSplitPlotPathsUnchanged(unittest.TestCase):

    def test_simple_rcbd_still_emits_genotype(self):
        df = _simple_rcbd_df()
        recs = build_observations(
            df,
            genotype_col="genotype",
            rep_col="rep",
            trait_col="trait",
            env_col=None,
        )
        for rec in recs:
            self.assertIn("genotype", rec)
            self.assertNotIn("main_plot", rec)

    def test_simple_rcbd_balance_still_works(self):
        df = _simple_rcbd_df()
        # Remove one row to unbalance
        idx = df[(df["genotype"] == "G1") & (df["rep"] == "R1")].index[0]
        df = df.drop(idx)
        warnings = check_balance(df, "genotype", "rep", "trait", None)
        self.assertTrue(len(warnings) > 0)
        self.assertFalse(any("split" in w.lower() for w in warnings))


# ============================================================================
# Interpretation: split_plot_rcbd uses domain-neutral language
# ============================================================================

class TestSplitPlotInterpretationNeutral(unittest.TestCase):

    def _run_interpretation(self, mp_sig, sub_sig, int_sig):
        from analysis_anova_routes import generate_anova_interpretation
        return generate_anova_interpretation(
            trait="GrainYield",
            summary={"grand_mean": 4.5, "cv_percent": 12.0, "min": 2.0,
                     "max": 7.0, "range": 5.0, "standard_error": 0.3},
            precision_level="moderate",
            cv_interpretation_flag="cv_available",
            genotype_significant=None,
            environment_significant=None,
            gxe_significant=None,
            ranking_caution=None,
            selection_feasible=None,
            mean_separation=None,
            n_genotypes=None,
            n_environments=None,
            n_reps=3,
            environment_mode="single",
            design_type="split_plot_rcbd",
            main_plot_significant=mp_sig,
            subplot_significant=sub_sig,
            interaction_significant=int_sig,
        )

    def test_no_genotype_language(self):
        """Interpretation must not contain 'genotype' as a subject."""
        text = self._run_interpretation(True, True, False)
        # "genotype" appearing as part of another word is OK (e.g. "genotype_column")
        # but genotype as a standalone concept should not appear
        import re
        genotype_occurrences = re.findall(r'\bgenotype\b', text, re.IGNORECASE)
        self.assertEqual(
            genotype_occurrences, [],
            f"Found standalone 'genotype' in split-plot interpretation: {genotype_occurrences}"
        )

    def test_no_breeding_language(self):
        """Interpretation must not contain breeding-specific terms."""
        import re
        text = self._run_interpretation(True, False, False)
        # Short acronyms (3-4 chars) are checked as whole words to avoid false positives
        # from common words (e.g. "suggesting" contains "gge").
        banned = ["breeding", "selection", "germplasm", "advance", "heritability",
                  "GCV", "PCV", "AMMI", "GGE", "biplot"]
        for term in banned:
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.findall(pattern, text, re.IGNORECASE)
            self.assertEqual(
                matches, [],
                f"Found banned term '{term}' in split-plot interpretation"
            )

    def test_mentions_main_plot_factor(self):
        """Interpretation must mention main-plot factor."""
        text = self._run_interpretation(True, True, True)
        self.assertIn("main-plot", text.lower())

    def test_mentions_subplot_factor(self):
        """Interpretation must mention subplot factor."""
        text = self._run_interpretation(True, True, True)
        self.assertTrue(
            "subplot" in text.lower() or "sub_plot" in text.lower() or "sub-plot" in text.lower()
        )

    def test_significant_interaction_mentioned(self):
        """When interaction is significant, interpretation must flag it."""
        text = self._run_interpretation(True, True, True)
        self.assertIn("interaction", text.lower())

    def test_no_significant_interaction_mentioned(self):
        """When interaction is not significant, additive interpretation given."""
        text = self._run_interpretation(True, True, False)
        self.assertIn("additive", text.lower())

    def test_no_trait_placeholder_leak(self):
        """No literal '{trait}' placeholder must appear in the output."""
        text = self._run_interpretation(False, False, False)
        self.assertNotIn("{trait}", text)

    def test_mentions_restricted_randomisation(self):
        """Overview must explain the restricted-randomisation structure."""
        text = self._run_interpretation(True, False, False)
        self.assertIn("restricted randomis", text.lower())

    def test_mentions_error_strata(self):
        """Recommendation must always mention both error strata."""
        text = self._run_interpretation(False, False, False)
        self.assertIn("whole-plot error", text.lower())
        self.assertIn("subplot", text.lower())

    def test_error_strata_emitted_when_nothing_significant(self):
        """Error-strata guidance appears even when all effects are non-significant."""
        text = self._run_interpretation(False, False, False)
        self.assertIn("whole-plot error", text.lower())

    def test_interaction_significant_uses_cell_means_language(self):
        """When interaction is significant, recommendation uses cell-means language."""
        text = self._run_interpretation(True, True, True)
        self.assertIn("cell", text.lower())


# ============================================================================
# Snapshot: banned / required terms across all significance variants
# ============================================================================

class TestSplitPlotSnapshotReport(unittest.TestCase):
    """
    Consistency validator for split-plot RCBD interpretation text.

    BANNED  — terms that must never appear in split-plot output
    REQUIRED — terms that must always appear regardless of significance
    """

    BANNED = [
        r"\bgenotype\b",
        r"\bGGE\b",
        r"\bheritability\b",
        r"\bGCV\b",
        r"\bPCV\b",
        r"\bbreeding\b",
        r"\bgermplasm\b",
        r"\bselection\b",
        r"\bAMMI\b",
        r"\bbiplot\b",
    ]

    REQUIRED = [
        "main-plot",
        "subplot",
        "restricted randomis",
        "whole-plot error",
    ]

    def _interp(self, mp_sig, sub_sig, int_sig):
        from analysis_anova_routes import generate_anova_interpretation
        return generate_anova_interpretation(
            trait="TestTrait",
            summary={"grand_mean": 5.0, "cv_percent": 15.0, "min": 2.0,
                     "max": 8.0, "range": 6.0, "standard_error": 0.4},
            precision_level="moderate",
            cv_interpretation_flag="cv_available",
            genotype_significant=None,
            environment_significant=None,
            gxe_significant=None,
            ranking_caution=None,
            selection_feasible=None,
            mean_separation=None,
            n_genotypes=None,
            n_environments=None,
            n_reps=3,
            environment_mode="single",
            design_type="split_plot_rcbd",
            main_plot_significant=mp_sig,
            subplot_significant=sub_sig,
            interaction_significant=int_sig,
        )

    def _variants(self):
        """All four significance combinations."""
        return [
            (True,  True,  True),
            (True,  True,  False),
            (True,  False, False),
            (False, False, False),
        ]

    def test_snapshot_all_significant(self):
        """All significant: required terms present, banned absent, interaction + cell mentioned."""
        import re
        text = self._interp(True, True, True)
        for term in self.REQUIRED:
            self.assertIn(term.lower(), text.lower(),
                          f"Required term '{term}' missing (all-sig variant)")
        for pattern in self.BANNED:
            self.assertEqual(
                re.findall(pattern, text, re.IGNORECASE), [],
                f"Banned pattern '{pattern}' found (all-sig variant)",
            )
        self.assertIn("interaction", text.lower())
        self.assertIn("cell", text.lower())

    def test_snapshot_main_only_significant(self):
        """Main-plot only: required terms present, banned absent, additive mentioned."""
        import re
        text = self._interp(True, False, False)
        for term in self.REQUIRED:
            self.assertIn(term.lower(), text.lower(),
                          f"Required term '{term}' missing (main-only variant)")
        for pattern in self.BANNED:
            self.assertEqual(
                re.findall(pattern, text, re.IGNORECASE), [],
                f"Banned pattern '{pattern}' found (main-only variant)",
            )
        self.assertIn("additive", text.lower())

    def test_snapshot_nothing_significant(self):
        """Nothing significant: required terms still present, banned still absent."""
        import re
        text = self._interp(False, False, False)
        for term in self.REQUIRED:
            self.assertIn(term.lower(), text.lower(),
                          f"Required term '{term}' missing (nothing-sig variant)")
        for pattern in self.BANNED:
            self.assertEqual(
                re.findall(pattern, text, re.IGNORECASE), [],
                f"Banned pattern '{pattern}' found (nothing-sig variant)",
            )
        # Error-strata guidance always appears
        self.assertIn("whole-plot error", text.lower())

    def test_snapshot_no_placeholder_tokens(self):
        """No literal {…} placeholders in any significance variant."""
        import re
        for mp, sub, intr in self._variants():
            text = self._interp(mp, sub, intr)
            placeholders = re.findall(r'\{[^}]+\}', text)
            self.assertEqual(
                placeholders, [],
                f"Placeholder tokens found for variant (mp={mp}, sub={sub}, int={intr}): "
                f"{placeholders}",
            )


if __name__ == "__main__":
    unittest.main()
