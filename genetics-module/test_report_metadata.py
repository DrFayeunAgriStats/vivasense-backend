"""
Report metadata and assumption wording (Parts A and B).

Two confirmed defects, both reporting-layer only:

A. A non-significant Levene/Bartlett test was narrated as "Homogeneity of
   variance supported ... Equal variance assumption is not violated", which
   asserts equality of variances from a failure to detect a difference. The
   export table also placed a "Passed" column immediately left of a sentence
   beginning with the finding, so a heterogeneous row read across as
   "No | Heterogeneity of variance detected".

B. Clients send environment_column for every design as a legacy field. A plain
   treatment factor sitting in that slot was counted as environments, so a
   3-level factor became "3 environments" in a factorial report while the model
   correctly ignored the column. The metadata label was likewise derived from
   the binary mode, labelling every factorial run "Single-environment".

Nothing here touches the statistical model, df, denominators, replication
structure, FAC-01..07, or the Levene/Bartlett calculations.

Run from inside genetics-module/:
    python -m pytest test_report_metadata.py -v
"""

import base64
import io
import unittest

import pandas as pd

from export_module_routes import _design_metadata_label


# ── Part A: assumption wording ───────────────────────────────────────────────

def _homogeneity_interpretations(trait_values, group):
    """Run the real R engine and return its homogeneity interpretation."""
    import subprocess, json, tempfile, os
    df = pd.DataFrame({"grp": group, "y": trait_values})
    with tempfile.TemporaryDirectory() as td:
        csv = os.path.join(td, "d.csv")
        df.to_csv(csv, index=False)
        script = f'''
suppressMessages(source("vivasense_genetics.R"))
d <- read.csv("{csv.replace(os.sep, "/")}")
dat <- data.frame(genotype=factor(d$grp), rep=factor(rep(1:3, length.out=nrow(d))),
                  trait_value=d$y)
r <- compute_single_environment(dat, trait_name="Y")
h <- r$assumption_tests$homogeneity
cat(sprintf("%s|%.6f|%s", h$interpretation, h$p_value, h$passed))
'''
        sf = os.path.join(td, "s.R")
        open(sf, "w", encoding="utf-8").write(script)
        out = subprocess.run(["Rscript", sf], capture_output=True, text=True,
                             cwd=os.getcwd(), timeout=180)
        tail = [ln for ln in out.stdout.splitlines() if "|" in ln]
        if not tail:
            raise unittest.SkipTest(f"R did not return homogeneity: {out.stderr[-300:]}")
        text, p, passed = tail[-1].rsplit("|", 2)
        return text, float(p), passed.strip() == "TRUE"


class TestAssumptionWordingIsConservative(unittest.TestCase):

    def test_a_heterogeneous_data_reports_evidence_of_heterogeneity(self):
        """p < 0.05 -> evidence of heterogeneity."""
        vals = [1, 1.1, 0.9, 1.05, 0.95, 1.0] + [50, 5, 95, 2, 80, 30]
        grp = ["A"] * 6 + ["B"] * 6
        text, p, passed = _homogeneity_interpretations(vals, grp)
        self.assertLess(p, 0.05)
        self.assertFalse(passed)
        self.assertIn("Heterogeneity of variance detected", text)
        self.assertNotIn("No evidence", text)

    def test_b_homogeneous_data_reports_no_evidence_not_proof(self):
        """p >= 0.05 -> 'no evidence', never 'supported'/'not violated'."""
        vals = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1] + [12.0, 12.5, 11.5, 12.2, 11.8, 12.1]
        grp = ["A"] * 6 + ["B"] * 6
        text, p, passed = _homogeneity_interpretations(vals, grp)
        self.assertGreaterEqual(p, 0.05)
        self.assertTrue(passed)
        self.assertIn("No evidence of heterogeneity of variance was detected", text)

    def test_c_wording_never_claims_equal_variances_are_proven(self):
        vals = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1] + [12.0, 12.5, 11.5, 12.2, 11.8, 12.1]
        grp = ["A"] * 6 + ["B"] * 6
        text, _, _ = _homogeneity_interpretations(vals, grp)
        for claim in ("Homogeneity of variance supported",
                      "Equal variance assumption is not violated"):
            self.assertNotIn(claim, text)
        self.assertIn("does not establish that the variances are equal", text)


class TestAssumptionTableHeader(unittest.TestCase):
    """The 'Passed' header sat beside a finding sentence and inverted its reading."""

    def test_header_renamed_to_break_the_adjacency(self):
        src = open("genetics_export.py", encoding="utf-8").read()
        self.assertIn('"Assumption upheld", "Interpretation"', src)
        self.assertNotIn('"p-value", "Passed", "Interpretation"', src)


# ── Part B: design metadata ──────────────────────────────────────────────────

class _Result:
    def __init__(self, design):
        self.design = design


class _Trait:
    def __init__(self, design):
        self.analysis_result = type("AR", (), {"result": _Result(design)})()


class _Payload:
    def __init__(self, mode, designs=()):
        self.mode = mode
        self.trait_results = {f"t{i}": _Trait(d) for i, d in enumerate(designs)}


class TestDesignMetadataLabel(unittest.TestCase):

    def test_a_generic_factorial_is_not_called_single_environment(self):
        self.assertEqual(
            _design_metadata_label(_Payload("single", ["factorial_rcbd"])),
            "Factorial RCBD")

    def test_a_factorial_crd_labelled_as_such(self):
        self.assertEqual(
            _design_metadata_label(_Payload("single", ["factorial_crd"])),
            "Factorial CRD")

    def test_c_genuine_met_metadata_unchanged(self):
        self.assertEqual(_design_metadata_label(_Payload("multi", ["rcbd"])),
                         "Multi-environment")

    def test_d_genuine_single_environment_unchanged(self):
        self.assertEqual(_design_metadata_label(_Payload("single", ["rcbd"])),
                         "Single-environment")

    def test_split_plot_described_by_its_design(self):
        self.assertEqual(
            _design_metadata_label(_Payload("single", ["split_plot_rcbd"])),
            "Split-Plot RCBD")

    def test_missing_design_falls_back_to_mode(self):
        self.assertEqual(_design_metadata_label(_Payload("single", [])),
                         "Single-environment")


class TestEnvironmentCountNotInvented(unittest.TestCase):
    """A treatment factor in the legacy environment slot must not be counted."""

    @staticmethod
    def _summary(design_type, env_col):
        import asyncio
        from unittest.mock import patch
        import app_genetics
        import multitrait_upload_routes as routes
        from multitrait_upload_schemas import UploadAnalysisRequest

        rows = []
        for rep in ["R1", "R2", "R3"]:
            for a in ["A1", "A2"]:
                for b in ["B1", "B2", "B3"]:   # 3 levels — the trap
                    rows.append({"REP": rep, "A": a, "B": b, "Y": 10.0})
        buf = io.StringIO(); pd.DataFrame(rows).to_csv(buf, index=False)
        kwargs = dict(
            base64_content=base64.b64encode(buf.getvalue().encode()).decode(),
            file_type="csv", rep_column="REP", trait_columns=["Y"],
            mode="single", design_type=design_type, module="anova",
            environment_column=env_col,
        )
        if design_type == "factorial":
            kwargs.update(genotype_column=None, factor_a_column="A", factor_b_column="B")
        else:
            kwargs.update(genotype_column="A")

        class Stub:
            class Reached(Exception): pass
            def run_analysis(self, *a, **k): raise Stub.Reached()

        with patch.object(app_genetics, "r_engine", Stub()):
            resp = asyncio.run(routes.analyze_upload(UploadAnalysisRequest(**kwargs)))
        return resp.dataset_summary

    def test_a_factorial_with_no_environment_role_reports_none(self):
        self.assertIsNone(self._summary("factorial", None).n_environments)

    def test_b_three_level_factor_in_legacy_slot_is_not_three_environments(self):
        """The exact reported defect: B has 3 levels and sat in environment_column."""
        summary = self._summary("factorial", "B")
        self.assertIsNone(
            summary.n_environments,
            "a treatment factor must not be reported as an environment count")

    def test_c_non_factorial_with_environment_column_still_reports_it(self):
        self.assertEqual(self._summary("rcbd", "B").n_environments, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestFactorialGenotypeLabelling(unittest.TestCase):
    """FACT-01: Factor A of a generic factorial must not be called 'genotypes'.

    Factor A is a real model term, unlike the environment column — so this is
    mislabelling rather than invention. The principle is the same: an upload
    position must not confer a biological role.
    """

    @staticmethod
    def _summary(**over):
        import asyncio
        from unittest.mock import patch
        import app_genetics
        import multitrait_upload_routes as routes
        from multitrait_upload_schemas import UploadAnalysisRequest

        rows = []
        for rep in ["R1", "R2", "R3"]:
            for a in ["Full", "Deficit"]:
                for b in ["V1", "V2", "V3"]:
                    rows.append({"REP": rep, "Irrigation": a, "Variety": b, "Y": 10.0})
        buf = io.StringIO(); pd.DataFrame(rows).to_csv(buf, index=False)
        kwargs = dict(
            base64_content=base64.b64encode(buf.getvalue().encode()).decode(),
            file_type="csv", rep_column="REP", trait_columns=["Y"],
            mode="single", design_type="factorial", module="anova",
            genotype_column=None, factor_a_column="Irrigation",
            factor_b_column="Variety",
        )
        kwargs.update(over)

        class Stub:
            class Reached(Exception): pass
            def run_analysis(self, *a, **k): raise Stub.Reached()

        with patch.object(app_genetics, "r_engine", Stub()):
            return asyncio.run(routes.analyze_upload(UploadAnalysisRequest(**kwargs))).dataset_summary

    def test_b_factor_a_irrigation_is_not_counted_as_genotypes(self):
        self.assertIsNone(self._summary().n_genotypes)

    def test_c_factor_a_variety_still_not_auto_genotype(self):
        """Even a biologically genotype-like name must not be inferred."""
        s = self._summary(factor_a_column="Variety", factor_b_column="Irrigation")
        self.assertIsNone(s.n_genotypes)

    def test_a_genuine_genotype_role_is_reported(self):
        s = self._summary(genotype_column="Variety", factor_a_column="Irrigation",
                          factor_b_column="Variety")
        self.assertEqual(s.n_genotypes, 3)

    def test_e_non_factorial_genotype_metadata_unchanged(self):
        s = self._summary(design_type="rcbd", genotype_column="Variety",
                          factor_a_column=None, factor_b_column=None)
        self.assertEqual(s.n_genotypes, 3)

    def test_replication_inference_still_works_without_genotype_role(self):
        """The rep fallback groups by Factor A regardless of its role."""
        self.assertEqual(self._summary().n_reps, 3)

    def test_d_export_uses_factor_labels_not_genotype_terminology(self):
        src = open("genetics_export.py", encoding="utf-8").read()
        self.assertIn('rows.append((f"No. {_lbl} Levels", str(len(set(_levels)))))', src)
