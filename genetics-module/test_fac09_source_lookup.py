"""
FAC-09: the factorial ANOVA starter sentence must not look up a literal
"genotype" source.

Factorial ANOVA sources are relabelled to the researcher's own column names, so
`_extract_source_stats(at, "genotype")` missed, returned (None, None, None), and
the caller's else-branch reported a real effect as "not significant" with
em-dash statistics — a false negative stated as a finding.

The fix resolves the primary treatment source from mean_separation.treatment_label
for factorial designs only, keeping "genotype" for everything else and as a
fallback. Scope is the one call site; _extract_source_stats and
_eta_squared_for_source are untouched.

Run from inside genetics-module/:
    python -m pytest test_fac09_source_lookup.py -v
"""

import asyncio
import base64
import io
import unittest
from unittest.mock import patch

import pandas as pd

import app_genetics
import multitrait_upload_routes as routes
from genetics_export import _extract_source_stats, _eta_squared_for_source
from multitrait_upload_schemas import UploadAnalysisRequest


def _resolve_source(result):
    """Mirror of the resolution at genetics_export.py:2252 under test."""
    source = "genotype"
    if (getattr(result, "design", None) or "") in ("factorial_crd", "factorial_rcbd"):
        label = (getattr(result.mean_separation, "treatment_label", None)
                 if result.mean_separation else None)
        if label:
            source = label
    return source


def _analyse(**kw):
    app_genetics.r_engine = app_genetics.RGeneticsEngine("vivasense_genetics.R")
    path = kw.pop("path", None)
    if path:
        b64 = base64.b64encode(open(path, "rb").read()).decode()
    else:
        buf = io.StringIO(); kw.pop("df").to_csv(buf, index=False)
        b64 = base64.b64encode(buf.getvalue().encode()).decode()
    base = dict(base64_content=b64, file_type="csv", mode="single", module="anova")
    base.update(kw)
    resp = asyncio.run(routes.analyze_upload(UploadAnalysisRequest(**base)))
    trait = resp.trait_results[base["trait_columns"][0]]
    assert trait.status == "success", trait.error
    return trait.analysis_result.result


def _noisy_two_factor():
    import random
    random.seed(11)
    rows = []
    for rep in ["R1", "R2", "R3"]:
        for i, a in enumerate(["Full", "Deficit"]):
            for j, b in enumerate(["V1", "V2", "V3"]):
                rows.append({"REP": rep, "Irrigation": a, "Variety": b,
                             "Y": 10 + 4 * i + 1.5 * j + random.gauss(0, 0.3)})
    return pd.DataFrame(rows)


class TestFac09SourceResolution(unittest.TestCase):

    def test_a_genuine_genotype_analysis_unchanged(self):
        df = _noisy_two_factor().rename(columns={"Irrigation": "Genotype"})
        res = _analyse(df=df, rep_column="REP", trait_columns=["Y"],
                       genotype_column="Genotype", design_type="rcbd")
        self.assertEqual(_resolve_source(res), "genotype")
        f, p, _ = _extract_source_stats(res.anova_table, _resolve_source(res))
        self.assertIsNotNone(f)
        self.assertIsNotNone(p)
        self.assertLess(p, 0.05)

    def test_b_two_factor_factorial_retrieves_real_statistics(self):
        res = _analyse(df=_noisy_two_factor(), rep_column="REP", trait_columns=["Y"],
                       genotype_column=None, design_type="factorial",
                       factor_a_column="Irrigation", factor_b_column="Variety")
        source = _resolve_source(res)
        self.assertEqual(source, "Irrigation")
        self.assertNotEqual(source, "genotype")

        f, p, _ = _extract_source_stats(res.anova_table, source)
        eta = _eta_squared_for_source(res.anova_table, source)
        for value, name in ((f, "F"), (p, "p"), (eta, "eta2")):
            self.assertIsNotNone(value, f"{name} must not be em-dash")
        # The effect is real; the old lookup called it "not significant".
        self.assertLess(p, 0.05)

        old_f, old_p, _ = _extract_source_stats(res.anova_table, "genotype")
        self.assertIsNone(old_f, "regression guard: 'genotype' must still miss here")
        self.assertIsNone(old_p)

    def test_c_three_factor_factorial_retrieves_real_statistics(self):
        res = _analyse(path="testdata/f3_threeway.csv", rep_column="Rep",
                       trait_columns=["Yield"], genotype_column=None,
                       design_type="factorial", factor_a_column="Irrigation",
                       factor_b_column="Variety", factor_c_column="Spacing")
        source = _resolve_source(res)
        self.assertEqual(source, "Irrigation")
        f, p, _ = _extract_source_stats(res.anova_table, source)
        self.assertIsNotNone(f)
        self.assertLess(p, 0.05)
        self.assertNotIn("genotype", res.anova_table.source)

    def test_d_fallback_when_treatment_label_missing(self):
        """No label -> fall back to 'genotype' without crashing."""
        class _R:
            design = "factorial_rcbd"
            mean_separation = None
        self.assertEqual(_resolve_source(_R()), "genotype")

        class _R2:
            design = "factorial_rcbd"
            mean_separation = type("MS", (), {"treatment_label": ""})()
        self.assertEqual(_resolve_source(_R2()), "genotype")

    def test_non_factorial_never_uses_a_factor_label(self):
        class _R:
            design = "rcbd"
            mean_separation = type("MS", (), {"treatment_label": "Irrigation"})()
        self.assertEqual(_resolve_source(_R()), "genotype")


class TestCallSiteWiring(unittest.TestCase):
    def test_export_uses_the_resolved_source_for_both_lookups(self):
        src = open("genetics_export.py", encoding="utf-8").read()
        self.assertIn("_extract_source_stats(at, _primary_source)", src)
        self.assertIn("_eta_squared_for_source(at, _primary_source)", src)
        self.assertIn("f\"the {_effect_label} was {sig_word} \"", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
