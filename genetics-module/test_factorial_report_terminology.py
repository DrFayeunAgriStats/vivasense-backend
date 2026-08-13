"""
Generic factorial reports must contain no genotype terminology.

Factor A of a factorial is whatever the researcher mapped. Four independent
report paths inferred a biological role from that position: the Word header
(design label and entry unit), the Descriptive Statistics row, the ANOVA
starter's entry clause, and the mean-separation prose. result.n_genotypes comes
from R as nlevels(data$genotype) — which for a factorial IS Factor A — so
suppressing the dataset-summary count alone was not enough.

Genuine genotype designs are unaffected: every gate keys on design, not domain.
"""
import asyncio, base64, io, re, unittest

import pandas as pd

import app_genetics
import genetics_export as ge
import multitrait_upload_routes as routes
from multitrait_upload_schemas import UploadAnalysisRequest

FIXTURE = "testdata/f3_threeway.csv"


def _render(**kw):
    app_genetics.r_engine = app_genetics.RGeneticsEngine("vivasense_genetics.R")
    df = pd.read_csv(kw.pop("path", FIXTURE))
    if kw.pop("rename", None):
        df = df.rename(columns={"Irrigation": "Genotype"})
    buf = io.StringIO(); df.to_csv(buf, index=False)
    base = dict(base64_content=base64.b64encode(buf.getvalue().encode()).decode(),
                file_type="csv", trait_columns=["Yield"], mode="single", module="anova")
    base.update(kw)
    resp = asyncio.run(routes.analyze_upload(UploadAnalysisRequest(**base)))
    req = ge.DownloadReportRequest(
        summary_table=resp.summary_table, trait_results=resp.trait_results,
        dataset_summary=resp.dataset_summary, module="anova", domain="plant_breeding")
    return ge._collect_doc_text(ge._build_document(req)), resp


def _factorial(**over):
    kw = dict(rep_column="Rep", design_type="factorial", genotype_column="Irrigation",
              factor_a_column="Irrigation", factor_b_column="Variety",
              factor_c_column="Spacing")
    kw.update(over)
    return _render(**kw)


class TestGenericFactorial(unittest.TestCase):
    def setUp(self):
        self.text, self.resp = _factorial()

    def test_no_genotype_terminology_anywhere(self):
        hits = re.findall(r"[^.\n]*genotype[s]?[^.\n]*", self.text, re.I)
        self.assertEqual(hits, [], f"genotype wording leaked: {hits}")

    def test_header_names_the_design_not_single_environment(self):
        header = [l for l in self.text.split("\n") if "·" in l][0]
        self.assertIn("Factorial RCBD", header)
        self.assertNotIn("Single-environment", header)
        self.assertNotIn("genotypes", header)

    def test_header_lists_factor_levels(self):
        header = [l for l in self.text.split("\n") if "·" in l][0]
        for lbl, n in (("Irrigation", 2), ("Variety", 3), ("Spacing", 2)):
            self.assertIn(f"{lbl} ({n} levels)", header)

    def test_no_genotypes_row_but_level_rows_remain(self):
        self.assertNotIn("No. Genotypes", self.text)
        for lbl in ("No. Irrigation Levels", "No. Variety Levels", "No. Spacing Levels"):
            self.assertIn(lbl, self.text)

    def test_no_environment_count_invented(self):
        """No COUNT. Generic advisory prose ('validated across additional
        environments and seasons') is not a structural claim and stays."""
        self.assertIsNone(self.resp.dataset_summary.n_environments)
        header = [l for l in self.text.split("\n") if "·" in l][0]
        self.assertNotIn("environment", header)

    def test_dataset_summary_genotype_count_suppressed(self):
        """genotype_column was sent as a legacy field; it must not be trusted."""
        self.assertIsNone(self.resp.dataset_summary.n_genotypes)


class TestBiologicalSoundingLabel(unittest.TestCase):
    def test_factor_named_genotype_does_not_acquire_the_role(self):
        text, resp = _factorial(rename=True, genotype_column="Genotype",
                                factor_a_column="Genotype")
        self.assertIsNone(resp.dataset_summary.n_genotypes)
        self.assertNotIn("No. Genotypes", text)
        header = [l for l in text.split("\n") if "·" in l][0]
        self.assertIn("Factorial RCBD", header)
        self.assertIn("Genotype (2 levels)", header)


class TestGenuineGenotypeDesignUnchanged(unittest.TestCase):
    def test_rcbd_keeps_genotype_reporting(self):
        text, resp = _render(rep_column="Rep", design_type="rcbd",
                             genotype_column="Variety")
        header = [l for l in text.split("\n") if "·" in l][0]
        self.assertIn("Single-environment", header)
        self.assertIn("genotypes", header)
        self.assertIn("No. Genotypes", text)
        self.assertEqual(resp.dataset_summary.n_genotypes, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
