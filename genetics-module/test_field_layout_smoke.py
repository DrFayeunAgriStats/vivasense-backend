"""Smoke tests for the VivaSense field layout generator.

Covers:
  1. Valid CRD — correct plot count, no duplicate plot IDs, all treatments present.
  2. Valid RCBD — same assertions, plus each block contains every treatment exactly once.
  3. Valid split-plot — plot count and fieldbook schema.
  4. Valid factorial RCBD — combination count and no duplicate plot IDs.
  5. Invalid treatment count (zero / empty list).
  6. Invalid replication count (zero / negative).
  7. Missing seed defaults to 0 (deterministic).
  8. Repeated requests with the same seed produce identical layouts.
  9. Response schema — standardized fields are present.
 10. Export-ready structures — printable_plot_numbers, row_col_matrix, field_map_labels.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from field_layout_generator import generate_field_layout


# ── Helper utilities ──────────────────────────────────────────────────────────

def _plot_ids(result: Dict[str, Any]) -> List[int]:
    """Return a sorted list of all plot IDs from the fieldbook."""
    return sorted(row["plot_id"] for row in result["fieldbook"])


def _treatment_set(result: Dict[str, Any]) -> set:
    """Return the set of treatment labels present in the fieldbook."""
    return {
        row.get("treatment") or row.get("treatment_combination") or ""
        for row in result["fieldbook"]
    }


# ── Test cases ────────────────────────────────────────────────────────────────

class TestCRDSmoke(unittest.TestCase):
    """Smoke tests for Completely Randomized Design."""

    def setUp(self):
        self.request = {
            "design_type": "crd",
            "treatments": ["T1", "T2", "T3", "T4"],
            "replications": 3,
            "seed": 42,
        }
        self.result = generate_field_layout(self.request)

    def test_total_plots_correct(self):
        """CRD: total plots == treatments × replications."""
        self.assertEqual(self.result["total_plots"], 12)
        self.assertEqual(len(self.result["fieldbook"]), 12)

    def test_no_duplicate_plot_ids(self):
        """CRD: every plot ID is unique."""
        ids = _plot_ids(self.result)
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_treatments_present(self):
        """CRD: each treatment appears exactly replications times."""
        counts: Dict[str, int] = {}
        for row in self.result["fieldbook"]:
            t = row["treatment"]
            counts[t] = counts.get(t, 0) + 1
        for t in ["T1", "T2", "T3", "T4"]:
            self.assertEqual(counts[t], 3, f"Treatment {t} should appear 3 times")

    def test_standardized_fields_present(self):
        """CRD: response contains all required standardized keys."""
        for key in ("design", "seed", "treatments", "replications", "total_plots",
                    "layout_matrix", "plot_labels", "timestamp", "export_ready"):
            self.assertIn(key, self.result, f"Missing key: {key}")

    def test_design_field(self):
        self.assertEqual(self.result["design"], "crd")

    def test_seed_field(self):
        self.assertEqual(self.result["seed"], 42)

    def test_replications_field(self):
        self.assertEqual(self.result["replications"], 3)

    def test_plot_labels_length(self):
        """CRD: plot_labels has one entry per plot."""
        self.assertEqual(len(self.result["plot_labels"]), 12)

    def test_backward_compat_fields_present(self):
        """CRD: legacy fields design_type, plot_matrix, fieldbook, layout_summary still present."""
        for key in ("design_type", "plot_matrix", "fieldbook", "layout_summary", "alpha_value"):
            self.assertIn(key, self.result, f"Missing legacy key: {key}")

    def test_layout_matrix_same_as_plot_matrix(self):
        """CRD: layout_matrix and plot_matrix refer to equivalent data."""
        self.assertEqual(self.result["layout_matrix"], self.result["plot_matrix"])


class TestRCBDSmoke(unittest.TestCase):
    """Smoke tests for Randomized Complete Block Design."""

    def setUp(self):
        self.request = {
            "design_type": "rcbd",
            "treatments": ["A", "B", "C", "D", "E"],
            "replications": 4,
            "seed": 99,
        }
        self.result = generate_field_layout(self.request)

    def test_total_plots_correct(self):
        """RCBD: total plots == treatments × replications."""
        self.assertEqual(self.result["total_plots"], 20)

    def test_no_duplicate_plot_ids(self):
        """RCBD: every plot ID is unique."""
        ids = _plot_ids(self.result)
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_treatments_in_every_block(self):
        """RCBD: each block contains every treatment exactly once."""
        blocks: Dict[int, List[str]] = {}
        for row in self.result["fieldbook"]:
            b = row["block"]
            blocks.setdefault(b, []).append(row["treatment"])
        expected = sorted(["A", "B", "C", "D", "E"])
        for b, treatments in blocks.items():
            self.assertEqual(sorted(treatments), expected,
                             f"Block {b} does not contain all treatments exactly once")

    def test_standardized_fields(self):
        for key in ("design", "seed", "treatments", "total_plots", "layout_matrix",
                    "plot_labels", "timestamp", "export_ready"):
            self.assertIn(key, self.result)


class TestSplitPlotSmoke(unittest.TestCase):
    """Smoke tests for Split-Plot design."""

    def setUp(self):
        self.request = {
            "design_type": "split_plot",
            "main_treatments": ["M1", "M2", "M3"],
            "sub_treatments": ["S1", "S2"],
            "replications": 3,
            "seed": 7,
        }
        self.result = generate_field_layout(self.request)

    def test_total_plots_correct(self):
        """Split-plot: total = main × sub × reps."""
        self.assertEqual(self.result["total_plots"], 18)

    def test_no_duplicate_plot_ids(self):
        ids = _plot_ids(self.result)
        self.assertEqual(len(ids), len(set(ids)))

    def test_fieldbook_schema(self):
        """Split-plot: fieldbook rows contain required keys."""
        required = {"plot_id", "rep", "main_plot", "main_treatment", "sub_plot", "sub_treatment"}
        for row in self.result["fieldbook"]:
            self.assertTrue(required.issubset(row.keys()),
                            f"Row missing keys: {required - row.keys()}")


class TestFactorialRCBDSmoke(unittest.TestCase):
    """Smoke tests for Factorial RCBD design."""

    def setUp(self):
        self.request = {
            "design_type": "factorial_rcbd",
            "factors": {
                "Nitrogen": ["N0", "N1", "N2"],
                "Irrigation": ["I0", "I1"],
            },
            "replications": 3,
            "seed": 21,
        }
        self.result = generate_field_layout(self.request)

    def test_total_plots_correct(self):
        """Factorial RCBD: total = combinations × reps."""
        n_combos = 3 * 2  # Nitrogen × Irrigation
        self.assertEqual(self.result["total_plots"], n_combos * 3)

    def test_no_duplicate_plot_ids(self):
        ids = _plot_ids(self.result)
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_combinations_in_every_rep(self):
        """Factorial RCBD: every treatment combination present in each rep."""
        reps: Dict[int, List[str]] = {}
        for row in self.result["fieldbook"]:
            reps.setdefault(row["rep"], []).append(row["treatment_combination"])
        for rep_id, combos in reps.items():
            self.assertEqual(len(combos), 6,
                             f"Rep {rep_id} should have 6 combinations, got {len(combos)}")
            self.assertEqual(len(set(combos)), 6,
                             f"Rep {rep_id} has duplicate combinations")


# ── Validation failure tests ──────────────────────────────────────────────────

class TestValidationFailures(unittest.TestCase):
    """Ensure invalid inputs raise ValueError with human-readable messages."""

    def test_empty_treatments_raises(self):
        """Empty treatment list is rejected."""
        with self.assertRaises(ValueError) as ctx:
            generate_field_layout({
                "design_type": "crd",
                "treatments": [],
                "replications": 3,
                "seed": 1,
            })
        self.assertIn("treatments", str(ctx.exception).lower())

    def test_zero_replications_raises(self):
        """Zero replications is rejected."""
        with self.assertRaises(ValueError) as ctx:
            generate_field_layout({
                "design_type": "crd",
                "treatments": ["T1", "T2"],
                "replications": 0,
                "seed": 1,
            })
        self.assertIn("replications", str(ctx.exception).lower())

    def test_negative_replications_raises(self):
        """Negative replications is rejected."""
        with self.assertRaises(ValueError):
            generate_field_layout({
                "design_type": "rcbd",
                "treatments": ["T1", "T2", "T3"],
                "replications": -1,
                "seed": 1,
            })

    def test_non_integer_replications_raises(self):
        """String replications value is rejected."""
        with self.assertRaises(ValueError):
            generate_field_layout({
                "design_type": "crd",
                "treatments": ["T1", "T2"],
                "replications": "three",
                "seed": 1,
            })

    def test_boolean_replications_raises(self):
        """Boolean replications value is rejected (bool is a subtype of int in Python)."""
        with self.assertRaises(ValueError):
            generate_field_layout({
                "design_type": "crd",
                "treatments": ["T1", "T2"],
                "replications": True,
                "seed": 1,
            })

    def test_duplicate_treatments_raises(self):
        """Duplicate treatment labels are rejected."""
        with self.assertRaises(ValueError) as ctx:
            generate_field_layout({
                "design_type": "crd",
                "treatments": ["T1", "T1", "T2"],
                "replications": 2,
                "seed": 1,
            })
        self.assertIn("duplicate", str(ctx.exception).lower())

    def test_invalid_design_type_raises(self):
        """Unrecognized design type is rejected."""
        with self.assertRaises(ValueError) as ctx:
            generate_field_layout({
                "design_type": "magic_design",
                "treatments": ["T1", "T2"],
                "replications": 2,
                "seed": 1,
            })
        self.assertIn("unsupported", str(ctx.exception).lower())

    def test_non_integer_seed_raises(self):
        """Float seed is rejected."""
        with self.assertRaises(ValueError) as ctx:
            generate_field_layout({
                "design_type": "crd",
                "treatments": ["T1", "T2"],
                "replications": 2,
                "seed": 3.14,
            })
        self.assertIn("seed", str(ctx.exception).lower())


# ── Missing seed default test ─────────────────────────────────────────────────

class TestMissingSeed(unittest.TestCase):
    """Missing seed should default to 0 and still produce a valid layout."""

    def test_missing_seed_produces_valid_layout(self):
        result = generate_field_layout({
            "design_type": "crd",
            "treatments": ["T1", "T2", "T3"],
            "replications": 2,
            # seed deliberately omitted
        })
        self.assertEqual(result["total_plots"], 6)
        self.assertEqual(result["seed"], 0)

    def test_explicit_zero_seed_matches_missing_seed(self):
        """Explicit seed=0 and missing seed should produce identical layouts."""
        req = {
            "design_type": "rcbd",
            "treatments": ["A", "B", "C"],
            "replications": 2,
        }
        r_default = generate_field_layout(req)
        r_explicit = generate_field_layout({**req, "seed": 0})
        self.assertEqual(
            [row["treatment"] for row in r_default["fieldbook"]],
            [row["treatment"] for row in r_explicit["fieldbook"]],
        )


# ── Reproducibility tests ─────────────────────────────────────────────────────

class TestReproducibility(unittest.TestCase):
    """Same seed + same payload must always produce identical layouts."""

    def _run(self, design_type: str, extra: dict, seed: int) -> Dict[str, Any]:
        return generate_field_layout({
            "design_type": design_type,
            "seed": seed,
            **extra,
        })

    def test_crd_same_seed_identical(self):
        extra = {"treatments": ["T1", "T2", "T3", "T4"], "replications": 3}
        r1 = self._run("crd", extra, 42)
        r2 = self._run("crd", extra, 42)
        self.assertEqual(r1["plot_labels"], r2["plot_labels"])
        self.assertEqual(
            [row["treatment"] for row in r1["fieldbook"]],
            [row["treatment"] for row in r2["fieldbook"]],
        )

    def test_rcbd_same_seed_identical(self):
        extra = {"treatments": ["A", "B", "C", "D"], "replications": 4}
        r1 = self._run("rcbd", extra, 99)
        r2 = self._run("rcbd", extra, 99)
        self.assertEqual(r1["plot_labels"], r2["plot_labels"])

    def test_different_seeds_produce_different_layouts(self):
        """Different seeds (usually) produce different treatment orderings."""
        extra = {"treatments": ["T1", "T2", "T3", "T4", "T5"], "replications": 4}
        r1 = self._run("rcbd", extra, 1)
        r2 = self._run("rcbd", extra, 2)
        # With 5 treatments × 4 reps, it is astronomically unlikely that two
        # different seeds yield the same ordering.
        self.assertNotEqual(r1["plot_labels"], r2["plot_labels"])

    def test_repeated_requests_stable_across_three_runs(self):
        """Calling generate_field_layout three times with the same seed is stable."""
        req = {"design_type": "crd", "treatments": ["X", "Y", "Z"], "replications": 5, "seed": 77}
        results = [generate_field_layout(req) for _ in range(3)]
        labels_0 = results[0]["plot_labels"]
        for i, r in enumerate(results[1:], start=1):
            self.assertEqual(r["plot_labels"], labels_0,
                             f"Run {i + 1} produced a different layout than run 1")


# ── Export-ready structure tests ──────────────────────────────────────────────

class TestExportReady(unittest.TestCase):
    """Validate export_ready structure for CRD and RCBD."""

    def test_crd_printable_plot_numbers(self):
        result = generate_field_layout({
            "design_type": "crd",
            "treatments": ["T1", "T2", "T3"],
            "replications": 2,
            "seed": 1,
        })
        pr = result["export_ready"]["printable_plot_numbers"]
        self.assertEqual(len(pr), 6)
        # Each entry is "Plot N | Treatment"
        for entry in pr:
            self.assertTrue(entry.startswith("Plot "), entry)
            self.assertIn("|", entry)

    def test_rcbd_row_col_matrix_shape(self):
        result = generate_field_layout({
            "design_type": "rcbd",
            "treatments": ["A", "B", "C"],
            "replications": 3,
            "seed": 5,
        })
        rcm = result["export_ready"]["row_col_matrix"]
        # RCBD fieldbook has row/column fields; should produce a 2-D matrix
        self.assertGreater(len(rcm), 0)
        for row in rcm:
            for cell in row:
                self.assertIn("row", cell)
                self.assertIn("col", cell)
                self.assertIn("plot_id", cell)
                self.assertIn("treatment", cell)

    def test_rcbd_field_map_labels_matches_row_col_matrix(self):
        result = generate_field_layout({
            "design_type": "rcbd",
            "treatments": ["A", "B", "C"],
            "replications": 3,
            "seed": 5,
        })
        fml = result["export_ready"]["field_map_labels"]
        rcm = result["export_ready"]["row_col_matrix"]
        self.assertEqual(len(fml), len(rcm))
        for label_row, cell_row in zip(fml, rcm):
            self.assertEqual(len(label_row), len(cell_row))
            for label, cell in zip(label_row, cell_row):
                self.assertIn(str(cell["plot_id"]), label)
                self.assertIn(cell["treatment"], label)


if __name__ == "__main__":
    unittest.main()
