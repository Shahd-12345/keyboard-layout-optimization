"""Tests for repeated-experiment statistical analysis."""

from __future__ import annotations

import unittest

import pandas as pd

from keyboard_optimization.analysis import (
    cliffs_delta,
    cliffs_delta_magnitude,
    paired_method_comparison,
    summarize_results,
)


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = pd.DataFrame(
            {
                "algorithm": ["A", "A", "A", "B", "B", "B"],
                "seed": [1, 2, 3, 1, 2, 3],
                "test_cost": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
            }
        )

    def test_cliffs_delta_direction_for_minimization(self) -> None:
        delta = cliffs_delta([1.0, 1.1, 1.2], [2.0, 2.1, 2.2])
        self.assertEqual(delta, -1.0)
        self.assertEqual(cliffs_delta_magnitude(delta), "large")

    def test_summary_has_one_row_per_algorithm(self) -> None:
        summary = summarize_results(self.results, metric="test_cost")
        self.assertEqual(set(summary["algorithm"]), {"A", "B"})
        self.assertTrue((summary["n"] == 3).all())
        self.assertIn("minimum", summary.columns)
        self.assertIn("maximum", summary.columns)
        self.assertNotIn("best_min", summary.columns)
        self.assertNotIn("worst_max", summary.columns)
        self.assertTrue((summary["ci95_low"] <= summary["mean"]).all())
        self.assertTrue((summary["ci95_high"] >= summary["mean"]).all())

    def test_paired_comparison_matches_by_seed(self) -> None:
        comparison = paired_method_comparison(
            self.results,
            first_method="A",
            second_method="B",
            metric="test_cost",
        )
        self.assertEqual(comparison["matched_runs"], 3)
        self.assertEqual(comparison["cliffs_delta"], -1.0)
        self.assertLessEqual(comparison["p_value"], 0.25)


if __name__ == "__main__":
    unittest.main()
