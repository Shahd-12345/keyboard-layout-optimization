"""Tests for the matched-budget benchmark suite."""

from __future__ import annotations

import unittest

from keyboard_optimization.corpus import count_text_bigrams
from keyboard_optimization.experiments import (
    ALGORITHM_ORDER,
    ExperimentConfig,
    run_benchmark_suite,
)
from keyboard_optimization.objective import KeyboardObjective


class ExperimentTests(unittest.TestCase):
    def test_small_suite_has_matched_runs_and_exact_histories(self) -> None:
        training_matrix, _ = count_text_bigrams(
            ["adaptive keyboard layout optimization training text"]
        )
        testing_matrix, _ = count_text_bigrams(
            ["unseen keyboard layout evaluation text"]
        )
        config = ExperimentConfig(
            runs=2,
            evaluation_budget=120,
            first_seed=10,
            ga_population_size=20,
            sa_adaptation_window=20,
        )
        suite = run_benchmark_suite(
            KeyboardObjective(training_matrix),
            KeyboardObjective(testing_matrix),
            config=config,
            progress=None,
        )

        self.assertEqual(len(suite.runs), 2 * len(ALGORITHM_ORDER))
        self.assertEqual(set(suite.runs["seed"]), {10, 11})
        self.assertTrue((suite.runs["evaluations"] == 120).all())
        self.assertTrue(suite.runs["best_layout"].str.len().eq(26).all())
        for algorithm in ALGORITHM_ORDER:
            self.assertEqual(suite.histories[algorithm].shape, (2, 120))


if __name__ == "__main__":
    unittest.main()
