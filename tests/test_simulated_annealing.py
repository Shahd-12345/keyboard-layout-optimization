"""Tests for standard and adaptive Simulated Annealing."""

from __future__ import annotations

import unittest

from keyboard_optimization.corpus import count_text_bigrams
from keyboard_optimization.keyboard import QWERTY_LAYOUT, validate_layout
from keyboard_optimization.objective import KeyboardObjective
from keyboard_optimization.simulated_annealing import (
    AdaptiveSAConfig,
    SAConfig,
    run_adaptive_simulated_annealing,
    run_simulated_annealing,
)


class SimulatedAnnealingTests(unittest.TestCase):
    def setUp(self) -> None:
        matrix, _ = count_text_bigrams(
            [
                "the quick brown fox jumps over the lazy dog while adaptive "
                "simulated annealing optimizes a keyboard layout"
            ]
        )
        self.objective = KeyboardObjective(matrix)

    def test_standard_sa_uses_exact_budget_and_valid_layout(self) -> None:
        config = SAConfig(
            evaluation_budget=137,
            initial_temperature=1.0,
            cooling_rate=0.98,
            temperature_record_interval=20,
        )
        result = run_simulated_annealing(self.objective, config=config, seed=21)
        self.assertEqual(result.evaluations, 137)
        self.assertEqual(len(result.best_cost_history), 137)
        self.assertEqual(validate_layout(result.best_layout), result.best_layout)
        self.assertTrue(
            all(
                later <= earlier
                for earlier, later in zip(
                    result.best_cost_history,
                    result.best_cost_history[1:],
                )
            )
        )
        self.assertLessEqual(result.best_cost, self.objective.cost(QWERTY_LAYOUT))

    def test_standard_sa_is_reproducible_for_fixed_seed(self) -> None:
        config = SAConfig(evaluation_budget=180, cooling_rate=0.99)
        first = run_simulated_annealing(self.objective, config=config, seed=22)
        second = run_simulated_annealing(self.objective, config=config, seed=22)
        self.assertEqual(first.best_layout, second.best_layout)
        self.assertEqual(first.best_cost, second.best_cost)
        self.assertEqual(first.best_cost_history, second.best_cost_history)
        self.assertEqual(first.temperature_history, second.temperature_history)

    def test_adaptive_sa_uses_exact_budget_and_is_reproducible(self) -> None:
        config = AdaptiveSAConfig(
            evaluation_budget=183,
            initial_temperature=1.0,
            cooling_rate=0.99,
            adaptation_window=20,
        )
        first = run_adaptive_simulated_annealing(
            self.objective,
            config=config,
            seed=23,
        )
        second = run_adaptive_simulated_annealing(
            self.objective,
            config=config,
            seed=23,
        )
        self.assertEqual(first.evaluations, 183)
        self.assertEqual(len(first.best_cost_history), 183)
        self.assertEqual(validate_layout(first.best_layout), first.best_layout)
        self.assertEqual(first.best_layout, second.best_layout)
        self.assertEqual(first.best_cost_history, second.best_cost_history)
        self.assertEqual(first.temperature_history, second.temperature_history)
        self.assertEqual(
            first.acceptance_ratio_history,
            second.acceptance_ratio_history,
        )
        self.assertTrue(
            all(
                later <= earlier
                for earlier, later in zip(
                    first.best_cost_history,
                    first.best_cost_history[1:],
                )
            )
        )

    def test_adaptive_sa_changes_schedule_when_acceptance_is_high(self) -> None:
        config = AdaptiveSAConfig(
            evaluation_budget=61,
            initial_temperature=1_000.0,
            cooling_rate=0.999,
            adaptation_window=5,
            target_acceptance_low=0.0,
            target_acceptance_high=0.01,
            accelerated_cooling_factor=0.5,
            maximum_temperature=1_000.0,
        )
        result = run_adaptive_simulated_annealing(
            self.objective,
            config=config,
            seed=24,
        )
        actions = [action for _, action in result.adaptation_history]
        self.assertIn("accelerated_cooling", actions)
        self.assertGreater(len(result.acceptance_ratio_history), 1)

    def test_invalid_adaptive_acceptance_targets_are_rejected(self) -> None:
        config = AdaptiveSAConfig(
            target_acceptance_low=0.7,
            target_acceptance_high=0.4,
        )
        with self.assertRaises(ValueError):
            run_adaptive_simulated_annealing(
                self.objective,
                config=config,
                seed=25,
            )


if __name__ == "__main__":
    unittest.main()
