"""Tests for the adaptive permutation Genetic Algorithm."""

from __future__ import annotations

import unittest

from keyboard_optimization.adaptive_genetic_algorithm import (
    AdaptiveGAConfig,
    population_positional_diversity,
    run_adaptive_genetic_algorithm,
)
from keyboard_optimization.corpus import count_text_bigrams
from keyboard_optimization.keyboard import QWERTY_LAYOUT, validate_layout
from keyboard_optimization.objective import KeyboardObjective


class AdaptiveGeneticAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        matrix, _ = count_text_bigrams(
            [
                "the quick brown fox jumps over the lazy dog while keyboard "
                "optimization moves frequent letter pairs closer together"
            ]
        )
        self.objective = KeyboardObjective(matrix)

    def test_positional_diversity_has_expected_bounds(self) -> None:
        self.assertAlmostEqual(population_positional_diversity([QWERTY_LAYOUT] * 4), 0.0)
        reversed_layout = QWERTY_LAYOUT[::-1]
        mixed = population_positional_diversity([QWERTY_LAYOUT, reversed_layout])
        self.assertGreater(mixed, 0.0)
        self.assertLessEqual(mixed, 1.0)

    def test_exact_budget_valid_layout_and_monotone_history(self) -> None:
        config = AdaptiveGAConfig(population_size=20, evaluation_budget=137)
        result = run_adaptive_genetic_algorithm(
            self.objective,
            config=config,
            seed=21,
        )
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
        self.assertEqual(
            len(result.mutation_rate_history),
            result.generations_completed,
        )
        self.assertEqual(
            len(result.diversity_history),
            result.generations_completed,
        )

    def test_fixed_seed_is_reproducible(self) -> None:
        config = AdaptiveGAConfig(population_size=20, evaluation_budget=180)
        first = run_adaptive_genetic_algorithm(self.objective, config=config, seed=22)
        second = run_adaptive_genetic_algorithm(self.objective, config=config, seed=22)
        self.assertEqual(first.best_layout, second.best_layout)
        self.assertEqual(first.best_cost, second.best_cost)
        self.assertEqual(first.best_cost_history, second.best_cost_history)
        self.assertEqual(first.mutation_rate_history, second.mutation_rate_history)
        self.assertEqual(first.diversity_history, second.diversity_history)

    def test_controller_changes_mutation_rate(self) -> None:
        config = AdaptiveGAConfig(
            population_size=20,
            evaluation_budget=220,
            initial_mutation_rate=0.10,
            minimum_mutation_rate=0.05,
            maximum_mutation_rate=0.80,
            mutation_increase_factor=2.0,
            mutation_decrease_factor=0.90,
            diversity_threshold=1.0,
            stagnation_generations=1,
        )
        result = run_adaptive_genetic_algorithm(self.objective, config=config, seed=23)
        self.assertGreater(len(result.mutation_rate_history), 1)
        self.assertNotEqual(result.mutation_rate_history[0], config.initial_mutation_rate)
        self.assertTrue(
            all(
                config.minimum_mutation_rate
                <= rate
                <= config.maximum_mutation_rate
                for rate in result.mutation_rate_history
            )
        )


if __name__ == "__main__":
    unittest.main()
