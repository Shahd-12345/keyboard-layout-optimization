"""Unit tests for the shared keyboard-layout benchmark."""

from __future__ import annotations

import string
import random
import unittest

import numpy as np

from keyboard_optimization.baselines import random_search
from keyboard_optimization.corpus import count_text_bigrams, count_token_bigrams
from keyboard_optimization.genetic_algorithm import (
    GAConfig,
    order_crossover,
    run_genetic_algorithm,
    swap_mutation,
)
from keyboard_optimization.keyboard import (
    QWERTY_LAYOUT,
    build_distance_matrix,
    validate_layout,
)
from keyboard_optimization.objective import KeyboardObjective


class CorpusTests(unittest.TestCase):
    def test_counts_within_words_without_crossing_space(self) -> None:
        matrix, total = count_text_bigrams(["abba cab"], normalize=False)
        index = {letter: ord(letter) - ord("a") for letter in string.ascii_lowercase}
        self.assertEqual(total, 5)
        self.assertEqual(matrix[index["a"], index["b"]], 2)
        self.assertEqual(matrix[index["b"], index["b"]], 1)
        self.assertEqual(matrix[index["b"], index["a"]], 1)
        self.assertEqual(matrix[index["c"], index["a"]], 1)

    def test_normalized_matrix_sums_to_one(self) -> None:
        matrix, total = count_token_bigrams(["keyboard", "layout"])
        self.assertGreater(total, 0)
        self.assertAlmostEqual(float(matrix.sum()), 1.0)

    def test_no_alphabetic_bigrams_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            count_text_bigrams(["1 ! ? a"])


class KeyboardTests(unittest.TestCase):
    def test_qwerty_is_valid(self) -> None:
        self.assertEqual(validate_layout(QWERTY_LAYOUT), QWERTY_LAYOUT)

    def test_duplicate_letter_is_rejected(self) -> None:
        invalid = QWERTY_LAYOUT[:-1] + QWERTY_LAYOUT[0]
        with self.assertRaises(ValueError):
            validate_layout(invalid)

    def test_distance_matrix_properties(self) -> None:
        distances = build_distance_matrix()
        self.assertEqual(distances.shape, (26, 26))
        self.assertTrue(np.allclose(distances, distances.T))
        self.assertTrue(np.allclose(np.diag(distances), 0.0))


class ObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        matrix, _ = count_text_bigrams(
            ["the quick brown fox jumps over the lazy dog keyboard optimization"]
        )
        self.objective = KeyboardObjective(matrix)

    def test_cost_is_deterministic_and_nonnegative(self) -> None:
        first = self.objective.cost(QWERTY_LAYOUT)
        second = self.objective.cost(QWERTY_LAYOUT)
        self.assertGreaterEqual(first, 0.0)
        self.assertAlmostEqual(first, second)

    def test_qwerty_improvement_over_itself_is_zero(self) -> None:
        self.assertAlmostEqual(
            self.objective.improvement_pct(QWERTY_LAYOUT, QWERTY_LAYOUT),
            0.0,
        )

    def test_random_search_history_is_monotone(self) -> None:
        result = random_search(self.objective, evaluations=50, seed=7)
        self.assertEqual(result.evaluations, 50)
        self.assertEqual(len(result.best_cost_history), 50)
        self.assertTrue(
            all(
                later <= earlier
                for earlier, later in zip(
                    result.best_cost_history,
                    result.best_cost_history[1:],
                )
            )
        )
        self.assertEqual(result.best_cost, result.best_cost_history[-1])


class GeneticAlgorithmTests(unittest.TestCase):
    def setUp(self) -> None:
        matrix, _ = count_text_bigrams(
            ["the quick brown fox jumps over the lazy dog keyboard optimization"]
        )
        self.objective = KeyboardObjective(matrix)

    def test_order_crossover_preserves_permutations(self) -> None:
        other = QWERTY_LAYOUT[::-1]
        first, second = order_crossover(QWERTY_LAYOUT, other, random.Random(11))
        self.assertEqual(validate_layout(first), first)
        self.assertEqual(validate_layout(second), second)

    def test_swap_mutation_preserves_permutation_and_changes_layout(self) -> None:
        mutated = swap_mutation(QWERTY_LAYOUT, random.Random(12))
        self.assertEqual(validate_layout(mutated), mutated)
        self.assertNotEqual(mutated, QWERTY_LAYOUT)

    def test_ga_uses_exact_budget_and_has_monotone_history(self) -> None:
        config = GAConfig(population_size=20, evaluation_budget=137)
        result = run_genetic_algorithm(self.objective, config=config, seed=13)
        self.assertEqual(result.evaluations, 137)
        self.assertEqual(len(result.best_cost_history), 137)
        self.assertTrue(
            all(
                later <= earlier
                for earlier, later in zip(
                    result.best_cost_history,
                    result.best_cost_history[1:]
                )
            )
        )
        self.assertLessEqual(result.best_cost, self.objective.cost(QWERTY_LAYOUT))

    def test_ga_is_reproducible_for_a_fixed_seed(self) -> None:
        config = GAConfig(population_size=20, evaluation_budget=120)
        first = run_genetic_algorithm(self.objective, config=config, seed=14)
        second = run_genetic_algorithm(self.objective, config=config, seed=14)
        self.assertEqual(first.best_layout, second.best_layout)
        self.assertEqual(first.best_cost, second.best_cost)
        self.assertEqual(first.best_cost_history, second.best_cost_history)


if __name__ == "__main__":
    unittest.main()

