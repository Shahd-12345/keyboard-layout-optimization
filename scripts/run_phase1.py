#!/usr/bin/env python3
"""Verify the common benchmark and run QWERTY/Random Search baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keyboard_optimization.baselines import random_search
from keyboard_optimization.corpus import count_text_bigrams, load_brown_train_test
from keyboard_optimization.keyboard import QWERTY_LAYOUT
from keyboard_optimization.objective import KeyboardObjective

SAMPLE_TRAINING_TEXTS = [
    (
        "Heuristic optimization searches for high quality keyboard layouts. "
        "A genetic algorithm explores many permutations while preserving valid layouts."
    ),
    (
        "Frequent letter pairs should be placed on nearby keys to reduce modeled movement."
    ),
]

SAMPLE_TESTING_TEXTS = [
    (
        "An unseen text checks whether an optimized keyboard generalizes beyond "
        "the text used to estimate letter pair frequencies."
    )
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--brown",
        action="store_true",
        help="Use a reproducible 80/20 Brown Corpus document split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-evaluations", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1_baseline.json"),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.brown:
        split = load_brown_train_test(seed=arguments.seed)
        training_matrix = split.training_matrix
        testing_matrix = split.testing_matrix
        dataset_name = "NLTK Brown Corpus"
        metadata = {
            "training_documents": list(split.training_document_ids),
            "testing_documents": list(split.testing_document_ids),
            "training_bigrams": split.training_bigrams,
            "testing_bigrams": split.testing_bigrams,
        }
    else:
        training_matrix, training_total = count_text_bigrams(SAMPLE_TRAINING_TEXTS)
        testing_matrix, testing_total = count_text_bigrams(SAMPLE_TESTING_TEXTS)
        dataset_name = "built-in smoke-test text (not for final reporting)"
        metadata = {
            "training_bigrams": training_total,
            "testing_bigrams": testing_total,
        }

    training_objective = KeyboardObjective(training_matrix)
    testing_objective = KeyboardObjective(testing_matrix)

    qwerty_training_cost = training_objective.cost(QWERTY_LAYOUT)
    qwerty_testing_cost = testing_objective.cost(QWERTY_LAYOUT)
    random_result = random_search(
        training_objective,
        evaluations=arguments.random_evaluations,
        seed=arguments.seed,
    )
    random_test_cost = testing_objective.cost(random_result.best_layout)

    result = {
        "dataset": dataset_name,
        "seed": arguments.seed,
        "distance_metric": "euclidean",
        "objective": "normalized average within-word bigram movement",
        "random_search_evaluations": arguments.random_evaluations,
        "qwerty": {
            "layout": QWERTY_LAYOUT,
            "training_cost": qwerty_training_cost,
            "testing_cost": qwerty_testing_cost,
        },
        "random_search": {
            "best_layout": random_result.best_layout,
            "training_cost": random_result.best_cost,
            "testing_cost": random_test_cost,
            "training_improvement_over_qwerty_pct": (
                100.0
                * (qwerty_training_cost - random_result.best_cost)
                / qwerty_training_cost
            ),
            "testing_improvement_over_qwerty_pct": (
                100.0 * (qwerty_testing_cost - random_test_cost) / qwerty_testing_cost
            ),
        },
        "corpus_metadata": metadata,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Dataset: {dataset_name}")
    print(f"QWERTY training cost: {qwerty_training_cost:.6f}")
    print(f"QWERTY testing cost:  {qwerty_testing_cost:.6f}")
    print(f"Random best layout:  {random_result.best_layout}")
    print(f"Random training cost: {random_result.best_cost:.6f}")
    print(f"Random testing cost:  {random_test_cost:.6f}")
    print(
        "Random train improvement over QWERTY: "
        f"{result['random_search']['training_improvement_over_qwerty_pct']:.3f}%"
    )
    print(
        "Random test improvement over QWERTY:  "
        f"{result['random_search']['testing_improvement_over_qwerty_pct']:.3f}%"
    )
    print(f"Saved: {arguments.output}")


if __name__ == "__main__":
    main()

