#!/usr/bin/env python3
"""Run and compare the Genetic Algorithm against common baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from keyboard_optimization.baselines import random_search
from keyboard_optimization.corpus import count_text_bigrams, load_brown_train_test
from keyboard_optimization.genetic_algorithm import GAConfig, run_genetic_algorithm
from keyboard_optimization.keyboard import QWERTY_LAYOUT
from keyboard_optimization.objective import KeyboardObjective

from run_phase1 import SAMPLE_TESTING_TEXTS, SAMPLE_TRAINING_TEXTS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brown", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluation-budget", type=int, default=10_000)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.90)
    parser.add_argument("--mutation-rate", type=float, default=0.20)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ga_smoke_test.json"),
    )
    return parser.parse_args()


def improvement_pct(baseline_cost: float, candidate_cost: float) -> float:
    return 100.0 * (baseline_cost - candidate_cost) / baseline_cost


def main() -> None:
    arguments = parse_arguments()
    if arguments.brown:
        split = load_brown_train_test(seed=arguments.seed)
        training_matrix = split.training_matrix
        testing_matrix = split.testing_matrix
        dataset_name = "NLTK Brown Corpus"
        corpus_metadata = {
            "training_documents": list(split.training_document_ids),
            "testing_documents": list(split.testing_document_ids),
            "training_bigrams": split.training_bigrams,
            "testing_bigrams": split.testing_bigrams,
        }
    else:
        training_matrix, training_total = count_text_bigrams(SAMPLE_TRAINING_TEXTS)
        testing_matrix, testing_total = count_text_bigrams(SAMPLE_TESTING_TEXTS)
        dataset_name = "built-in smoke-test text (not for final reporting)"
        corpus_metadata = {
            "training_bigrams": training_total,
            "testing_bigrams": testing_total,
        }

    training_objective = KeyboardObjective(training_matrix)
    testing_objective = KeyboardObjective(testing_matrix)
    qwerty_train = training_objective.cost(QWERTY_LAYOUT)
    qwerty_test = testing_objective.cost(QWERTY_LAYOUT)

    random_result = random_search(
        training_objective,
        evaluations=arguments.evaluation_budget,
        seed=arguments.seed,
    )
    random_test = testing_objective.cost(random_result.best_layout)

    config = GAConfig(
        population_size=arguments.population_size,
        evaluation_budget=arguments.evaluation_budget,
        tournament_size=arguments.tournament_size,
        crossover_rate=arguments.crossover_rate,
        mutation_rate=arguments.mutation_rate,
        elite_count=arguments.elite_count,
    )
    ga_result = run_genetic_algorithm(
        training_objective,
        config=config,
        seed=arguments.seed,
    )
    ga_test = testing_objective.cost(ga_result.best_layout)

    result = {
        "dataset": dataset_name,
        "seed": arguments.seed,
        "evaluation_budget_per_algorithm": arguments.evaluation_budget,
        "distance_metric": "euclidean",
        "objective": "normalized average within-word bigram movement",
        "qwerty": {
            "layout": QWERTY_LAYOUT,
            "training_cost": qwerty_train,
            "testing_cost": qwerty_test,
        },
        "random_search": {
            "best_layout": random_result.best_layout,
            "training_cost": random_result.best_cost,
            "testing_cost": random_test,
            "training_improvement_over_qwerty_pct": improvement_pct(
                qwerty_train, random_result.best_cost
            ),
            "testing_improvement_over_qwerty_pct": improvement_pct(
                qwerty_test, random_test
            ),
        },
        "genetic_algorithm": {
            "configuration": {
                "population_size": config.population_size,
                "tournament_size": config.tournament_size,
                "crossover_rate": config.crossover_rate,
                "mutation_rate": config.mutation_rate,
                "elite_count": config.elite_count,
                "seed_layout": config.seed_layout,
            },
            "best_layout": ga_result.best_layout,
            "training_cost": ga_result.best_cost,
            "testing_cost": ga_test,
            "training_improvement_over_qwerty_pct": improvement_pct(
                qwerty_train, ga_result.best_cost
            ),
            "testing_improvement_over_qwerty_pct": improvement_pct(
                qwerty_test, ga_test
            ),
            "evaluations": ga_result.evaluations,
            "generations_completed": ga_result.generations_completed,
            "runtime_seconds": ga_result.runtime_seconds,
            "best_cost_history": list(ga_result.best_cost_history),
        },
        "corpus_metadata": corpus_metadata,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Dataset: {dataset_name}")
    print(f"QWERTY train/test: {qwerty_train:.6f} / {qwerty_test:.6f}")
    print(
        "Random train/test: "
        f"{random_result.best_cost:.6f} / {random_test:.6f} "
        f"({improvement_pct(qwerty_test, random_test):.3f}% test improvement)"
    )
    print(f"GA best layout: {ga_result.best_layout}")
    print(
        "GA train/test: "
        f"{ga_result.best_cost:.6f} / {ga_test:.6f} "
        f"({improvement_pct(qwerty_test, ga_test):.3f}% test improvement)"
    )
    print(
        f"GA evaluations/generations/runtime: {ga_result.evaluations} / "
        f"{ga_result.generations_completed} / {ga_result.runtime_seconds:.4f} s"
    )
    print(f"Saved: {arguments.output}")


if __name__ == "__main__":
    main()
