#!/usr/bin/env python3
"""Run the complete matched-seed experiment from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from keyboard_optimization.analysis import summarize_results
from keyboard_optimization.corpus import count_text_bigrams, load_brown_train_test
from keyboard_optimization.experiments import (
    ExperimentConfig,
    run_benchmark_suite,
    save_benchmark_suite,
)
from keyboard_optimization.objective import KeyboardObjective

from run_phase1 import SAMPLE_TESTING_TEXTS, SAMPLE_TRAINING_TEXTS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the tiny built-in software-check text instead of Brown.",
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--evaluation-budget", type=int, default=10_000)
    parser.add_argument("--first-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/final"),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.smoke:
        training_matrix, training_bigrams = count_text_bigrams(
            SAMPLE_TRAINING_TEXTS
        )
        testing_matrix, testing_bigrams = count_text_bigrams(SAMPLE_TESTING_TEXTS)
        dataset_name = "built-in smoke-test text (not for final reporting)"
        corpus_metadata: dict[str, object] = {
            "dataset": dataset_name,
            "training_bigrams": training_bigrams,
            "testing_bigrams": testing_bigrams,
        }
    else:
        split = load_brown_train_test(seed=arguments.first_seed)
        training_matrix = split.training_matrix
        testing_matrix = split.testing_matrix
        dataset_name = "NLTK Brown Corpus"
        corpus_metadata = {
            "dataset": dataset_name,
            "training_document_ids": list(split.training_document_ids),
            "testing_document_ids": list(split.testing_document_ids),
            "training_bigrams": split.training_bigrams,
            "testing_bigrams": split.testing_bigrams,
        }

    print(f"Dataset: {dataset_name}")
    config = ExperimentConfig(
        runs=arguments.runs,
        evaluation_budget=arguments.evaluation_budget,
        first_seed=arguments.first_seed,
    )
    suite = run_benchmark_suite(
        KeyboardObjective(training_matrix),
        KeyboardObjective(testing_matrix),
        config=config,
    )
    paths = save_benchmark_suite(
        suite,
        arguments.output_dir,
        corpus_metadata=corpus_metadata,
    )

    summary = summarize_results(suite.runs, metric="test_cost")
    print("\nHeld-out test-cost summary (lower is better):")
    print(summary.to_string(index=False))
    print("\nSaved files:")
    for path in paths.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
