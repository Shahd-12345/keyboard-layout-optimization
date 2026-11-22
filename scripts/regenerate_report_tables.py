#!/usr/bin/env python3
"""Regenerate compact report tables from preserved per-run results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from keyboard_optimization.analysis import (
    paired_method_comparison,
    summarize_results,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_directory = project_root / "results" / "final"
    results = pd.read_csv(results_directory / "run_results.csv")

    tables = {
        "test_cost_summary.csv": summarize_results(results, metric="test_cost"),
        "test_improvement_summary.csv": summarize_results(
            results, metric="test_improvement_over_qwerty_pct"
        ),
        "runtime_summary.csv": summarize_results(
            results, metric="runtime_seconds"
        ),
    }
    for filename, table in tables.items():
        table.to_csv(results_directory / filename, index=False)

    comparisons = pd.DataFrame(
        [
            paired_method_comparison(
                results,
                first_method="Adaptive GA",
                second_method="Standard GA",
                metric="test_cost",
            ),
            paired_method_comparison(
                results,
                first_method="Adaptive SA",
                second_method="Standard SA",
                metric="test_cost",
            ),
        ]
    )
    comparisons.to_csv(
        results_directory / "paired_statistical_tests.csv", index=False
    )

    print(f"Regenerated report tables in {results_directory}")


if __name__ == "__main__":
    main()
