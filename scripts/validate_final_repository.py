#!/usr/bin/env python3
"""Validate preserved final results and the executed notebook."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from keyboard_optimization.experiments import ALGORITHM_ORDER
from keyboard_optimization.keyboard import validate_layout


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    final_directory = project_root / "results" / "final"
    results = pd.read_csv(final_directory / "run_results.csv")

    expected_rows = 30 * len(ALGORITHM_ORDER)
    if len(results) != expected_rows:
        raise ValueError(f"Expected {expected_rows} run rows; received {len(results)}.")
    counts = results.groupby("algorithm").size().to_dict()
    if any(counts.get(algorithm) != 30 for algorithm in ALGORITHM_ORDER):
        raise ValueError(f"Unexpected algorithm run counts: {counts}")
    if set(results["seed"]) != set(range(100, 130)):
        raise ValueError("Final seeds must be exactly 100 through 129.")
    if not results["evaluations"].eq(10_000).all():
        raise ValueError("Every final run must use exactly 10,000 evaluations.")
    if results.isna().any().any():
        raise ValueError("Final run table contains missing values.")
    for layout in results["best_layout"]:
        validate_layout(layout)

    histories = np.load(final_directory / "convergence_histories.npz")
    for algorithm in ALGORITHM_ORDER:
        history = histories[algorithm]
        if history.shape != (30, 10_000):
            raise ValueError(f"Unexpected {algorithm} history shape: {history.shape}")
        if not np.isfinite(history).all():
            raise ValueError(f"{algorithm} history has a non-finite value.")
        if not (np.diff(history, axis=1) <= 1e-12).all():
            raise ValueError(f"{algorithm} best-so-far history is not monotone.")

    notebook_path = project_root / "notebooks" / "keyboard_layout_optimization.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if len(code_cells) != 15:
        raise ValueError(f"Expected 15 code cells; received {len(code_cells)}.")
    if any(cell.get("execution_count") is None for cell in code_cells):
        raise ValueError("The final notebook contains an unexecuted code cell.")
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise ValueError(f"The final notebook contains {len(errors)} error outputs.")

    print("Final repository validation passed.")
    print(f"- {len(results)} complete stochastic runs")
    print("- 30 runs and 10,000 evaluations per method")
    print("- valid layouts and monotone convergence histories")
    print("- 15/15 notebook code cells executed without errors")


if __name__ == "__main__":
    main()
