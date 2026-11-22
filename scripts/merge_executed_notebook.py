#!/usr/bin/env python3
"""Merge verified outputs into the current canonical notebook source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("executed_notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def replace_legacy_labels(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("best_min", "minimum").replace(
            "worst_max", "maximum"
        )
    if isinstance(value, list):
        return [replace_legacy_labels(item) for item in value]
    if isinstance(value, dict):
        return {
            key: replace_legacy_labels(item) for key, item in value.items()
        }
    return value


def main() -> None:
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    canonical_path = project_root / "notebooks" / "keyboard_layout_optimization.ipynb"

    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    executed = json.loads(arguments.executed_notebook.read_text(encoding="utf-8"))
    if len(canonical["cells"]) != len(executed["cells"]):
        raise ValueError("Canonical and executed notebooks have different cell counts.")

    for index, (canonical_cell, executed_cell) in enumerate(
        zip(canonical["cells"], executed["cells"], strict=True)
    ):
        if canonical_cell["cell_type"] != executed_cell["cell_type"]:
            raise ValueError(f"Cell type mismatch at index {index}.")
        executed_cell["source"] = canonical_cell["source"]
        if executed_cell["cell_type"] == "code":
            executed_cell["outputs"] = replace_legacy_labels(
                executed_cell.get("outputs", [])
            )

    executed["metadata"] = canonical["metadata"]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(executed, indent=1), encoding="utf-8")
    print(arguments.output)


if __name__ == "__main__":
    main()
