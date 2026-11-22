"""Simple baselines used to validate the objective before advanced heuristics."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .keyboard import ALPHABET
from .objective import KeyboardObjective


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    best_layout: str
    best_cost: float
    evaluations: int
    best_cost_history: tuple[float, ...]


def random_layout(random_generator: random.Random) -> str:
    letters = list(ALPHABET)
    random_generator.shuffle(letters)
    return "".join(letters)


def random_search(
    objective: KeyboardObjective,
    *,
    evaluations: int,
    seed: int,
) -> SearchResult:
    """Evaluate independent random layouts under a fixed budget."""
    if evaluations <= 0:
        raise ValueError("evaluations must be positive.")

    random_generator = random.Random(seed)
    best_layout = ""
    best_cost = float("inf")
    history: list[float] = []

    for _ in range(evaluations):
        candidate = random_layout(random_generator)
        candidate_cost = objective.cost(candidate)
        if candidate_cost < best_cost:
            best_layout = candidate
            best_cost = candidate_cost
        history.append(best_cost)

    return SearchResult(
        algorithm="Random Search",
        best_layout=best_layout,
        best_cost=best_cost,
        evaluations=evaluations,
        best_cost_history=tuple(history),
    )

