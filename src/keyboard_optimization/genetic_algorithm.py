"""Permutation-safe Genetic Algorithm for keyboard layout optimization."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from .baselines import random_layout
from .keyboard import QWERTY_LAYOUT, validate_layout
from .objective import KeyboardObjective


@dataclass(frozen=True)
class GAConfig:
    """Configuration for one Genetic Algorithm run."""

    population_size: int = 100
    evaluation_budget: int = 10_000
    tournament_size: int = 3
    crossover_rate: float = 0.90
    mutation_rate: float = 0.20
    elite_count: int = 2
    seed_layout: str | None = QWERTY_LAYOUT

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.evaluation_budget < self.population_size:
            raise ValueError(
                "evaluation_budget must be at least as large as population_size."
            )
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError(
                "tournament_size must be between 2 and population_size."
            )
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1.")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1.")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be in [0, population_size).")
        if self.seed_layout is not None:
            validate_layout(self.seed_layout)


@dataclass(frozen=True)
class GAResult:
    """Result and convergence data for one GA run."""

    best_layout: str
    best_cost: float
    evaluations: int
    generations_completed: int
    runtime_seconds: float
    best_cost_history: tuple[float, ...]


def _ordered_child(
    segment_parent: str,
    fill_parent: str,
    start: int,
    end: int,
) -> str:
    """Build one order-crossover child using the half-open segment [start, end)."""
    child: list[str | None] = [None] * len(segment_parent)
    child[start:end] = segment_parent[start:end]
    retained = set(segment_parent[start:end])

    fill_letters = [
        letter
        for letter in fill_parent[end:] + fill_parent[:end]
        if letter not in retained
    ]
    fill_positions = list(range(end, len(child))) + list(range(0, start))
    for position, letter in zip(fill_positions, fill_letters, strict=True):
        child[position] = letter

    result = "".join(letter for letter in child if letter is not None)
    return validate_layout(result)


def order_crossover(
    first_parent: str,
    second_parent: str,
    random_generator: random.Random,
) -> tuple[str, str]:
    """Return two valid children using order crossover (OX)."""
    first = validate_layout(first_parent)
    second = validate_layout(second_parent)
    start, end = sorted(random_generator.sample(range(len(first) + 1), 2))
    if start == end:
        raise RuntimeError("Crossover points must be distinct.")
    return (
        _ordered_child(first, second, start, end),
        _ordered_child(second, first, start, end),
    )


def swap_mutation(layout: str, random_generator: random.Random) -> str:
    """Swap two positions while preserving the permutation constraint."""
    letters = list(validate_layout(layout))
    first, second = random_generator.sample(range(len(letters)), 2)
    letters[first], letters[second] = letters[second], letters[first]
    return validate_layout("".join(letters))


def _tournament_select(
    population: list[str],
    costs: list[float],
    tournament_size: int,
    random_generator: random.Random,
) -> str:
    contestants = random_generator.sample(range(len(population)), tournament_size)
    winner = min(contestants, key=lambda index: costs[index])
    return population[winner]


def _initial_population(config: GAConfig, random_generator: random.Random) -> list[str]:
    population: list[str] = []
    seen: set[str] = set()

    if config.seed_layout is not None:
        normalized_seed = validate_layout(config.seed_layout)
        population.append(normalized_seed)
        seen.add(normalized_seed)

    while len(population) < config.population_size:
        candidate = random_layout(random_generator)
        if candidate not in seen:
            population.append(candidate)
            seen.add(candidate)
    return population


def run_genetic_algorithm(
    objective: KeyboardObjective,
    *,
    config: GAConfig,
    seed: int,
) -> GAResult:
    """Run a minimization GA under an exact objective-evaluation budget."""
    config.validate()
    random_generator = random.Random(seed)
    start_time = time.perf_counter()

    population = _initial_population(config, random_generator)
    costs: list[float] = []
    best_layout = ""
    best_cost = float("inf")
    history: list[float] = []

    for layout in population:
        cost = objective.cost(layout)
        costs.append(cost)
        if cost < best_cost:
            best_layout = layout
            best_cost = cost
        history.append(best_cost)

    evaluations = len(population)
    generations_completed = 0

    while evaluations < config.evaluation_budget:
        ranked_indices = sorted(range(len(population)), key=lambda index: costs[index])
        elite_indices = ranked_indices[: config.elite_count]
        next_population = [population[index] for index in elite_indices]
        next_costs = [costs[index] for index in elite_indices]

        generation_finished = True
        while len(next_population) < config.population_size:
            first_parent = _tournament_select(
                population,
                costs,
                config.tournament_size,
                random_generator,
            )
            second_parent = _tournament_select(
                population,
                costs,
                config.tournament_size,
                random_generator,
            )

            if random_generator.random() < config.crossover_rate:
                children = order_crossover(
                    first_parent,
                    second_parent,
                    random_generator,
                )
            else:
                children = (first_parent, second_parent)

            for child in children:
                if len(next_population) >= config.population_size:
                    break
                if random_generator.random() < config.mutation_rate:
                    child = swap_mutation(child, random_generator)

                child_cost = objective.cost(child)
                evaluations += 1
                next_population.append(child)
                next_costs.append(child_cost)

                if child_cost < best_cost:
                    best_layout = child
                    best_cost = child_cost
                history.append(best_cost)

                if evaluations >= config.evaluation_budget:
                    generation_finished = False
                    break

            if evaluations >= config.evaluation_budget:
                break

        if generation_finished and len(next_population) == config.population_size:
            generations_completed += 1

        population = next_population
        costs = next_costs

    return GAResult(
        best_layout=best_layout,
        best_cost=best_cost,
        evaluations=evaluations,
        generations_completed=generations_completed,
        runtime_seconds=time.perf_counter() - start_time,
        best_cost_history=tuple(history),
    )
