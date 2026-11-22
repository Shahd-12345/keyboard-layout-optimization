"""Adaptive permutation Genetic Algorithm for keyboard layout optimization.

The controller increases mutation when the population loses positional
diversity or when the global best cost stagnates.  When search is progressing
with sufficient diversity, it gradually returns mutation toward its minimum.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass

from .baselines import random_layout
from .genetic_algorithm import order_crossover, swap_mutation
from .keyboard import QWERTY_LAYOUT, validate_layout
from .objective import KeyboardObjective


@dataclass(frozen=True)
class AdaptiveGAConfig:
    """Configuration for one adaptive Genetic Algorithm run."""

    population_size: int = 100
    evaluation_budget: int = 10_000
    tournament_size: int = 3
    crossover_rate: float = 0.90
    initial_mutation_rate: float = 0.15
    minimum_mutation_rate: float = 0.05
    maximum_mutation_rate: float = 0.60
    mutation_increase_factor: float = 1.50
    mutation_decrease_factor: float = 0.90
    diversity_threshold: float = 0.30
    stagnation_generations: int = 8
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
        if not 0.0 <= self.minimum_mutation_rate <= 1.0:
            raise ValueError("minimum_mutation_rate must be between 0 and 1.")
        if not 0.0 <= self.maximum_mutation_rate <= 1.0:
            raise ValueError("maximum_mutation_rate must be between 0 and 1.")
        if self.minimum_mutation_rate > self.maximum_mutation_rate:
            raise ValueError(
                "minimum_mutation_rate cannot exceed maximum_mutation_rate."
            )
        if not (
            self.minimum_mutation_rate
            <= self.initial_mutation_rate
            <= self.maximum_mutation_rate
        ):
            raise ValueError(
                "initial_mutation_rate must lie between its minimum and maximum."
            )
        if self.mutation_increase_factor <= 1.0:
            raise ValueError("mutation_increase_factor must be greater than 1.")
        if not 0.0 < self.mutation_decrease_factor <= 1.0:
            raise ValueError("mutation_decrease_factor must be in (0, 1].")
        if not 0.0 <= self.diversity_threshold <= 1.0:
            raise ValueError("diversity_threshold must be between 0 and 1.")
        if self.stagnation_generations < 1:
            raise ValueError("stagnation_generations must be positive.")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be in [0, population_size).")
        if self.seed_layout is not None:
            validate_layout(self.seed_layout)


@dataclass(frozen=True)
class AdaptiveGAResult:
    """Best solution and diagnostic histories from one adaptive GA run."""

    best_layout: str
    best_cost: float
    evaluations: int
    generations_completed: int
    runtime_seconds: float
    best_cost_history: tuple[float, ...]
    mutation_rate_history: tuple[float, ...]
    diversity_history: tuple[float, ...]


def population_positional_diversity(population: list[str]) -> float:
    """Return mean normalized positional entropy in the interval [0, 1].

    A value of zero means every layout assigns the same letter to every
    position.  Larger values indicate that letters remain distributed across
    different physical key positions.  Entropy is normalized by log(26).
    """
    if not population:
        raise ValueError("population cannot be empty.")
    normalized_population = [validate_layout(layout) for layout in population]
    population_size = len(normalized_population)
    maximum_entropy = math.log(26)
    position_entropies: list[float] = []

    for position in range(26):
        counts = Counter(layout[position] for layout in normalized_population)
        entropy = -sum(
            (count / population_size) * math.log(count / population_size)
            for count in counts.values()
        )
        position_entropies.append(entropy / maximum_entropy)

    diversity = sum(position_entropies) / len(position_entropies)
    return min(1.0, max(0.0, diversity))


def _initial_population(
    config: AdaptiveGAConfig,
    random_generator: random.Random,
) -> list[str]:
    population: list[str] = []
    seen: set[str] = set()

    if config.seed_layout is not None:
        seed_layout = validate_layout(config.seed_layout)
        population.append(seed_layout)
        seen.add(seed_layout)

    while len(population) < config.population_size:
        candidate = random_layout(random_generator)
        if candidate not in seen:
            population.append(candidate)
            seen.add(candidate)
    return population


def _tournament_select(
    population: list[str],
    costs: list[float],
    tournament_size: int,
    random_generator: random.Random,
) -> str:
    contestants = random_generator.sample(range(len(population)), tournament_size)
    winner_index = min(contestants, key=lambda index: costs[index])
    return population[winner_index]


def run_adaptive_genetic_algorithm(
    objective: KeyboardObjective,
    *,
    config: AdaptiveGAConfig,
    seed: int,
) -> AdaptiveGAResult:
    """Run the adaptive GA under an exact objective-evaluation budget."""
    config.validate()
    random_generator = random.Random(seed)
    start_time = time.perf_counter()

    population = _initial_population(config, random_generator)
    costs: list[float] = []
    best_layout = ""
    best_cost = float("inf")
    best_cost_history: list[float] = []

    for layout in population:
        cost = objective.cost(layout)
        costs.append(cost)
        if cost < best_cost:
            best_layout = layout
            best_cost = cost
        best_cost_history.append(best_cost)

    evaluations = len(population)
    generations_completed = 0
    stagnant_generations = 0
    current_mutation_rate = config.initial_mutation_rate
    mutation_rate_history: list[float] = []
    diversity_history: list[float] = []

    while evaluations < config.evaluation_budget:
        generation_start_best = best_cost
        ranked_indices = sorted(range(len(population)), key=lambda index: costs[index])
        elite_indices = ranked_indices[: config.elite_count]
        next_population = [population[index] for index in elite_indices]
        next_costs = [costs[index] for index in elite_indices]

        while (
            len(next_population) < config.population_size
            and evaluations < config.evaluation_budget
        ):
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
                if (
                    len(next_population) >= config.population_size
                    or evaluations >= config.evaluation_budget
                ):
                    break
                if random_generator.random() < current_mutation_rate:
                    child = swap_mutation(child, random_generator)

                child_cost = objective.cost(child)
                evaluations += 1
                next_population.append(child)
                next_costs.append(child_cost)

                if child_cost < best_cost:
                    best_layout = child
                    best_cost = child_cost
                best_cost_history.append(best_cost)

        if len(next_population) < config.population_size:
            break

        population = next_population
        costs = next_costs
        generations_completed += 1

        if best_cost < generation_start_best - 1e-12:
            stagnant_generations = 0
        else:
            stagnant_generations += 1

        diversity = population_positional_diversity(population)
        if (
            diversity < config.diversity_threshold
            or stagnant_generations >= config.stagnation_generations
        ):
            current_mutation_rate = min(
                config.maximum_mutation_rate,
                current_mutation_rate * config.mutation_increase_factor,
            )
            if stagnant_generations >= config.stagnation_generations:
                stagnant_generations = 0
        else:
            current_mutation_rate = max(
                config.minimum_mutation_rate,
                current_mutation_rate * config.mutation_decrease_factor,
            )

        diversity_history.append(diversity)
        mutation_rate_history.append(current_mutation_rate)

    return AdaptiveGAResult(
        best_layout=validate_layout(best_layout),
        best_cost=best_cost,
        evaluations=evaluations,
        generations_completed=generations_completed,
        runtime_seconds=time.perf_counter() - start_time,
        best_cost_history=tuple(best_cost_history),
        mutation_rate_history=tuple(mutation_rate_history),
        diversity_history=tuple(diversity_history),
    )
