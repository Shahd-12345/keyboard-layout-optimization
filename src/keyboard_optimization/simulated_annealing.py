"""Standard and acceptance-ratio-adaptive Simulated Annealing."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .keyboard import QWERTY_LAYOUT, validate_layout
from .objective import KeyboardObjective


@dataclass(frozen=True)
class SAConfig:
    """Configuration for standard geometric-cooling Simulated Annealing."""

    evaluation_budget: int = 10_000
    initial_temperature: float = 1.0
    cooling_rate: float = 0.995
    minimum_temperature: float = 1e-6
    temperature_record_interval: int = 100
    start_layout: str = QWERTY_LAYOUT

    def validate(self) -> None:
        if self.evaluation_budget < 1:
            raise ValueError("evaluation_budget must be at least 1.")
        if self.initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive.")
        if not 0.0 < self.cooling_rate < 1.0:
            raise ValueError("cooling_rate must be strictly between 0 and 1.")
        if self.minimum_temperature <= 0.0:
            raise ValueError("minimum_temperature must be positive.")
        if self.minimum_temperature > self.initial_temperature:
            raise ValueError(
                "minimum_temperature cannot exceed initial_temperature."
            )
        if self.temperature_record_interval < 1:
            raise ValueError("temperature_record_interval must be at least 1.")
        validate_layout(self.start_layout)


@dataclass(frozen=True)
class AdaptiveSAConfig(SAConfig):
    """Configuration for acceptance-ratio-adaptive Simulated Annealing."""

    adaptation_window: int = 100
    target_acceptance_low: float = 0.20
    target_acceptance_high: float = 0.50
    reheat_factor: float = 1.50
    accelerated_cooling_factor: float = 0.80
    maximum_temperature: float = 10.0

    def validate(self) -> None:
        super().validate()
        if self.adaptation_window < 1:
            raise ValueError("adaptation_window must be at least 1.")
        if not (
            0.0
            <= self.target_acceptance_low
            < self.target_acceptance_high
            <= 1.0
        ):
            raise ValueError(
                "Acceptance targets must satisfy "
                "0 <= low < high <= 1."
            )
        if self.reheat_factor <= 1.0:
            raise ValueError("reheat_factor must be greater than 1.")
        if not 0.0 < self.accelerated_cooling_factor < 1.0:
            raise ValueError(
                "accelerated_cooling_factor must be strictly between 0 and 1."
            )
        if self.maximum_temperature < self.initial_temperature:
            raise ValueError(
                "maximum_temperature cannot be below initial_temperature."
            )


@dataclass(frozen=True)
class SAResult:
    """Outcome and recorded trajectory for standard Simulated Annealing."""

    best_layout: str
    best_cost: float
    final_layout: str
    final_cost: float
    evaluations: int
    accepted_moves: int
    runtime_seconds: float
    best_cost_history: tuple[float, ...]
    temperature_history: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class AdaptiveSAResult(SAResult):
    """Outcome plus acceptance feedback used by adaptive SA."""

    acceptance_ratio_history: tuple[tuple[int, float], ...]
    adaptation_history: tuple[tuple[int, str], ...]


def _swap_neighbor(layout: str, random_generator: random.Random) -> str:
    letters = list(validate_layout(layout))
    first, second = random_generator.sample(range(len(letters)), 2)
    letters[first], letters[second] = letters[second], letters[first]
    return "".join(letters)


def _accept_move(
    cost_change: float,
    temperature: float,
    random_generator: random.Random,
) -> bool:
    if cost_change <= 0.0:
        return True
    return random_generator.random() < math.exp(-cost_change / temperature)


def run_simulated_annealing(
    objective: KeyboardObjective,
    *,
    config: SAConfig,
    seed: int,
) -> SAResult:
    """Run standard SA under an exact objective-evaluation budget."""
    config.validate()
    random_generator = random.Random(seed)
    start_time = time.perf_counter()

    current_layout = validate_layout(config.start_layout)
    current_cost = objective.cost(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    temperature = config.initial_temperature
    evaluations = 1
    accepted_moves = 0
    best_cost_history = [best_cost]
    temperature_history = [(evaluations, temperature)]

    while evaluations < config.evaluation_budget:
        candidate_layout = _swap_neighbor(current_layout, random_generator)
        candidate_cost = objective.cost(candidate_layout)
        evaluations += 1

        if _accept_move(
            candidate_cost - current_cost,
            temperature,
            random_generator,
        ):
            current_layout = candidate_layout
            current_cost = candidate_cost
            accepted_moves += 1

        if current_cost < best_cost:
            best_layout = current_layout
            best_cost = current_cost
        best_cost_history.append(best_cost)

        temperature = max(
            config.minimum_temperature,
            temperature * config.cooling_rate,
        )
        if (
            evaluations % config.temperature_record_interval == 0
            or evaluations == config.evaluation_budget
        ):
            temperature_history.append((evaluations, temperature))

    return SAResult(
        best_layout=best_layout,
        best_cost=best_cost,
        final_layout=current_layout,
        final_cost=current_cost,
        evaluations=evaluations,
        accepted_moves=accepted_moves,
        runtime_seconds=time.perf_counter() - start_time,
        best_cost_history=tuple(best_cost_history),
        temperature_history=tuple(temperature_history),
    )


def run_adaptive_simulated_annealing(
    objective: KeyboardObjective,
    *,
    config: AdaptiveSAConfig,
    seed: int,
) -> AdaptiveSAResult:
    """Run SA whose temperature responds to recent acceptance behavior.

    Every adaptation window, a low acceptance ratio triggers reheating to
    promote exploration. A high ratio triggers additional cooling to focus the
    search. The temperature is always clamped to configured bounds.
    """
    config.validate()
    random_generator = random.Random(seed)
    start_time = time.perf_counter()

    current_layout = validate_layout(config.start_layout)
    current_cost = objective.cost(current_layout)
    best_layout = current_layout
    best_cost = current_cost
    temperature = config.initial_temperature
    evaluations = 1
    accepted_moves = 0
    window_attempts = 0
    window_acceptances = 0

    best_cost_history = [best_cost]
    temperature_history = [(evaluations, temperature)]
    acceptance_ratio_history: list[tuple[int, float]] = []
    adaptation_history: list[tuple[int, str]] = []

    while evaluations < config.evaluation_budget:
        candidate_layout = _swap_neighbor(current_layout, random_generator)
        candidate_cost = objective.cost(candidate_layout)
        evaluations += 1
        window_attempts += 1

        if _accept_move(
            candidate_cost - current_cost,
            temperature,
            random_generator,
        ):
            current_layout = candidate_layout
            current_cost = candidate_cost
            accepted_moves += 1
            window_acceptances += 1

        if current_cost < best_cost:
            best_layout = current_layout
            best_cost = current_cost
        best_cost_history.append(best_cost)

        temperature = max(
            config.minimum_temperature,
            temperature * config.cooling_rate,
        )

        if window_attempts == config.adaptation_window:
            acceptance_ratio = window_acceptances / window_attempts
            if acceptance_ratio < config.target_acceptance_low:
                temperature = min(
                    config.maximum_temperature,
                    temperature * config.reheat_factor,
                )
                action = "reheat"
            elif acceptance_ratio > config.target_acceptance_high:
                temperature = max(
                    config.minimum_temperature,
                    temperature * config.accelerated_cooling_factor,
                )
                action = "accelerated_cooling"
            else:
                action = "base_cooling"

            acceptance_ratio_history.append((evaluations, acceptance_ratio))
            adaptation_history.append((evaluations, action))
            temperature_history.append((evaluations, temperature))
            window_attempts = 0
            window_acceptances = 0

    if window_attempts:
        acceptance_ratio = window_acceptances / window_attempts
        acceptance_ratio_history.append((evaluations, acceptance_ratio))
        adaptation_history.append((evaluations, "partial_window_no_adaptation"))
        if temperature_history[-1][0] != evaluations:
            temperature_history.append((evaluations, temperature))

    return AdaptiveSAResult(
        best_layout=best_layout,
        best_cost=best_cost,
        final_layout=current_layout,
        final_cost=current_cost,
        evaluations=evaluations,
        accepted_moves=accepted_moves,
        runtime_seconds=time.perf_counter() - start_time,
        best_cost_history=tuple(best_cost_history),
        temperature_history=tuple(temperature_history),
        acceptance_ratio_history=tuple(acceptance_ratio_history),
        adaptation_history=tuple(adaptation_history),
    )
