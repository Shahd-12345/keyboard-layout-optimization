"""Fair repeated-experiment runner for all stochastic search methods."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .adaptive_genetic_algorithm import (
    AdaptiveGAConfig,
    run_adaptive_genetic_algorithm,
)
from .baselines import random_search
from .genetic_algorithm import GAConfig, run_genetic_algorithm
from .keyboard import QWERTY_LAYOUT
from .objective import KeyboardObjective
from .simulated_annealing import (
    AdaptiveSAConfig,
    SAConfig,
    run_adaptive_simulated_annealing,
    run_simulated_annealing,
)

ALGORITHM_ORDER = (
    "Random Search",
    "Standard GA",
    "Adaptive GA",
    "Standard SA",
    "Adaptive SA",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Common settings applied to every stochastic method."""

    runs: int = 30
    evaluation_budget: int = 10_000
    first_seed: int = 42
    ga_population_size: int = 100
    ga_tournament_size: int = 3
    ga_crossover_rate: float = 0.90
    ga_mutation_rate: float = 0.15
    ga_elite_count: int = 2
    sa_initial_temperature: float = 1.0
    sa_cooling_rate: float = 0.995
    sa_minimum_temperature: float = 1e-6
    sa_adaptation_window: int = 100

    def validate(self) -> None:
        if self.runs < 1:
            raise ValueError("runs must be at least 1.")
        if self.ga_population_size < 2:
            raise ValueError("ga_population_size must be at least 2.")
        if self.evaluation_budget < self.ga_population_size:
            raise ValueError(
                "evaluation_budget must be at least ga_population_size."
            )
        if self.first_seed < 0:
            raise ValueError("first_seed must be nonnegative.")


@dataclass
class BenchmarkSuiteResult:
    """Tables, trajectories, and diagnostics from a benchmark suite."""

    runs: pd.DataFrame
    histories: dict[str, np.ndarray]
    qwerty: dict[str, float | str]
    adaptive_ga_diagnostics: list[dict[str, object]]
    adaptive_sa_diagnostics: list[dict[str, object]]
    configuration: dict[str, object]


def _improvement_pct(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / baseline


def _result_row(
    *,
    algorithm: str,
    run_number: int,
    seed: int,
    best_layout: str,
    train_cost: float,
    test_cost: float,
    qwerty_train_cost: float,
    qwerty_test_cost: float,
    evaluations: int,
    runtime_seconds: float,
) -> dict[str, float | int | str]:
    return {
        "algorithm": algorithm,
        "run": run_number,
        "seed": seed,
        "best_layout": best_layout,
        "train_cost": train_cost,
        "test_cost": test_cost,
        "generalization_gap": test_cost - train_cost,
        "train_improvement_over_qwerty_pct": _improvement_pct(
            qwerty_train_cost, train_cost
        ),
        "test_improvement_over_qwerty_pct": _improvement_pct(
            qwerty_test_cost, test_cost
        ),
        "evaluations": evaluations,
        "runtime_seconds": runtime_seconds,
    }


def run_benchmark_suite(
    training_objective: KeyboardObjective,
    testing_objective: KeyboardObjective,
    *,
    config: ExperimentConfig,
    progress: Callable[[str], None] | None = print,
) -> BenchmarkSuiteResult:
    """Run matched-seed comparisons under one exact evaluation budget."""
    config.validate()
    qwerty_train_cost = training_objective.cost(QWERTY_LAYOUT)
    qwerty_test_cost = testing_objective.cost(QWERTY_LAYOUT)

    ga_config = GAConfig(
        population_size=config.ga_population_size,
        evaluation_budget=config.evaluation_budget,
        tournament_size=config.ga_tournament_size,
        crossover_rate=config.ga_crossover_rate,
        mutation_rate=config.ga_mutation_rate,
        elite_count=config.ga_elite_count,
    )
    adaptive_ga_config = AdaptiveGAConfig(
        population_size=config.ga_population_size,
        evaluation_budget=config.evaluation_budget,
        tournament_size=config.ga_tournament_size,
        crossover_rate=config.ga_crossover_rate,
        initial_mutation_rate=config.ga_mutation_rate,
        elite_count=config.ga_elite_count,
    )
    sa_config = SAConfig(
        evaluation_budget=config.evaluation_budget,
        initial_temperature=config.sa_initial_temperature,
        cooling_rate=config.sa_cooling_rate,
        minimum_temperature=config.sa_minimum_temperature,
    )
    adaptive_sa_config = AdaptiveSAConfig(
        evaluation_budget=config.evaluation_budget,
        initial_temperature=config.sa_initial_temperature,
        cooling_rate=config.sa_cooling_rate,
        minimum_temperature=config.sa_minimum_temperature,
        adaptation_window=config.sa_adaptation_window,
    )

    rows: list[dict[str, float | int | str]] = []
    histories: dict[str, list[tuple[float, ...]]] = {
        algorithm: [] for algorithm in ALGORITHM_ORDER
    }
    adaptive_ga_diagnostics: list[dict[str, object]] = []
    adaptive_sa_diagnostics: list[dict[str, object]] = []

    for run_index in range(config.runs):
        run_number = run_index + 1
        seed = config.first_seed + run_index
        if progress is not None:
            progress(f"Run {run_number}/{config.runs}, seed {seed}")

        random_start = time.perf_counter()
        random_result = random_search(
            training_objective,
            evaluations=config.evaluation_budget,
            seed=seed,
        )
        random_runtime = time.perf_counter() - random_start
        random_test_cost = testing_objective.cost(random_result.best_layout)
        rows.append(
            _result_row(
                algorithm="Random Search",
                run_number=run_number,
                seed=seed,
                best_layout=random_result.best_layout,
                train_cost=random_result.best_cost,
                test_cost=random_test_cost,
                qwerty_train_cost=qwerty_train_cost,
                qwerty_test_cost=qwerty_test_cost,
                evaluations=random_result.evaluations,
                runtime_seconds=random_runtime,
            )
        )
        histories["Random Search"].append(random_result.best_cost_history)

        ga_result = run_genetic_algorithm(
            training_objective,
            config=ga_config,
            seed=seed,
        )
        ga_test_cost = testing_objective.cost(ga_result.best_layout)
        rows.append(
            _result_row(
                algorithm="Standard GA",
                run_number=run_number,
                seed=seed,
                best_layout=ga_result.best_layout,
                train_cost=ga_result.best_cost,
                test_cost=ga_test_cost,
                qwerty_train_cost=qwerty_train_cost,
                qwerty_test_cost=qwerty_test_cost,
                evaluations=ga_result.evaluations,
                runtime_seconds=ga_result.runtime_seconds,
            )
        )
        histories["Standard GA"].append(ga_result.best_cost_history)

        adaptive_ga_result = run_adaptive_genetic_algorithm(
            training_objective,
            config=adaptive_ga_config,
            seed=seed,
        )
        adaptive_ga_test_cost = testing_objective.cost(
            adaptive_ga_result.best_layout
        )
        rows.append(
            _result_row(
                algorithm="Adaptive GA",
                run_number=run_number,
                seed=seed,
                best_layout=adaptive_ga_result.best_layout,
                train_cost=adaptive_ga_result.best_cost,
                test_cost=adaptive_ga_test_cost,
                qwerty_train_cost=qwerty_train_cost,
                qwerty_test_cost=qwerty_test_cost,
                evaluations=adaptive_ga_result.evaluations,
                runtime_seconds=adaptive_ga_result.runtime_seconds,
            )
        )
        histories["Adaptive GA"].append(adaptive_ga_result.best_cost_history)
        adaptive_ga_diagnostics.append(
            {
                "run": run_number,
                "seed": seed,
                "mutation_rate_history": list(
                    adaptive_ga_result.mutation_rate_history
                ),
                "diversity_history": list(adaptive_ga_result.diversity_history),
            }
        )

        sa_result = run_simulated_annealing(
            training_objective,
            config=sa_config,
            seed=seed,
        )
        sa_test_cost = testing_objective.cost(sa_result.best_layout)
        rows.append(
            _result_row(
                algorithm="Standard SA",
                run_number=run_number,
                seed=seed,
                best_layout=sa_result.best_layout,
                train_cost=sa_result.best_cost,
                test_cost=sa_test_cost,
                qwerty_train_cost=qwerty_train_cost,
                qwerty_test_cost=qwerty_test_cost,
                evaluations=sa_result.evaluations,
                runtime_seconds=sa_result.runtime_seconds,
            )
        )
        histories["Standard SA"].append(sa_result.best_cost_history)

        adaptive_sa_result = run_adaptive_simulated_annealing(
            training_objective,
            config=adaptive_sa_config,
            seed=seed,
        )
        adaptive_sa_test_cost = testing_objective.cost(
            adaptive_sa_result.best_layout
        )
        rows.append(
            _result_row(
                algorithm="Adaptive SA",
                run_number=run_number,
                seed=seed,
                best_layout=adaptive_sa_result.best_layout,
                train_cost=adaptive_sa_result.best_cost,
                test_cost=adaptive_sa_test_cost,
                qwerty_train_cost=qwerty_train_cost,
                qwerty_test_cost=qwerty_test_cost,
                evaluations=adaptive_sa_result.evaluations,
                runtime_seconds=adaptive_sa_result.runtime_seconds,
            )
        )
        histories["Adaptive SA"].append(adaptive_sa_result.best_cost_history)
        adaptive_sa_diagnostics.append(
            {
                "run": run_number,
                "seed": seed,
                "temperature_history": list(adaptive_sa_result.temperature_history),
                "acceptance_ratio_history": list(
                    adaptive_sa_result.acceptance_ratio_history
                ),
                "adaptation_history": list(adaptive_sa_result.adaptation_history),
            }
        )

    history_arrays = {
        algorithm: np.asarray(algorithm_histories, dtype=float)
        for algorithm, algorithm_histories in histories.items()
    }
    run_table = pd.DataFrame.from_records(rows)
    run_table["algorithm"] = pd.Categorical(
        run_table["algorithm"], categories=ALGORITHM_ORDER, ordered=True
    )
    run_table = run_table.sort_values(["run", "algorithm"]).reset_index(drop=True)

    return BenchmarkSuiteResult(
        runs=run_table,
        histories=history_arrays,
        qwerty={
            "layout": QWERTY_LAYOUT,
            "training_cost": qwerty_train_cost,
            "testing_cost": qwerty_test_cost,
        },
        adaptive_ga_diagnostics=adaptive_ga_diagnostics,
        adaptive_sa_diagnostics=adaptive_sa_diagnostics,
        configuration={
            "experiment": asdict(config),
            "standard_ga": asdict(ga_config),
            "adaptive_ga": asdict(adaptive_ga_config),
            "standard_sa": asdict(sa_config),
            "adaptive_sa": asdict(adaptive_sa_config),
        },
    )


def save_benchmark_suite(
    suite: BenchmarkSuiteResult,
    output_directory: str | Path,
    *,
    corpus_metadata: dict[str, object] | None = None,
) -> dict[str, Path]:
    """Persist compact reproducibility files for a completed suite."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    runs_path = output / "run_results.csv"
    histories_path = output / "convergence_histories.npz"
    metadata_path = output / "experiment_metadata.json"
    diagnostics_path = output / "adaptation_diagnostics.json"

    suite.runs.to_csv(runs_path, index=False)
    np.savez_compressed(histories_path, **suite.histories)
    metadata = {
        "configuration": suite.configuration,
        "qwerty": suite.qwerty,
        "corpus": corpus_metadata or {},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    diagnostics = {
        "adaptive_ga": suite.adaptive_ga_diagnostics,
        "adaptive_sa": suite.adaptive_sa_diagnostics,
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    return {
        "runs": runs_path,
        "histories": histories_path,
        "metadata": metadata_path,
        "diagnostics": diagnostics_path,
    }
