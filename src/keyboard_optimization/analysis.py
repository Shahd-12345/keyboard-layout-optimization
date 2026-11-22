"""Statistical summaries for repeated optimization experiments."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def cliffs_delta(first: Iterable[float], second: Iterable[float]) -> float:
    """Return Cliff's delta for first versus second.

    A negative value means values in ``first`` tend to be smaller. For a
    minimization objective, that direction favors the first method.
    """
    first_values = np.asarray(tuple(first), dtype=float)
    second_values = np.asarray(tuple(second), dtype=float)
    if first_values.size == 0 or second_values.size == 0:
        raise ValueError("Both samples must contain at least one value.")
    if not np.isfinite(first_values).all() or not np.isfinite(second_values).all():
        raise ValueError("Samples must contain only finite values.")

    greater = np.greater.outer(first_values, second_values).sum()
    smaller = np.less.outer(first_values, second_values).sum()
    return float((greater - smaller) / (first_values.size * second_values.size))


def cliffs_delta_magnitude(delta: float) -> str:
    """Classify the absolute Cliff's delta using common thresholds."""
    absolute = abs(delta)
    if absolute < 0.147:
        return "negligible"
    if absolute < 0.330:
        return "small"
    if absolute < 0.474:
        return "medium"
    return "large"


def summarize_results(
    results: pd.DataFrame,
    *,
    metric: str,
    group_column: str = "algorithm",
) -> pd.DataFrame:
    """Return descriptive statistics and a t-based 95% confidence interval."""
    required = {group_column, metric}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    records: list[dict[str, float | int | str]] = []
    for group_name, group in results.groupby(
        group_column, sort=False, observed=True
    ):
        values = group[metric].to_numpy(dtype=float)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Group {group_name!r} has invalid {metric} values.")

        count = int(values.size)
        mean = float(values.mean())
        standard_deviation = float(values.std(ddof=1)) if count > 1 else 0.0
        if count > 1:
            margin = float(
                stats.t.ppf(0.975, df=count - 1)
                * standard_deviation
                / np.sqrt(count)
            )
        else:
            margin = 0.0

        records.append(
            {
                group_column: str(group_name),
                "n": count,
                "mean": mean,
                "std": standard_deviation,
                "median": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
            }
        )
    return pd.DataFrame.from_records(records)


def paired_method_comparison(
    results: pd.DataFrame,
    *,
    first_method: str,
    second_method: str,
    metric: str,
    seed_column: str = "seed",
    method_column: str = "algorithm",
) -> dict[str, float | int | str]:
    """Compare two methods on matched seeds with Wilcoxon and Cliff's delta."""
    required = {method_column, seed_column, metric}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    first = results.loc[
        results[method_column] == first_method,
        [seed_column, metric],
    ].rename(columns={metric: "first_value"})
    second = results.loc[
        results[method_column] == second_method,
        [seed_column, metric],
    ].rename(columns={metric: "second_value"})
    paired = first.merge(second, on=seed_column, how="inner", validate="one_to_one")
    if paired.empty:
        raise ValueError("The selected methods have no matched seeds.")

    first_values = paired["first_value"].to_numpy(dtype=float)
    second_values = paired["second_value"].to_numpy(dtype=float)
    differences = first_values - second_values
    if np.allclose(differences, 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        wilcoxon_result = stats.wilcoxon(
            first_values,
            second_values,
            alternative="two-sided",
            zero_method="wilcox",
        )
        statistic = float(wilcoxon_result.statistic)
        p_value = float(wilcoxon_result.pvalue)

    delta = cliffs_delta(first_values, second_values)
    return {
        "first_method": first_method,
        "second_method": second_method,
        "metric": metric,
        "matched_runs": int(paired.shape[0]),
        "median_paired_difference_first_minus_second": float(
            np.median(differences)
        ),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "cliffs_delta": delta,
        "effect_magnitude": cliffs_delta_magnitude(delta),
    }
