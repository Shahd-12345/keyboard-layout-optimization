"""Common keyboard-layout objective used by every search algorithm."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .keyboard import QWERTY_LAYOUT, build_distance_matrix, layout_positions, validate_layout


@dataclass(frozen=True)
class LayoutEvaluation:
    layout: str
    cost: float
    improvement_over_qwerty_pct: float


class KeyboardObjective:
    """Frequency-weighted average movement distance for a layout."""

    def __init__(
        self,
        frequency_matrix: np.ndarray,
        distance_matrix: np.ndarray | None = None,
    ) -> None:
        frequency = np.asarray(frequency_matrix, dtype=float)
        if frequency.shape != (26, 26):
            raise ValueError(
                f"frequency_matrix must have shape (26, 26); received {frequency.shape}."
            )
        if (frequency < 0).any() or not np.isfinite(frequency).all():
            raise ValueError("frequency_matrix must contain finite nonnegative values.")
        if frequency.sum() <= 0:
            raise ValueError("frequency_matrix must contain at least one transition.")

        distances = (
            build_distance_matrix()
            if distance_matrix is None
            else np.asarray(distance_matrix, dtype=float)
        )
        if distances.shape != (26, 26):
            raise ValueError(
                f"distance_matrix must have shape (26, 26); received {distances.shape}."
            )
        if (distances < 0).any() or not np.isfinite(distances).all():
            raise ValueError("distance_matrix must contain finite nonnegative values.")

        self.frequency_matrix = frequency.copy()
        self.distance_matrix = distances.copy()

    def cost(self, layout: str) -> float:
        """Return the lower-is-better movement cost for a valid layout."""
        positions = layout_positions(layout)
        pairwise_distances = self.distance_matrix[np.ix_(positions, positions)]
        return float(np.sum(self.frequency_matrix * pairwise_distances))

    def improvement_pct(
        self,
        layout: str,
        baseline_layout: str = QWERTY_LAYOUT,
    ) -> float:
        """Return percentage cost reduction relative to a baseline layout."""
        candidate_cost = self.cost(layout)
        baseline_cost = self.cost(baseline_layout)
        if baseline_cost == 0:
            raise ZeroDivisionError("The baseline layout has zero objective cost.")
        return 100.0 * (baseline_cost - candidate_cost) / baseline_cost

    def evaluate(self, layout: str) -> LayoutEvaluation:
        normalized = validate_layout(layout)
        return LayoutEvaluation(
            layout=normalized,
            cost=self.cost(normalized),
            improvement_over_qwerty_pct=self.improvement_pct(normalized),
        )

