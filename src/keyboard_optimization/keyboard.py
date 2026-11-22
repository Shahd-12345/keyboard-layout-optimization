"""Keyboard representation, geometry, and distance calculations."""

from __future__ import annotations

import string
from typing import Literal

import numpy as np

ALPHABET = string.ascii_lowercase

# A layout lists the letter assigned to each physical position in this order:
# top row (10 keys), home row (9 keys), bottom row (7 keys).
QWERTY_LAYOUT = "qwertyuiopasdfghjklzxcvbnm"

ROW_SPECS = (
    ("qwertyuiop", 0.00, 2.0),
    ("asdfghjkl", 0.25, 1.0),
    ("zxcvbnm", 0.75, 0.0),
)


def validate_layout(layout: str) -> str:
    """Return a normalized valid layout or raise ValueError."""
    normalized = layout.strip().lower()
    if len(normalized) != len(ALPHABET):
        raise ValueError(
            f"A layout must contain exactly 26 letters; received {len(normalized)}."
        )
    if set(normalized) != set(ALPHABET):
        missing = "".join(sorted(set(ALPHABET) - set(normalized)))
        repeated = "".join(
            sorted({letter for letter in normalized if normalized.count(letter) > 1})
        )
        raise ValueError(
            "A layout must contain every letter a-z exactly once. "
            f"Missing: {missing or 'none'}; repeated: {repeated or 'none'}."
        )
    return normalized


def keyboard_coordinates() -> np.ndarray:
    """Return 26 physical key-center coordinates in layout-string order."""
    coordinates: list[tuple[float, float]] = []
    for letters, horizontal_offset, y_coordinate in ROW_SPECS:
        coordinates.extend(
            (horizontal_offset + index, y_coordinate)
            for index, _ in enumerate(letters)
        )
    result = np.asarray(coordinates, dtype=float)
    if result.shape != (26, 2):
        raise RuntimeError(f"Expected 26 key coordinates; received {result.shape}.")
    return result


def build_distance_matrix(
    metric: Literal["euclidean", "manhattan"] = "euclidean",
) -> np.ndarray:
    """Return the symmetric 26-by-26 physical-position distance matrix."""
    coordinates = keyboard_coordinates()
    deltas = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    if metric == "euclidean":
        distances = np.sqrt(np.square(deltas).sum(axis=2))
    elif metric == "manhattan":
        distances = np.abs(deltas).sum(axis=2)
    else:
        raise ValueError("metric must be either 'euclidean' or 'manhattan'.")

    if not np.allclose(distances, distances.T):
        raise RuntimeError("Distance matrix must be symmetric.")
    if not np.allclose(np.diag(distances), 0.0):
        raise RuntimeError("Distance from a key to itself must be zero.")
    if not np.isfinite(distances).all():
        raise RuntimeError("Distance matrix contains a non-finite value.")
    return distances


def layout_positions(layout: str) -> np.ndarray:
    """Map alphabet indices a=0,...,z=25 to physical position indices."""
    normalized = validate_layout(layout)
    positions = np.empty(26, dtype=int)
    for physical_position, letter in enumerate(normalized):
        positions[ord(letter) - ord("a")] = physical_position
    return positions


def key_distance(
    first_position: int,
    second_position: int,
    metric: Literal["euclidean", "manhattan"] = "euclidean",
) -> float:
    """Return a single distance; useful for transparent spot checks."""
    if not (0 <= first_position < 26 and 0 <= second_position < 26):
        raise ValueError("Physical position indices must be between 0 and 25.")
    return float(build_distance_matrix(metric)[first_position, second_position])

