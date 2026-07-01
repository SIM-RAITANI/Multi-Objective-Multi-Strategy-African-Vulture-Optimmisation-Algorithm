"""
Objective 1 — Delivery Time
==============================
T(path) = (1/v) × Σ ||P_{i+1} - P_i||

Pure Euclidean path length divided by constant cruise speed.
Independent of wind, obstacles, or turning — represents pure urgency.
"""

from __future__ import annotations

import numpy as np

import config
from environment.models import Path


def compute_time(path: Path) -> float:
    """
    Compute delivery time for the given path.

    T = total_path_length / cruise_speed

    Returns:
        Delivery time in seconds (to be minimized).
    """
    points = path.all_points  # shape (n+2, 2)

    # Sum of Euclidean distances between consecutive points
    diffs = np.diff(points, axis=0)  # shape (n+1, 2)
    segment_lengths = np.linalg.norm(diffs, axis=1)  # shape (n+1,)
    total_length = np.sum(segment_lengths)

    return total_length / config.DRONE_CRUISE_SPEED
