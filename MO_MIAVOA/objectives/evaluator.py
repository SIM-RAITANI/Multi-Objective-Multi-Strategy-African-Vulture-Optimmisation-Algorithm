"""
Combined Objective Evaluator
==============================
Computes the full (T, E, R) objective triple for a path.
Caches results on the Path object to avoid redundant computation.
"""

from __future__ import annotations

from typing import Tuple

from environment.models import Path, Environment
from objectives.time_obj import compute_time
from objectives.energy_obj import compute_energy
from objectives.risk_obj import compute_risk


def evaluate_path(path: Path, env: Environment,
                  force: bool = False) -> Tuple[float, float, float]:
    """
    Evaluate all three objectives for a path.

    Args:
        path: The candidate drone path.
        env: The simulation environment.
        force: If True, recompute even if cached.

    Returns:
        (T, E, R) tuple — all to be minimized.
    """
    if path.objectives is not None and not force:
        return path.objectives

    t = compute_time(path)
    e = compute_energy(path, env.wind_zones)
    r = compute_risk(path, env.buildings, env.wind_zones)

    path.objectives = (t, e, r)
    return path.objectives
