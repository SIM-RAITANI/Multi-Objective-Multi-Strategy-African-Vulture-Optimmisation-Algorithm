"""
Objective 2 — Flight Energy Expenditure
==========================================
E(path) = k1 × D_total + k2 × Σ_z (I_z × d_z(path))

Two components:
    1. Distance-driven cost (proportional to total path length)
    2. Wind-driven cost (proportional to distance × intensity inside each wind zone)
"""

from __future__ import annotations

import numpy as np
from typing import List

import config
from environment.models import Path, WindZone
from environment.wind import wind_exposure


def compute_energy(path: Path, wind_zones: List[WindZone]) -> float:
    """
    Compute flight energy expenditure for the given path.

    Returns:
        Energy cost (arbitrary units, to be minimized).
    """
    points = path.all_points

    # Component 1: Distance-driven energy
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_distance = np.sum(segment_lengths)
    energy_distance = config.K1_ENERGY_DISTANCE * total_distance

    # Component 2: Wind-driven energy
    exposure = wind_exposure(path, wind_zones)
    energy_wind = 0.0
    for intensity_label, dist_in_zone in exposure.items():
        intensity_value = config.WIND_INTENSITY.get(intensity_label, 1.0)
        energy_wind += intensity_value * dist_in_zone

    energy_wind *= config.K2_ENERGY_WIND

    return energy_distance + energy_wind
