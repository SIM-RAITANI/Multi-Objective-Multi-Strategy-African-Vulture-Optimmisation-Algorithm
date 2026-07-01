"""
Objective 3 — Medical Payload Risk
=====================================
R(path) = w_c × R_clear + w_t × R_turn + w_w × R_wind

Three sub-components:
    A. Obstacle Clearance Risk — exp(-α × c_min) per segment
    B. Turning Severity Risk — sum of turning angles at waypoints
    C. Wind Exposure Risk — same wind map as Energy but independent weight
"""

from __future__ import annotations

import numpy as np
from typing import List

import config
from environment.models import Path, Building, WindZone
from environment.collision import minimum_clearance
from environment.wind import wind_exposure


# In risk_obj.py

def _compute_clearance_risk(path: Path, buildings: List[Building]) -> float:
    """
    A. Obstacle Clearance Risk.
    Strictly penalizes any path coming within the safety buffer.
    """
    clearances = minimum_clearance(path, buildings)

    risk = 0.0
    for c in clearances:
        # HARD CONSTRAINT: If clearance is less than or equal to the safety buffer,
        # treat it as a hard collision so it is dominated and removed.
        # (Assuming your building safety_buffer is 15.0)
        if c <= 15.0:  
            risk += 10000.0  
        else:
            # Only apply exponential decay for paths safely outside the buffer
            risk += np.exp(-config.CLEARANCE_DECAY_ALPHA * c)

    return risk

def _compute_turning_risk(path: Path) -> float:
    """
    B. Turning Severity Risk.
    Sum of turning angles (radians) at each intermediate waypoint.
    """
    points = path.all_points  # shape (n+2, 2)
    total_angle = 0.0

    # Iterate over intermediate points (indices 1 to n)
    for i in range(1, len(points) - 1):
        v_in = points[i] - points[i - 1]    # incoming vector
        v_out = points[i + 1] - points[i]   # outgoing vector

        len_in = np.linalg.norm(v_in)
        len_out = np.linalg.norm(v_out)

        if len_in < 1e-12 or len_out < 1e-12:
            continue

        # Cosine of turning angle
        cos_angle = np.dot(v_in, v_out) / (len_in * len_out)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)
        total_angle += angle

    return total_angle


def _compute_wind_risk(path: Path, wind_zones: List[WindZone]) -> float:
    """
    C. Wind Exposure Risk (payload stability perspective).
    Same wind zone map as Energy, but weighted independently by w_w.
    """
    exposure = wind_exposure(path, wind_zones)

    risk = 0.0
    for intensity_label, dist_in_zone in exposure.items():
        intensity_value = config.WIND_INTENSITY.get(intensity_label, 1.0)
        risk += intensity_value * dist_in_zone

    return risk


def compute_risk(path: Path, buildings: List[Building],
                 wind_zones: List[WindZone]) -> float:
    """
    Compute composite medical payload risk.

    R = w_c × R_clear + w_t × R_turn + w_w × R_wind

    Returns:
        Risk score (to be minimized).
    """
    r_clear = _compute_clearance_risk(path, buildings)
    r_turn = _compute_turning_risk(path)
    r_wind = _compute_wind_risk(path, wind_zones)

    return (config.W_CLEARANCE * r_clear +
            config.W_TURNING * r_turn +
            config.W_WIND_RISK * r_wind)
