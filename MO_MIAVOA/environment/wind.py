"""
Wind Zone Utilities
====================
Computes how much of a drone path passes through each wind zone,
using segment-rectangle clipping (Liang-Barsky algorithm).

Used by both Objective 2 (Energy) and Objective 3 (Risk).
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict

from environment.models import WindZone, Path


def _liang_barsky_clip_length(p1: np.ndarray, p2: np.ndarray,
                              x_min: float, y_min: float,
                              x_max: float, y_max: float) -> float:
    """
    Compute the length of the portion of segment (p1→p2) that lies
    inside the axis-aligned rectangle [x_min, x_max] × [y_min, y_max].

    Uses the Liang-Barsky line clipping algorithm.

    Returns:
        Length of the clipped segment (0.0 if no intersection).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # p and q arrays for the four edges
    p = [-dx, dx, -dy, dy]
    q = [p1[0] - x_min, x_max - p1[0], p1[1] - y_min, y_max - p1[1]]

    t_enter = 0.0
    t_exit = 1.0

    for i in range(4):
        if abs(p[i]) < 1e-12:
            # Segment is parallel to this edge
            if q[i] < 0:
                return 0.0  # Outside and parallel — no intersection
            continue

        t = q[i] / p[i]

        if p[i] < 0:
            # Entering edge
            t_enter = max(t_enter, t)
        else:
            # Leaving edge
            t_exit = min(t_exit, t)

        if t_enter > t_exit:
            return 0.0  # No valid intersection

    if t_enter > t_exit:
        return 0.0

    # Compute the length of the clipped portion
    seg_length = np.sqrt(dx * dx + dy * dy)
    return (t_exit - t_enter) * seg_length


def segment_wind_exposure(p1: np.ndarray, p2: np.ndarray,
                          wind_zones: List[WindZone]) -> Dict[str, float]:
    """
    Compute wind zone exposure for a single path segment.

    Returns:
        Dictionary mapping intensity label → distance traveled inside zones
        of that intensity. Example: {"green": 45.2, "yellow": 0.0, "red": 12.8}
    """
    exposure: Dict[str, float] = {}

    for zone in wind_zones:
        clip_len = _liang_barsky_clip_length(
            p1, p2,
            zone.x, zone.y, zone.right, zone.top,
        )

        if clip_len > 0:
            key = zone.intensity
            exposure[key] = exposure.get(key, 0.0) + clip_len

    return exposure


def wind_exposure(path: Path, wind_zones: List[WindZone]) -> Dict[str, float]:
    """
    Compute total wind zone exposure for an entire path.

    This is the core utility used by both:
    - Objective 2 (Energy): weighted by k2 for battery cost
    - Objective 3 (Risk):   weighted by w_w for payload stability

    Returns:
        Dictionary mapping intensity label → total distance inside zones
        of that intensity across the entire path.
        Example: {"green": 120.5, "yellow": 45.0, "red": 30.2}
    """
    total_exposure: Dict[str, float] = {}

    points = path.all_points

    for i in range(len(points) - 1):
        seg_exposure = segment_wind_exposure(points[i], points[i + 1], wind_zones)

        for key, dist in seg_exposure.items():
            total_exposure[key] = total_exposure.get(key, 0.0) + dist

    return total_exposure
