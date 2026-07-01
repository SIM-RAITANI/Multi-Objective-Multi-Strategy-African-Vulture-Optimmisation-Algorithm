"""
Collision Detection Utilities
===============================
Segment-vs-rectangle intersection, minimum clearance computation,
and feasibility checks for drone paths against building obstacles.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple, Optional

from environment.models import Building, Path


# =============================================================================
# Core Geometry
# =============================================================================

def _cross_2d(ax: float, ay: float, bx: float, by: float) -> float:
    """2D cross product of vectors a and b."""
    return ax * by - ay * bx

def segments_intersect(p1: np.ndarray, p2: np.ndarray,
                       p3: np.ndarray, p4: np.ndarray) -> bool:
    return _segments_intersect_coords(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1])

def _segments_intersect_coords(x1: float, y1: float, x2: float, y2: float,
                               x3: float, y3: float, x4: float, y4: float) -> bool:
    """Check if line segment (x1,y1)->(x2,y2) intersects (x3,y3)->(x4,y4)."""
    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = x4 - x3, y4 - y3

    denom = _cross_2d(d1x, d1y, d2x, d2y)
    if abs(denom) < 1e-12:
        return False

    v31x, v31y = x3 - x1, y3 - y1
    t = _cross_2d(v31x, v31y, d2x, d2y) / denom
    u = _cross_2d(v31x, v31y, d1x, d1y) / denom

    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0

def segment_intersects_rect(p1: np.ndarray, p2: np.ndarray,
                            x_min: float, y_min: float,
                            x_max: float, y_max: float) -> bool:
    """
    Check if a line segment (p1->p2) intersects an axis-aligned rectangle.
    """
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    
    # Check if either endpoint is inside the rectangle
    if (x_min <= x1 <= x_max and y_min <= y1 <= y_max):
        return True
    if (x_min <= x2 <= x_max and y_min <= y2 <= y_max):
        return True

    # Four edges of the rectangle
    if _segments_intersect_coords(x1, y1, x2, y2, x_min, y_min, x_max, y_min): return True
    if _segments_intersect_coords(x1, y1, x2, y2, x_max, y_min, x_max, y_max): return True
    if _segments_intersect_coords(x1, y1, x2, y2, x_max, y_max, x_min, y_max): return True
    if _segments_intersect_coords(x1, y1, x2, y2, x_min, y_max, x_min, y_min): return True

    return False


def point_to_segment_distance(point: np.ndarray,
                              seg_start: np.ndarray,
                              seg_end: np.ndarray) -> float:
    """Minimum distance from a point to a line segment (scalar math for speed)."""
    px, py = point[0], point[1]
    sx, sy = seg_start[0], seg_start[1]
    ex, ey = seg_end[0], seg_end[1]
    
    dx = ex - sx
    dy = ey - sy
    length_sq = dx * dx + dy * dy

    if length_sq < 1e-12:
        return math.sqrt((px - sx)**2 + (py - sy)**2)

    t = ((px - sx) * dx + (py - sy) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    
    proj_x = sx + t * dx
    proj_y = sy + t * dy
    return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)


def minimum_clearance_to_rect(seg_start: np.ndarray, seg_end: np.ndarray,
                              building: Building) -> float:
    """
    Compute minimum distance from a line segment to a building rectangle.
    Returns a negative value if the segment intersects the building to trigger
    an inescapable exponential risk penalty in the cost evaluation function.
    """
    # Quick intersection check
    if segment_intersects_rect(seg_start, seg_end,
                               building.x, building.y,
                               building.right, building.top):
        return -1000.0  # Forces exp(-alpha * -1000) to explode to infinity

    corners = [np.array(c) for c in building.corners]
    min_dist = float('inf')

    # Distance from each building corner to the path segment
    for corner in corners:
        dist = point_to_segment_distance(corner, seg_start, seg_end)
        min_dist = min(min_dist, dist)

    # Distance from each path endpoint to each building edge
    for i in range(4):
        edge_start = corners[i]
        edge_end = corners[(i + 1) % 4]
        for point in [seg_start, seg_end]:
            dist = point_to_segment_distance(point, edge_start, edge_end)
            min_dist = min(min_dist, dist)

    return min_dist


# =============================================================================
# Path-Level Checks
# =============================================================================

def segment_is_feasible(p1: np.ndarray, p2: np.ndarray,
                        buildings: List[Building]) -> bool:
    """Check if a single segment doesn't intersect any buffered building."""
    for b in buildings:
        # Add a small epsilon (0.1) to the buffer to mathematically prevent 
        # floating-point errors from allowing edge-grazing
        buf = b.safety_buffer + 0.1
        if segment_intersects_rect(p1, p2, 
                                   b.x - buf, b.y - buf, 
                                   b.right + buf, b.top + buf):
            return False
    return True


def path_is_feasible(path: Path, buildings: List[Building]) -> bool:
    """
    Check if the entire path avoids all building interiors.
    Tests every consecutive segment against every building.
    """
    points = path.all_points
    for i in range(len(points) - 1):
        if not segment_is_feasible(points[i], points[i + 1], buildings):
            return False
    return True


def minimum_clearance(path: Path, buildings: List[Building]) -> List[float]:
    """
    Compute minimum clearance from each path segment to the nearest building.

    Returns:
        List of clearance values, one per segment (length = num_segments).
    """
    points = path.all_points
    clearances = []

    for i in range(len(points) - 1):
        seg_start = points[i]
        seg_end = points[i + 1]

        min_clear = float('inf')
        for b in buildings:
            clear = minimum_clearance_to_rect(seg_start, seg_end, b)
            # If a collision is found (-1000.0), make sure it takes absolute priority
            if clear < 0:
                min_clear = clear
                break
            min_clear = min(min_clear, clear)

        clearances.append(min_clear)

    return clearances


# =============================================================================
# Repair Utilities
# =============================================================================

def find_blocking_building(p1: np.ndarray, waypoint: np.ndarray,
                           p2: np.ndarray,
                           buildings: List[Building]) -> Optional[Building]:
    """
    Find the first building that blocks either segment (p1→waypoint) or
    (waypoint→p2).
    """
    for b in buildings:
        if (segment_intersects_rect(p1, waypoint, b.x, b.y, b.right, b.top) or
                segment_intersects_rect(waypoint, p2, b.x, b.y, b.right, b.top)):
            return b
    return None


def get_corners_with_clearance(building: Building,
                               clearance: float) -> List[np.ndarray]:
    """
    Get the four corners of a building expanded by a clearance margin.
    Used for repair routing — path routes around these corners.
    """
    b = building
    # Force the repair routing points to be 1.5x further out than the safety buffer.
    # This completely eliminates corner clipping during crossover repairs.
    eff_clearance = max(clearance, b.safety_buffer * 1.5)
    
    return [
        np.array([b.x - eff_clearance, b.y - eff_clearance]),
        np.array([b.right + eff_clearance, b.y - eff_clearance]),
        np.array([b.right + eff_clearance, b.top + eff_clearance]),
        np.array([b.x - eff_clearance, b.top + eff_clearance])
    ]