"""
Crossover & Repair
===================
Implements the "splice by waypoint index" crossover and 
nearest-feasible-location repair mechanism.
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple

import config
from environment.models import Path, Building, Environment
from environment.collision import (
    segment_is_feasible,
    find_blocking_building,
    get_corners_with_clearance
)


def repair_to_feasible(waypoint: np.ndarray, 
                       prev_wp: np.ndarray, 
                       next_wp: np.ndarray, 
                       buildings: List[Building],
                       max_retries: int = config.REPAIR_MAX_RETRIES) -> Optional[np.ndarray]:
    """
    Move waypoint to nearest position where both adjacent edges clear all buildings.
    Routes around the blocking obstacle via its nearest visible corner.
    """
    current_wp = waypoint.copy()
    
    for _ in range(max_retries):
        if segment_is_feasible(prev_wp, current_wp, buildings) and \
           segment_is_feasible(current_wp, next_wp, buildings):
            return current_wp  # Feasible
        
        # Find the blocking building
        blocker = find_blocking_building(prev_wp, current_wp, next_wp, buildings)
        
        if blocker is None:
            # Should not happen if it wasn't feasible, but fallback
            return current_wp
            
        # Move waypoint to the nearest corner of the blocker + clearance offset
        corners = get_corners_with_clearance(blocker, config.REPAIR_CLEARANCE)
        
        # Pick the corner that minimizes total detour distance from prev to next
        best_corner = min(corners, key=lambda c: 
            np.linalg.norm(prev_wp - c) + np.linalg.norm(c - next_wp))
        
        # Only accept if this corner has line-of-sight to both neighbors
        if segment_is_feasible(prev_wp, best_corner, buildings) and \
           segment_is_feasible(best_corner, next_wp, buildings):
            return best_corner
            
        # Try again from new position
        current_wp = best_corner
    
    return None  # Repair failed — discard this offspring


def splice_crossover(parent1: Path, 
                     parent2: Path, 
                     env: Environment, 
                     rng: np.random.Generator) -> Optional[Path]:
    """
    Perform a splice crossover between two parent paths.
    Creates exactly one new joint edge. If the joint intersects a building,
    attempts to repair it.
    
    Returns a new Path if successful, or None if repair fails.
    """
    n = len(parent1.waypoints)
    if n < 2:
        return None
        
    # Choose cut index k (1 to n-1)
    k = rng.integers(1, n)
    
    # Offspring keeps P1's waypoints up to k (exclusive), and P2's from k to end
    offspring_waypoints = np.vstack([
        parent1.waypoints[:k],
        parent2.waypoints[k:]
    ])
    
    # The joint is between offspring_waypoints[k-1] and offspring_waypoints[k]
    # To repair, we can try moving either k-1 or k
    
    # We first check if the whole path is feasible
    # Actually, we only need to check the joint segment and adjacent segments
    # For simplicity and robustness, we will check the joint specifically
    
    # Previous waypoint to the joint is k-2 (or source if k=1)
    if k == 1:
        prev_wp = np.array(env.source)
    else:
        prev_wp = offspring_waypoints[k - 2]
        
    wp1 = offspring_waypoints[k - 1]
    wp2 = offspring_waypoints[k]
    
    # Next waypoint to the joint is k+1 (or dest if k=n-1)
    if k == n - 1:
        next_wp = np.array(env.destination)
    else:
        next_wp = offspring_waypoints[k + 1]
        
    # Is the joint feasible?
    if segment_is_feasible(wp1, wp2, env.buildings):
        # We also need to make sure we didn't break line of sight from prev to wp1 or wp2 to next
        # But those came directly from parents, so they should be feasible, unless the cut itself
        # is the only new thing. Wait, the ONLY new edge is wp1 -> wp2.
        return Path(waypoints=offspring_waypoints, source=env.source, destination=env.destination)
        
    # If not feasible, try to repair
    # Strategy: try moving wp1, if fails try moving wp2
    
    repaired_wp1 = repair_to_feasible(wp1, prev_wp, wp2, env.buildings)
    if repaired_wp1 is not None:
        offspring_waypoints[k - 1] = repaired_wp1
        return Path(waypoints=offspring_waypoints, source=env.source, destination=env.destination)
        
    repaired_wp2 = repair_to_feasible(wp2, wp1, next_wp, env.buildings)
    if repaired_wp2 is not None:
        offspring_waypoints[k] = repaired_wp2
        return Path(waypoints=offspring_waypoints, source=env.source, destination=env.destination)
        
    # Repair failed completely
    return None
