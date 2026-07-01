"""
Leader Selection
=================
Selects leaders from the Pareto archive using a crowding-distance tournament.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

import config
from environment.models import Path
from optimizer.archive import compute_crowding_distances


def select_leader_pool(archive: List[Path], 
                       rng: np.random.Generator) -> Tuple[List[Path], List[float]]:
    """
    Build a pool of leaders from the archive using crowding-distance tournaments.
    
    Returns:
        A tuple of (leader_pool, their_crowding_distances).
    """
    pool_size = min(config.LEADER_POOL_SIZE, len(archive))
    if pool_size == 0:
        return [], []
        
    distances = compute_crowding_distances(archive)
    
    leader_pool = []
    pool_distances = []
    
    for _ in range(pool_size):
        # Pick 2 archive members at random
        idx1, idx2 = rng.choice(len(archive), size=2, replace=False) if len(archive) >= 2 else (0, 0)
        
        # Keep the one with greater crowding distance
        if distances[idx1] >= distances[idx2]:
            best_idx = idx1
        else:
            best_idx = idx2
            
        leader_pool.append(archive[best_idx])
        pool_distances.append(distances[best_idx])
        
    return leader_pool, pool_distances


def select_leader_roulette(leader_pool: List[Path], 
                           pool_distances: List[float], 
                           rng: np.random.Generator) -> Path:
    """
    Draw a reference vulture R(i) from the leader pool via roulette wheel,
    proportional to crowding distance.
    """
    if not leader_pool:
        raise ValueError("Leader pool is empty.")
        
    # Handle infinite distances (boundary points)
    # Replace inf with a large number relative to the max finite distance
    finite_dists = [d for d in pool_distances if d != float('inf')]
    max_finite = max(finite_dists) if finite_dists else 1.0
    
    safe_distances = np.array([d if d != float('inf') else max_finite * 2.0 
                               for d in pool_distances])
    
    # If all distances are 0, use uniform distribution
    total_dist = np.sum(safe_distances)
    if total_dist == 0:
        probs = np.ones(len(leader_pool)) / len(leader_pool)
    else:
        probs = safe_distances / total_dist
        
    chosen_idx = rng.choice(len(leader_pool), p=probs)
    return leader_pool[chosen_idx]
