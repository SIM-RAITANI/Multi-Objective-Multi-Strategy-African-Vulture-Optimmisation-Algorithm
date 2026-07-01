"""
Mutation Operators
===================
Implements Adaptive Lévy perturbation and Bounded Detour mutation.
Designed to explore local spaces without disrupting the entire path.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional

import config
from environment.models import Path, Environment
from environment.collision import segment_is_feasible
from optimizer.crossover import repair_to_feasible


def _levy_step_2d(rng: np.random.Generator) -> np.ndarray:
    """Generate a 2D Lévy-distributed step vector."""
    beta = config.BETA_LEVY
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (num / den) ** (1 / beta)
    
    u = rng.normal(0, sigma, size=2)
    v = rng.normal(0, 1, size=2)
    
    epsilon = 1e-8
    step = u / (np.abs(v) ** (1 / beta) + epsilon)
    return step


def adaptive_levy_mutation(path: Path, 
                           iteration: int, 
                           max_iterations: int, 
                           env: Environment, 
                           rng: np.random.Generator) -> Optional[Path]:
    """
    Adaptive Lévy Perturbation.
    Perturbs a random waypoint using a Lévy-distributed step.
    The scale shrinks over time for convergence.
    """
    n = len(path.waypoints)
    if n == 0:
        return None
        
    idx = rng.integers(0, n)
    
    # Adaptive scale shrinks over iterations
    scale = config.LEVY_SCALE_BASE * (1.0 - iteration / max_iterations)
    step = scale * _levy_step_2d(rng)
    
    new_waypoints = path.waypoints.copy()
    new_waypoints[idx] = new_waypoints[idx] + step
    
    # Clamp to boundaries
    new_waypoints[idx, 0] = np.clip(new_waypoints[idx, 0], 0, env.world_width)
    new_waypoints[idx, 1] = np.clip(new_waypoints[idx, 1], 0, env.world_height)
    
    # Check and repair
    prev_wp = np.array(env.source) if idx == 0 else new_waypoints[idx - 1]
    next_wp = np.array(env.destination) if idx == n - 1 else new_waypoints[idx + 1]
    
    if not (segment_is_feasible(prev_wp, new_waypoints[idx], env.buildings) and 
            segment_is_feasible(new_waypoints[idx], next_wp, env.buildings)):
            
        repaired = repair_to_feasible(new_waypoints[idx], prev_wp, next_wp, env.buildings)
        if repaired is None:
            return None
        new_waypoints[idx] = repaired
        
    return Path(waypoints=new_waypoints, source=env.source, destination=env.destination)


def bounded_detour_mutation(path: Path, 
                            env: Environment, 
                            rng: np.random.Generator) -> Optional[Path]:
    """
    Bounded Detour.
    Bows out a short run of consecutive waypoints along a perpendicular offset.
    """
    n = len(path.waypoints)
    if n < 2:
        return None
        
    length = rng.integers(1, min(n, config.DETOUR_MAX_WAYPOINTS) + 1)
    start_idx = rng.integers(0, n - length + 1)
    
    # Determine base segment (prev to next) to find perpendicular
    prev_wp = np.array(env.source) if start_idx == 0 else path.waypoints[start_idx - 1]
    end_idx = start_idx + length - 1
    next_wp = np.array(env.destination) if end_idx == n - 1 else path.waypoints[end_idx + 1]
    
    direction = next_wp - prev_wp
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return None
        
    direction = direction / norm
    perp = np.array([-direction[1], direction[0]])
    
    offset_dist = rng.uniform(-config.DETOUR_OFFSET_RANGE, config.DETOUR_OFFSET_RANGE)
    offset_vec = perp * offset_dist
    
    new_waypoints = path.waypoints.copy()
    
    # Apply offset smoothly (sine window)
    for i in range(length):
        progress = (i + 1) / (length + 1)
        window = math.sin(progress * math.pi)
        new_waypoints[start_idx + i] += offset_vec * window
        
        # Clamp
        new_waypoints[start_idx + i, 0] = np.clip(new_waypoints[start_idx + i, 0], 0, env.world_width)
        new_waypoints[start_idx + i, 1] = np.clip(new_waypoints[start_idx + i, 1], 0, env.world_height)
        
    # Check feasibility of affected segments
    # For simplicity, if any segment breaks, we just discard rather than complex repair
    # Because a detour is multiple waypoints, repairing all of them is tricky.
    
    for i in range(start_idx, end_idx + 1):
        wp_prev = np.array(env.source) if i == 0 else new_waypoints[i - 1]
        wp_curr = new_waypoints[i]
        
        if not segment_is_feasible(wp_prev, wp_curr, env.buildings):
            return None
            
    # Check the last closing segment
    wp_last = new_waypoints[end_idx]
    if not segment_is_feasible(wp_last, next_wp, env.buildings):
        return None
        
    return Path(waypoints=new_waypoints, source=env.source, destination=env.destination)
