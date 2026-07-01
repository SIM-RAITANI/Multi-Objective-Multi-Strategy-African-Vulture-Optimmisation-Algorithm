"""
MIAVOA Core Equations
======================
Implements the core position-update equations from the MIAVOA base paper.
Includes the improved hunger factor, adaptive control, and elite candidate pool.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Callable, List, Tuple

import config
from environment.models import Path


def _levy_flight(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Lévy flight step vector."""
    beta = config.BETA_LEVY
    num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (num / den) ** (1 / beta)
    
    u = rng.normal(0, sigma, size=dim)
    v = rng.normal(0, 1, size=dim)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    step = 0.01 * u / (np.abs(v) ** (1 / beta) + epsilon)
    return step


def compute_hunger(iteration: int, max_iterations: int, rng: np.random.Generator) -> float:
    """
    Compute the improved hunger factor F (Eq 25).
    """
    omega = (math.pi / 2) * (iteration / max_iterations)
    h = rng.uniform(-2, 2)
    
    t = h * (math.sin(omega) * abs(math.cos(omega)) - math.cos(omega) * abs(math.sin(omega)))
    
    z = rng.uniform(-1, 1)
    rand1 = rng.random()
    
    decay_term = -math.exp(((iteration / max_iterations) * (math.e - 1)) ** 2 - 1)
    
    F = (2 * rand1 + 1) * z * decay_term + t
    return F


def clip_position(pos: np.ndarray) -> np.ndarray:
    """Clamp the position vector to the world boundaries."""
    # pos is shaped as [x1, y1, x2, y2, ...]
    # We clip x to [0, WORLD_WIDTH] and y to [0, WORLD_HEIGHT]
    pos_2d = pos.reshape(-1, 2)
    pos_2d[:, 0] = np.clip(pos_2d[:, 0], 0, config.WORLD_WIDTH)
    pos_2d[:, 1] = np.clip(pos_2d[:, 1], 0, config.WORLD_HEIGHT)
    return pos_2d.flatten()


def update_position(p_i: np.ndarray, 
                    r_i: np.ndarray, 
                    leader1: np.ndarray, 
                    leader2: np.ndarray,
                    F: float, 
                    iteration: int, 
                    max_iterations: int,
                    eval_func: Callable[[np.ndarray], Tuple[float, float, float]],
                    dominates_func: Callable[[Tuple[float, float, float], Tuple[float, float, float]], bool],
                    rng: np.random.Generator) -> np.ndarray:
    """
    Update the position of a single vulture based on MIAVOA equations.
    
    Args:
        p_i: Current position vector.
        r_i: Reference vulture (leader) chosen via roulette wheel.
        leader1: Best leader (e.g., highest crowding distance).
        leader2: Second best leader.
        F: Hunger factor.
        iteration: Current iteration.
        max_iterations: Total iterations.
        eval_func: Function to evaluate a position vector (returns T, E, R).
        dominates_func: Function to check Pareto dominance.
        rng: Random generator.
        
    Returns:
        The new position vector (clipped to bounds).
    """
    dim = len(p_i)
    
    if abs(F) >= 1.0:
        # Exploration Phase
        if config.P1 >= rng.random():
            # Strategy 1
            X = 2 * rng.random()
            D_i = np.abs(X * r_i - p_i)
            new_pos = r_i - D_i * F
        else:
            # Strategy 2
            lb = 0.0
            # For simplicity, we assume ub is WORLD_WIDTH (though technically it's a mix of W and H)
            # Since we clip anyway, this approximation is fine for exploration bounding.
            ub = max(config.WORLD_WIDTH, config.WORLD_HEIGHT)
            new_pos = r_i - F + rng.random() * ((ub - lb) * rng.random() + lb)
            
    else:
        # Exploitation Phase
        if abs(F) >= 0.5:
            # Sub-phase 1
            if config.P2 >= rng.random():
                # Food protection
                d_t = r_i - p_i
                X = 2 * rng.random()
                D_i = np.abs(X * r_i - p_i)
                new_pos = D_i * (F + rng.random()) - d_t
            else:
                # Spiral flight + Elite Candidate Pool
                S1 = r_i * (rng.random() / (2 * math.pi)) * np.cos(p_i)
                S2 = r_i * (rng.random() / (2 * math.pi)) * np.sin(p_i)
                
                R1 = r_i + S1
                R2 = r_i - S1
                R3 = r_i + S2
                R4 = r_i - S2
                
                candidates = [R1, R2, R3, R4]
                best_cand = candidates[0]
                best_obj = eval_func(clip_position(best_cand))
                
                for cand in candidates[1:]:
                    cand_obj = eval_func(clip_position(cand))
                    if dominates_func(cand_obj, best_obj):
                        best_cand = cand
                        best_obj = cand_obj
                        
                new_pos = best_cand
        else:
            # Sub-phase 2
            if config.P3 >= rng.random():
                # Accumulation + Adaptive Control
                epsilon = 1e-8
                A1 = leader1 * (leader1 * p_i) / (leader1 - p_i ** 2 + epsilon) * F
                A2 = leader2 * (leader2 * p_i) / (leader2 - p_i ** 2 + epsilon) * F
                
                alpha = config.ALPHA_CONTROL
                alpha1 = alpha + (1 - alpha) * (iteration / max_iterations)
                alpha2 = (1 - alpha) - (1 - alpha) * (iteration / max_iterations)
                
                new_pos = alpha1 * A1 + alpha2 * A2
            else:
                # Lévy flight
                d_t = r_i - p_i
                new_pos = r_i - np.abs(d_t) * F * _levy_flight(dim, rng)
                
    return clip_position(new_pos)


def gqrbl_initialization(lb: float, ub: float, dim: int, num_vultures: int, 
                         rng: np.random.Generator) -> List[np.ndarray]:
    """
    Generate initial population using Gaussian Quasi-Reflection-Based Learning.
    Returns a list of position vectors.
    """
    population = []
    
    # Gaussian chaos map generation for gamma
    gamma = rng.random()
    if gamma == 0:
        gamma = rng.random()
        
    for _ in range(num_vultures):
        # Generate random position
        x = rng.uniform(lb, ub, size=dim)
        
        # Update gamma using chaos map
        gamma = (1 / gamma) % 1 if gamma != 0 else rng.random()
        
        # Generate quasi-reflected position
        # Bound it between center and reflected point
        center = (lb + ub) / 2
        reflected = lb + ub - gamma * x
        
        x_gqr = np.zeros(dim)
        for i in range(dim):
            min_val = min(center, reflected[i])
            max_val = max(center, reflected[i])
            x_gqr[i] = rng.uniform(min_val, max_val)
            
        # We will keep both and let the archive sorting handle it later,
        # or we just pick one. The paper evaluates both and keeps the best.
        # Since evaluation requires the environment, we'll just yield the
        # initial random one and the reflected one, and the main loop will
        # evaluate and insert into archive. To keep population size N,
        # we'll return both x and x_gqr and take the best N later, OR we can
        # just return a pool of 2N and evaluate them.
        population.append(x)
        population.append(x_gqr)
        
    return population
