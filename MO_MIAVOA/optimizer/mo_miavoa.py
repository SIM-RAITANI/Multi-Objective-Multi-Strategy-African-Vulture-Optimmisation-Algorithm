"""
MO-MIAVOA Main Orchestrator
=============================
Executes the MO-MIAVOA algorithm, combining the base MIAVOA equations
with multi-objective Pareto archive and GA components.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple

import config
from environment.models import Path, Environment
from environment.collision import segment_is_feasible
from objectives.evaluator import evaluate_path
from optimizer.archive import try_insert, dominates
from optimizer.leader_selection import select_leader_pool, select_leader_roulette
from optimizer.miavoa_core import gqrbl_initialization, update_position, compute_hunger
from optimizer.crossover import splice_crossover, repair_to_feasible
from optimizer.mutation import adaptive_levy_mutation, bounded_detour_mutation


class MOMIAVOA:
    def __init__(self, env: Environment, seed: int = None):
        self.env = env
        self.rng = np.random.default_rng(seed if seed is not None else config.RANDOM_SEED)
        
        self.population: List[Path] = []
        self.archive: List[Path] = []
        self.dim = config.NUM_WAYPOINTS * 2
        
    def _eval_func(self, vector: np.ndarray) -> Tuple[float, float, float]:
        """Wrapper to evaluate a flattened vector as a path."""
        p = Path.from_vector(vector, self.env.source, self.env.destination)
        return evaluate_path(p, self.env)
        
    def _repair_path(self, path: Path) -> Path:
        """
        Attempts to repair infeasible waypoints by routing around obstacles.
        Ensures continuous math updates don't permanently drag paths into buildings.
        """
        waypoints = path.waypoints.copy()
        n = len(waypoints)
        changed = False
        
        for i in range(n):
            prev_wp = np.array(self.env.source) if i == 0 else waypoints[i - 1]
            next_wp = np.array(self.env.destination) if i == n - 1 else waypoints[i + 1]
            
            # Check feasibility and run existing repair loop if an obstacle is hit
            if not (segment_is_feasible(prev_wp, waypoints[i], self.env.buildings) and 
                    segment_is_feasible(waypoints[i], next_wp, self.env.buildings)):
                
                repaired = repair_to_feasible(waypoints[i], prev_wp, next_wp, self.env.buildings)
                if repaired is not None:
                    waypoints[i] = repaired
                    changed = True
                    
        if changed:
            return Path(waypoints=waypoints, source=path.source, destination=path.destination)
        return path
        
    def initialize_population(self):
        """Generate initial population using GQRBL."""
        raw_vectors = gqrbl_initialization(
            0.0, max(config.WORLD_WIDTH, config.WORLD_HEIGHT), 
            self.dim, 
            config.POPULATION_SIZE, 
            self.rng
        )
        
        valid_paths = []
        for vec in raw_vectors:
            p = Path.from_vector(vec, self.env.source, self.env.destination)
            
            # Enforce path geometry to escape obstacles right at initialization
            p = self._repair_path(p) 
            
            evaluate_path(p, self.env)
            valid_paths.append(p)
            
        for p in valid_paths:
            try_insert(p, self.archive, config.ARCHIVE_CAPACITY)
            
        self.population = []
        for p in self.archive:
            if len(self.population) < config.POPULATION_SIZE:
                self.population.append(p.copy())
                
        while len(self.population) < config.POPULATION_SIZE:
            idx = self.rng.integers(0, len(valid_paths))
            self.population.append(valid_paths[idx].copy())
            
    def run(self) -> List[Path]:
        """Execute the optimization loop."""
        print("Initializing MO-MIAVOA population...")
        self.initialize_population()
        
        print(f"Starting optimization for {config.MAX_ITERATIONS} iterations...")
        
        for iteration in range(1, config.MAX_ITERATIONS + 1):
            leader_pool, pool_distances = select_leader_pool(self.archive, self.rng)
            
            if not leader_pool:
                leader_pool = [self.population[0]]
                pool_distances = [1.0]
                
            leader1 = leader_pool[0].to_vector()
            leader2 = leader_pool[min(1, len(leader_pool)-1)].to_vector()
            
            # Position Updates
            new_population = []
            for i, vulture in enumerate(self.population):
                r_i = select_leader_roulette(leader_pool, pool_distances, self.rng).to_vector()
                p_i = vulture.to_vector()
                F = compute_hunger(iteration, config.MAX_ITERATIONS, self.rng)
                
                new_vec = update_position(
                    p_i, r_i, leader1, leader2, F, 
                    iteration, config.MAX_ITERATIONS,
                    self._eval_func,
                    dominates,
                    self.rng
                )
                
                new_path = Path.from_vector(new_vec, self.env.source, self.env.destination)
                
                # Intercept the mathematical update and snap to geometry
                new_path = self._repair_path(new_path)
                
                evaluate_path(new_path, self.env)
                new_population.append(new_path)
                
                try_insert(new_path, self.archive, config.ARCHIVE_CAPACITY)
                
            self.population = new_population
            
            # Crossover
            if iteration % config.CROSSOVER_INTERVAL == 0 and len(self.archive) >= 2:
                p1, p2 = self.rng.choice(self.archive, size=2, replace=False)
                offspring = splice_crossover(p1, p2, self.env, self.rng)
                
                if offspring:
                    evaluate_path(offspring, self.env)
                    try_insert(offspring, self.archive, config.ARCHIVE_CAPACITY)
                    
                    replace_idx = self.rng.integers(0, config.POPULATION_SIZE)
                    self.population[replace_idx] = offspring.copy()
                    
            # Mutation
            num_mutations = int(config.POPULATION_SIZE * config.MUTATION_RATE)
            for _ in range(num_mutations):
                idx = self.rng.integers(0, config.POPULATION_SIZE)
                cand = self.population[idx]
                
                if self.rng.random() < 0.5:
                    mutated = adaptive_levy_mutation(cand, iteration, config.MAX_ITERATIONS, self.env, self.rng)
                else:
                    mutated = bounded_detour_mutation(cand, self.env, self.rng)
                    
                if mutated:
                    evaluate_path(mutated, self.env)
                    try_insert(mutated, self.archive, config.ARCHIVE_CAPACITY)
                    self.population[idx] = mutated
                    
            if iteration % 1 == 0:
                print(f"Iteration {iteration}/{config.MAX_ITERATIONS} | Archive Size: {len(self.archive)}")
                
        print("Optimization complete.")
        return self.archive