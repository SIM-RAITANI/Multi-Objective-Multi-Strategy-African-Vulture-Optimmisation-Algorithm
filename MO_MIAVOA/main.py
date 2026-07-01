"""
MO-MIAVOA Entry Point
======================
Sets up the environment, runs the optimizer, selects the best path,
and visualizes the results.
"""

import time
import argparse

import config
from environment.generator import generate_environment
from optimizer.mo_miavoa import MOMIAVOA
from selection.final_selection import select_best_compromise
from visualization.plotting import plot_environment, plot_pareto_front


def main():
    parser = argparse.ArgumentParser(description="MO-MIAVOA Path Planner")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()
    
    print(f"Generating Environment (Seed: {args.seed})...")
    env = generate_environment(seed=args.seed)
    
    print(f"Initializing MO-MIAVOA...")
    optimizer = MOMIAVOA(env, seed=args.seed)
    
    start_time = time.time()
    archive = optimizer.run()
    elapsed = time.time() - start_time
    print(f"Optimization finished in {elapsed:.2f} seconds.")
    
    if not archive:
        print("No feasible paths found!")
        return
        
    print(f"Found {len(archive)} Pareto-optimal paths.")
    
    best_path = select_best_compromise(archive)
    T, E, R = best_path.objectives
    print(f"\nBest Compromise Solution:")
    print(f"  Time:   {T:.2f} s")
    print(f"  Energy: {E:.2f}")
    print(f"  Risk:   {R:.2f}")
    
    if not args.no_plot:
        print("\nPlotting results...")
        plot_environment(env, path=best_path, archive=archive, show=True)
        plot_pareto_front(archive, show=True)


if __name__ == "__main__":
    main()
