"""
Plotting Utilities
===================
Visualizes the environment, wind zones, buildings, and paths.
Also plots the 3D Pareto front.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import List

import config
from environment.models import Environment, Path


def plot_environment(env: Environment, path: Path = None, archive: List[Path] = None, show: bool = True):
    """Plot the 2D environment top-down view."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, env.world_width)
    ax.set_ylim(0, env.world_height)
    ax.set_aspect('equal')
    
    # Colors for wind zones
    wind_colors = {
        "green": "#ccebc5",   # light green
        "yellow": "#ffffcc",  # light yellow
        "red": "#fbb4ae"      # light red
    }
    
    # Plot wind zones
    for wz in env.wind_zones:
        color = wind_colors.get(wz.intensity, "white")
        rect = patches.Rectangle((wz.x, wz.y), wz.width, wz.height, 
                                 linewidth=1, edgecolor='none', facecolor=color, alpha=0.5)
        ax.add_patch(rect)
        
    # Plot buildings
    for b in env.buildings:
        rect = patches.Rectangle((b.x, b.y), b.width, b.height, 
                                 linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rect)
        
    # Plot source and destination
    ax.plot(env.source[0], env.source[1], 'bo', markersize=10, label='Source')
    ax.plot(env.destination[0], env.destination[1], 'r*', markersize=15, label='Destination')
    
    # Plot archive paths (light grey)
    if archive:
        for p in archive:
            pts = p.all_points
            ax.plot(pts[:, 0], pts[:, 1], color='gray', alpha=0.3, linewidth=1)
            
    # Plot best path (blue)
    if path:
        pts = path.all_points
        ax.plot(pts[:, 0], pts[:, 1], color='blue', linewidth=2.5, marker='.', label='Best Path')
        
    ax.set_title("MO-MIAVOA Path Planning Environment")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    if show:
        plt.show()


def plot_pareto_front(archive: List[Path], show: bool = True):
    """Plot the 3D Pareto front of Time, Energy, Risk."""
    if not archive:
        print("Archive is empty. Cannot plot Pareto front.")
        return
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    objs = np.array([p.objectives for p in archive])
    
    times = objs[:, 0]
    energies = objs[:, 1]
    risks = objs[:, 2]
    
    scatter = ax.scatter(times, energies, risks, c=risks, cmap='viridis', s=50, edgecolors='k')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')
    ax.set_zlabel('Risk')
    ax.set_title(f'Pareto Front (Size: {len(archive)})')
    
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Risk')
    
    if show:
        plt.show()
