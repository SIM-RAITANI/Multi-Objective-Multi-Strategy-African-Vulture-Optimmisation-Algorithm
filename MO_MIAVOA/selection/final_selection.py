"""
Final Solution Selection
=========================
Selects the 'best' compromise solution from the Pareto archive
using a weighted sum of normalized objectives.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional

import config
from environment.models import Path


def select_best_compromise(archive: List[Path], weights: Dict[str, float] = None) -> Optional[Path]:
    """
    Selects a single solution from the Pareto archive using weighted sum 
    of min-max normalized objectives.
    """
    if not archive:
        return None
        
    if len(archive) == 1:
        return archive[0]
        
    if weights is None:
        weights = config.DEFAULT_WEIGHTS
        
    w_array = np.array([weights["time"], weights["energy"], weights["risk"]])
    
    # Extract objectives
    obj_array = np.array([p.objectives for p in archive])  # shape (N, 3)
    
    # Min-max normalization
    obj_min = np.min(obj_array, axis=0)
    obj_max = np.max(obj_array, axis=0)
    
    # Avoid division by zero if all values are the same
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0
    
    norm_obj = (obj_array - obj_min) / obj_range
    
    # Weighted sum
    scores = np.dot(norm_obj, w_array)
    
    # We want to minimize the score
    best_idx = np.argmin(scores)
    
    return archive[best_idx]
