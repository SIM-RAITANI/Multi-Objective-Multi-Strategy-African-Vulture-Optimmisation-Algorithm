"""
Environment Package
====================
Defines the simulation world: buildings, wind zones, source/destination,
and utility functions for collision detection and wind exposure computation.

This package is algorithm-agnostic — it describes the world only.
"""

from environment.models import Building, WindZone, Environment, Path
from environment.generator import generate_environment
from environment.collision import (
    path_is_feasible,
    segment_is_feasible,
    minimum_clearance,
)
from environment.wind import wind_exposure
