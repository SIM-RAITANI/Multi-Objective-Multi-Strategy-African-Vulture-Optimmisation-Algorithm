"""
MO-MIAVOA Configuration
========================
Central configuration for all tunable parameters.
All values are derived from the MIAVOA base paper (Li et al. 2024)
and the MO-MIAVOA Design Document.
"""

import math

# =============================================================================
# Population & Iterations
# =============================================================================
POPULATION_SIZE = 30          # N — number of vultures (candidate paths)
MAX_ITERATIONS = 500          # T — maximum optimization iterations
NUM_WAYPOINTS = 25            # n — intermediate waypoints per path

# =============================================================================
# World
# =============================================================================
WORLD_WIDTH = 1000.0          # meters
WORLD_HEIGHT = 1000.0         # meters

# =============================================================================
# MIAVOA Parameters (from base paper Table 6)
# =============================================================================
P1 = 0.6                     # Exploration strategy selection threshold
P2 = 0.4                     # Exploitation sub-phase threshold
P3 = 0.6                     # Late exploitation strategy threshold
ALPHA_CONTROL = 0.8           # Adaptive control parameter (α)
BETA_LEVY = 1.5               # Lévy flight exponent (β)

# =============================================================================
# Pareto Archive
# =============================================================================
ARCHIVE_CAPACITY = 75         # C_max — maximum archive size
LEADER_POOL_SIZE = 4          # Number of leaders selected per iteration

# =============================================================================
# Drone Parameters
# =============================================================================
DRONE_CRUISE_SPEED = 15.0     # v (m/s) — constant cruise speed

# =============================================================================
# Objective 2 — Energy Coefficients
# =============================================================================
K1_ENERGY_DISTANCE = 1.0     # k1 — weight for distance-driven energy cost
K2_ENERGY_WIND = 0.5          # k2 — weight for wind-driven energy cost

# =============================================================================
# Wind Zone Intensities
# =============================================================================
WIND_INTENSITY = {
    "green": 1.0,             # Low wind — minor effect
    "yellow": 3.0,            # Medium wind — moderate effect
    "red": 6.0,               # High wind — severe effect
}

# =============================================================================
# Objective 3 — Risk Sub-Component Weights
# =============================================================================
W_CLEARANCE = 10.0           # w_c — obstacle clearance risk weight
W_TURNING = 0.3               # w_t — turning severity risk weight
W_WIND_RISK = 0.3             # w_w — wind exposure risk weight
CLEARANCE_DECAY_ALPHA = 0.05  # α in f(c) = exp(-α·c) for clearance penalty
SAFETY_BUFFER = 15.0          # meters — buffer zone around buildings

# =============================================================================
# GA Component — Crossover
# =============================================================================
CROSSOVER_INTERVAL = 12       # Iterations between crossover events
REPAIR_MAX_RETRIES = 3        # Max crossover repair attempts per offspring
REPAIR_CLEARANCE = 20.0      # meters — clearance offset for corner routing

# =============================================================================
# GA Component — Mutation (Adaptive Lévy)
# =============================================================================
MUTATION_RATE = 0.1           # Fraction of population mutated per iteration
LEVY_SCALE_BASE = 1.0         # Base scale factor for Adaptive Lévy mutation
DETOUR_MAX_WAYPOINTS = 3      # Max consecutive waypoints for bounded detour
DETOUR_OFFSET_RANGE = 50.0    # Max perpendicular offset for detour (meters)

# =============================================================================
# Environment Generation Defaults
# =============================================================================
NUM_BUILDINGS = 15            # Default number of buildings
BUILDING_WIDTH_RANGE = (30, 100)    # (min, max) meters
BUILDING_HEIGHT_RANGE = (30, 150)   # (min, max) meters
NUM_WIND_ZONES = 4            # Default number of wind zones
WIND_ZONE_SIZE_RANGE = (100, 300)   # (min, max) meters per dimension
SOURCE_DEST_MIN_SEPARATION = 500.0  # Minimum distance between source & dest

# =============================================================================
# Final Solution Selection
# =============================================================================
DEFAULT_WEIGHTS = {
    "time": 0.4,
    "energy": 0.3,
    "risk": 0.3,
}

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_SEED = 42              # Default seed for reproducible runs
