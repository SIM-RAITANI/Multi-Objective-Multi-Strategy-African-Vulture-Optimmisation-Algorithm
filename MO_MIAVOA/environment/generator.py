"""
Environment Generator
======================
Generates random but valid simulation environments with configurable seed.

Generation order (per SES Section 15):
    1. World boundary
    2. Buildings (non-overlapping)
    3. Wind zones
    4. Source & Destination (in opposite quadrants, outside buildings)
    5. Validation
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple

import config
from environment.models import Building, WindZone, Environment


def _rects_overlap(x1: float, y1: float, w1: float, h1: float,
                   x2: float, y2: float, w2: float, h2: float,
                   margin: float = 0.0) -> bool:
    """Check if two axis-aligned rectangles overlap (with optional margin)."""
    return not (x1 + w1 + margin <= x2 or x2 + w2 + margin <= x1 or
                y1 + h1 + margin <= y2 or y2 + h2 + margin <= y1)


def _point_in_any_building(px: float, py: float,
                           buildings: List[Building],
                           margin: float = 0.0) -> bool:
    """Check if a point is inside any building (with optional margin)."""
    for b in buildings:
        if (b.x - margin < px < b.right + margin and
                b.y - margin < py < b.top + margin):
            return True
    return False


def generate_buildings(rng: np.random.Generator,
                       world_w: float, world_h: float,
                       num_buildings: int,
                       safety_buffer: float) -> List[Building]:
    """
    Generate non-overlapping axis-aligned rectangular buildings.
    Buildings are placed with margins to ensure gaps between them.
    """
    buildings: List[Building] = []
    max_attempts = num_buildings * 20  # prevent infinite loops
    attempts = 0

    while len(buildings) < num_buildings and attempts < max_attempts:
        attempts += 1

        w = rng.uniform(*config.BUILDING_WIDTH_RANGE)
        h = rng.uniform(*config.BUILDING_HEIGHT_RANGE)

        # Keep buildings away from world edges
        edge_margin = 50.0
        x = rng.uniform(edge_margin, world_w - w - edge_margin)
        y = rng.uniform(edge_margin, world_h - h - edge_margin)

        # Check overlap with existing buildings (including buffer gap)
        overlaps = False
        for existing in buildings:
            if _rects_overlap(x, y, w, h,
                              existing.x, existing.y,
                              existing.width, existing.height,
                              margin=safety_buffer * 2):
                overlaps = True
                break

        if not overlaps:
            buildings.append(Building(
                x=x, y=y, width=w, height=h,
                safety_buffer=safety_buffer,
            ))

    return buildings


def generate_wind_zones(rng: np.random.Generator,
                        world_w: float, world_h: float,
                        num_zones: int) -> List[WindZone]:
    """
    Generate wind zones with varying intensities.
    Zones may overlap with buildings (wind exists around buildings).
    """
    intensities = list(config.WIND_INTENSITY.keys())
    zones: List[WindZone] = []

    for _ in range(num_zones):
        w = rng.uniform(*config.WIND_ZONE_SIZE_RANGE)
        h = rng.uniform(*config.WIND_ZONE_SIZE_RANGE)

        x = rng.uniform(0, world_w - w)
        y = rng.uniform(0, world_h - h)

        intensity = rng.choice(intensities)
        intensity_value = config.WIND_INTENSITY[intensity]

        zones.append(WindZone(
            x=x, y=y, width=w, height=h,
            intensity=intensity,
            intensity_value=intensity_value,
        ))

    return zones


def generate_endpoints(rng: np.random.Generator,
                       world_w: float, world_h: float,
                       buildings: List[Building],
                       min_separation: float) -> Tuple[Tuple[float, float],
                                                        Tuple[float, float]]:
    """
    Place source and destination in opposite quadrants with minimum separation.
    Both must be outside all buildings (including buffer).
    """
    max_attempts = 200

    for _ in range(max_attempts):
        # Source in bottom-left quadrant region
        sx = rng.uniform(50, world_w * 0.3)
        sy = rng.uniform(50, world_h * 0.3)

        # Destination in top-right quadrant region
        dx = rng.uniform(world_w * 0.7, world_w - 50)
        dy = rng.uniform(world_h * 0.7, world_h - 50)

        # Check separation
        dist = np.sqrt((dx - sx) ** 2 + (dy - sy) ** 2)
        if dist < min_separation:
            continue

        # Check not inside any building
        if (_point_in_any_building(sx, sy, buildings, margin=config.SAFETY_BUFFER) or
                _point_in_any_building(dx, dy, buildings, margin=config.SAFETY_BUFFER)):
            continue

        return (sx, sy), (dx, dy)

    # Fallback: use fixed positions far from center
    return (80.0, 80.0), (920.0, 920.0)


def validate_environment(env: Environment) -> List[str]:
    """
    Validate environment constraints (SES Section 14).
    Returns list of violation messages (empty = valid).
    """
    violations = []

    # Source and destination outside buildings
    for b in env.buildings:
        if b.contains_point(*env.source):
            violations.append(f"Source {env.source} is inside building at ({b.x}, {b.y})")
        if b.contains_point(*env.destination):
            violations.append(f"Destination {env.destination} is inside building at ({b.x}, {b.y})")

    # Buildings inside world boundary
    for b in env.buildings:
        if b.x < 0 or b.y < 0 or b.right > env.world_width or b.top > env.world_height:
            violations.append(f"Building at ({b.x}, {b.y}) exceeds world boundary")

    # Wind zones inside world boundary
    for wz in env.wind_zones:
        if wz.x < 0 or wz.y < 0 or wz.right > env.world_width or wz.top > env.world_height:
            violations.append(f"Wind zone at ({wz.x}, {wz.y}) exceeds world boundary")

    # Buildings don't overlap
    for i, b1 in enumerate(env.buildings):
        for j, b2 in enumerate(env.buildings):
            if i >= j:
                continue
            if _rects_overlap(b1.x, b1.y, b1.width, b1.height,
                              b2.x, b2.y, b2.width, b2.height):
                violations.append(
                    f"Buildings {i} and {j} overlap"
                )

    return violations


def generate_environment(seed: Optional[int] = None,
                         num_buildings: Optional[int] = None,
                         num_wind_zones: Optional[int] = None) -> Environment:
    """
    Generate a complete, validated simulation environment.

    Args:
        seed: Random seed for reproducibility. Uses config default if None.
        num_buildings: Number of buildings. Uses config default if None.
        num_wind_zones: Number of wind zones. Uses config default if None.

    Returns:
        A validated Environment instance.
    """
    if seed is None:
        seed = config.RANDOM_SEED
    if num_buildings is None:
        num_buildings = config.NUM_BUILDINGS
    if num_wind_zones is None:
        num_wind_zones = config.NUM_WIND_ZONES

    rng = np.random.default_rng(seed)

    world_w = config.WORLD_WIDTH
    world_h = config.WORLD_HEIGHT

    # Step 1: Generate buildings
    buildings = generate_buildings(rng, world_w, world_h,
                                   num_buildings, config.SAFETY_BUFFER)

    # Step 2: Generate wind zones
    wind_zones = generate_wind_zones(rng, world_w, world_h, num_wind_zones)

    # Step 3: Place source and destination
    source, destination = generate_endpoints(rng, world_w, world_h,
                                              buildings,
                                              config.SOURCE_DEST_MIN_SEPARATION)

    # Step 4: Assemble and validate
    env = Environment(
        world_width=world_w,
        world_height=world_h,
        source=source,
        destination=destination,
        buildings=buildings,
        wind_zones=wind_zones,
        seed=seed,
    )

    violations = validate_environment(env)
    if violations:
        print(f"[WARNING] Environment validation issues:")
        for v in violations:
            print(f"  - {v}")

    return env
