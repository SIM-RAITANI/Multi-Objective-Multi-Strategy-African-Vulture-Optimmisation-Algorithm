"""
Data Models
============
Core data structures for the simulation environment.
These are pure data containers — no optimization logic.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Building:
    """
    An axis-aligned rectangular obstacle.

    Attributes:
        x: Left edge x-coordinate (meters).
        y: Bottom edge y-coordinate (meters).
        width: Width of the building (meters).
        height: Height of the building (meters).
        safety_buffer: Additional clearance margin around the building (meters).
    """
    x: float
    y: float
    width: float
    height: float
    safety_buffer: float = 15.0

    @property
    def right(self) -> float:
        """Right edge x-coordinate."""
        return self.x + self.width

    @property
    def top(self) -> float:
        """Top edge y-coordinate."""
        return self.y + self.height

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the building."""
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def corners(self) -> List[Tuple[float, float]]:
        """Four corners of the building: [bottom-left, bottom-right, top-right, top-left]."""
        return [
            (self.x, self.y),
            (self.right, self.y),
            (self.right, self.top),
            (self.x, self.top),
        ]

    @property
    def buffered_corners(self) -> List[Tuple[float, float]]:
        """Four corners expanded by the safety buffer — used for repair routing."""
        b = self.safety_buffer
        return [
            (self.x - b, self.y - b),
            (self.right + b, self.y - b),
            (self.right + b, self.top + b),
            (self.x - b, self.top + b),
        ]

    def contains_point(self, px: float, py: float) -> bool:
        """Check if a point is strictly inside the building rectangle."""
        return self.x < px < self.right and self.y < py < self.top

    def contains_point_buffered(self, px: float, py: float) -> bool:
        """Check if a point is inside the building + safety buffer zone."""
        b = self.safety_buffer
        return (self.x - b < px < self.right + b and
                self.y - b < py < self.top + b)


@dataclass
class WindZone:
    """
    An axis-aligned rectangular wind region.

    Attributes:
        x: Left edge x-coordinate (meters).
        y: Bottom edge y-coordinate (meters).
        width: Width of the zone (meters).
        height: Height of the zone (meters).
        intensity: Wind intensity label — "green", "yellow", or "red".
        intensity_value: Numerical intensity value for computation.
    """
    x: float
    y: float
    width: float
    height: float
    intensity: str  # "green", "yellow", or "red"
    intensity_value: float = 1.0

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y + self.height

    def as_rect(self) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) for clipping computations."""
        return (self.x, self.y, self.right, self.top)


@dataclass
class Environment:
    """
    Complete simulation environment — the single source of truth.

    Contains everything needed to evaluate a drone path:
    world boundaries, obstacles, wind zones, and endpoints.
    """
    world_width: float
    world_height: float
    source: Tuple[float, float]
    destination: Tuple[float, float]
    buildings: List[Building]
    wind_zones: List[WindZone]
    seed: Optional[int] = None

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) of the world."""
        return (0.0, 0.0, self.world_width, self.world_height)


@dataclass
class Path:
    """
    A candidate drone path from source to destination.

    The path is:  source → waypoints[0] → waypoints[1] → ... → waypoints[n-1] → destination

    Attributes:
        waypoints: NumPy array of shape (n, 2) — the intermediate waypoint coordinates.
        source: Starting point (from environment).
        destination: Ending point (from environment).
        objectives: Cached (T, E, R) tuple, or None if not yet evaluated.
    """
    waypoints: np.ndarray  # shape (NUM_WAYPOINTS, 2)
    source: Tuple[float, float]
    destination: Tuple[float, float]
    objectives: Optional[Tuple[float, float, float]] = None

    @property
    def all_points(self) -> np.ndarray:
        """
        Full sequence of points including source and destination.
        Returns array of shape (n+2, 2).
        """
        src = np.array(self.source).reshape(1, 2)
        dst = np.array(self.destination).reshape(1, 2)
        return np.vstack([src, self.waypoints, dst])

    @property
    def num_segments(self) -> int:
        """Number of line segments in the path."""
        return len(self.waypoints) + 1

    def to_vector(self) -> np.ndarray:
        """Flatten waypoints to 1D vector [x1, y1, x2, y2, ...] for MIAVOA equations."""
        return self.waypoints.flatten()

    @staticmethod
    def from_vector(vector: np.ndarray, source: Tuple[float, float],
                    destination: Tuple[float, float]) -> 'Path':
        """Create a Path from a flattened 1D vector."""
        waypoints = vector.reshape(-1, 2)
        return Path(waypoints=waypoints, source=source, destination=destination)

    def copy(self) -> 'Path':
        """Return a deep copy of this path."""
        return Path(
            waypoints=self.waypoints.copy(),
            source=self.source,
            destination=self.destination,
            objectives=self.objectives,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Path):
            return NotImplemented
        return (
            np.array_equal(self.waypoints, other.waypoints) and
            self.source == other.source and
            self.destination == other.destination
        )
