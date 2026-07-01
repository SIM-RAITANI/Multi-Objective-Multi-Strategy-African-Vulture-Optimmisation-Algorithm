import unittest
import numpy as np

import config
from environment.models import Path, WindZone
from objectives.time_obj import compute_time
from objectives.energy_obj import compute_energy
from objectives.risk_obj import _compute_turning_risk

class TestObjectives(unittest.TestCase):

    def setUp(self):
        # 3 points forming a right angle triangle path: (0,0) -> (10,0) -> (10,10)
        # 1 waypoint in the middle
        self.path = Path(
            waypoints=np.array([[10, 0]]),
            source=(0, 0),
            destination=(10, 10)
        )
        self.wind_zones = [WindZone(x=5, y=-5, width=10, height=10, intensity="yellow", intensity_value=3.0)]

    def test_time(self):
        t = compute_time(self.path)
        # length is 10 + 10 = 20
        # time = 20 / cruise_speed
        self.assertAlmostEqual(t, 20.0 / config.DRONE_CRUISE_SPEED)

    def test_energy(self):
        e = compute_energy(self.path, self.wind_zones)
        # dist_energy = 20 * k1
        # wind_energy: path enters wind zone at x=5, goes to x=10. Length in zone = 5.
        # wind_energy = 5 * intensity(3) * k2
        expected = 20.0 * config.K1_ENERGY_DISTANCE + 30.0 * config.K2_ENERGY_WIND
        self.assertAlmostEqual(e, expected)

    def test_turning_risk(self):
        r = _compute_turning_risk(self.path)
        # angle is 90 degrees = pi/2
        self.assertAlmostEqual(r, np.pi / 2)

if __name__ == '__main__':
    unittest.main()
