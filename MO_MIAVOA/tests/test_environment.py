import unittest
import numpy as np

from environment.models import Building, WindZone
from environment.collision import segment_intersects_rect, minimum_clearance_to_rect
from environment.wind import _liang_barsky_clip_length

class TestEnvironment(unittest.TestCase):

    def test_segment_intersects_rect(self):
        # Rectangle 10x10 at origin
        rect = (0, 0, 10, 10)
        
        # Inside entirely
        self.assertTrue(segment_intersects_rect(np.array([2,2]), np.array([8,8]), *rect))
        
        # Crossing
        self.assertTrue(segment_intersects_rect(np.array([-5,5]), np.array([15,5]), *rect))
        
        # Outside completely
        self.assertFalse(segment_intersects_rect(np.array([-5,15]), np.array([15,15]), *rect))

    def test_minimum_clearance(self):
        b = Building(x=0, y=0, width=10, height=10)
        
        # Segment 5 units above
        c = minimum_clearance_to_rect(np.array([-5,15]), np.array([15,15]), b)
        self.assertAlmostEqual(c, 5.0)
        
        # Intersecting segment
        c2 = minimum_clearance_to_rect(np.array([5,5]), np.array([15,5]), b)
        self.assertEqual(c2, 0.0)

    def test_wind_clipping(self):
        rect = (0, 0, 10, 10)
        
        # Crossing horizontally
        l = _liang_barsky_clip_length(np.array([-5,5]), np.array([15,5]), *rect)
        self.assertAlmostEqual(l, 10.0)
        
        # Crossing diagonally
        l2 = _liang_barsky_clip_length(np.array([-5,-5]), np.array([15,15]), *rect)
        self.assertAlmostEqual(l2, 10.0 * np.sqrt(2))

if __name__ == '__main__':
    unittest.main()
