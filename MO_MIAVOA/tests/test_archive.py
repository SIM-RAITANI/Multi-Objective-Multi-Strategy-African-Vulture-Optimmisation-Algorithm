import unittest
import numpy as np

from environment.models import Path
from optimizer.archive import dominates, compute_crowding_distances, try_insert

class TestArchive(unittest.TestCase):

    def test_dominance(self):
        self.assertTrue(dominates((1, 1, 1), (2, 2, 2)))
        self.assertTrue(dominates((1, 2, 2), (2, 2, 2)))
        self.assertFalse(dominates((2, 2, 2), (1, 1, 1)))
        self.assertFalse(dominates((1, 3, 1), (2, 2, 2))) # incomparable

    def test_crowding_distance(self):
        p1 = Path(np.zeros((1,2)), (0,0), (1,1), objectives=(0, 10, 0))
        p2 = Path(np.zeros((1,2)), (0,0), (1,1), objectives=(5, 5, 5))
        p3 = Path(np.zeros((1,2)), (0,0), (1,1), objectives=(10, 0, 10))
        
        archive = [p1, p2, p3]
        dists = compute_crowding_distances(archive)
        
        self.assertEqual(len(dists), 3)
        self.assertEqual(dists[0], float('inf'))
        self.assertEqual(dists[2], float('inf'))
        self.assertGreater(dists[1], 0)

    def test_try_insert(self):
        archive = []
        p1 = Path(np.zeros((1,2)), (0,0), (1,1), objectives=(2, 2, 2))
        p2 = Path(np.zeros((1,2)), (0,0), (1,1), objectives=(1, 1, 1))
        
        self.assertTrue(try_insert(p1, archive, 10))
        self.assertTrue(try_insert(p2, archive, 10))
        # p2 should dominate p1 and remove it
        self.assertEqual(len(archive), 1)
        self.assertEqual(archive[0].objectives, (1, 1, 1))

if __name__ == '__main__':
    unittest.main()
