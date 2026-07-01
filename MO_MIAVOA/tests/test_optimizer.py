import unittest
import numpy as np

import config
from environment.generator import generate_environment
from optimizer.mo_miavoa import MOMIAVOA

class TestOptimizer(unittest.TestCase):

    def test_optimizer_initialization(self):
        env = generate_environment(seed=42)
        opt = MOMIAVOA(env, seed=42)
        
        # Test initialization completes
        opt.initialize_population()
        self.assertEqual(len(opt.population), config.POPULATION_SIZE)
        self.assertTrue(len(opt.archive) > 0)

if __name__ == '__main__':
    unittest.main()
