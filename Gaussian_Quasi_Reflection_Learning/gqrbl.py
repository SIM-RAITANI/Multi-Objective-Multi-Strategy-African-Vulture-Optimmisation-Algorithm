import numpy as np


class GQRBL:
    """
    Gaussian Quasi Reflection Based Learning (GQRBL)

    Generates an improved initial population for optimization algorithms.
    """

    def __init__(self, dim, lb, ub, pop_size):

        self.dim = dim
        self.pop_size = pop_size

        if np.isscalar(lb):
            self.lb = np.full(dim, lb)
        else:
            self.lb = np.array(lb)

        if np.isscalar(ub):
            self.ub = np.full(dim, ub)
        else:
            self.ub = np.array(ub)

    def random_population(self):
        """
        Generate random population.
        """

        return self.lb + np.random.rand(
            self.pop_size,
            self.dim
        ) * (self.ub - self.lb)

    def quasi_reflection(self, population):
        """
        Generate quasi-reflected population.
        """

        center = (self.lb + self.ub) / 2

        reflected = np.zeros_like(population)

        for i in range(len(population)):

            x = population[i]

            opposite = self.lb + self.ub - x

            # quasi reflection:
            reflected[i] = center + np.random.rand(
                self.dim
            ) * (opposite - center)

        return reflected

    def gaussian_perturbation(self, population, sigma=0.1):
        """
        Apply Gaussian perturbation.
        """

        noise = np.random.normal(
            loc=0,
            scale=sigma,
            size=population.shape
        )

        perturbed = population + noise * (
            self.ub - self.lb
        )

        return np.clip(
            perturbed,
            self.lb,
            self.ub
        )

    def generate(self, objective_function):
        """
        Generate GQRBL population.

        Returns:
            population
            fitness
        """

        # Step 1
        original_pop = self.random_population()

        # Step 2
        reflected_pop = self.quasi_reflection(
            original_pop
        )

        # Step 3
        reflected_pop = self.gaussian_perturbation(
            reflected_pop
        )

        # Step 4
        combined_pop = np.vstack(
            [original_pop, reflected_pop]
        )

        # Step 5
        fitness = np.array([
            objective_function(ind)
            for ind in combined_pop
        ])

        # Step 6
        best_indices = np.argsort(fitness)[
            :self.pop_size
        ]

        final_population = combined_pop[
            best_indices
        ]

        final_fitness = fitness[
            best_indices
        ]

        return final_population, final_fitness