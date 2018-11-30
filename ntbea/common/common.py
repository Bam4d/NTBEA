import numpy as np
import random
from ntbea import Mutator


class DefaultMutator(Mutator):
    """
    A simple mutator used by Examples
    """

    def __init__(self, search_space, swap_mutate=False, random_chaos_mutate=False, mutation_point_probability=1.0, flip_at_least_one=True):
        super(DefaultMutator, self).__init__("Default Mutator")

        self._search_space = search_space
        self._swap_mutate = swap_mutate
        self._random_chaos_mutate = random_chaos_mutate
        self._mutation_point_probability = mutation_point_probability
        self._flip_at_least_one = flip_at_least_one

    def _swap_mutation(self, point):
        length = len(point)

        idx = random.sample(range(length), size=2)

        a = point[idx[0]]
        b = point[idx[1]]

        point[idx[0]] = b
        point[idx[1]] = a

        return point

    def _mutate_value(self, point, dim):
        """
        mutate the value of x at the given dimension 'dim'
        """
        point[dim] = np.random.choice(self._search_space.get_valid_values_in_dim(dim))

    def mutate(self, point):

        new_point = np.copy(point)
        length = len(point)

        # Perform swap mutation operation
        if self._swap_mutate:
            return self._swap_mutation(point)

        # Random mutation i.e just return a random search point
        if self._random_chaos_mutate:
            return self._search_space.get_random_point()

        # For each of the dimensions, we mutate it based on mutation_probability
        for dim in range(length):
            if self._mutation_point_probability > np.random.uniform():
                self._mutate_value(new_point, dim)

        # If we want to force flip at least one of the points then we do this here
        if self._flip_at_least_one:
            self._mutate_value(new_point, random.sample(range(length), 1)[0])

        return new_point






