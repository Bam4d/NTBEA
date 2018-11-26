import logging
import numpy as np

from ntbea import SearchSpace, Evaluator, NTupleLandscape, NTupleEvolutionaryAlgorithm
from examples.common import DefaultMutator


class MMaxSearchSpace(SearchSpace):

    def __init__(self, ndims, m):
        super(MMaxSearchSpace, self).__init__("M-Max Search Space", ndims)

        self._search_space = np.reshape(np.arange(0, m), [m, 1]) + np.ones(self._ndims)
        self._m = m

    def get_value_at(self, idx):
        assert len(idx) == 2
        assert idx[0] < self._ndims
        assert idx[1] < self._m

        return self._search_space[idx]

    def get_random_point(self):
        return np.random.choice(np.arange(1, self._m + 1), size=(self._ndims))

    def get_valid_values_in_dim(self, dim):
        return self._search_space[:, dim]

    def get_size(self):
        return self._m ** self._ndims


class MMaxEvaluator(Evaluator):

    def __init__(self):
        super(MMaxEvaluator, self).__init__("M-Max Evalutator")

    def evaluate(self, x):
        return np.sum(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    max_dims = 6
    max_m = 4

    # Set up the problem domain as one-max problem
    search_space = MMaxSearchSpace(max_dims, max_m)
    evaluator = MMaxEvaluator()

    # 1-tuple, 2-tuple, 3-tuple and N-tuple
    tuple_landscape = NTupleLandscape(search_space, [1,2,max_dims])

    # Set the mutator type
    mutator = DefaultMutator(search_space, mutation_point_probability=0.5)

    evolutionary_algorithm = NTupleEvolutionaryAlgorithm(tuple_landscape, evaluator, search_space, mutator,
                                                         k_explore=2.0, eval_neighbours=50)

    evolutionary_algorithm.run(5000)
