import logging
import numpy as np

from ntbea import SearchSpace, Evaluator, NTupleLandscape, NTupleEvolutionaryAlgorithm
from examples.common import DefaultMutator


class OneMaxSearchSpace(SearchSpace):

    def __init__(self, ndims):
        super(OneMaxSearchSpace, self).__init__("OneMax Search Space", ndims)

        # Binary data for 0 and 1
        self._search_space = np.zeros([2, self._ndims], dtype=np.bool)
        self._search_space[1, :] = True

    def get_value_at(self, idx):
        assert len(idx) == 2
        assert idx[0] < self._ndims
        assert idx[1] < 2

        return self._search_space[idx]

    def get_random_point(self):
        return np.random.choice([0, 1], size=(self._ndims))

    def get_valid_values_in_dim(self, dim):
        """
        The only valid values are 0 and 1 in all dimensions
        """
        return [0, 1]

    def get_size(self):
        return 2 ** self._ndims


class OneMaxEvaluator(Evaluator):

    def __init__(self):
        super(OneMaxEvaluator, self).__init__("OneMax Evalutator")

    def evaluate(self, x):
        return np.sum(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    max_dims = 10

    # Set up the problem domain as one-max problem
    search_space = OneMaxSearchSpace(max_dims)
    evaluator = OneMaxEvaluator()

    # 1-tuple, 2-tuple, 3-tuple and N-tuple
    tuple_landscape = NTupleLandscape(search_space, [1, 2, max_dims])

    # Set the mutator type
    mutator = DefaultMutator(search_space)

    evolutionary_algorithm = NTupleEvolutionaryAlgorithm(tuple_landscape, evaluator, search_space, mutator,
                                                         k_explore=2.0)

    evolutionary_algorithm.run(200)
