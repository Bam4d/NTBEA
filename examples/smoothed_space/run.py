import logging
import numpy as np

from ntbea import SearchSpace, Evaluator, NTupleLandscape, NTupleEvolutionaryAlgorithm
from ntbea.common import DefaultMutator


class GaussianSpace(SearchSpace):

    def __init__(self, ndims, slices):
        super().__init__("Goldstein Search Space", ndims * 2)

        mu, sig = self._generate_gaussians_population((0, 0), (1, 1), (0.05, 0.05), (0.1, 0.1), slices)
        self._slices = slices
        self._search_space = np.concatenate([mu.T, sig.T])

    def sample_value_at(self, idx):
        nout = self._ndims // 2
        values = np.zeros([nout])
        for d in range(nout):
            mu = self._search_space[d, idx[d]]
            sig = self._search_space[d + nout, idx[d + nout]]
            values[d] = np.random.normal(mu, sig)
        return values

    def get_random_point(self):
        return np.random.choice(np.arange(1, self._slices), size=(self._ndims))

    def get_valid_values_in_dim(self, dim):
        return np.arange(len(self._search_space[dim]))

    def get_size(self):
        return self._slices ** self._ndims

    def _generate_gaussians_population(self, min_mu, max_mu, min_sig, max_sig, steps):
        mu = np.linspace(min_mu, max_mu, steps)
        sig = np.linspace(min_sig, max_sig, steps)
        return mu, sig


class GoldsteinEvaluator(Evaluator):

    def __init__(self):
        super().__init__("Goldstein Price")

    def evaluate(self, x):
        x1 = x[0] * 4.0 - 2.0
        x2 = x[1] * 4.0 - 2.0
        return np.maximum(0, ((400.0 - (
                1.0 + (x1 + x2 + 1) ** 2 * (19.0 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)) *
                               (30.0 + (2 * x1 - 3 * x2) ** 2 * (
                                       18.0 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))) / 500.0))


class Braninvaluator(Evaluator):

    def __init__(self):
        super().__init__("Goldstein Price")

    def evaluate(self, x):
        x1 = x[0] * 4.0 - 2.0
        x2 = x[1] * 4.0 - 2.0
        return np.maximum(0, (400.0 - (1.0 + (x1 + x2 + 1) ** 2 * (
                19.0 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)) *
                              (30.0 + (2 * x1 - 3 * x2) ** 2 * (
                                      18.0 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))) / 500.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dims = 2
    slices = 30

    # Set up the problem domain as one-max problem
    search_space = GaussianSpace(dims, slices)
    evaluator = GoldsteinEvaluator()

    # 1-tuple, 2-tuple, 3-tuple and N-tuple
    tuple_landscape = NTupleLandscape(search_space, [1, 2, dims])

    # Set the mutator type
    mutator = DefaultMutator(search_space)

    evolutionary_algorithm = NTupleEvolutionaryAlgorithm(tuple_landscape, evaluator, search_space, mutator,
                                                         n_samples=100,
                                                         k_explore=2.0)

    evolutionary_algorithm.run(2000)
