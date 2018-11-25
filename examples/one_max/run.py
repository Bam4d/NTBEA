import numpy as np
from ntbea import SearchSpace, Evaluator, NTupleLandscape, NTupleEvolutionaryAlgorithm
from examples.common import DefaultMutator

class OneMaxSearchSpace(SearchSpace):

    def __init__(self, ndims):
        super(OneMaxSearchSpace).__init__("OneMax SearchSpace", ndims)

        # Binary data for 0 and 1
        self._search_space = np.zeros([2,self._ndims], dtype=np.bool)
        self._search_space[1,:] = True

    def get_value_at(self, idx):
        assert len(idx) == 2
        assert idx[0] < self._ndims
        assert idx[1] < 2

        return self._search_space[idx]

    def get_random_point(self):
        return np.random.uniform(size=self._ndims)



class OneMaxEvaluator(Evaluator):

    def __init__(self):
        super(OneMaxEvaluator).__init__("OneMax Evalutator")

    def evaluate(self, x):
        return np.sum(x)


if __name__ == "__main__":

    # Set up the problem domain as one-max problem
    search_space = OneMaxSearchSpace(5)
    evaluator = OneMaxEvaluator()

    # 1-tuple, 2-tuple, 3-tuple and N-tuple
    tuple_landscape = NTupleLandscape(search_space, [1, 2, 3, 5])

    # Set the mutator type
    mutator = DefaultMutator()

    evolutionary_algorithm = NTupleEvolutionaryAlgorithm(tuple_landscape, search_space, evaluator, mutator)

    evolutionary_algorithm.run(2000)

