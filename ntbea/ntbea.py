import numpy as np
from collections import defaultdict

class Mutator(object):
    """
    Inherited by other mutator objects
    """

    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def mutate(self, x):
        raise NotImplementedError()


class SearchSpace(object):
    """
    Inherited by other search space objects
    """

    def __init__(self, name, ndims):

        self._ndims = ndims
        self._name = name

    def get_name(self):
        return self._name

    def get_num_dims(self):
        return self._ndims

    def get_random_point(self):
        raise NotImplementedError()

    def get_size(self):
        raise NotImplementedError()

    def get_dim_size(self, j):
        raise NotImplementedError()


class Evaluator(object):
    """
    Inherited by other evalutator objects
    """

    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def evaluate(self, x):
        raise NotImplementedError()

class BanditLandscapeModel(object):

    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name

    def reset(self):
        raise NotImplementedError()

    def init(self):
        raise NotImplementedError()

    def add_evaluated_point(self, point, fitness):
        raise NotImplementedError()

class NTupleLandscape(BanditLandscapeModel):
    """
    The N-tuple landscape implementation
    """

    def __init__(self, search_space, tuple_config=None):
        super(BanditLandscapeModel).__init__("N-Tuple Bandit Landscape")

        # If we dont have a tuple config, we just create a default tuple config, the 1-tuples and N-tuples
        if tuple_config == None:
            tuple_config = [1, search_space.get_num_dims()]

        self._tuple_config = set(tuple_config)
        self._tuples = list()
        self._ndims = search_space.get_num_dims()

        self._tuple_stats = defaultdict(defaultdict(dict()))

    def reset(self):
        self._tuples.clear()

    def get_tuple_combinations(self, r, ndims):
        """
        Get the unique combinations of tuples for the n-tuple landscape
        :param n: the 'n' value of this tuple
        :param ndims: the number of dimensions in the search space
        :return:
        """
        return self._get_unique_combinations(r, range(0,ndims))


    def _get_unique_combinations(self, idx, r, source_array):

        result = []
        for i in range(idx, len(source_array)):
            if r - 1 > 0:
                next_level = self._get_unique_combinations(i + 1, r - 1, source_array)
                for x in next_level:
                    value = [source_array[i]]
                    value.extend(x)
                    result.append(value)

            else:
                result.append([source_array[i]])

        return result


    def init(self):
        """
        Create the index combinations for each of the n-tules
        """
        # Create all possible tuples for each
        for t in self._tuple_config:
            for i in range(0, t):
                self._tuples.extend([frozenset(tuple) for tuple in self.get_tuple_combinations(i, self._ndims)])


    def add_evaluated_point(self, point, fitness):

        for tuple in self._tuples:
            search_space_value = point[tuple]

            tuple_stats = self._tuple_stats[tuple][search_space_value]
            tuple_stats["n"] += 1
            tuple_stats["max"] = np.max(tuple_stats["max"], fitness)
            tuple_stats["min"] = np.min(tuple_stats["max"], fitness)
            tuple_stats["sum"] += fitness
            tuple_stats["sumsq"] += fitness**2



class NTupleEvolutionaryAlgorithm():

    def __init__(self, tuple_landscape, evaluator, searchSpace, mutator, k_explore=100, n_samples=1):

        self._tuple_landscape = tuple_landscape
        self._evaluator = evaluator
        self._searchSpace = searchSpace
        self._mutator = mutator
        self._k_explore = k_explore
        self._n_samples = n_samples

    def run(self, n_evaluations):

        # Get a random point to start
        point = self._searchSpace.get_random_point()

        # Repeat many times
        for eval in range(0, n_evaluations):

            # Evaluate the point (is repeated several times if n_sameples > 0)
            fitness = np.mean([self._evaluator.evaluate(point) for sample_count in range(0, self._n_samples)])

            self._tuple_landscape.add_evaluated_point(point, fitness)

