[![PyPI version](https://badge.fury.io/py/NTBEA.svg)](https://badge.fury.io/py/NTBEA)

# NTBEA
An implementation of the [N-Tuple Evolutionary Algorithm](https://arxiv.org/pdf/1802.05991.pdf)

## Installation

### using pip

```
pip install ntbea
```

## Usage

To use the NTBEA algorithm, you will need to define the following:

#### Search Space
The search space defines the potential parameters in their respective dimensions

#### Evaluator
The evaluator scores is used to score the combination of parameters for the optimiztion problem

#### NTupleLandscape
The NTuple landscape is the set of tuples which are used to choose combinations of parameters to test and score



### Examples

Examples of setting up the `Search Space`, `Evaluator` and `NTupleLandscape` can be found in the `examples` directory and run with:
```
python run.py
```

#### m_max example

```
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
```


## Cite

If you want to cite this library, please use the following DOI

