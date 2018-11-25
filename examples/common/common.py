from ntbea import Mutator


class DefaultMutator(Mutator):
    """
    A simple mutator used by Examples
    """

    def __init__(self):
        super(Mutator).__init__("Default Mutator")

    def mutate(self, x):
        pass