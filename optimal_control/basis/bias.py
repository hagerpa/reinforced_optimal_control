import numpy as np

from optimal_control.basis import Basis


class Bias(Basis):
    """A constant basis which always returns a vector of ones."""

    def __init__(self):
        def f(X):
            return np.ones_like(X[:, 0])

        super().__init__(functions=[f], dimension=1)
