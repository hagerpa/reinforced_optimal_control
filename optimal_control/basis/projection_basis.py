import numpy as np

from .basis import Basis


class ProjectionBasis(Basis):
    def __init__(self, coordinates):
        self.coordinates = np.atleast_1d(coordinates)

        def fun(i):
            def fun_(X):
                return X[:, i]

            return fun_

        super().__init__([fun(i) for i in self.coordinates], dimension=len(self.coordinates))

    def transform(self, X, out=None):
        if out is None:
            out = X[:, self.coordinates]
        else:
            out[...] = X[:, self.coordinates]
        return out
