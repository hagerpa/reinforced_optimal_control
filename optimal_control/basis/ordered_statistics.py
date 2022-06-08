import numpy as np

from .basis import Basis


class OrderStatisticsBasis(Basis):
    def __init__(self, dimension):
        def max_(i):
            def fun(X):  # These functions are not used in the transform method.
                return np.sort(X, axis=1)[:, i]

            return fun

        super().__init__(functions=[max_(i) for i in range(dimension)],
                         dimension=dimension)

    def transform(self, X, out=None):
        out = self.__check_out__(X.shape[0], out)
        if self.dimension == X.shape[1]:
            out[...] = X
            out.sort(axis=1)
        else:
            # TODO: Memory Improvement?
            out = np.sort(X, axis=1)[:self.dimension]
        return out
