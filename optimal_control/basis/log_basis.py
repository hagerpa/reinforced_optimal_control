import numpy as np

from .basis import Basis


class LogBasis(Basis):
    def __init__(self, dimensions, intercept=None):
        self.intercept = intercept

        def fun(i):
            def fun_(X):
                return np.log(X[:, i])

            return fun_

        super().__init__(functions=[fun(i) for i in range(dimensions)], dimension=dimensions)

    def transform(self, X, out=None):
        out = self.__check_out__(X.shape[0], out)
        if X.shape[1] != self.dimension:
            raise ValueError("The dimension of this identity bases {} does not "
                             "equal the dimension of the input {}.".format(self.dimension, X.shape[1]))
        out[...] = np.log(X) if (self.intercept is None) else np.log(X * self.intercept)
        return out
