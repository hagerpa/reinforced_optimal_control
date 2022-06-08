from .basis import Basis


class IdentityBasis(Basis):
    def __init__(self, dimensions):
        def fun(i):
            def fun_(X):
                return X[:, i]

            return fun_

        super().__init__(functions=[fun(i) for i in range(dimensions)], dimension=dimensions)

    def transform(self, X, out=None):
        out = self.__check_out__(X.shape[0], out)
        if X.shape[1] != self.dimension:
            raise ValueError("The dimension of this identity bases {} does not "
                             "equal the dimension of the input {}.".format(self.dimension, X.shape[1]))
        out[...] = X
        return out
