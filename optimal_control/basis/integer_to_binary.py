from optimal_control.basis import Basis


class IntegerToBinary(Basis):
    def __init__(self, max_integer, min_integer=0):
        self.max_integer = max_integer
        self.min_integer = min_integer

        def f_i(i: int):
            def f_(X):
                return X[:, 0] == i

            return f_

        super().__init__(functions=[f_i(i) for i in range(min_integer, max_integer + 1)],
                         dimension=max_integer - min_integer + 1)

    def transform(self, X, out=None):
        if X.shape[1] != 1:
            raise ValueError("Input of IntegerToBinary basis is assumed to be one-dimensional.")

        out = self.__check_out__(X.shape[0], out)

        for i, k in enumerate(range(self.min_integer, self.max_integer + 1)):
            out[:, i] = X[:, 0] == k
        return out
