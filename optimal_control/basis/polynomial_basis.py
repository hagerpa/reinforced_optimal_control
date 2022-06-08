from itertools import combinations_with_replacement
from typing import Optional

import numpy as np
from optimal_control.basis.memory_efficient_product import memory_efficient_product
from sklearn.preprocessing import PolynomialFeatures

from optimal_control.basis import Basis


class PolynomialBasis(Basis):
    def __init__(self, degrees):
        def pol(deg_):
            def fun(X):
                return X ** deg_

            return pol

        super().__init__(functions=[pol(deg) for deg in degrees], dimension=len(degrees))
        if np.ndim(degrees) == 1:
            self.degrees = np.reshape(degrees, (-1, 1))
        else:
            self.degrees = np.array(degrees)

    def transform(self, X, out=None):
        out = self.__check_out__(X.shape[0], out)
        m, d = X.shape

        if d != self.degrees.shape[1]:
            raise ValueError("Number of degrees must equal number of dimensions: {0} != {1}".format(
                self.degrees.shape[1], d))

        out[...] = (X.reshape(m, d, 1) ** self.degrees.T).prod(axis=1)
        return out


class SciKitPolynomial(Basis):
    def __init__(self, input_dimension, degree: int, pre_basis: Basis = None):
        """
        Specify if a pre_basis is specified, the output of this pre-basis is efficiently stored in the out memory
        reservoir.
        :param input_dimension:
        :param pre_basis:
        """

        self.sk_pol = PolynomialFeatures(degree=degree)
        self.sk_pol.fit(np.full((1, input_dimension), 1))

        def pol(*w):
            if len(w) == 0:
                def fun(X_):
                    return np.ones((X_.shape[0], 1))
            else:
                if pre_basis:
                    def fun_(X):
                        out = pre_basis.functions[w[0]](X)
                        for i in w[1:]: out *= pre_basis.functions[w[i]](X)
                        return out
                else:
                    def fun_(X):
                        out = X[:, w[0]]
                        for i in w[1:]:
                            out *= X[:, i]
                        return out
                return fun_

        functions = [pol(*w) for deg in range(degree + 1) for w in
                     combinations_with_replacement(range(input_dimension), deg)]
        dimension = len(functions)
        super().__init__(functions=functions, dimension=dimension)
        self.pre_basis: Basis = pre_basis
        self.input_dimension = input_dimension

    def transform(self, X, out=None):
        if (self.pre_basis is None) and (X.shape[1] != self.input_dimension):
            raise ValueError("X must have the number of dimensions "
                             "as specified by the initialization {}!={}".format(X.shape[1], self.input_dimension))

        out = self.__check_out__(X.shape[0], out)
        if self.pre_basis is None:
            out[...] = self.sk_pol.transform(X)
        else:
            self.pre_basis.transform(X, out[:, 1:self.input_dimension + 1])
            out[...] = self.sk_pol.transform(out[:, 1:self.input_dimension + 1])
        return out


class SecondDegreePolynomialBasis(Basis):
    def __init__(self, input_dimension, pre_basis: Basis = None):
        """
        Second order polynomials using a more memory efficient product.
        :param input_dimension:
        :param pre_basis:
        """
        if not (pre_basis is None) and input_dimension != pre_basis.dimension:
            raise AttributeError("Pre-basis must have the number of expect input dimensions.")

        def zero():
            def fun(X_): return np.ones((X_.shape[0], 1))

        def one(i):
            def fun_(X): return pre_basis.functions[i](X) if pre_basis else X[:, i]

        def two(i, j):
            def fun_(X): return pre_basis.functions[i](X) * pre_basis.functions[j](X) if pre_basis else X[:, i]

        first_degree = [one(i) for i in range(input_dimension)]
        second_degree = [two(i, j) for i in range(input_dimension) for j in range(i, input_dimension)]
        functions = [zero()] + first_degree + second_degree
        dimension = 1 + input_dimension + input_dimension * (input_dimension + 1) // 2
        super().__init__(functions=functions, dimension=dimension)
        self.pre_basis: Optional[Basis] = pre_basis
        self.input_dimension = input_dimension

    def transform(self, X, out=None):
        if (self.pre_basis is None) and (X.shape[1] != self.input_dimension):
            raise ValueError("X must have the number of dimensions "
                             "as specified by the initialization {}!={}".format(X.shape[1], self.input_dimension))

        out = self.__check_out__(X.shape[0], out)
        out[:, 0] = 1
        if self.pre_basis is None:
            out[:, 1:self.input_dimension + 1] = X
            memory_efficient_product(X, out[:, self.input_dimension + 1:])
        else:
            self.pre_basis.transform(X, out[:, 1:self.input_dimension + 1])
            memory_efficient_product(out[:, 1:self.input_dimension + 1], out[:, self.input_dimension + 1:])


class ThirdDegreePolynomialBasis(SciKitPolynomial):
    def __init__(self, input_dimension, pre_basis: Basis = None):
        super().__init__(input_dimension, degree=3, pre_basis=pre_basis)
