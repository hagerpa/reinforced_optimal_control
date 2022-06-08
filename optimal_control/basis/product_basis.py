import numpy as np
from optimal_control.basis.memory_efficient_product import memory_efficient_product

from optimal_control.basis import Basis


class ProductBasis(Basis):
    """
    Takes the product of two function bases, i.e. all products of functions in the respective bases.
    :param basis1: First basis.
    :param n_features_1: Input dimension (number of features) of the first basis.
    :param basis2: Second basis.
    :return: A new basis which has |basis1.functions|*|basis2.functions| functions. The input dimension is the sum of
    the input dimensions of basis1 and basis2.
    """

    def __init__(self, basis1: Basis, basis2: Basis):
        """
        Product of all pairs of functions from the given bases.
        :param split_input:
        :param basis1: First basis.
        :param basis2: Second basis.
        fed to the first/second basis respectively.
        """
        self.basis1 = basis1
        self.basis2 = basis2

        def function_product(f, g):
            def f_times_g(X):
                return (f(X) * g(X)).flatten()

            return f_times_g

        dimension = self.basis1.dimension * self.basis2.dimension
        functions = [function_product(f, g) for f in basis1.functions for g in basis2.functions]

        super().__init__(functions=functions, dimension=dimension)

    def transform(self, X, out=None):
        """
        Since evaluating all functions first and then using Einstein sums yields performance improvements we are not
        using the function that are in basis.functions.
        :param X: Original feature matrix (n_samples, n_features).
        :return: Transformed feature matrix (n_samples, |basis1.functions|*|basis2.functions|).
        """
        out = self.__check_out__(X.shape[0], out)
        F1 = self.basis1.transform(X)
        F2 = self.basis2.transform(X)

        ProductBasis.product(F1, F2, out)
        return out

    @staticmethod
    def product(F1, F2, out):  # TODO: sparseness?
        if F2.shape[1] < F1.shape[1]:
            ProductBasis.product(F2, F1, out)
        elif F1.shape[1] == 1:
            np.multiply(F2, F1, out=out)
        else:
            np.einsum('...j,...l -> ...jl', F1, F2,
                      out=out.reshape(F1.shape[0], F1.shape[1], F2.shape[1]))


class SplitProductBasis(Basis):
    def __init__(self, basis1: Basis, basis2: Basis, split_input_at: int):
        def function_product(f, g):
            def f_times_g(X):
                return (f(X[:, :basis1.dimension]) * g(X[:, basis1.dimension:])).flatten()

        dimension = basis1.dimension * basis2.dimension
        functions = [function_product(f, g) for f in basis1.functions for g in basis2.functions]
        super().__init__(functions=functions, dimension=dimension)
        self.basis1 = basis1
        self.basis2 = basis2
        self.split_input_at = split_input_at

    def transform(self, X, out=None):
        """
        Since evaluating all functions first and then using Einstein sums yields performance improvements we are not
        using the function that are in basis.functions.
        :param X: Original feature matrix (n_samples, n_features).
        :return: Transformed feature matrix (n_samples, |basis1.functions|*|basis2.functions|).
        """
        out = self.__check_out__(X.shape[0], out)

        F1 = self.basis1.transform(X[:, :self.split_input_at])
        F2 = self.basis2.transform(X[:, self.split_input_at:])

        ProductBasis.product(F1, F2, out)
        return out


class SelfProductBasis(Basis):
    def __init__(self, basis1: Basis):
        """
        Product of all pairs of functions from the given bases.
        :param split_input:
        :param basis1: First basis.
        :param basis2: Second basis.
        fed to the first/second basis respectively.
        """
        self.basis1 = basis1

        def function_product(f, g):
            def f_times_g(X):
                return (f(X) * g(X)).flatten()

            return f_times_g

        dimension = (basis1.dimension * (basis1.dimension + 1)) // 2
        functions = [function_product(f, g)
                     for (j, f) in enumerate(basis1.functions)
                     for (l, g) in enumerate(basis1.functions)
                     if j >= l]
        super().__init__(functions=functions, dimension=dimension)

    def transform(self, X, out=None):
        """
        Since evaluating all functions first and then using Einstein sums yields performance improvements we are not
        using the function that are in basis.functions.
        :param X: Original feature matrix (n_samples, n_features).
        :return: Transformed feature matrix (n_samples, |basis1.functions|*|basis2.functions|).
        """
        out = self.__check_out__(X.shape[0], out)
        F1 = self.basis1.transform(X)
        memory_efficient_product(F1, out=out)
        return out
