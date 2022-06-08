import unittest

import numpy as np

from optimal_control.examples import IntegerToBinary


class TestReinforcedSolver(unittest.TestCase):

    def test_poly_basis(self):
        pass

    def test_piecewise_poly_basis(self):
        from optimal_control.basis import PiecewisePolynomialBasis
        basis = PiecewisePolynomialBasis(grid=[0, 0.5, 1.0], max_degree=2)
        assert basis.dimension == 1 + 2 * 2

        X = np.zeros((5, 1))
        X[1, 0] = 0.2
        X[2, 0] = 0.5
        X[3, 0] = 0.7
        X[4, 0] = 1

        Y = np.array([
            [1, 0, 0, 0.5, 0.5 ** 2],
            [1, 0.2, 0.2 ** 2, 0.5, 0.5 ** 2],
            [1, 0.5, 0.5 ** 2, 0.5, 0.5 ** 2],
            [1, 0.5, 0.5 ** 2, 0.7, 0.7 ** 2],
            [1, 0.5, 0.5 ** 2, 1.0, 1.0 ** 2]
        ])
        assert np.all(Y == basis.transform(X))

        basis = PiecewisePolynomialBasis(grid=[0, 0.5, 1.0], max_degree=2, min_degree=1)
        assert np.all(Y[:, 1:] == basis.transform(X))

    def test_product_basis(self):
        from optimal_control.basis import PolynomialBasis, ProductBasis, SelfProductBasis, SplitProductBasis
        basis1 = PolynomialBasis([[0, 0], [1, 0], [1, 1]])  # 1, x, xy
        basis2 = PolynomialBasis([[1, 1], [0, 1]])  # xy, y
        prod_basis = ProductBasis(basis1, basis2)  # xy, y, x^2y, xy, x^2y^2, xy^2
        assert prod_basis.dimension == basis1.dimension * basis2.dimension
        X = np.array([
            [0, 1],
            [-1, 2]
        ])
        Z = np.array([
            [0, 1, 0, 0, 0, 0],
            [-1 * 2, 2, 2, -2, 1 * 2 ** 2, -1 * 2 ** 2]
        ])
        assert np.all(prod_basis.transform(X) == Z)

        # Testing basis1 == basis2
        basis1 = PolynomialBasis([[1, 0], [0, 1]])  # x, y
        prod_basis = SelfProductBasis(basis1)  # x**2, x*y, y**2
        assert prod_basis.dimension == basis1.dimension * (basis1.dimension + 1) // 2
        X = np.array([
            [0, 1],
            [-1, 2]
        ])
        Z = np.array([
            [0, 0, 1],
            [1, -2, 4]
        ])
        assert np.all(prod_basis.transform(X) == Z)

        # Testing basis split
        basis1 = PolynomialBasis([[0], [1]])  # 1, x
        basis2 = PolynomialBasis([[0], [2]])  # 1, y^2
        prod_basis = SplitProductBasis(basis1, basis2, split_input_at=1)  # 1, y^2, x, xy^2
        assert prod_basis.dimension == basis1.dimension * basis2.dimension
        X = np.array([
            [0, 1],
            [-1, 2]
        ])
        Z = np.array([
            [1, 1, 0, 0],
            [1, 4, -1, -4]
        ])
        assert np.all(prod_basis.transform(X) == Z)

    def test_integerToBinary(self):
        basis = IntegerToBinary(10)
        assert np.all(np.eye(10) == basis.transform(np.arange(10).reshape(-1, 1)))


if __name__ == '__main__':
    unittest.main()
