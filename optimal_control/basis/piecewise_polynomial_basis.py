import numpy as np

from optimal_control.basis import Basis


class PiecewisePolynomialBasis(Basis):
    def __init__(self, grid, max_degree=0, min_degree=0):
        """
        Piecewise polynomial basis for a one-dimensional input. One polynomial with the given degree for each interval of
        the grid. The polynomials will be extrapolated flat outside their corresponding interval.
        :param grid: defining the intervals.
        :param max_degree: maximal degree of the polynomials.
        :return: Function basis with indicator functions corresponding to each interval, i.e. the output dimension is the
        number of intervals (#grid_points - 1).
        """

        def poly_piece(a, b, deg):
            def poly_piece_(X):
                X_ = np.clip(X, a, b)
                return X_ ** deg

            return poly_piece_

        self.bias = True if min_degree == 0 else False
        self.degrees = np.arange(max(min_degree, 1), max_degree + 1)

        functions = [
            poly_piece(a, b, deg) for b, a in zip(grid[1:], grid) for deg in range(max(min_degree, 1), max_degree + 1)
        ]
        dimension = (len(grid) - 1) * self.degrees.size

        if self.bias:
            def bias(X):
                m, _ = X.shape
                return np.ones(m)

            functions = [bias] + functions
            dimension += 1

        super().__init__(functions=functions, dimension=dimension)

        self.grid = grid

    def transform(self, X, out=None):
        """
        Since evaluating all functions first and then using Einstein sums yields performance improvements we are not
        using the function that are in basis.functions.
        :param X: Original feature matrix (n_samples, n_features).
        :param out: If provided, the result is saved in this memory space.
        :return: Transformed feature matrix (n_samples, |basis1.functions|*|basis2.functions|).
        """
        if X.ndim != 1:
            if X.shape[1] == 1:
                X = X.flatten()
            else:
                raise AttributeError("Only one-dimensional array in piecewise defined functions.")

        out = self.__check_out__(X.shape[0], out)

        if self.bias:
            out[:, 0] = 1.0

        # X_ = np.empty_like(X)
        masks = [X < x for x in self.grid]
        n_degrees = np.size(self.degrees)
        for (i, (b, a)) in enumerate(zip(self.grid[1:], self.grid)):
            slice_ = slice(self.bias + i * n_degrees, self.bias + (i + 1) * n_degrees)

            if np.any(masks[i]):
                out[masks[i], slice_] = np.power(a, self.degrees)

            r_out = np.logical_not(masks[i + 1])
            if np.any(r_out):
                out[r_out, slice_] = np.power(b, self.degrees)  # TODO: this could be pre-evaluated

            interval = np.logical_and(np.logical_not(masks[i]), masks[i + 1])
            if np.any(interval):
                out[interval, slice_] = np.power(X[interval].reshape(-1, 1), self.degrees)

        return out
