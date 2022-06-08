import numpy as np
from scipy.linalg import lstsq

from optimal_control.basis import Basis
from optimal_control.examples.discrete import StoppingExample
from optimal_control.solvers.discrete import DiscreteValueFunction


class ModifiedForStopping(DiscreteValueFunction):

    def __init__(self, example: StoppingExample, x_basis: Basis, I: int = 0, positive_continuation=True):
        super().__init__(example)
        self.positive_continuation = positive_continuation
        J = self.n_time_steps - 2
        self.x_basis = x_basis
        self.y_max = 1
        self.regression_coefficients = np.zeros((J + 1, x_basis.dimension + 1, I + 1))
        self.I = I if (I <= J) else J

        self.basis_normalization = np.ones((J + 1, x_basis.dimension))
        self.reinforced_basis_normalization = np.ones((J + 1, I + 1))

    def value_and_policy(self, j, Y_j, X_j, depth=0, **kwargs):
        m, _ = X_j.shape
        J = self.n_time_steps - 2
        VH = np.zeros((m, 2))
        mask = (Y_j[:, 0] == 0)

        FX = self.x_basis.transform(X_j[mask])
        m_, _ = FX.shape
        I_ = min(self.I, J - j)
        H = np.zeros((m_, I_ + 1))
        for i in range(I_ + 1):
            H[:, i] = self.example.g(j + i, X_j[mask])
        VH[mask] = self.__vh__(j, FX, I_, H)
        return VH

    def fit(self, X):
        if np.ndim(X) == 2:
            m, n = X.shape
            X = X.reshape(m, n, 1)

        m, n, d = X.shape
        J = self.n_time_steps - 2
        I = self.I

        x_basis_dimension = self.x_basis.dimension

        H = np.zeros((m, 2, I + 1))
        H[:, 0, 0] = self.example.g(J + 1, X[:, J + 1])

        FX = np.zeros((m, 2, self.regression_coefficients.shape[1]))
        FX[:, 0, :x_basis_dimension] = self.x_basis.transform(X[:, J + 1, :])

        for j in range(J, -1, -1):
            ModifiedForStopping.__print_progression__(j, J)
            FX[:, 1, :x_basis_dimension] = FX[:, 0, :x_basis_dimension]
            FX[:, 0, :x_basis_dimension] = self.x_basis.transform(X[:, j, :])

            H[:, 1] = H[:, 0]
            for i in range(min(I, J - j) + 1):
                H[:, 0, i] = self.example.g(j + i, X[:, j])

            z = self.__vh__(j + 1, FX[:, 1, :x_basis_dimension], min(I, J - (j + 1)), H[:, 1])[:, 0]
            if (j == 0) and (FX[:, 0, 1].var() == 0):  # Only if index 0 basis function is the constant function!
                z_mean = z.mean()
                self.regression_coefficients[0, 0, I] = z_mean
            else:
                for i in range(min(I, J - j) + 1):
                    if i < I - j:
                        continue
                    if i == 0:
                        res = lstsq(FX[:, 0, :x_basis_dimension], z)[0]
                        self.regression_coefficients[j, :x_basis_dimension, 0] = res
                    else:
                        f = self.__vh__(j + 1, FX[:, 0, :x_basis_dimension], i - 1, H[:, 0, 1:])[:, 0]
                        FX[:, 0, -1] = f
                        res = lstsq(FX[:, 0, :], z)[0]
                        self.regression_coefficients[j, :, i] = res

    def __vh__(self, j: int, FX, i: int, H):
        m, basis_dimension = FX.shape
        J = self.n_time_steps - 2
        VH = np.zeros((m, 2))
        VI = np.zeros((m, 2))
        V = np.zeros((m, i + 1))
        C = np.zeros((m, i + 1))
        if j == J + 1:
            VH[:, 1] = 0
            VH[:, 0] = 0
        else:
            assert J - j >= i, "Only {}-steps to go backwards, but depth is {}.".format(J - j, i)
            for u in range(0, i + 1):
                s = j + i - u
                C[:, s - j] = np.dot(FX, self.regression_coefficients[s, :basis_dimension, u])
                if u > 0:
                    C[:, s - j] += V[:, s - j + 1] * self.regression_coefficients[s, -1, u]
                if self.positive_continuation:
                    C[:, s - j] = np.maximum(C[:, s - j], 0)

                VI[:, 0] = C[:, s - j]
                VI[:, 1] = H[:, s - j]

                if s > j:
                    V[:, s - j] = np.max(VI, axis=1)
                if s == j:
                    arg_max = np.expand_dims(np.argmax(VI, axis=1), axis=1)
                    VH[:, 0] = np.take_along_axis(VI, arg_max, axis=1)[:, 0]
                    VH[:, 1] = arg_max[:, 0]
        return VH

    def value_all_y(self, j, X_j):
        m = X_j.shape[0]
        V = np.zeros((m, 2))
        V[:, 0] = self.evaluate(j, np.zeros((m, 1)), X_j)
        return V

    @staticmethod
    def __print_progression__(i, n):
        print("{}/{} <-".format(i, n), flush=True, end="")
        print(end="\r", flush=True)
