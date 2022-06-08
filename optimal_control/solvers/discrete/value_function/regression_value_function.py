import numpy as np
from scipy.linalg import lstsq

from optimal_control.basis import Basis
from optimal_control.examples.discrete import DiscreteExample
from optimal_control.solvers.discrete import DiscreteValueFunction

class RegressionValueFunction(DiscreteValueFunction):

    def __init__(self, example: DiscreteExample, y_max: int, x_basis: Basis, D: int = 0, L_=None):
        """
        The modified reinforced regression algorithm for discrete state control problems.
        :param example: optimal control problem with discrete control state.
        :param x_basis: basis functions in the x-variable for the regression.
        :param D: number of iterations
        :param L_: function that maps from the control state to the subset of controls for reinforced value functions.
        """
        super().__init__(example)

        self.x_basis = x_basis
        self.y_max = y_max

        self.regression_coefficients = np.zeros(
            (self.n_time_steps - 1, x_basis.dimension + self.y_max + 1, self.y_max + 1, D + 1)
        )

        self.x_basis_normalization = np.ones((self.n_time_steps, self.x_basis.dimension))
        self.reinforced_basis_normalization = np.ones((self.n_time_steps, self.y_max + 1, D + 1))

        self.D = D if (D <= self.n_time_steps - 2) else D

        self.L_ = L_

        self.h_space_max_length = max([len(self.example.h_space(j, y))
                                       for y in range(self.y_max + 1)
                                       for j in range(0, self.n_time_steps - 1)])

        self.normalization = False

    def value_and_policy(self, j, Y_j, X_j, depth=0, **kwargs):
        m, d = X_j.shape
        y = np.array(Y_j[:, 0], dtype=int)
        VH = self.value_and_policy_all_y(j, X_j, depth, **kwargs)[np.arange(m), :, y]
        return VH

    def value_and_policy_all_y(self, j, X_j, depth=0, **kwargs):
        J = self.n_time_steps - 2

        if j == J + 1:
            FX = None
        else:
            FX = self.x_basis.transform(X_j)
            if self.normalization: FX *= self.x_basis_normalization[j]

        VH = self.__vh__(j, X_j, FX, k=min(self.D, J - j))
        return VH

    def value_all_y(self, j, X_j, depth=0, **kwargs):
        return self.value_and_policy_all_y(j, X_j, depth=depth, **kwargs)[:, 0]

    def fit(self, X, print_progression=True):
        if np.ndim(X) == 2:
            m, n = X.shape
            X = X.reshape(m, n, 1)

        m, n, d = X.shape
        J = self.n_time_steps - 2

        x_basis_dimension = self.x_basis.dimension

        FX = np.zeros((m, 2, self.regression_coefficients.shape[1]))
        FX[:, 0, :x_basis_dimension] = self.x_basis.transform(X[:, J + 1, :])

        D = self.D
        for j in range(J, -1, -1):  # j = J, J-1, ..., 0
            self.__print_progression__(j, J)

            FX[:, 1, :x_basis_dimension] = FX[:, 0, :x_basis_dimension]
            FX[:, 0, :x_basis_dimension] = self.x_basis.transform(X[:, j, :])

            if self.normalization:
                self.__normalize_x_basis__(j, X[:, j])
                FX[:, 0, :x_basis_dimension] *= self.x_basis_normalization

            z = self.__vh__(j + 1, X[:, j + 1], FX[:, 1, :x_basis_dimension], k=min(D, J - (j + 1)))[:, 0, :]

            for k in range(min(D, J - j) + 1):
                if k < D - j:
                    continue
                if k == 0:
                    res = lstsq(FX[:, 0, :x_basis_dimension], z)
                    self.regression_coefficients[j, :x_basis_dimension, :, 0] = res[0]
                else:
                    f = self.__vh__(j + 1, X[:, j], FX[:, 0, :x_basis_dimension], k - 1)[:, 0, :]
                    if self.normalization:
                        self.__normalize_reinforced_basis__(j, X[:, j], k)
                        f *= self.reinforced_basis_normalization[j, :, k]

                    FX[:, 0, self.x_basis.dimension:] = f
                    for y in range(0, self.y_max + 1):
                        reinforced_selection = x_basis_dimension + self.L_(y)
                        basis_selection = np.concatenate([np.arange(x_basis_dimension), reinforced_selection])
                        res = lstsq(FX[:, 0, basis_selection], z[:, y])
                        self.regression_coefficients[j, basis_selection, y, k] = res[0]

                        # if j>0: assert np.isclose(
                        #    np.linalg.norm(FX[:, j].dot(self.regression_coefficients[j, :, y, k])-z[:, y])**2,
                        #    res[1]
                        # )

    def __vh__(self, j, X_j, FX, k):
        m = X_j.shape[0]
        J = self.n_time_steps - 2

        VH = np.empty((m, 2, self.y_max + 1))

        if j == J + 1:
            for y in range(self.y_max + 1):
                VH[:, 1, y] = 0
                VH[:, 0, y] = self.payoff(J + 1, None, np.full((m, 1), y), X_j)

        else:
            assert J - j >= k, "Only {}-steps to go backwards, but depth is {}.".format(J - j, k)

            VI = np.zeros((m, self.h_space_max_length, self.y_max + 1))

            V = np.zeros((m, k + 1, self.y_max + 1))

            t_range = j + np.arange(k + 1)
            k_range = np.arange(k + 1)[::-1]

            # Evaluating all non-reinforced parts of the continuation functions C_s^{(s - j + k)}(X) for s = j, ..., j+k and
            # y = 0, ..., y_max
            C = FX.dot(self.regression_coefficients[t_range, :self.x_basis.dimension, :, k_range])

            for u in range(0, k + 1):
                s = j + k - u  # s = j + k, j + k - 1, ..., j
                if u > 0:
                    f = V[:, s - j + 1]
                    if self.normalization: f *= self.reinforced_basis_normalization[s, :, u]
                    for y in range(self.y_max + 1):
                        C[:, s - j, y] += f.dot(self.regression_coefficients[s, self.x_basis.dimension:, y, u])

                C[:, s - j] = np.maximum(C[:, s - j], 0)

                for y in range(self.y_max + 1):
                    h_space = self.example.h_space(s, y)
                    for (i, h_i) in enumerate(h_space):
                        H = self.example.payoff(s, H=np.full((m, 1), h_i), Y=np.full((m, 1), y), X=X_j)
                        VI[:, i, y] = C[:, s - j, self.example.phi(s, h_i, y)] + H

                    index_h = np.arange(len(h_space))
                    if s > j:
                        V[:, s - j, y] = np.max(VI[:, index_h, y], axis=1)
                    if s == j:
                        arg_max = np.argmax(VI[:, index_h, y], axis=1)
                        VH[:, 0, y] = VI[np.arange(m), arg_max, y]
                        VH[:, 1, y] = h_space[arg_max]
        return VH

    def __normalize_x_basis__(self, j, X_j):
        if X_j.var() == 0:
            self.x_basis_normalization[j, :] = 0
            self.x_basis_normalization[j, 0] = 1
        else:
            x_mean = np.mean(X_j, axis=0).reshape(1, -1)
            f = self.x_basis.transform(x_mean)
            f[f == 0] = 1
            self.x_basis_normalization[j, :self.x_basis.dimension] = f ** (-1)

    def __normalize_reinforced_basis__(self, j, X_j, k):
        if X_j.var() == 0:
            self.reinforced_basis_normalization[j] = 0
        if X_j.var() != 0:  # otherwise already normalized
            x_mean = np.mean(X_j, axis=0).reshape(1, -1)
            fx_mean = self.x_basis.transform(x_mean)
            f = self.__vh__(j + 1, x_mean, fx_mean, k - 1)[:, 0, :]
            f[f == 0] = 1
            self.reinforced_basis_normalization[j, :, k] = f ** (-1)

    @staticmethod
    def __print_progression__(i, n):
        print("{}/{} <-".format(i, n), flush=True, end="")
        print(end="\r", flush=True)
