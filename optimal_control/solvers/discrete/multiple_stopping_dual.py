import numpy as np

from optimal_control.solvers.discrete import RegressionValueFunction
from optimal_control.evaluation.test_environment import TestEnvironment


def multiple_stopping_dual(X,
                           value_function: RegressionValueFunction,
                           test_environment: TestEnvironment,
                           n_sub_samples):
    m, n_, d = X.shape

    if m * n_sub_samples > 10 ** 6:
        means = []
        stds = []
        batches = (m * n_sub_samples - 1) // 10 ** 6 + 1
        batch_size = (m - 1) // batches + 1
        for i in range(batches):
            print("batch {}/{}".format(i + 1, batches))
            a = i * batch_size
            b = (i + 1) * batch_size if (i < batches - 1) else m
            mean, std = __multiple_stopping_dual__(X[a:b], value_function, test_environment, n_sub_samples)
            means += [mean]
            stds += [std]
        return np.mean(means), np.mean(stds) / np.sqrt(batches)
    else:
        return __multiple_stopping_dual__(X, value_function, test_environment, n_sub_samples)


def __multiple_stopping_dual__(X,
                               value_function: RegressionValueFunction,
                               test_environment: TestEnvironment,
                               n_sub_samples):
    m, n_, d = X.shape
    n = n_ - 1
    dt = test_environment.example.T / (test_environment.example.n_time_steps - 1)

    L = value_function.y_max

    V = np.zeros((m, n, L + 1))
    Z = np.zeros((m, n))

    dM = np.zeros((m, n, L + 1))

    for t in range(n):
        V[:, t] = value_function.value_all_y(t, X[:, t])
        Z[:, t] = test_environment.example.payoff(t, np.ones((m, 1)), np.zeros((m, 1)), X[:, t])

    for t in range(n - 1):
        __print_progression__(t, n - 2)
        X_sub = test_environment.price_model.sub_sample_paths(X[:, t], dt, n_sub_samples)
        V_ = value_function.value_all_y(t + 1, X_sub.reshape(n_sub_samples * m, d))
        E_ = V_.reshape(n_sub_samples, m, -1).mean(axis=0)
        dM[:, t] = V[:, t + 1] - E_

    if L == 1:
        M = np.insert(np.cumsum(dM[:, :-1, 0], axis=1), 0, 0, axis=1)
        eta0 = np.max(Z - M, axis=1)
        return eta0.mean(), eta0.std() / np.sqrt(m)
    else:
        eta = np.zeros((m, n + 1, L + 1))
        eta[:, n - 1, :L] = Z[:, n - 1].reshape((m, 1))
        for p in range(L - 1, -1, -1):
            for i in range(n - 2, -1, -1):
                eta[:, i, p] = np.maximum(
                    Z[:, i] - dM[:, i, p + 1] + eta[:, i + 1, p + 1],
                    - dM[:, i, p] + eta[:, i + 1, p])

        return eta[:, 0, 0].mean(), eta[:, 0, 0].std() / np.sqrt(m)

    # Naive implementation
    #
    # def objective(J):  # J = (0, j_1, j_2, ..., j_L)
    #     return sum([
    #         Z[:, J[k]] + V[:, J[k], k] - V[:, J[k], k - 1] \
    #         + (E[:, J[k - 1]:J[k], k - 1] - V[:, J[k - 1]:J[k], k - 1]).sum(axis=1)
    #         for k in range(1, L + 1)
    #     ])
    #
    #
    # xi = None
    # ch = np.zeros(L + 1, dtype=int)
    #
    # for c in combinations(range(1, n), L):  # Stopping begins from 1
    #     ch[1:] = c
    #     xi = objective(ch) if xi is None else np.maximum(xi, objective(ch))
    #
    # V0_dual = V[:, 0, 0].mean() + xi.mean()
    # V0_dual_std = (V[:, 0, 0].std() + xi.std())/(np.sqrt(2*m))
    # return V0_dual, V0_dual_std


def __print_progression__(i, n):
    print("martingale increments {}/{} ->".format(i, n), flush=True, end="")
    print(end="\r", flush=True)
