import numpy as np

from optimal_control.examples.discrete.swing.swing_option import SwingOption


class DiscretizedGasStorage(SwingOption):
    def __init__(self, T: float, n_time_steps: int, interest_rate: float, n_y_states: int,
                 y_max: float, y_min: float, c_min_0: float, c_max_y_max: float):
        super().__init__(T, n_time_steps)

        self.interest_rate = interest_rate
        self.time_delta = self.T / (self.n_time_steps - 1)

        self.n_y_states = n_y_states

        self.y_delta = y_max / n_y_states

        self.y_min = y_min
        self.y_max = y_max

        self.c_min_0 = c_min_0
        self.c_max_y_max = c_max_y_max

        self.C0 = c_max_y_max / np.sqrt(y_max)
        self.C2 = - 1 / (y_min + y_max)
        self.C1 = c_min_0 / np.sqrt(1 / y_min + self.C2)

    def phi(self, t: int, H, Y):
        return ((Y * self.y_delta - H) / self.y_delta).astype(int)

    def payoff(self, t: int, H, Y, X):
        if t == 0 or t == self.n_time_steps - 1:
            return np.zeros(X.shape[0])
        else:
            h = H[:, 0]
            return h * X[:, 0] * np.exp(-self.interest_rate * t * self.time_delta)

    def h_space(self, t, y):
        h_min = np.ceil(self.daily_minimum(y) / self.y_delta) * self.y_delta
        h_max = np.floor(self.daily_maximum(y) / self.y_delta) * self.y_delta

        if (h_min >= 0) or (h_max <= 0):
            return np.array([h_min, h_max])
        else:
            return np.array([h_min, 0, h_max])

    def daily_minimum(self, y):
        real_y = y * self.y_delta
        d_min = - self.C1 * np.sqrt(1 / (real_y + self.y_min) + self.C2) * self.time_delta * 365
        return - np.min([d_min, self.y_max - real_y], axis=0)

    def daily_maximum(self, y):
        real_y = y * self.y_delta
        d_max = self.C0 * np.sqrt(real_y) * self.time_delta * 365
        return np.min([d_max, real_y], axis=0)
