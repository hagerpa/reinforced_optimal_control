import numpy as np

from optimal_control.examples.continuous.swing_type_example import SwingTypeExample


class SwingOption(SwingTypeExample):
    def __init__(self, T, n_time_steps, total_maximum, daily_maximum, strike, interest_rate=0.0):
        super().__init__(T, n_time_steps)
        self.interest_rate = interest_rate
        self.h_max = daily_maximum
        self.y_max = total_maximum
        self.strike = strike

        self.time_delta = T / (n_time_steps - 1)

    def daily_maximum(self, t, y, X):
        return np.minimum(self.h_max, self.y_max - y)

    def daily_minimum(self, t: int, y, X):
        return np.zeros_like(y)

    def payoff(self, t: int, H, Y, X):
        if t == self.n_time_steps - 1:
            return np.zeros((X.shape[0], 1))
        else:
            h = H[:, 0]
            x = X[:, 0]
            return h * np.maximum(x - self.strike, 0) * np.exp(
                -self.interest_rate * t * self.time_delta)

    def phi(self, t: int, H, Y):
        return H + Y

    def optimization_grid(self, t, lower_bounds, upper_bounds):
        n = lower_bounds.size
        h_space = np.arange(self.h_max + 1).reshape((-1, 1)).repeat(n, axis=1)
        h_space = [np.minimum(np.maximum(h, lower_bounds), upper_bounds) for h in h_space]
        return h_space
