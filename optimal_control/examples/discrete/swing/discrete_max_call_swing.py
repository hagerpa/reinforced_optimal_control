import numpy as np

from optimal_control.examples.discrete import SwingOption


class DiscreteMaxCallSwing(SwingOption):
    def __init__(self, T: float, n_time_steps: int, y_max: int, strike: float, interest_rate: float, h_max: int):
        super().__init__(T, n_time_steps, y_max)
        self.h_max = h_max
        self.strike = strike
        self.interest_rate = interest_rate
        self.time_delta = self.T / (self.n_time_steps - 1)

    def payoff(self, t: int, H, Y, X):
        if t == 0 or t == self.n_time_steps - 1:
            return np.zeros(X.shape[0])
        else:
            h = H[:, 0]
            return h * np.maximum(np.max(X, axis=1) - self.strike, 0) * np.exp(
                -self.interest_rate * t * self.time_delta)

    def h_space(self, t, y):
        return np.arange(0, min(self.h_max, self.y_max - y) + 1)
