import numpy as np

from optimal_control.examples.discrete.swing.swing_option import SwingOption


class DiscreteGasStraddle(SwingOption):
    def __init__(self, T: float, n_time_steps: int, interest_rate: float, y_max: int, h_max: int, h_min: int,
                 call_strike: float, put_strike: float):
        super().__init__(T, n_time_steps, y_max)
        self.put_strike = put_strike
        self.call_strike = call_strike
        self.h_min = h_min
        self.h_max = h_max
        self.y_max = y_max
        self.interest_rate = interest_rate
        self.time_delta = self.T / (self.n_time_steps - 1)

    def phi(self, t: int, H, Y):
        return Y - H

    def payoff(self, t: int, H, Y, X):
        if t == 0 or t == self.n_time_steps - 1:
            return np.zeros(X.shape[0])
        else:
            h = H[:, 0]

            return np.abs(h) * np.where(h <= 0,
                                        np.maximum(X[:, 0] - self.call_strike, 0),
                                        np.maximum(self.put_strike - X[:, 0], 0)
                                        ) * np.exp(-self.interest_rate * t * self.time_delta)

    def h_space(self, t, y):
        return np.arange(max(self.h_min, y - self.y_max), min(self.h_max, y) + 1)
