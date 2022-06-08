import numpy as np

from optimal_control.examples.continuous.swing_type_example import SwingTypeExample


class DiscreteStraddle(SwingTypeExample):
    def __init__(self, T, n_time_steps, strike_low, strike_high):
        super().__init__(T, n_time_steps)
        self.strike_low = strike_low
        self.strike_high = strike_high

    def daily_minimum(self, t, y, X):
        res = -np.ones_like(y)
        res[y == -1] = 0
        return res

    def daily_maximum(self, t: int, y, X):
        res = np.ones_like(y)
        res[y == 1] = 0
        return res

    def payoff(self, t: int, H, Y, X):
        if t == self.n_time_steps - 1:
            return np.zeros_like(Y[:, 0])
        else:
            h = H[:, 0]
            return np.where(h > 0,
                            h * np.maximum(X[:, 0] - self.strike_high, 0),
                            h * np.minimum(X[:, 0] - self.strike_low, 0))

    def phi(self, t: int, H, Y):
        return Y + H
