from abc import ABC, abstractmethod

import numpy as np

from optimal_control.examples.continuous.swing_type_example import SwingTypeExample


class MinMaxStraddle(SwingTypeExample, ABC):
    def __init__(self, T, n_time_steps, strike_low, strike_high, interest_rate):
        super().__init__(T, n_time_steps)
        self.interest_rate = interest_rate
        self.dt = T / (n_time_steps - 1)
        self.strike_low = strike_low
        self.strike_high = strike_high

    @abstractmethod
    def y_states(self):
        pass

    def payoff(self, t: int, H, Y, X):
        if t == self.n_time_steps - 1:
            return np.zeros_like(Y[:, 0])
        else:
            h = H[:, 0]
            res = np.zeros_like(h)
            res[h == 1] = np.maximum(np.max(X[h == 1], axis=1) - self.strike_high, 0)
            res[h == -1] = np.maximum(self.strike_low - np.min(X[h == -1], axis=1), 0)
            return res * np.exp(- self.interest_rate * self.dt * t)
