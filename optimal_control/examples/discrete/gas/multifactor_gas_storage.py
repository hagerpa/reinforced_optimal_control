import numpy as np

from optimal_control.examples.discrete.swing.swing_option import SwingOption


class MultifactorGasStorage(SwingOption):
    def __init__(self, T: float, n_time_steps: int, interest_rate: float, y_max: int, h_max: int, h_min: int,
                 log_utility: bool = True):
        super().__init__(T, n_time_steps, y_max)
        self.log_utility = log_utility
        self.h_min = h_min
        self.h_max = h_max
        self.interest_rate = interest_rate
        self.time_delta = self.T / (self.n_time_steps - 1)

    def phi(self, t: int, H, Y):
        return Y - H

    def payoff(self, t: int, H, Y, X):
        if t == 0 or t == self.n_time_steps - 1:
            return np.zeros(X.shape[0])
        else:
            h = H[:, 0]
            p = h * X[:, 1] * np.exp(-self.interest_rate * t * self.time_delta)
            if self.log_utility:
                def log_utility_f(p_):
                    np.sign(p_) * np.log(1 + np.abs(p_))  # sgn(x)log(1 +|x|)

                return log_utility_f(p)
            else:
                return p

    def h_space(self, t, y):
        return np.arange(max(self.h_min, y - self.y_max), min(self.h_max, y) + 1)
