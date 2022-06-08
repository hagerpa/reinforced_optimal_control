import numpy as np

from optimal_control.examples.continuous.swing import SwingOption


class OffPeakSwingOption(SwingOption):
    def __init__(self, T, n_time_steps, total_maximum, strike):
        super().__init__(T, n_time_steps, daily_maximum=2, total_maximum=total_maximum, strike=strike)

    def __h_max__(self, t):
        if (t % 5 == 0) or (t % 6 == 0):
            return 2
        else:
            return 1

    def daily_maximum(self, t, y, X):
        return np.minimum(self.__h_max__(t), self.y_max - y)

    def daily_minimum(self, t: int, y, X):
        return np.zeros_like(y)

    def optimization_grid(self, t, lower_bounds, upper_bounds):
        n = lower_bounds.size
        h_space = np.arange(self.__h_max__(t) + 1).reshape((-1, 1)).repeat(n, axis=1)
        h_space = [ np.minimum(np.maximum(h, lower_bounds), upper_bounds) for h in h_space]
        return h_space