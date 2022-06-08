import numpy as np

from optimal_control.examples import MinMaxStraddle


class MaxCall(MinMaxStraddle):
    def __init__(self, T, n_time_steps, strike_low, strike_high, interest_rate):
        super().__init__(T, n_time_steps, strike_low, strike_high, interest_rate)

    def y_states(self):
        return [0]

    def optimization_grid(self, t, lower_bounds, upper_bounds):
        return [np.zeros_like(lower_bounds), upper_bounds]

    def daily_minimum(self, t, y, X):
        return np.zeros_like(y)

    def daily_maximum(self, t: int, y, X):
        res = np.ones_like(y)
        res[y == 1] = 0
        return res

    def phi(self, t: int, H, Y):
        return Y + np.abs(H)
