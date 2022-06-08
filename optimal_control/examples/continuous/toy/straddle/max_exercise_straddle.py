import numpy as np

from optimal_control.examples import MinMaxStraddle


class MaxExerciseStraddle(MinMaxStraddle):
    def __init__(self, T, n_time_steps, strike_low, strike_high, interest_rate, max_exercises=5):
        super().__init__(T, n_time_steps, strike_low, strike_high, interest_rate)
        self.max_exercises = max_exercises

    def y_states(self):
        return np.arange(0, self.max_exercises)

    def optimization_grid(self, t, lower_bounds, upper_bounds):
        return [lower_bounds, np.zeros_like(lower_bounds), upper_bounds]

    def daily_minimum(self, t, y, X):
        res = -np.ones_like(y)
        res[y == self.max_exercises] = 0
        return res

    def daily_maximum(self, t: int, y, X):
        res = np.ones_like(y)
        res[y == self.max_exercises] = 0
        return res

    def phi(self, t: int, H, Y):
        return Y + np.abs(H)
