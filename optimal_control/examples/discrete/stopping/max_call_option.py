import numpy as np

from optimal_control.examples.discrete.stopping.stopping_example import StoppingExample


class MaxCallOption(StoppingExample):
    def __init__(self, T: float, n_time_steps: int, strike: float, interest_rate: float):
        super().__init__(T, n_time_steps)
        self.strike = strike
        self.interest_rate = interest_rate
        self.time_delta = self.T / (self.n_time_steps - 1)

    def g(self, t: int, X):
        return np.maximum(np.max(X, axis=1) - self.strike, 0) * np.exp(
            -self.interest_rate * t * self.time_delta)