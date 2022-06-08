import numpy as np

from optimal_control.examples.continuous.stopping_example import StoppingExample


class MaxCallOption(StoppingExample):
    def __init__(self, T: float, n_steps: int, strike: float, interest_rate: float):
        super().__init__(T, n_steps)
        self.strike = strike
        self.time_delta = T / (n_steps - 1)
        self.interest_rate = interest_rate

    def payoff(self, t: int, H, Y, X):
        if t == 0 or t == self.n_time_steps -1:
            return np.zeros(X.shape[0])
        else:
            h = H[:, 0]
            return h * np.maximum(np.max(X, axis=1) - self.strike, 0) * np.exp(
                -self.interest_rate * t * self.time_delta)
