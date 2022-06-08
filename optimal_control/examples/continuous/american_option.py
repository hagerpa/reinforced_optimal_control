import numpy as np

from optimal_control.examples.continuous.stopping_example import StoppingExample


class AmericanOption(StoppingExample):
    def __init__(self, T: float, n_steps: int, strike: float):
        super().__init__(T, n_steps)
        self.strike = strike

    def payoff(self, t: int, H, Y, X) -> np.ndarray:
        if t == self.n_time_steps-1:
            return np.zeros(X.shape[0])
        else:
            return H[:,0] * np.maximum(X[:,0] - self.strike, 0)
