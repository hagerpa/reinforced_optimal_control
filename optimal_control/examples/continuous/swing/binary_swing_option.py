from optimal_control.examples.continuous.multiple_stopping import MultipleStoppingExample

import numpy as np

class BinarySwingOption(MultipleStoppingExample):
    def __init__(self, T: float, n_steps: int, y_max: int, strike: float):
        super().__init__(T=T, n_steps=n_steps, y_max=y_max)
        self.strike = strike

    def payoff(self, t: int, H, Y, X):
        if t == self.n_time_steps - 1:
            return np.zeros(X.shape[0])
        else:
            h, y, x = H[:, 0], Y[:, 0], X[:, 0]
            if np.logical_and(h == 1, y >= self.y_max).any():
                raise RuntimeError("Constrained not respected.")
            pay_off = h*np.maximum((x - self.strike), 0)
            return pay_off
