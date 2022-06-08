import numpy as np

from optimal_control.examples.continuous import SwingOption

class MaxCallSwing(SwingOption):

    def payoff(self, t: int, H, Y, X):
        if t == 0 or t == self.n_time_steps - 1:
            return np.zeros(X.shape[0])
        else:
            h = H[:, 0]
            return h * np.maximum(np.max(X, axis=1) - self.strike, 0) * np.exp(
                -self.interest_rate * t * self.time_delta)
