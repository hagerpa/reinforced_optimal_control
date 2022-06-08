from optimal_control.examples.continuous.stopping_example import StoppingExample

import numpy as np
from numpy import logical_not as npnot
from numpy import logical_or as npor


class BermudaCancelableSwap(StoppingExample):
    def __init__(self, T: float, n_steps: int, alpha, n1, n2, s1, s2, s3, interest_rate):
        super().__init__(T, n_steps)
        self.alpha = alpha
        self.n1 = n1
        self.n2 = n2
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.interest_rate = interest_rate
        self.time_delta = T / (n_steps - 1)

    def payoff(self, t: int, H, Y, X):
        # if t == 0: #TODO: The if is only outcommented for comparison with the paper.
        #    return np.zeros((Y.shape[0]))
        if t == self.n_time_steps - 1:
            return np.zeros((Y.shape[0]))

        def Ct(X_):
            Nt = np.sum(X_ <= (1 - self.alpha), axis=1)
            mask1 = Nt <= self.n1
            mask3 = Nt > self.n2
            mask2 = npnot(npor(mask1, mask3))

            assert (np.sum(mask1 + mask2 + mask3) == np.size(mask1))
            assert (self.time_delta == 0.5)

            at = self.s1 * mask1 + self.s2 * mask2 + self.s3 * mask3
            res_ = (np.exp(self.interest_rate * self.time_delta) - 1) - at * self.time_delta

            return res_ * np.exp(-self.interest_rate * t * self.time_delta)

        y = Y.flatten()
        res = np.zeros_like(y)
        mask = (y == 0)
        if mask.any():
            res[mask] = Ct(X[mask, :])
        return res * 10000  # Times 1000 for comparison with the paper
