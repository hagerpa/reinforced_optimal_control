from optimal_control.examples.continuous.gas import GasStorage
from optimal_control.some_types import *


class GasStorageStraddle(GasStorage):
    def __init__(self, T, n_time_steps, y_min, y_max, c_min_0,
                 c_max_y_max, gas_loss, y_prior_range, strike_low, strike_up):
        """
        :param y_max: Maximum gas storage capacity.
        :param y_min: Base gas storage requirement.
        :param c_max_y_max: Maximum injection rate at storage level y_max.
        :param c_min_0: Maximum injection rate at storage level 0.
        :param gas_loss: Gas loss rate as a function of the injection rate c and the stored gas y.
        """
        super().__init__(T, n_time_steps, y_min, y_max, c_min_0,
                         c_max_y_max, gas_loss, y_prior_range)
        self.strike_up = strike_up
        self.strike_low = strike_low

    def payoff(self, t: int, H, Y: np.ndarray, X):
        if (Y > self.y_max).any():
            raise RuntimeError("Y exceeded constrained.")
        if t == self.n_time_steps - 1:
            return np.zeros(X.shape[0])
        else:
            h = H[:, 0]
            y = Y[:, 0]

            payoff = np.zeros_like(h)
            pos = h > 0
            neg = h < 0
            payoff[pos] = (h[pos] - self.time_delta * self.gas_loss(y[pos], h[pos])) * 1000 \
                          * np.minimum(X[pos, 0], self.strike_low)
            payoff[neg] = (h[neg] - self.time_delta * self.gas_loss(y[neg], h[neg])) * 1000 \
                          * np.maximum(X[neg, 0], self.strike_up)
            return payoff
