from optimal_control.examples.continuous.swing_type_example import SwingTypeExample
from optimal_control.some_types import *


class GasStorage(SwingTypeExample):
    def __init__(self, T, n_time_steps, y_min, y_max, c_min_0,
                 c_max_y_max, gas_loss, interest_rate):
        """
        :param y_max: Maximum gas storage capacity.
        :param y_min: Base gas storage requirement.
        :param c_max_y_max: Maximum injection rate at storage level y_max.
        :param c_min_0: Maximum injection rate at storage level 0.
        :param gas_loss: Gas loss rate as a function of the injection rate c and the stored gas y.
        """
        super().__init__(T, n_time_steps)
        self.y_min = y_min
        self.y_max = y_max

        self.c_min_0 = c_min_0
        self.c_max_y_max = c_max_y_max

        self.time_delta_year = self.T / (self.n_time_steps - 1)
        self.time_delta_day = 365 * self.time_delta_year
        self.gas_loss = gas_loss

        self.C0 = c_max_y_max / np.sqrt(y_max)
        self.C2 = - 1 / (y_min + y_max)
        self.C1 = c_min_0 / np.sqrt(1 / y_min + self.C2)

        self.interest_rate = interest_rate

    def daily_minimum(self, t: int, y, X):
        d_min = - self.C1 * np.sqrt(1 / (y + self.y_min) + self.C2) * self.time_delta_day
        return - np.min([d_min, self.y_max - y], axis=0)

    def daily_maximum(self, t, y, X):
        d_max = self.C0 * np.sqrt(y) * self.time_delta_day
        return np.min([d_max, y], axis=0)

    def payoff(self, t: int, H, Y: np.ndarray, X):
        if (Y > self.y_max).any():
            raise RuntimeError("Y exceeded constrained.")
        if t == self.n_time_steps - 1:
            return np.zeros(X.shape[0])
        else:
            h = H[:, 0]
            y = Y[:, 0]
            return (h - self.gas_loss(y, h) * self.time_delta_day) * X[:, 0] \
                   * np.exp(- self.time_delta_year * self.interest_rate * t)

    def phi(self, t: int, H, Y):
        return Y - H
