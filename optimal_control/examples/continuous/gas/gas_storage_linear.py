from optimal_control.examples.continuous.gas import GasStorage
from optimal_control.some_types import *


class GasStorageLinear(GasStorage):
    def __init__(self, T, n_time_steps, y_min, y_max, c_min_0,
                 c_max_y_max, gas_loss, y_prior_range):
        """
        :param y_max: Maximum gas storage capacity.
        :param y_min: Base gas storage requirement.
        :param c_max_y_max: Maximum injection rate at storage level y_max.
        :param c_min_0: Maximum injection rate at storage level 0.
        :param gas_loss: Gas loss rate as a function of the injection rate c and the stored gas y.
        """
        super().__init__(T, n_time_steps, y_min, y_max, c_min_0, c_max_y_max, gas_loss, y_prior_range)

    def daily_minimum(self, t: int, y, X):
        d_min = 0.1 * (self.y_max - self.y_min)  # 10% of total volume
        return - np.minimum(self.y_max - y, d_min)

    def daily_maximum(self, t, y, X):
        d_max = 0.1 * (self.y_max - self.y_min)  # 10% of total volume
        return np.minimum(y, d_max)
