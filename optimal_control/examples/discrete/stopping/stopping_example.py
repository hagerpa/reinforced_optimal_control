from abc import ABC, abstractmethod
import numpy as np

from optimal_control.examples.discrete import SwingOption


class StoppingExample(SwingOption, ABC):
    def __init__(self, T: float, n_time_steps: int):
        super().__init__(T, n_time_steps, 1)

    @abstractmethod
    def g(self, t: int, X):
        pass

    def payoff(self, t: int, H, Y, X):
        m, _ = X.shape
        if (t == 0) or (t == (self.n_time_steps - 1)):
            return np.zeros(m)
        else:
            if t == (self.n_time_steps - 2):
                mask = Y[:, 0] == 0
            else:
                mask = np.logical_and(H[:, 0] == 1, Y[:, 0] == 0)
            return np.where(mask, self.g(t, X), 0)

    def h_space(self, t, y):
        if y == 0:
            return np.array([0, 1])
        else:
            return np.array([0])
