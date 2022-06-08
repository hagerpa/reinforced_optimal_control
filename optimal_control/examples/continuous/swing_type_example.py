from abc import abstractmethod
import numpy as np

from . import ContinuousExample


class SwingTypeExample(ContinuousExample):
    @abstractmethod
    def daily_minimum(self, t: int, y, X):
        pass

    @abstractmethod
    def daily_maximum(self, t, y, X):
        pass

    def optimization_grid(self, t, lower_bounds, upper_bounds):
        return [lower_bounds, np.zeros_like(lower_bounds), upper_bounds]

    def constrained_maximizer(self, t: int, Y, X):
        if Y.shape[1] != 1:
            raise RuntimeError("Y needs to bee 1d.")

        n_samples = Y.shape[0]
        y = Y[:, 0]

        lower_bounds = self.daily_minimum(t, y, X)
        upper_bounds = self.daily_maximum(t, y, X)

        def maximizer(g):
            Vh = np.empty((n_samples, 2))
            h_space = self.optimization_grid(t, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
            v = [g(h_.reshape(-1, 1)) for h_ in h_space]
            arg_max = np.argmax(v, axis=0)
            rows = np.arange(n_samples)
            Vh[:, 1] = np.array(h_space).T[rows, arg_max]
            Vh[:, 0] = np.array(v).T[rows, arg_max]

            return Vh

        return maximizer
