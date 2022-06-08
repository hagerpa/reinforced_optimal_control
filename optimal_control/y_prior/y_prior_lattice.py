from .y_prior import YPrior

import numpy as np


class YPriorLattice(YPrior):
    def __init__(self, ny_samples, a, b, random_seed: int = None, end_points: bool = False):
        super().__init__(ny_samples, n_dimensions=1, random_seed=random_seed)
        self.a = a
        self.b = b
        self.end_points = end_points

    def sample(self, t: int, nx_samples: int):
        if not (self.random_seed is None):
            np.random.seed(self.random_seed + t)

        m = nx_samples
        if self.end_points:
            n = self.ny_samples - 2
            Y = np.empty((nx_samples, self.ny_samples))
            Y_ = (np.arange(n) / n).reshape(1, n).repeat(m, axis=0) + np.random.uniform(size=(m, 1))
            Y[:, 1:-1] = (self.b - self.a) * (Y_ - np.floor(Y_)) + self.a
            Y[:, 0] = self.a
            Y[:, -1] = self.b
        else:
            n = self.ny_samples
            Y_ = (np.arange(n) / n).reshape(1, n).repeat(m, axis=0) + np.random.uniform(size=(m, 1))
            Y = (self.b - self.a) * (Y_ - np.floor(Y_)) + self.a

        return Y.flatten().reshape(-1, 1)
