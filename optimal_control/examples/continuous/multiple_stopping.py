import numpy as np
from abc import abstractmethod

from optimal_control.basis import Basis
from optimal_control.some_types import RealFunction2
from optimal_control.examples.continuous.example import Example


class MultipleStoppingExample(Example):
    def __init__(self, T: float, n_steps: int, y_max: int):
        super().__init__(T, n_steps)
        self.y_max = y_max

    @abstractmethod
    def payoff(self, t: int, H, Y, X):
        pass

    def phi(self, t: int, H, Y) -> RealFunction2:
        return Y + H

    def constrained_maximizer(self, t: int, Y, X):
        n_samples = X.shape[0]
        VH = np.zeros((n_samples, 2))
        H = VH[:, 1:2]
        if t == 0:
            def maximizer_(g):
                VH[:, 0] = g(H)
                return VH

            return maximizer_
        else:
            def maximizer_(g):
                g0 = g(H)
                mask = (Y.flatten() < self.y_max)
                n_mask = np.logical_not(mask)
                VH[n_mask, 0] = g0[n_mask]

                if mask.any():
                    H_ = np.zeros_like(H)
                    H_[mask] = 1
                    g1 = g(H_)
                    argmax = np.argmax([g0[mask], g1[mask]], axis=0)
                    VH[mask, 1] = argmax
                    VH[mask, 0] = np.array([g0[mask], g1[mask]]).T[np.arange(np.size(argmax)), argmax]

                return VH
            return maximizer_

    def multiple_stopping_basis(self):
        def fy(y):
            return self.y_max - y.flatten()

        y_basis = Basis([fy], dimension=1)
        return y_basis
