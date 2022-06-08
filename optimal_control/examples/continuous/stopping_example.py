from abc import abstractmethod

import numpy as np

from optimal_control.basis import Basis
from . import ContinuousExample


class StoppingExample(ContinuousExample):
    def __init__(self, T: float, n_steps: int):
        super().__init__(T, n_steps)

    @abstractmethod
    def payoff(self, t: int, H, Y, X):
        pass

    def phi(self, t: int, H, Y):
        return Y + H

    def constrained_maximizer(self, t: int, Y, X):
        n_samples = X.shape[0]
        VH = np.zeros((n_samples, 2))
        H = VH[:, 1:2]
        if t == 0:  # In t=0 there is no stopping possibility
            def maximizer_(g):
                # H=0 by default
                VH[:, 0] = g(H)
                return VH

            return maximizer_
        else:
            def maximizer_(g):
                g0 = g(H)
                mask_1 = Y[:, 0] == 0  # Check where exercise is possible
                mask_2 = self.payoff(t, np.ones_like(H), Y, X) > 0  # Check where payoff is truly possible
                mask = np.logical_and(mask_1, mask_2)
                n_mask = np.logical_not(mask)
                VH[n_mask, 0] = g0[n_mask]

                if mask.any():
                    g1 = g(np.ones_like(H))
                    argmax = np.argmax([g0[mask], g1[mask]], axis=0)
                    VH[mask, 1] = argmax
                    VH[mask, 0] = np.array([g0[mask], g1[mask]]).T[np.arange(np.size(argmax)), argmax]

                return VH

            return maximizer_

    @staticmethod
    def stopping_basis():
        def fy(y):
            return 1 - y.flatten()

        y_basis = Basis([fy], dimension=1)
        return y_basis
