import unittest

import numpy as np


class TestReinforcedSolver(unittest.TestCase):

    def test_y_prior_lattice(self):
        from optimal_control.y_prior.y_prior_lattice import YPriorLattice

        ny_samples = 1000
        y_prior = YPriorLattice(ny_samples=ny_samples, a=2, b=10)
        nx_samples = 1000
        Y = y_prior.sample(0, nx_samples)

        # Check periodicity of the the lattice
        Y_shifted = Y - Y[::10, :].repeat(10, axis=0)
        delta = Y_shifted[1, 0]
        assert np.allclose(0, np.sin(Y_shifted * 2 * np.pi / delta))

        # Test mean and variance
        means = Y.reshape(ny_samples, nx_samples).mean(axis=0)
        vars_ = Y.reshape(ny_samples, nx_samples).var(axis=0)
        np.allclose(means.mean(), (y_prior.b + y_prior.a) / 2)
        np.allclose(vars_.mean(), (y_prior.b - y_prior.a) ** 2 / 12)


if __name__ == '__main__':
    unittest.main()
