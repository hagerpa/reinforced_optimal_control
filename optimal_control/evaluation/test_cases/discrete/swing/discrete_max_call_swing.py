import numpy as np
from price_processes import LogOrnsteinUhlenbeck

from optimal_control.basis import PolynomialBasis, UnionBasis, Basis
from optimal_control.examples.discrete.swing.discrete_max_call_swing import \
    DiscreteMaxCallSwing as DiscreteMaxCallSwingExample
from optimal_control.solvers.discrete import RegressionValueFunction
from optimal_control.evaluation.test_cases import TestCase
from optimal_control.evaluation.test_environment import TestEnvironment
from optimal_control.y_prior.y_prior_multiple_stopping import YPriorMultipleStopping


class DiscreteMaxCallSwing(TestCase):

    @staticmethod
    def test_environment(dates: int = 50,
                         exercise_rights: int = 1,
                         off_peak=False,
                         strike: float = 100.0,
                         volatility: float = 0.5,
                         mean: float = 0.0,
                         mean_revision: float = 0.9,
                         dimension: int = 1,
                         rho: float = 0.0):

        example = DiscreteMaxCallSwingExample(T=dates,
                                              n_time_steps=dates + 1,
                                              y_max=exercise_rights,
                                              h_max=0,
                                              strike=strike,
                                              interest_rate=0.0)

        def h_space(t, y):
            if off_peak:
                h_max = 1 if (t % 7) < 5 else 2
            else:
                h_max = 1
            return np.arange(0, min(h_max, example.y_max - y) + 1)

        example.h_space = h_space

        # C = np.ones((dimension, dimension)) * rho + np.eye(dimension) * (1 - rho)

        price = LogOrnsteinUhlenbeck(volatility=volatility, mean=mean, mean_revision=mean_revision)

        y_prior = YPriorMultipleStopping(example.y_max - 1)

        return TestEnvironment(name="Off peak swing option example from Bender, Schoenmakers, Zhang 2015.",
                               comment=""".""",
                               price_model=price,
                               example=example,
                               y_prior=y_prior,
                               random_seed=None)

    @staticmethod
    def value_function(test_environment, max_depth=None, x_deg=0):
        def payoff_basis_fun(X):
            m, d = X.shape
            return test_environment.example.payoff(1, np.full((m, 1), 1), np.full((m, 1), 0), X)

        payoff_basis = Basis(functions=[payoff_basis_fun], dimension=1)

        X_BASIS = {
            0: PolynomialBasis([0]),
            1: PolynomialBasis([0, 1]),
            2: PolynomialBasis([0, 1, 2]),
            "lin+pay": UnionBasis([PolynomialBasis([0, 1]), payoff_basis]),
        }

        return RegressionValueFunction(test_environment.example, test_environment.example.y_max, X_BASIS[x_deg],
                                       D=max_depth)
