import numpy as np
from price_processes import BlackScholesModel

from optimal_control.basis import OrderStatisticsBasis, UnionBasis, Bias, \
    SecondDegreePolynomialBasis, IdentityBasis, ThirdDegreePolynomialBasis, Basis
from optimal_control.evaluation.test_cases import TestCase
from optimal_control.evaluation.test_environment import TestEnvironment
from optimal_control.examples.discrete import DiscreteMaxCallSwing
from optimal_control.solvers.discrete import RegressionValueFunction


class MultipleMaxCallTestCase(TestCase):

    @staticmethod
    def test_environment(
            J: int = 9,
            T: float = 3.0,
            exercise_rights: int = 1,
            strike: float = 100.0,
            volatility: float = 0.2,
            interest_rate: float = 0.05,
            dividend: float = 0.1,
            rho: float = 0.0,
            dimension: int = 2):
        example = DiscreteMaxCallSwing(T=(T / J) * (J + 1),
                                       n_time_steps=J + 2,
                                       y_max=exercise_rights,
                                       strike=strike,
                                       interest_rate=interest_rate,
                                       h_max=1)

        C = np.ones((dimension, dimension)) * rho + np.eye(dimension) * (1 - rho)

        price = BlackScholesModel(volatility=volatility, drift=interest_rate - dividend, correlation=C)

        return TestEnvironment(name="A swing option with a max-call payoff.",
                               comment=""".""",
                               price_model=price,
                               example=example,
                               y_prior=None,
                               random_seed=None)

    @staticmethod
    def value_function(test_environment, discrete=True, deg=0, max_depth=None,
                       n_reinforced_functions=1, use_order_statistic=True) -> RegressionValueFunction:

        dimension = test_environment.price_model.dimension
        if use_order_statistic:
            osb = OrderStatisticsBasis(dimension)
        else:
            osb = IdentityBasis(dimension)

        def pay_off_fun(X):
            return np.maximum(np.max(X, axis=1) - 100, 0)

        pay_off_basis = Basis(functions=[pay_off_fun], dimension=1)

        X_BASIS = {
            0: UnionBasis([Bias()]),
            1: UnionBasis([Bias(), osb]),
            2: UnionBasis([SecondDegreePolynomialBasis(osb.dimension, pre_basis=osb)]),
            3: UnionBasis([ThirdDegreePolynomialBasis(osb.dimension, pre_basis=osb)]),
            '1+p': UnionBasis([Bias(), osb, pay_off_basis])
        }

        def reinforced_value_functions(y: int):
            return np.arange(n_reinforced_functions)

        return RegressionValueFunction(
            example=test_environment.example,
            y_max=test_environment.example.y_max,
            x_basis=X_BASIS[deg],
            D=max_depth,
            L_=reinforced_value_functions
        )
