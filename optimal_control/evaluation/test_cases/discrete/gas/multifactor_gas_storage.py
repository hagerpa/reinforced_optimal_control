import numpy as np
from price_processes import TwoFactorJumpDiffusion

from optimal_control.basis import PolynomialBasis, UnionBasis
from optimal_control.basis.polynomial_basis import SciKitPolynomial
from optimal_control.examples.discrete.gas.multifactor_gas_storage import MultifactorGasStorage
from optimal_control.solvers.discrete import RegressionValueFunction
from optimal_control.evaluation.test_cases import TestCase
from optimal_control.evaluation.test_environment import TestEnvironment


class MultifactorGasStorageTestCase(TestCase):

    @staticmethod
    def test_environment(J: int,
                         y_max: int, h_max: int, h_min: int,
                         interest_rate: float, volatility,
                         mean: float, mean_revision,
                         jump_rate: float, jump_mean: float, jump_std: float,
                         correlation: float, log_utility: bool, euler_refinement: bool):
        # def gas_loss(y, c):
        #    a = np.zeros_like(c)
        #    a[c < 0] = 1.7
        #    return a

        period = 365 // J
        T = J * period / 365.0

        # Setting up the gas storage example as in [Gyurko, Hambly, Witte](2011)
        example = MultifactorGasStorage(T=(T / J) * (J + 1),
                                        n_time_steps=J + 2,
                                        interest_rate=interest_rate,
                                        y_max=y_max,
                                        h_max=h_max,
                                        h_min=h_min,
                                        log_utility=log_utility)

        er = period if euler_refinement else 1
        jol = TwoFactorJumpDiffusion(volatility=volatility,
                                     mean=mean,
                                     mean_reversion=mean_revision,
                                     lam=jump_rate,
                                     jump_mean=jump_mean,
                                     jump_std=jump_std,
                                     correlation=correlation,
                                     euler_refinement=er)

        # Setting up the test environment
        return TestEnvironment(name="A discrete gas storage problem.",
                               comment="""We modified the example from (Gyurko, Hambly, Witte, 2011), so that the
                               control variable (the a mount of stored gas) is discrete.""",
                               price_model=jol,
                               example=example,
                               y_prior=None,
                               random_seed=None)

    @staticmethod
    def value_function(test_environment, deg=0, max_depth=None):
        X_BASIS = {
            0: SciKitPolynomial(2, 0),
            1: SciKitPolynomial(2, 1),
            2: SciKitPolynomial(2, 2),
            3: SciKitPolynomial(2, 3),
            4: SciKitPolynomial(2, 4),
            'x1': UnionBasis([PolynomialBasis([[0, 0], [0, 1]])]),
            'x2': UnionBasis([PolynomialBasis([[0, 0], [0, 1], [0, 2]])])
        }

        def reinforced_value_functions(y: int):
            return np.array([4])

        return RegressionValueFunction(
            example=test_environment.example,
            y_max=test_environment.example.y_max,
            x_basis=X_BASIS[deg],
            D=max_depth,
            L_=reinforced_value_functions
        )
