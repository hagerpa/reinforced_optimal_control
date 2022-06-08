import numpy as np
from price_processes import MeanRevertingJumpDiffusion

from optimal_control.basis import PolynomialBasis
from optimal_control.examples.discrete.gas.discrete_gas_storage import DiscreteGasStorage
from optimal_control.solvers.discrete import RegressionValueFunction
from optimal_control.evaluation.test_cases import TestCase
from optimal_control.evaluation.test_environment import TestEnvironment


class DiscreteGasStorageTestCase(TestCase):

    @staticmethod
    def test_environment(T: float, J: int,
                         y_max: int, h_max: int, h_min: int,
                         interest_rate: float, volatility: float,
                         mean: float, mean_revision: float,
                         jump_rate: float, jump_mean: float, jump_std: float,
                         log_utility: bool):
        # def gas_loss(y, c):
        #    a = np.zeros_like(c)
        #    a[c < 0] = 1.7
        #    return a

        # Setting up the gas storage example as in [Gyurko, Hambly, Witte](2011)
        example = DiscreteGasStorage(T=T,
                                     n_time_steps=J + 1,
                                     interest_rate=interest_rate,
                                     y_max=y_max,
                                     h_max=h_max,
                                     h_min=h_min,
                                     log_utility=log_utility)

        # Setting up the price model as in [Gyurko, Hambly, Witte](2011)
        jol = MeanRevertingJumpDiffusion(volatility=volatility,
                                         mean=mean,
                                         mean_reversion=mean_revision,
                                         lam=jump_rate,
                                         jump_mean=jump_mean,  # 6.4
                                         jump_std=jump_std,  # 2
                                         euler_refinement=1)

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
            i: PolynomialBasis(degrees=range(i + 1)) for i in range(5)
        }

        def reinforced_value_functions(y: int):
            # return np.arange(0, test_environment.example.y_max, 2)
            return np.array([4])
            # return test_environment.example.phi(0, test_environment.example.h_space(0, y), y)

        return RegressionValueFunction(
            example=test_environment.example,
            y_max=test_environment.example.y_max,
            x_basis=X_BASIS[deg],
            D=max_depth,
            L_=reinforced_value_functions
        )
