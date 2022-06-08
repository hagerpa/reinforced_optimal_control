import numpy as np
from price_processes import FractionalBrownianMotion

from optimal_control.basis import UnionBasis, Bias, \
    IdentityBasis, ProjectionBasis
from optimal_control.basis.polynomial_basis import SciKitPolynomial
from optimal_control.examples.discrete import StoppingPathComponent
from optimal_control.solvers.discrete import ModifiedForStopping
from optimal_control.evaluation.test_cases import TestCase
from optimal_control.evaluation.test_environment import TestEnvironment
from optimal_control.y_prior.y_prior_multiple_stopping import YPriorMultipleStopping


class FractionalBrownianMotionTestCase(TestCase):

    @staticmethod
    def test_environment(
            J: int = 100,
            T: float = 1.0,
            H: float = 0.1,
            past=0
    ):
        example = StoppingPathComponent(T=(T / J) * (J + 1),
                                        n_time_steps=J + 2,
                                        path_component=0)

        price = FractionalBrownianMotion(H, past)

        y_prior = YPriorMultipleStopping(example.y_max - 1)

        return TestEnvironment(name="Stopping a fractional Brownian motion by including the past.",
                               comment=""".""",
                               price_model=price,
                               example=example,
                               y_prior=y_prior,
                               random_seed=None)

    @staticmethod
    def value_function(test_environment, deg=0, max_depth=None, high_order=10):
        dimension = test_environment.price_model.dimension
        osb = IdentityBasis(dimension)

        hbs = ProjectionBasis(np.arange(0, high_order))
        lbs = ProjectionBasis(np.arange(high_order, test_environment.price_model.dimension))

        X_BASIS = {
            0: UnionBasis([Bias()]),
            1: UnionBasis([Bias(), osb]),
            2: UnionBasis([SciKitPolynomial(hbs.dimension, degree=2, pre_basis=hbs), lbs]),
            3: UnionBasis([SciKitPolynomial(hbs.dimension, degree=3, pre_basis=hbs), lbs])
        }

        return ModifiedForStopping(
            example=test_environment.example,
            x_basis=X_BASIS[deg],
            I=max_depth,
            positive_continuation=False
        )
