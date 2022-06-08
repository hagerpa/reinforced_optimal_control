from typing import Optional

from price_processes import PriceModel

from optimal_control.examples import Example
from optimal_control.y_prior import YPrior


class TestEnvironment:
    """
    Saves the relevant parameters of a specific test scenario and their description.
    """

    def __init__(self, name: str, comment: str, price_model: PriceModel, example: Example, y_prior: Optional[YPrior],
                 random_seed: int = None):
        self.name = name
        self.comment = comment
        self.example = example
        self.price_model = price_model
        self.random_seed = random_seed
        self.y_prior = y_prior