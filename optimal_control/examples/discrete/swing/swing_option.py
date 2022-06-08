from abc import ABC

from .. import DiscreteExample


class SwingOption(DiscreteExample, ABC):
    def __init__(self, T: float, n_time_steps: int, y_max: int):
        super().__init__(T, n_time_steps, y_max)

    def phi(self, t: int, H, Y):
        return H + Y
