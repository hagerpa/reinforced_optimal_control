from abc import ABC, abstractmethod

from .. import Example

class DiscreteExample(Example, ABC):
    """The prototype of any optimal control example with discrete control state-space."""
    def __init__(self, T:float, n_time_steps: int, y_max: int):
        super().__init__(T, n_time_steps)
        self.y_max = y_max

    @abstractmethod
    def h_space(self, t, y):
        pass