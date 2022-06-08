from abc import ABC, abstractmethod


class YPrior(ABC):
    def __init__(self, ny_samples: int, n_dimensions: int, random_seed: int = None):
        self.random_seed = random_seed
        self.n_dimensions = n_dimensions
        self.ny_samples = ny_samples

    @abstractmethod
    def sample(self, t: int, nx_samples: int):
        pass
