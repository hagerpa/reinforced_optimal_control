from optimal_control.examples.discrete import StoppingExample


class StoppingPathComponent(StoppingExample):
    def __init__(self, T: float, n_time_steps: int, path_component: int):
        super().__init__(T, n_time_steps)
        self.path_component = path_component

    def g(self, t: int, X):
        return X[:, self.path_component]
