from abc import ABC, abstractmethod
from time import time, process_time
from typing import Tuple

import numpy as np
from optimal_control.evaluation.test_environment import TestEnvironment

from optimal_control.solvers import ValueFunction
from optimal_control.solvers.solver import optimal_control_solution


class Monitor(ABC):
    def __init__(self, visual=False):
        self.visual = visual
        self.result = None

    @abstractmethod
    def test(self, test_environment: TestEnvironment, value_function: ValueFunction, test_set_file: None,
             train_set_file: None) -> Tuple[float, float, float]:
        pass


class LowerBoundMonitor(Monitor):
    """Re-sampling and estimation of lower-bounds for the value at 0.
    The solver is re-trained with a different training set "runs"-time, and tested on a fixed set."""

    def __init__(self, x0, y0, n_train_samples: int, runs: int, n_test_samples: int, start_law):
        super().__init__()
        self.x0 = x0
        self.y0 = y0
        self.n_train_samples = n_train_samples
        self.runs = runs
        self.n_test_samples = n_test_samples
        self.start_law = start_law
        self.result = {}

    def test(self, test_environment: TestEnvironment, value_function: ValueFunction, test_set_file: None,
             train_set_file: None):
        example = test_environment.example
        price_model = test_environment.price_model
        random_seed = test_environment.random_seed

        timing = {}
        for s in ["er", "cpu"]:
            for u in ["tr", "te"]:
                q = "{}_time_{}".format(s, u)
                timing[q] = []

        values = {}
        for s in ["tr", "te"]:
            for u in ["mean", "std"]:
                q = "{}_{}".format(s, u)
                values[q] = []

        if random_seed:
            np.random.seed(random_seed)

        X_test = []
        if test_set_file is None:
            X_test = price_model.sample_paths(example.T, example.n_time_steps, self.n_test_samples, ('dirac', self.x0))

        for i in range(self.runs):
            if random_seed:
                np.random.seed(random_seed + i + 1)

            if train_set_file is None:
                X_train = price_model.sample_paths(example.T, example.n_time_steps, self.n_train_samples,
                                                   self.start_law)
            else:
                X_train = np.load(train_set_file + "{}.npy".format(i))

            tm, tp = time(), process_time()
            value_function.fit_te(X=X_train, test_environment=test_environment)
            timing["er_time_tr"] += [time() - tm]
            timing["cpu_time_tr"] += [process_time() - tp]

            _, res_train = optimal_control_solution(X_train, self.y0, value_function, example)

            X_train = []
            if not (test_set_file is None):
                X_test = np.load(test_set_file + ".npy")

            tm, tp = time(), process_time()
            _, res_test = optimal_control_solution(X_test, self.y0, value_function, example)
            timing["er_time_te"] += [time() - tm]
            timing["cpu_time_te"] += [process_time() - tp]

            if not (test_set_file is None):
                X_test = []

            values["te_mean"] += [res_test.mean()]
            values["te_std"] += [res_test.std()]

            values["tr_mean"] += [res_train.mean()]
            values["tr_std"] += [res_train.std()]

        def monte_carlo_values(means, stds):
            return np.mean(means), np.std(means) / np.sqrt(self.runs), np.mean(stds) / np.sqrt(self.n_test_samples)

        for s in ["tr", "te"]:
            v = monte_carlo_values(values[s + "_mean"], values[s + "_std"])
            self.result[s + "_mean"] = v[0]
            self.result[s + "_tr_err"] = v[1]
            self.result[s + "_te_err"] = v[2]

        for s in ["er", "cpu"]:
            q = "{}_time".format(s)
            self.result[q + "_best"] = np.add(timing[q + "_te"], timing[q + "_tr"]).min()
            self.result[q + "_mean"] = np.add(timing[q + "_te"], timing[q + "_tr"]).mean()
            for u in ["tr", "te"]:
                q = "{}_time_{}".format(s, u)
                self.result[q + "_best"] = min(timing[q])
                self.result[q + "_mean"] = max(timing[q])

        return self.result
