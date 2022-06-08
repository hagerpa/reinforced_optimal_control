import os
import re
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from os import path

import numpy as np
import pandas as pd

from optimal_control.evaluation.test_environment import TestEnvironment
from optimal_control.solvers import ValueFunction
from optimal_control.evaluation.monitoring.lower_bound_monitor import LowerBoundMonitor


class TestCase(ABC):
    """
    An envelop structure to be used for running several test scenarios from several parameter setups and monitoring the
    results. It defines the maps from model / example parameters to the TestEnvironment and ValueFunction objects. In
    this way a TestCase object allows to save a numerical evaluation setup, which can be reproduced, e.g. on a
    computational cluster.
    """

    @staticmethod
    @abstractmethod
    def test_environment(**test_environment_params) -> TestEnvironment:
        pass

    @staticmethod
    @abstractmethod
    def value_function(test_environment, **value_function_params) -> ValueFunction:
        pass

    def test(self, test: LowerBoundMonitor, value_function_params: dict, test_environment_params: dict,
             file_name: str = None, random_seed=None):

        te = self.test_environment(**test_environment_params)
        te.random_seed = random_seed
        vf = self.value_function(te, **value_function_params)

        self.__test__(test, vf, te, value_function_params, test_environment_params, file_name, random_seed)

    @staticmethod
    def __test__(test: LowerBoundMonitor, vf: ValueFunction, te: TestEnvironment, value_function_params: dict,
                 test_environment_params: dict,
                 file_name: str = None,
                 test_set_file=None, train_set_file=None):

        test.test(test_environment=te, value_function=vf, test_set_file=test_set_file, train_set_file=train_set_file)

        if not (file_name is None):
            Y0 = np.array([[test.y0]])
            X0 = te.price_model.sample_paths(T=te.example.T / (te.example.n_time_steps - 1), n_steps=2,
                                             n_samples=1, start_law=('dirac', test.x0))[:, 0, :]
            v0 = vf.evaluate(0, Y0, X0)[0]

            row = dict()
            row.update({"vf_" + k: v for (k, v) in value_function_params.items()})
            row.update(test_environment_params)
            row.update(dict(
                v0=v0,
                date_time=datetime.now(),
                train_set_size=test.n_train_samples,
                test_set_size=test.n_test_samples,
                runs=test.runs,
                x0=test.x0,
                y0=test.y0)
            )
            row.update(test.result)

            if not path.exists(file_name):
                df = pd.DataFrame(columns=[k for k in row.keys()])
            else:
                df = pd.read_pickle(file_name)

            df.to_pickle(file_name)

            pd.to_pickle(pd.read_pickle(file_name).append(row, ignore_index=True), file_name)

        return test

    def run_test_from_script(self, test, PARAMS):
        if len(sys.argv) < 2:
            print("Choose an parameter index:")
            for (i, p) in enumerate(PARAMS):
                print("\t {}: {}".format(i + 1, p))
            raise SystemExit

        param_index = int(sys.argv[1]) - 1
        params = PARAMS[param_index]

        print(params)

        file_name = 'results_{}.pkl'.format(param_index)
        self.test(test, params["vf_params"], params["te_params"], file_name)

    def run_test_from_nb(self, test: LowerBoundMonitor, value_function_params: list, test_environment_params: list,
                         random_seed=None, read_test_sets=False, delete_test_sets=True):

        for ti, tp in enumerate(test_environment_params):
            te = self.test_environment(**tp)
            te.random_seed = random_seed

            if not read_test_sets:
                if random_seed: np.random.seed(random_seed)
                np.save('.test_case_run_x_te', te.price_model.sample_paths(te.example.T, te.example.n_time_steps,
                                                                           test.n_test_samples, ('dirac', test.x0)))

                for i in range(test.runs):
                    if random_seed: np.random.seed(random_seed + i + 1)
                    np.save('.test_case_run_x_tr_{}'.format(i),
                            te.price_model.sample_paths(te.example.T, te.example.n_time_steps, test.n_train_samples,
                                                        test.start_law))

            for vi, vp in enumerate(value_function_params):
                print(
                    "({}/{}) ({}/{})".format(ti + 1, len(test_environment_params), vi + 1, len(value_function_params)))
                vf = self.value_function(te, **vp)
                file_name = 'results_{}.pkl'.format(ti * len(value_function_params) + vi)
                self.__test__(test, vf, te, value_function_params=vp, test_environment_params=tp,
                              file_name=file_name, test_set_file='.test_case_run_x_te',
                              train_set_file='.test_case_run_x_tr_')

            if delete_test_sets:
                os.remove('.test_case_run_x_te.npy')
                for i in range(test.runs):
                    os.remove('.test_case_run_x_tr_{}.npy'.format(i))


def merge_results(file_name):
    prefix, suffix = file_name.split(".")

    result_files = [s for s in os.listdir() if re.match(r"{}_([0-9]+).{}$".format(prefix, suffix), s)]
    if len(result_files) == 0: print("No new results."); return;
    result_df = pd.concat([pd.read_pickle(s) for s in result_files])

    if os.path.exists(file_name):
        result_df = pd.concat([pd.read_pickle(file_name), result_df])

    result_df.reset_index(drop=True, inplace=True)
    pd.to_pickle(result_df, file_name)

    for s in result_files: os.remove(s)
