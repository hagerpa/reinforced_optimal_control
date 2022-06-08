from optimal_control.examples import Example
from . import ValueFunction
import numpy as np


def optimal_control_solution(X, y0, trained_value_function: ValueFunction, example: Example, print_progression=True,
                             max_batch_size=1_000_000):
    """
    Calculates the (approximate) solution, i.e. the sequence of optimal policies, to the optimal control problem given a
    sequence of value functions and sample paths.

    :param X: Test set of price paths.
    :param y0: Initial states of the control variable
    :param trained_value_function: -.
    :param example: Financial derivative (e.g. SwingOption).
    :param max_batch_size: a very large amount of samples will be split in batches of this size.

    :return: 3d tensor (n_samples, n_time_steps, 3): 1st slice is the sample matrix, 2nd slice the values, 3rd slice the
    policies. Also returns the totally achieved payoff along each sample path - for the calculation of lower bounds.
    """

    if np.ndim(X) == 1:
        X = np.reshape(X, (1, len(X), 1))
    if np.ndim(X) == 2:
        m, n = X.shape
        X = np.reshape(X, (m, n, 1))

    n_samples, n_time_steps, n_dim = X.shape

    if trained_value_function.n_time_steps != n_time_steps:
        raise ValueError("The number of sample time steps must be equal to the number of value functions.")

    if np.ndim(y0) == 1 and np.size(y0) != n_samples:
        raise ValueError("Number of initial control states y0 must equal the number of samples in X.")

    Y = np.ones((n_samples, 1),
                dtype=np.dtype(type(y0))) * y0  # TODO: This still expects 1-dimensionality of the control variable

    solution = np.zeros((n_samples, n_time_steps, 2))

    total_payoff = np.zeros(n_samples)

    batches = (n_samples - 1) // max_batch_size + 1
    for i in range(batches):
        a = i * max_batch_size
        b = (i + 1) * max_batch_size if (i < batches - 1) else n_samples

        for t in range(n_time_steps):
            if print_progression:
                print("{}/{} ->".format(t, n_time_steps - 1), flush=True, end="")
                print(end="\r", flush=True)
            solution[a:b, t, :] = trained_value_function.value_and_policy(t, Y[a:b], X[a:b, t])
            H = solution[a:b, t, 1].reshape(-1, 1)  # TODO: Also expected 1-dimensionality here
            total_payoff[a:b] += example.payoff(t, H, Y[a:b], X[a:b, t])
            Y[a:b] = example.phi(t, H, Y[a:b])
        if print_progression: print("\r", flush=True, end='')
    return solution, total_payoff
