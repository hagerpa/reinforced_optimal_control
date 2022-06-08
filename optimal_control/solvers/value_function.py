from abc import ABC

from optimal_control.examples import Example


class ValueFunction(ABC):
    """The prototype of a value function which poses the methods for evaluation and fitting."""

    def __init__(self, example: Example):
        self.example = example
        self.n_time_steps = example.n_time_steps

    def evaluate(self, *args, **kwargs):
        """
        Value function evaluated at certain time step t, control y_t and underlying X_t.
        For the function's signature see :func:`value_and_policy`.
        :return: V_t(y, X) as a 1-dim vector.
        """
        return self.value_and_policy(*args, **kwargs)[:, 0]

    def policy(self, *args, **kwargs):
        """
        The optimal policy h_t yielding the value of V_t(y_t, X_t), i.e. the argmax of the maximisation over all
        admissible policies:
            h_t = argmax{ H_t(h, y_t, X_t) + C_t(y_t, X_t) | h in K_t(y_t, X_t) }
        :return: h_t(y_t, X_t) as a 1-dim vector.
        """
        return self.value_and_policy(*args, **kwargs)[:, 1]

    def value_and_policy(self, t, Y, X_t):
        """
        Returns the value V_t(y_t, X_t) and the corresponding optimal policy h_t(y_t, X_t).
        :param t: Time step of evaluation.
        :param Y: Control states as a 2-dim array of shape (n_samples, dimension of y).
        :param X_t: Value of the underlying as a 2-dim array of shape (n_samples, dimension of X_t).
        :return: Values and policy as a 2-dim array of shape (n_samples, 2)
        """
        pass
