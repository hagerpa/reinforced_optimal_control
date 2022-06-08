from abc import abstractmethod

from optimal_control.examples.example import Example

class ContinuousExample(Example):
    """The prototype of any optimal control example."""
    def __init__(self, T, n_time_steps):
        super().__init__(T, n_time_steps)

    @abstractmethod
    def constrained_maximizer(self, t: int, Y, X):
        """
        The constrained maximizer, i.e. a maximization algorithm, which respects the policy constraints.
        :param t: Time step.
        :param Y: Control as 2-dim array (n_samples, dimension of the control).
        :param X: Underlying as 2-dim array (n_samples, dimension of the underlying)
        :return: Returns a function M, which given a function g = g(H) returns the max and argmax, i.e.
            M: g |-> (max, argmax){ g(h) | h in K_t(y_t, X_t) }
        """
        pass
