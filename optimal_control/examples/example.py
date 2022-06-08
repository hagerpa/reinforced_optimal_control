from abc import ABC, abstractmethod

class Example(ABC):
    """The prototype of any optimal control example."""
    def __init__(self, T, n_time_steps):
        self.T = T
        self.n_time_steps = n_time_steps

    @abstractmethod
    def payoff(self, t: int, H, Y, X):
        """
        Time dependent pay of function of policy, control and underlying.
        :param t: Time step.
        :param H: Policy as 2-dim array (n_samples, dimension of the control)
        :param Y: Control as 2-dim array (n_samples, dimension of the control)
        :param X: Underlying as 2-dim array (n_samples, dimension of the underlying)
        :return H_t(h, y, X)
        """
        pass

    @abstractmethod
    def phi(self, t: int, H, Y):
        """
        Link function for control and policy.
        :param t: Time step.
        :param H: Policy as 2-dim array (n_samples, dimension of the control).
        :param Y: Control as 2-dim array (n_samples, dimension of the control).
        :return phi_t(y_t, X_t).
        """
        pass