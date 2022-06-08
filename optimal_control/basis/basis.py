import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class Basis(BaseEstimator, TransformerMixin):
    """
    This class represents the prototype of a function basis which can be used for linear regression.
    A function basis consists of a set of functions {f_1, ..., f_d} where d="dimension of the basis" and each function
    has the following dimensionality f_i: (n) -> (), [() stands for a zero-dim array] which we understand vectorized as
        f_i: (m,n) -> (m), f([x_1, ..., x_m]) = [f(x_1), ..., f(x_n)].
    The transform method then maps
        basis.transform: (m, n) -> (m, d), X |-> [f_1(X), ..., f_n(X)].T .
    The idea behind the abstraction of the basis class, is that in general [f_1(X), ..., f_n(X)] can be evaluated more
    efficiently then with two python loops over d and m. For the particular implementations see the transform method of
    the classes that implement this class.
    Moreover we target a better efficiency for functions that are often re-evaluated with a large m. In particular if
    the fit method of the basis is called, a memory view is reserved for the basis output, which is reused in every
    evaluation as long as m stays the same.
    """

    def __init__(self, functions, dimension: int):
        """
        All functions are assumed to take an array of shape (m, n) as input and output an array of shape (m).
        :param functions: Basis of functions. Functions are assumed to return flat arrays!
        """
        if len(functions) != dimension:
            raise AttributeError("Number of functions must equal the number of dimensions, since functions are assumed "
                                 "to return flat arrays")
        self.functions = functions
        self.dimension = dimension
        self.out = None

    def fit(self, X, y=None, out=None):
        """
        Creates a memory view for the basis output, which is reused for consecutive basis evaluations with the same
        sample size m.
        :param X: Features.
        :param y: labels, not used.
        :param out: If a memory view is provided and of correct shape, then it is saved into the basis for evaluation.
        :return: self.
        """
        if not (out is None):
            if out.shape[0] != X.shape[0]:
                raise ValueError("Basis output memory view need same number of rows as input X.")
            self.out = out
        elif (self.out is None) or (self.out.shape[0] == X.shape[0]):
            self.out = np.empty((X.shape[0], self.dimension))
        return self

    def transform(self, X, out=None):
        """
        Applies the functions to the input array, returning a features matrix.
        :param X: Input array of shape (m, n).
        :param out: If provided, the result will be saved into the given memory space.
        :return: Transformed feature matrix (m, dimension).
        """
        if out is None: out = np.empty((X.shape[0], self.dimension))
        for (i, f) in enumerate(self.functions): out[:, i] = f(X)
        return out

    def __check_out__(self, n_samples: int, out):
        """
        Checks if the provided or stored output memory view is of the correct shape for the basis evaluation.
        :param n_samples: The sample size of the input array that is to be evaluated.
        :param out: The provided output memory view. If None, then the interval memory view is checked, or if there is
        not internal memory view, a new memory is created.
        :return: An output memory view of the correct size. Throws an error if the provided memory view is of incorrect
        shape.
        """
        shape = (n_samples, self.dimension)
        if out is None:
            if (not (self.out is None)) and (self.out.shape == shape):
                return self.out
            else:
                out = np.empty(shape)
        else:
            if out.shape != shape: raise ValueError("The shape of the out error need to agree with the sample matrix"
                                                    "and the dimension of the basis {0}!={1}".format(out.shape, shape))
        return out
