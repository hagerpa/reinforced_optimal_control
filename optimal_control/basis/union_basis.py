from typing import List

from optimal_control.basis import Basis


class UnionBasis(Basis):
    """
    Takes the union of to function bases, i.e. joins the function into one basis.
    :param basis1: First basis.
    :param basis2: Second basis which is assumed to have the same input dimension as basis1.
    :return: A new basis which has |basis1.functions|+|basis2.functions| functions. The input dimension equals
    """

    def __init__(self, bases: List[Basis]):
        self.bases = bases
        super().__init__(functions=[f for b in bases for f in b.functions],
                         dimension=sum([b.dimension for b in bases]))

    def transform(self, X, out=None):
        out = self.__check_out__(X.shape[0], out)
        i = 0
        for b in self.bases:
            j = i + b.dimension
            b.transform(X, out[:, i:j])
            i = j
        return out
