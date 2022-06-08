from .basis import Basis


class CompositionBasis(Basis):
    def __init__(self, basis_inner: Basis, basis_outer: Basis):
        def comp_(fi, fo):
            def fun(X):
                return fo(fi(X))

            return fun

        functions = [comp_(basis_inner.transform, fo) for fo in basis_outer.functions]
        super().__init__(functions=functions, dimension=basis_outer.dimension)
        self.basis_inner = basis_inner
        self.basis_outer = basis_outer

    def transform(self, X, out=None):
        out = self.__check_out__(X.shape[0], out)
        self.basis_outer.transform(self.basis_inner.transform(X), out)
        return out
