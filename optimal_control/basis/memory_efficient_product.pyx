import numpy as np

def memory_efficient_product(a, out):
    # One an example of shape (20, 1_000_000) this was 14 times faster then the sklearn Polynomial features.
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]

    cdef int i
    cdef int j
    cdef int k = 0

    for i in range(n):
        k = i*n - i*(i-1)//2
        np.multiply(a[:,i:], a[:, i:i+1], out=out[:, k:k + (n - i)])