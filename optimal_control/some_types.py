from typing import Callable, TypeVar, Union
import numpy as np

'''
Typing (this can be skipped and is not necessary for understanding the implementation)
We introduce type annotations for function parameters. This is only for the purpose of preventing runtime errors, as the
IDE will help to find mistakes. Also it helps in building the code structure.
'''

# A type for time dependent objects, which we model as integer functions
TimeDependentObject = TypeVar("TimeDependentObject")
TimeDependent = Callable[[int], TimeDependentObject]

# Types for vectorized real-valued (actually float-valued) functions
RealVector = Union[float, np.ndarray]
RealFunction3 = Callable[[RealVector, RealVector, RealVector], RealVector]  # R^3 -> R
RealFunction2 = Callable[[RealVector, RealVector], RealVector]  # R^2 -> R
RealFunction1 = Callable[[RealVector], RealVector]  # R -> R
