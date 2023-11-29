from typing import Optional

import numpy as np
from numpy.typing import DTypeLike

from .rust_fast_math import sum_32


def sum(array: np.ndarray, dtype: Optional[DTypeLike] = None) -> np.floating:
    """Sum the elements of an array.

    Arguments
    ---
    array: a NumPy array
    dtype: Numpy datatype, optional
        If provided, the array data will be converted to this datatype while
        summing. Only supported if the array's data type can be safely converted
        to `dtype`.
    """
    if not isinstance(array, np.ndarray):
        raise NotImplementedError("Only NumPy arrays are supported.")

    dtype = np.dtype(dtype or array.dtype)
    if dtype != np.float32:
        raise NotImplementedError("Only float32s are currently supported.")

    return np.float32(sum_32(array))
