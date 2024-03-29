from typing import Optional, Union

import numpy as np
from numpy.typing import DTypeLike

from .rust_fast_math import sum_32


def sum(
    array: np.ndarray,
    dtype: Optional[DTypeLike] = None,
    axis: Optional[int] = None,
) -> Union[np.floating, np.ndarray]:
    """Sum the elements of an array.

    Arguments
    ---
    array: a NumPy array
    dtype: Numpy datatype, optional
        If provided, the array data will be converted to this datatype while
        summing. Only supported if the array's data type can be safely converted
        to `dtype`.
    axis: int, optional
        If provided, the axis to sum over. The sum will be returned as an array
        with one less dimension than `array`.

    Returns
    ---
        The sum of the array values, rounded to floating-point precision.

        The output is not guaranteed to be accurate if summing a subset of the
    array values would overflow. For instance, an array with many large
    positive numbers along with the negative of all those numbers has a well-
    defined sum of zero. However, this function may return NaN, infinity, or
    negative infinity. If the function does return a finite number, then it
    will be the rounded sum.
    """
    if not isinstance(array, np.ndarray):
        raise NotImplementedError("Only NumPy arrays are supported.")

    dtype = np.dtype(dtype or array.dtype)
    if dtype != np.float32:
        raise NotImplementedError("Only float32s are currently supported.")

    out = sum_32(array, axis)

    return np.float32(out)
