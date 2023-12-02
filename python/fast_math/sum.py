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
    """
    if not isinstance(array, np.ndarray):
        raise NotImplementedError("Only NumPy arrays are supported.")

    dtype = np.dtype(dtype or array.dtype)
    if dtype != np.float32:
        raise NotImplementedError("Only float32s are currently supported.")

    if axis is not None:
        out_shape = (*array.shape[:axis], *array.shape[axis + 1 :])
        out = np.zeros(out_shape, dtype=dtype)
        array = np.moveaxis(array, axis, -1)

        for index in np.ndindex(out.shape):
            out[index] = np.float32(sum_32(array[index]))

        if out.shape == (1,):
            out = out[0]
    else:
        out = sum_32(array)

    return np.float32(out)
