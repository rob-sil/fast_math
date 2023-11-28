import numpy as np

from .rust_fast_math import sum_32_1d, sum_32_2d, sum_32_dyn


def sum(array: np.ndarray) -> np.floating:
    """Sum the elements of an array."""
    if not isinstance(array, np.ndarray):
        raise NotImplementedError("Only NumPy arrays are supported.")

    if array.dtype != np.float32:
        raise NotImplementedError("Only float32s are currently supported.")

    if array.ndim == 1:
        return np.float32(sum_32_1d(array))
    elif array.ndim == 2:
        return np.float32(sum_32_2d(array))
    else:
        return np.float32(sum_32_dyn(array))
