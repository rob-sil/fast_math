import numpy as np

from .rust_fast_math import sum_32


def sum(array: np.ndarray) -> np.floating:
    """Sum the elements of an array."""
    if array.ndim != 1 or not isinstance(array, np.ndarray):
        raise NotImplementedError("Only single-dimension arrays are supported.")

    if array.dtype == np.float32:
        return np.float32(sum_32(array))
    else:
        raise NotImplementedError("Only float32s are currently supported.")
