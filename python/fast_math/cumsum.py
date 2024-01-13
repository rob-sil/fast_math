from typing import Optional

import numpy as np

from .rust_fast_math import cumsum_32


def cumsum(
    array: np.ndarray,
    axis: Optional[int] = None,
) -> np.ndarray:
    """Cumulatively sum the elements of an array.

    Each element returned equals the cumulative sum of the input array, rounded
    to floating point precision.

    Results are not guaranteed if the cumulate sum overflows.

    Arguments
    ---
    array: a NumPy array
    axis: int, optional
        The axis to sum over. If no axis is provided, the array is treated as
        if it were flattened.
    """

    if not isinstance(array, np.ndarray):
        raise NotImplementedError("Only NumPy arrays are supported.")

    out = cumsum_32(array, axis)

    return np.float32(out)
