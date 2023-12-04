from math import fsum

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import fast_math as fm


def test_float32():
    # A 32-bit float (IEEE 754 binary32) has a 23-bit mantissa, so 2^24 is
    # stored without a "ones" place. Trying to add 1 to 2^24 rounds to just
    # 2^24.
    array = np.array([1, 2**24, 0, -(2**24)], dtype=np.float32)

    accurate = np.float32(fsum(array))
    assert accurate == fm.sum(array)


def test_arange():
    # Test adding many ascending numbers
    array = np.arange(1_000_000, dtype=np.float32)

    accurate = np.float32(fsum(array))
    assert accurate == fm.sum(array)


def test_arange_reverse():
    # Test adding many descending numbers
    array = np.arange(1_000_000, dtype=np.float32)[::-1]

    accurate = np.float32(fsum(array))
    assert accurate == fm.sum(array)


@pytest.mark.parametrize("ndims", [1, 2, 3, 4])
def test_dimensions(ndims):
    # Test summing arrays with multiple dimensions
    shape = tuple([5] * (ndims - 1) + [-1])
    array = np.arange(1_000_000, dtype=np.float32).reshape(shape)

    accurate = np.float32(fsum(array.flatten()))
    assert accurate == fm.sum(array)


def test_axis():
    array = np.arange(2_000_000, dtype=np.float32).reshape((2, -1))

    accurate = np.array(
        [fsum(array[0]), fsum(array[1])],
        dtype=array.dtype,
    )

    assert_array_almost_equal(
        accurate,
        fm.sum(array, axis=1),
    )
