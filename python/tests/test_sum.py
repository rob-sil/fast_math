from math import fsum

import numpy as np

import fast_math as fm


def test_float32():
    # A 32-bit float (IEEE 754 binary32) has a 23-bit mantissa, so 2^24 is
    # stored without a "ones" place. Trying to add 1 to 2^24 rounds to just
    # 2^24.
    array = np.array([1, 2**24, 0, -(2**24)], dtype=np.float32)

    assert np.float32(fsum(array)) == fm.sum(array)
    assert np.float32(fsum(array)) != np.sum(array)


def test_arange():
    # Test ascending numbers
    array = np.arange(1_000_000, dtype=np.float32)

    assert np.float32(fsum(array)) == fm.sum(array)
    assert np.float32(fsum(array)) != np.sum(array)


def test_arange_reverse():
    # Test descending numbers
    array = np.arange(1_000_000, dtype=np.float32)[::-1]

    assert np.float32(fsum(array)) == fm.sum(array)
    assert np.float32(fsum(array)) != np.sum(array)


def test_dtype():
    # Test that the sum maintains the proper dtype
    array = np.arange(1_000_000, dtype=np.float32)

    assert type(fm.sum(array)) == array.dtype
