from math import fsum, isinf

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
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


@pytest.mark.parametrize("length", [10, 1_000, 10_000])
def test_inf(length):
    """Test that sum handles infinity"""
    array = np.arange(length, dtype=np.float32)
    array[5] = np.inf

    assert fm.sum(array) == np.inf


@pytest.mark.parametrize("length", [10, 1_000, 10_000])
def test_neg_inf(length):
    """Test that sum handles negative infinity"""
    array = np.arange(length, dtype=np.float32)
    array[5] = -np.inf

    assert fm.sum(array) == -np.inf


@pytest.mark.parametrize("length", [10, 1_000, 10_000])
def test_nan(length):
    """Test that sum handles NaN"""
    array = np.ones(length, dtype=np.float32)
    array[5] = np.nan

    assert np.isnan(fm.sum(array))


@pytest.mark.parametrize("length", [10, 1_000, 10_000])
def test_mixed_inf(length):
    """Test that sum handles mixing positive and negative infinity"""
    array = np.ones(length, dtype=np.float32)
    array[3] = np.inf
    array[5] = -np.inf

    assert np.isnan(fm.sum(array))


@pytest.mark.parametrize("length", [10, 1_000, 10_000])
def test_mixed_nan(length):
    """Test that sum handles multiple NaN/infinities"""
    array = np.ones(length, dtype=np.float32)
    array[3] = np.nan
    array[5] = -np.inf

    assert np.isnan(fm.sum(array))


def test_overflow():
    """Test that summing can overflow into infinity"""
    array = np.array([2**126] * 8, dtype=np.float32)

    assert fm.sum(array) == np.inf


@pytest.mark.filterwarnings("ignore:overflow encountered")
@given(arrays(dtype=np.float32, shape=(100,)))
def test_accuracy_small(array):
    """Hypothesis tests for summing small arrays"""
    assume(not np.isnan(np.float32(fsum(np.abs(array)))))

    if np.inf in array and -np.inf in array:
        accurate = np.nan
    else:
        accurate = np.float32(fsum(array))

    # Catch overflow
    if np.isfinite(np.array(array)).all():
        assume(not np.isnan(accurate))

    result = fm.sum(array)

    if np.isnan(accurate):
        assert np.isnan(result)
    elif np.isinf(accurate):
        assert np.isinf(result)
    elif result != accurate:
        assert not np.isnan(result) and not np.isinf(result)

        # fsum can be inaccurate on the last bit
        base = max(np.abs(result), np.abs(accurate))
        error_exp = int(np.floor(np.log2(base))) - 23

        assert np.abs(result - accurate) <= 2**error_exp


@pytest.mark.filterwarnings("ignore:overflow encountered")
@given(arrays(dtype=np.float32, shape=(100_000,)))
def test_accuracy_large(array):
    """Hypothesis tests for summing large arrays"""
    if np.inf in array and -np.inf in array:
        accurate = np.nan
    else:
        accurate = np.float32(fsum(array))

    # Catch overflow
    if np.isfinite(np.array(array)).all():
        assume(not np.isnan(accurate))

    result = fm.sum(array)

    if np.isnan(accurate):
        assert np.isnan(result)
    elif np.isinf(accurate):
        assert np.isinf(result)
    elif result != accurate:
        assert not np.isnan(result) and not np.isinf(result)

        # fsum can be inaccurate on the last bit
        base = max(np.abs(result), np.abs(accurate))
        error_exp = int(np.floor(np.log2(base))) - 23

        assert np.abs(result - accurate) <= 2**error_exp


def test_multistep_rounding():
    """A regression test for a rounding bug."""
    array = np.array(
        [
            1.3510799e16,
            -1497869811e02,
            2.0000000e00,
        ],
        dtype=np.float32,
    )

    accurate = np.float32(fsum(array))
    result = fm.sum(array)
    assert result == accurate


def test_big_rounding():
    """Test rounding for values that exceed binary64.

    The 32-bit float doesn't have the mantissa bits to store 2**53 and 2**29 at
    the same time, so adding them must round. The final 1 and -1 force rounding
    up and down respectively.

    An implementation that uses 64-bit floats doesn't have the bits to store
    2**53 and 2**0=1 at the same time, and might incorrectly round.
    """
    array = np.array([2**29, 2**53, 1.0], dtype=np.float32)

    assert fm.sum(array) == 2**53 + 2**30

    array = np.array([2**29, 2**53, -1.0], dtype=np.float32)

    assert fm.sum(array) == 2**53
