from math import fsum

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers
from numpy.testing import assert_array_almost_equal

import fast_math as fm


def test_float32():
    """Test accuracy on binary32 floating-point numbers

    A 32-bit float (IEEE 754 binary32) has a 23-bit mantissa, so 2^24 is stored
    without a "ones" place. Trying to add 1 to 2^24 rounds to just 2^24.
    """

    array = np.array([1, 2**24, 0, -(2**24)], dtype=np.float32)

    assert 1 == fm.sum(array)


@pytest.mark.parametrize("N", [1, 10, 1_000, 1_000_000, 100_000_000])
def test_ascending(N):
    """Test adding a sequence of ascending numbers

    When the array is ascending, each additional number grows along with the
    running sum.
    """
    array = np.arange(N, dtype=np.float32)

    accurate = np.float32((N * (N - 1)) // 2)
    assert accurate == fm.sum(array)


@pytest.mark.parametrize("N", [1, 10, 1_000, 1_000_000, 100_000_000])
def test_descending(N):
    """Test adding a sequence of descending numbers

    When the array is descending, each additional number gets smaller and may
    get to a point where the remaining numbers are all within the rounding
    error of the current value in 32 bits.
    """
    array = np.arange(N, dtype=np.float32)[::-1]

    accurate = np.float32((N * (N - 1)) / 2)
    assert accurate == fm.sum(array)


@pytest.mark.parametrize("ndims", [1, 2, 3, 4])
def test_dimensions(ndims):
    """Test summing multi-dimensional arrays."""
    N = 1_000_000

    shape = tuple([5] * (ndims - 1) + [-1])
    array = np.arange(N, dtype=np.float32).reshape(shape)

    accurate = np.float32(N * (N - 1) / 2)
    assert accurate == fm.sum(array)


@pytest.mark.parametrize("first_dimension", [1, 2, 5, 1_000])
def test_axis(first_dimension):
    """Test summing along an axis."""
    array = np.arange(2_000_000, dtype=np.float32).reshape((first_dimension, -1))

    accurate = np.array(
        [fsum(array[i]) for i in range(first_dimension)],
        dtype=array.dtype,
    )

    assert_array_almost_equal(
        accurate,
        fm.sum(array, axis=1),
    )


@pytest.mark.parametrize("length", [10, 1_000, 10_000])
def test_inf(length):
    """Test that sum handles infinity in the argument."""
    array = np.arange(length, dtype=np.float32)
    array[5] = np.inf

    assert fm.sum(array) == np.inf


@pytest.mark.parametrize("length", [10, 1_000, 10_000])
def test_neg_inf(length):
    """Test that sum handles negative infinity in the argument."""
    array = np.arange(length, dtype=np.float32)
    array[5] = -np.inf

    assert fm.sum(array) == -np.inf


@pytest.mark.parametrize("length", [10, 1_000, 10_000])
def test_nan(length):
    """Test that sum handles NaN in the argument."""
    array = np.ones(length, dtype=np.float32)
    array[5] = np.nan

    assert np.isnan(fm.sum(array))


@pytest.mark.parametrize("length", [10, 1_000, 10_000])
def test_mixed_inf(length):
    """Test that sum handles mixing positive and negative infinity."""
    array = np.ones(length, dtype=np.float32)
    array[3] = np.inf
    array[5] = -np.inf

    assert np.isnan(fm.sum(array))


@pytest.mark.parametrize("length", [10, 1_000, 10_000])
def test_mixed_nan(length):
    """Test that sum handles multiple NaN/infinities."""
    array = np.ones(length, dtype=np.float32)
    array[3] = np.nan
    array[5] = -np.inf

    assert np.isnan(fm.sum(array))


def test_overflow():
    """Test that summing can overflow into infinity."""
    array = np.array([2**126] * 8, dtype=np.float32)

    assert fm.sum(array) == np.inf


@given(seed=integers(min_value=0))
def test_ordering(seed):
    """Test randomly-ordered sequence of consecutive numbers."""
    N = 1_000_000
    array = np.arange(N, dtype=np.float32)

    rng = np.random.default_rng(seed)
    rng.shuffle(array)

    accurate = np.float32((N * (N - 1)) / 2)
    assert accurate == fm.sum(array)


@pytest.mark.filterwarnings("ignore:overflow encountered")
@pytest.mark.filterwarnings("ignore:invalid value encountered in reduce")
@given(arrays(dtype=np.float32, shape=(100,)))
def test_accuracy_small(array):
    """Fuzzing test for summing short arrays.

    For shorter arrays, summing may use algorithms with low overhead cost.
    This test focuses on those cases.
    """
    result = fm.sum(array)
    np_result = np.sum(array)

    # If both infinities are present the the result is NAN
    if np.inf in array and -np.inf in array:
        assert np.isnan(result)

    # Overflow could yield NAN
    elif np.inf in array or -np.inf in array:
        assert not np.isfinite(result)

    # Undefined if np.sum is not finite
    elif not np.isfinite(np_result):
        pass

    # Otherwise, compare to math.fsum
    else:
        fsum_result = np.float32(fsum(array))

        assume(np.isfinite(fsum_result))

        assert np.isfinite(result)

        # fsum can be inaccurate on the last bit
        if result != fsum_result:
            base = max(np.abs(result), np.abs(fsum_result))
            error_exp = int(np.floor(np.log2(base))) - 23

            assert np.abs(result - fsum_result) <= 2**error_exp


@pytest.mark.filterwarnings("ignore:overflow encountered")
@pytest.mark.filterwarnings("ignore:invalid value encountered in reduce")
@given(arrays(dtype=np.float32, shape=(100_000,)))
def test_accuracy_large(array):
    """Fuzzing test for summing long arrays.

    For larger arrays, summing may use algorithms with high overhead cost but
    low scaling costs. This test focuses on those cases.
    """
    result = fm.sum(array)
    np_result = np.sum(array)

    # If both infinities are present the the result is NAN
    if np.inf in array and -np.inf in array:
        assert np.isnan(result)

    # Overflow could yield NAN
    elif np.inf in array or -np.inf in array:
        assert not np.isfinite(result)

    # Undefined if np.sum is not finite
    elif not np.isfinite(np_result):
        pass

    # Otherwise, compare to math.fsum
    else:
        fsum_result = np.float32(fsum(array))

        assume(np.isfinite(fsum_result))

        assert np.isfinite(result)

        # fsum can be inaccurate on the last bit
        if result != fsum_result:
            base = max(np.abs(result), np.abs(fsum_result))
            error_exp = int(np.floor(np.log2(base))) - 23

            assert np.abs(result - fsum_result) <= 2**error_exp


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
