from math import fsum

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_array_almost_equal

import fast_math as fm


def test_dtype():
    """Test that cumsum maintains the proper dtype."""
    array = np.arange(1_000_000, dtype=np.float32)

    assert fm.cumsum(array).dtype == array.dtype


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.float32, np.int8, np.int16, np.uint8, np.uint16]
)
def test_dtypes(dtype):
    """Test that cumsum can handle all supported array dtypes."""
    array = np.ones(1_000_000, dtype=dtype)

    assert fm.sum(array, dtype=np.float32) == fm.cumsum(array)[-1]


@pytest.mark.parametrize(
    "dtype",
    [
        np.int32,
        np.int64,
        np.uint32,
        np.uint64,
        np.float64,
    ],
)
def test_dtypes_unsafe(dtype):
    """Test that cumsum raises on unsupported dtypes."""
    array = np.ones(1_000_000, dtype=dtype)

    with pytest.raises(ValueError):
        fm.cumsum(array)


@pytest.mark.parametrize("size", [10, 100, 1000])
def test_small(size):
    """Test cumulative sums where NumPy is accurate."""
    array = np.arange(size, dtype=np.float32)

    assert_array_almost_equal(np.cumsum(array), fm.cumsum(array))


@pytest.mark.parametrize("size", [1_000, 10_000, 100_000])
def test_big(size):
    """Test cumulative sums reach the proper sum for large numbers."""
    array = np.arange(size, dtype=np.float32)

    assert fm.cumsum(array)[-1] == np.float32(fsum(array))


@pytest.mark.parametrize("size", [10, 100, 1_000, 10_000])
def test_axis_small(size):
    """Test cumulative sums along an axis.

    The input data is simple enough that NumPy's cumsum will be accurate.
    """
    array = np.arange(size, dtype=np.float32).reshape((10, -1))

    for axis in range(array.ndim):
        assert_array_almost_equal(
            np.cumsum(array, axis=axis),
            fm.cumsum(array, axis=axis),
        )


def test_axis_empty():
    """Test cumsum over the axes of an empty array.

    Behavior should match np.sum for compatibility.
    """
    array = np.array([], dtype=np.float32).reshape((-1, 5, 5))

    for axis in range(array.ndim):
        assert_array_almost_equal(
            np.cumsum(array, axis=axis),
            fm.cumsum(array, axis=axis),
        )


def test_inf():
    """Test that cumsum handles infinity in the argument."""
    array = np.array([1, 2, np.inf, 4], dtype=np.float32)

    accurate = np.cumsum(array)
    result = fm.cumsum(array)

    assert_array_almost_equal(accurate, result)


def test_neg_inf():
    """Test that cumsum handles negative infinity in the argument."""
    array = np.array([1, 2, -np.inf, 4], dtype=np.float32)

    accurate = np.cumsum(array)
    result = fm.cumsum(array)

    assert_array_almost_equal(accurate, result)


def test_nan():
    """Test that cumsum handles NaN in the argument."""
    array = np.array([1, 2, np.nan, 4], dtype=np.float32)

    accurate = np.cumsum(array)
    result = fm.cumsum(array)

    assert_array_almost_equal(accurate, result)


@pytest.mark.filterwarnings("ignore:invalid value encountered in accumulate")
def test_mixed_inf():
    """Test that cumsum handles mixing positive and negative infinity."""
    array = np.array([1, 2, -np.inf, 5, np.inf, 7], dtype=np.float32)

    accurate = np.cumsum(array)
    result = fm.cumsum(array)

    assert_array_almost_equal(accurate, result)


def test_mixed_nan():
    """Test that cumsum handles multiple NaN/infinities."""
    array = np.array([1, 2, -np.inf, 5, np.nan, 7], dtype=np.float32)

    accurate = np.cumsum(array)
    result = fm.cumsum(array)

    assert_array_almost_equal(accurate, result)


@pytest.mark.filterwarnings("ignore:overflow encountered")
def test_overflow():
    """Test that summing can overflow into infinity"""
    array = np.array([2**126] * 8, dtype=np.float32)

    accurate = np.cumsum(array)
    result = fm.cumsum(array)

    assert_array_almost_equal(accurate, result)


@given(arrays(dtype=np.float32, shape=(10_000,)))
def test_accuracy(array):
    """Fuzzing test for cumsum.

    Test against fast_math.sum.
    """
    result = fm.cumsum(array)

    accurate = np.array(
        [fm.sum(array[: length + 1]) for length in range(len(array))], dtype=np.float32
    )

    assume(np.isfinite(accurate[-1]))

    assert (accurate == result).all()
