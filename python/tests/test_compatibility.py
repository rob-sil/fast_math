import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import fast_math as fm


def test_dtype():
    # Test that the sum maintains the proper dtype
    array = np.arange(1_000_000, dtype=np.float32)

    assert type(fm.sum(array)) == array.dtype


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.float32, np.int8, np.int16, np.uint8, np.uint16]
)
def test_dtypes(dtype):
    # Test supported array dtypes
    array = np.ones(1_000_000, dtype=dtype)

    assert np.sum(array, dtype=np.float32) == fm.sum(array, dtype=np.float32)


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
    # Test unsupported array dtypes
    array = np.ones(1_000_000, dtype=dtype)

    with pytest.raises(ValueError):
        fm.sum(array, dtype=np.float32)


def test_axis():
    array = np.arange(10_000, dtype=np.float32).reshape((10, 20, -1))

    for axis in range(array.ndim):
        assert_array_almost_equal(
            np.sum(array, axis=axis),
            fm.sum(array, axis=axis),
        )


def test_axis_empty():
    array = np.array([], dtype=np.float32).reshape((-1, 5, 5))

    for axis in range(array.ndim):
        assert_array_almost_equal(
            np.sum(array, axis=axis),
            fm.sum(array, axis=axis),
        )
