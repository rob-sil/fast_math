use std::fmt::Display;

use ndarray::{Axis, Dimension, Ix1, Ix2, Ix3, Ix4, IxDyn, RemoveAxis};
use numpy::{dtype, Element, PyArray, PyArrayDescr, PyReadonlyArray, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

mod accumulator;
mod array;
mod cumsum;
mod expansion;
mod float;
mod online_sum;

use accumulator::MultiAccumulator;
use array::map_axis;
use cumsum::cumsum_32;
use expansion::Expansion;
use online_sum::OnlineSumAlgorithm;

#[inline(always)]
fn online_sum<'a, I, T>(values: I, len: usize) -> f32
where
    I: IntoIterator,
    I::Item: Into<&'a T>,
    T: Into<f32> + 'a + Copy,
{
    if len < 1024 {
        // Shewchuk's approach has low overhead and is fast for small arrays
        Expansion::online_sum(values)
    } else {
        // Zhu and Hayes' OnlineExactSum has higher overhead to start
        MultiAccumulator::<8>::online_sum(values)
    }
}

/// Sum all the elements of an array into a single float.
///
/// This function checks whether the array is contiguous, in which case it can
/// be summed using its underlying slice. Using the slice is faster than using
/// the array iterator.
fn sum_full_array<'a, T, D>(py: Python, array: PyReadonlyArray<'a, T, D>) -> PyResult<Py<PyAny>>
where
    T: Into<f32> + Element + Copy + Display,
    D: Dimension,
{
    let value = match array.as_slice() {
        Ok(s) => online_sum(s, array.len()),
        Err(_) => online_sum(array.as_array(), array.len()),
    };
    Ok(value.to_object(py))
}

/// Sum an array along an axis, creating the result as a NumPy array.
fn sum_along_axis<'py, T, D>(
    py: Python,
    array: PyReadonlyArray<'py, T, D>,
    axis: usize,
) -> PyResult<Py<PyAny>>
where
    T: Into<f32> + Element + Copy + Display,
    D: Dimension + RemoveAxis,
{
    let reduced = map_axis(&array.as_array(), Axis(axis), |view| {
        online_sum(view, view.len())
    });

    Ok(PyArray::from_array(py, &reduced).to_object(py))
}

/// Sums an array of a known type into a 32-bit float result
///
/// This function has two responsibilities.
///
///  - First, it determines the dimension of the input array, which is enough
/// to represent the Python object as a Rust `PyReadonlyArray` (assuming
/// `array` is of type `T`).
///
///  - Second, this function splits by-axis summation from full-array
/// summation.
fn sum_with_type<'py, T>(py: Python, array: &'py PyAny, axis: Option<usize>) -> PyResult<Py<PyAny>>
where
    T: Into<f32> + 'py + Element + Copy + Display,
{
    let ndim = PyReadonlyArrayDyn::<T>::extract(array)?.shape().len();

    if let Some(axis) = axis {
        if axis >= ndim {
            return Err(PyValueError::new_err(format!(
                "Invalid axis {} for array of dimension {}",
                axis, ndim
            )));
        } else {
            match ndim {
                1 => sum_along_axis(py, PyReadonlyArray::<T, Ix1>::extract(array)?, axis),
                2 => sum_along_axis(py, PyReadonlyArray::<T, Ix2>::extract(array)?, axis),
                3 => sum_along_axis(py, PyReadonlyArray::<T, Ix3>::extract(array)?, axis),
                4 => sum_along_axis(py, PyReadonlyArray::<T, Ix4>::extract(array)?, axis),
                _ => sum_along_axis(py, PyReadonlyArray::<T, IxDyn>::extract(array)?, axis),
            }
        }
    } else {
        match ndim {
            1 => sum_full_array(py, PyReadonlyArray::<T, Ix1>::extract(array)?),
            2 => sum_full_array(py, PyReadonlyArray::<T, Ix2>::extract(array)?),
            3 => sum_full_array(py, PyReadonlyArray::<T, Ix3>::extract(array)?),
            4 => sum_full_array(py, PyReadonlyArray::<T, Ix4>::extract(array)?),
            _ => sum_full_array(py, PyReadonlyArray::<T, IxDyn>::extract(array)?),
        }
    }
}

/// Main function for summing 32-bit float data.
///
/// This function takes inputs from PyO3 and finds the type of the input
/// array `array`. Many valid NumPy numeric types cannot safely be cast
/// to 32-bit floats without possible loss (e.g., float64/f64 or int32/f32).
///
/// For valid array types, the sum moves on to the appropriate version of
/// `sum_with_type`.
#[pyfunction]
fn sum_32<'py>(py: Python, array: &'py PyAny, axis: Option<usize>) -> PyResult<Py<PyAny>> {
    match array.getattr("dtype") {
        Ok(py_dtype) => {
            let descr = py_dtype.extract::<&PyArrayDescr>()?;
            if descr.is_equiv_to(dtype::<f32>(py)) {
                sum_with_type::<f32>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<bool>(py)) {
                sum_with_type::<bool>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<i8>(py)) {
                sum_with_type::<i8>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<i16>(py)) {
                sum_with_type::<i16>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<u8>(py)) {
                sum_with_type::<u8>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<u16>(py)) {
                sum_with_type::<u16>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<u16>(py)) {
                sum_with_type::<u16>(py, array, axis)
            } else {
                Err(PyValueError::new_err(format!(
                    "Cannot safely convert {} to a 32-bit float.",
                    descr,
                )))
            }
        }
        Err(e) => Err(e),
    }
}

#[pymodule]
fn rust_fast_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_32, m)?)?;
    m.add_function(wrap_pyfunction!(cumsum_32, m)?)?;
    Ok(())
}
