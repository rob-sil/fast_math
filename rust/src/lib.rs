use std::{fmt::Display, mem};

use ndarray::{ArrayView, Dimension, IntoDimension, Ix1, Ix2, IxDyn, Shape, ShapeBuilder};
use numpy::{dtype, Element, PyArray, PyArrayDescr, PyReadonlyArray, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

mod accumulator;
mod expansion;
mod online_sum;

use expansion::Expansion;
use online_sum::OnlineSumAlgorithm;

const F32_EXPONENTS: usize = (f32::MAX_EXP - f32::MIN_EXP + 3) as usize;

// Zero-dimension return values are replaced with floats
enum SumReturn<'a> {
    Float(f32),
    Array(&'a PyAny),
}

/// Sum all the elements of an array into a single float.
///
/// This function chooses:
///
/// 1. Which summation algorithm to use. Zhu and Hayes' OnlineExactSum is amazingly
/// fast, but requires a large, fixed amount of memory to start. On smaller arrays,
/// overall performance is faster for one of Shewchuk's algorithms.
/// 2. Which `IntoIterator` implementation to use. If the array is contiguous, then
/// adding up elements of a slice is possible and faster than iterating over the view.
fn sum_full_array<'a, T, D>(array: PyReadonlyArray<'a, T, D>) -> PyResult<SumReturn>
where
    T: Into<f32> + Element + Copy + Display,
    D: Dimension,
{
    let value = if array.len() < 1024 {
        // Shewchuk's approach has low overhead and is fast for small arrays
        match array.as_slice() {
            Ok(s) => Expansion::online_sum(s),
            Err(_) => Expansion::online_sum(array.as_array()),
        }
    } else {
        // Zhu and Hayes' OnlineExactSum has higher overhead to start
        match array.as_slice() {
            Ok(s) => accumulator::online_sum::<_, T, 7, F32_EXPONENTS>(s),
            Err(_) => accumulator::online_sum::<_, T, 7, F32_EXPONENTS>(array.as_array()),
        }
    };
    Ok(SumReturn::Float(value))
}

/// Sum an array along an axis, filling the values of `out`.
fn sum_along_axis<'py, T, D>(
    array: PyReadonlyArray<'py, T, D>,
    axis: u32,
    out: &'py PyAny,
) -> PyResult<SumReturn<'py>>
where
    T: Into<f32> + Element + Copy + Display,
    D: Dimension,
{
    let out = out.downcast::<PyArray<f32, D::Smaller>>()?;

    let strides = array.strides();

    let mem_size = mem::size_of::<T>() as isize;

    let mut axis_stride = strides[axis as usize] / mem_size;
    let mut ptr = array.data() as *mut T;

    // Need to make sure axis_stride is positive for ArrayView construction
    if axis_stride < 0 {
        ptr = unsafe { ptr.offset(axis_stride * (array.shape()[axis as usize] as isize - 1)) };
        axis_stride = -axis_stride;
    }

    let mut out_view = unsafe { out.as_array_mut() };
    for (index, value) in out_view.indexed_iter_mut() {
        let mut offset = 0 as isize;
        for (mut ax, &index_ax) in index.into_dimension().slice().iter().enumerate() {
            if ax >= axis as usize {
                ax += 1
            }
            offset += (index_ax as isize) * strides[ax];
        }

        offset /= mem_size;

        let shape: Shape<Ix1> = array.shape()[axis as usize].into_dimension().into();
        let to_sum = unsafe {
            ArrayView::<T, Ix1>::from_shape_ptr(
                shape.strides(Ix1(axis_stride as usize)),
                ptr.offset(offset),
            )
        };
        *value = if to_sum.len() < 1024 {
            Expansion::online_sum(to_sum)
        } else {
            accumulator::online_sum::<_, _, 7, F32_EXPONENTS>(to_sum)
        };
    }

    Ok(SumReturn::Array(out))
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
///
/// This function also handles
fn sum_with_type<'py, T>(
    array: &'py PyAny,
    axis: Option<u32>,
    out: Option<&'py PyAny>,
) -> PyResult<SumReturn<'py>>
where
    T: Into<f32> + 'py + Element + Copy + Display,
{
    let ndim = PyReadonlyArrayDyn::<T>::extract(array)?.shape().len();

    if let Some(axis) = axis {
        if axis as usize >= ndim {
            return Err(PyValueError::new_err(format!(
                "Invalid axis {} for array of dimension {}",
                axis, ndim
            )));
        } else {
            let out = match out {
                Some(out) => out,
                None => return Err(PyValueError::new_err("axis was supplied but out was not.")),
            };

            match ndim {
                1 => sum_along_axis(PyReadonlyArray::<T, Ix1>::extract(array)?, axis, out),
                2 => sum_along_axis(PyReadonlyArray::<T, Ix2>::extract(array)?, axis, out),
                _ => sum_along_axis(PyReadonlyArray::<T, IxDyn>::extract(array)?, axis, out),
            }
        }
    } else {
        match ndim {
            1 => sum_full_array(PyReadonlyArray::<T, Ix1>::extract(array)?),
            2 => sum_full_array(PyReadonlyArray::<T, Ix2>::extract(array)?),
            _ => sum_full_array(PyReadonlyArray::<T, IxDyn>::extract(array)?),
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
fn sum_32<'py>(
    py: Python,
    array: &'py PyAny,
    axis: Option<u32>,
    out: Option<&'py PyAny>,
) -> PyResult<Py<PyAny>> {
    let result = match array.getattr("dtype") {
        Ok(py_dtype) => {
            let descr = py_dtype.extract::<&PyArrayDescr>()?;
            if descr.is_equiv_to(dtype::<f32>(py)) {
                sum_with_type::<f32>(array, axis, out)
            } else if descr.is_equiv_to(dtype::<bool>(py)) {
                sum_with_type::<bool>(array, axis, out)
            } else if descr.is_equiv_to(dtype::<i8>(py)) {
                sum_with_type::<i8>(array, axis, out)
            } else if descr.is_equiv_to(dtype::<i16>(py)) {
                sum_with_type::<i16>(array, axis, out)
            } else if descr.is_equiv_to(dtype::<u8>(py)) {
                sum_with_type::<u8>(array, axis, out)
            } else if descr.is_equiv_to(dtype::<u16>(py)) {
                sum_with_type::<u16>(array, axis, out)
            } else if descr.is_equiv_to(dtype::<u16>(py)) {
                sum_with_type::<u16>(array, axis, out)
            } else {
                Err(PyValueError::new_err(format!(
                    "Cannot safely convert {} to a 32-bit float.",
                    descr,
                )))
            }
        }
        Err(e) => Err(e),
    };

    match result? {
        SumReturn::Float(float_value) => Ok(float_value.to_object(py)),
        SumReturn::Array(array_value) => Ok(array_value.to_object(py)),
    }
}

#[pymodule]
fn rust_fast_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_32, m)?)?;
    Ok(())
}
