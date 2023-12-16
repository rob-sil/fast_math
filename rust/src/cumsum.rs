use std::fmt::Display;

use ndarray::{
    Array, ArrayView1, Axis, Dim, Dimension, Ix, Ix1, Ix2, Ix3, Ix4, IxDyn, RemoveAxis, Shape,
    ShapeBuilder,
};
use numpy::{dtype, Element, PyArray, PyArrayDescr, PyReadonlyArray};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::expansion::Expansion;
use crate::online_sum::OnlineSumAlgorithm;

/// Calculate the cumulative sum, flattening the array
fn cumsum_flattened<'py, T, D>(py: Python, array: PyReadonlyArray<'py, T, D>) -> PyResult<Py<PyAny>>
where
    T: Into<f32> + Element + Copy + Display,
    D: Dimension,
{
    let mut expansion = Expansion::new();

    let array = array.as_array();
    let results_iter = array.iter().map(|&value| {
        expansion.add(value.into());
        expansion.round()
    });

    let out = PyArray::from_iter(py, results_iter);
    Ok(out.to_object(py))
}

/// Calculate the cumulative sum along an axis
fn cumsum_along_axis<'py, T, D>(
    py: Python,
    array: PyReadonlyArray<'py, T, D>,
    axis: Axis,
) -> PyResult<Py<PyAny>>
where
    T: Into<f32> + Element + Copy + Display,
    D: Dimension + RemoveAxis,
{
    let array = array.as_array();

    if array.len() == 0 {
        return Ok(PyArray::from_array(py, &array).to_object(py));
    }

    let axis_len = array.len_of(axis);

    let shape: Shape<_> = Dim::<Ix>(axis_len).into();
    let stride = shape.strides(Ix1(array.stride_of(axis) as usize));

    let results_vec: Vec<f32> = array
        .index_axis(axis, 0)
        .iter()
        .flat_map(|ptr| {
            let axis_array = unsafe { ArrayView1::from_shape_ptr(stride, ptr) };
            let mut expansion = Expansion::new();
            axis_array
                .iter()
                .map(|&value| {
                    expansion.add(value.into());
                    expansion.round()
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let dim = array.raw_dim();

    let mut strides: Vec<usize> = dim
        .remove_axis(axis)
        .default_strides()
        .slice()
        .iter()
        .map(|v| axis_len * v)
        .collect();
    strides.insert(axis.0, 1);

    let mut dim_strides = dim.default_strides();
    for i in 0..strides.len() {
        dim_strides[i] = strides[i];
    }

    let out = Array::from_shape_vec(dim.strides(dim_strides), results_vec).unwrap();
    Ok(PyArray::from_array(py, &out).to_object(py))
}

/// Process the array into an PyReadonlyArray
fn cumsum_with_type<'py, T>(
    py: Python,
    array: &'py PyAny,
    axis: Option<usize>,
) -> PyResult<Py<PyAny>>
where
    T: Into<f32> + 'py + Element + Copy + Display,
{
    let ndim = PyReadonlyArray::<T, IxDyn>::extract(array)?.shape().len();

    if let Some(axis) = axis {
        if axis >= ndim {
            return Err(PyValueError::new_err(format!(
                "Invalid axis {} for array of dimension {}",
                axis, ndim
            )));
        } else {
            match ndim {
                1 => cumsum_along_axis(py, PyReadonlyArray::<T, Ix1>::extract(array)?, Axis(axis)),
                2 => cumsum_along_axis(py, PyReadonlyArray::<T, Ix2>::extract(array)?, Axis(axis)),
                3 => cumsum_along_axis(py, PyReadonlyArray::<T, Ix3>::extract(array)?, Axis(axis)),
                4 => cumsum_along_axis(py, PyReadonlyArray::<T, Ix4>::extract(array)?, Axis(axis)),
                _ => {
                    cumsum_along_axis(py, PyReadonlyArray::<T, IxDyn>::extract(array)?, Axis(axis))
                }
            }
        }
    } else {
        match ndim {
            1 => cumsum_flattened(py, PyReadonlyArray::<T, Ix1>::extract(array)?),
            2 => cumsum_flattened(py, PyReadonlyArray::<T, Ix2>::extract(array)?),
            3 => cumsum_flattened(py, PyReadonlyArray::<T, Ix3>::extract(array)?),
            4 => cumsum_flattened(py, PyReadonlyArray::<T, Ix4>::extract(array)?),
            _ => cumsum_flattened(py, PyReadonlyArray::<T, IxDyn>::extract(array)?),
        }
    }
}

/// Main function for cumulative sums of 32-bit float data.
///
/// This function takes inputs from PyO3 and finds the type of the input
/// array `array`. Many valid NumPy numeric types cannot safely be cast
/// to 32-bit floats without possible loss (e.g., float64/f64 or int32/f32).
///
/// For valid array types, cumsum moves on to the appropriate version of
/// `cumsum_with_type`.
#[pyfunction]
pub(crate) fn cumsum_32<'py>(
    py: Python,
    array: &'py PyAny,
    axis: Option<usize>,
) -> PyResult<Py<PyAny>> {
    match array.getattr("dtype") {
        Ok(py_dtype) => {
            let descr = py_dtype.extract::<&PyArrayDescr>()?;
            if descr.is_equiv_to(dtype::<f32>(py)) {
                cumsum_with_type::<f32>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<bool>(py)) {
                cumsum_with_type::<bool>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<i8>(py)) {
                cumsum_with_type::<i8>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<i16>(py)) {
                cumsum_with_type::<i16>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<u8>(py)) {
                cumsum_with_type::<u8>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<u16>(py)) {
                cumsum_with_type::<u16>(py, array, axis)
            } else if descr.is_equiv_to(dtype::<u16>(py)) {
                cumsum_with_type::<u16>(py, array, axis)
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
