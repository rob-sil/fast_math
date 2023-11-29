use ndarray::{Ix1, Ix2, IxDyn};
use pyo3::{exceptions::PyValueError, prelude::*};

use numpy::{dtype, Element, PyArrayDescr, PyReadonlyArray, PyReadonlyArrayDyn};

mod expansion;
mod accumulator;

const F32_EXPONENTS: usize = (f32::MAX_EXP - f32::MIN_EXP + 1) as usize;

fn sum_32_typed<'a, T>(array: &PyAny) -> PyResult<f32>
where
    T: Into<f32> + Element + Copy,
{
    let ndims = match PyReadonlyArrayDyn::<T>::extract(array) {
        Ok(a) => a.shape().len(),
        Err(_) => return Err(PyValueError::new_err("Only NumPy arrays are supported.")),
    };

    match ndims {
        1 => Ok(accumulator::online_sum::<_, T, 7, F32_EXPONENTS>(
            PyReadonlyArray::<T, Ix1>::extract(array)?.as_array(),
        )),
        2 => Ok(accumulator::online_sum::<_, T, 7, F32_EXPONENTS>(
            PyReadonlyArray::<T, Ix2>::extract(array)?.as_array(),
        )),
        _ => Ok(accumulator::online_sum::<_, T, 7, F32_EXPONENTS>(
            PyReadonlyArray::<T, IxDyn>::extract(array)?.as_array(),
        )),
    }
}

#[pyfunction]
fn sum_32(py: Python, array: &PyAny) -> PyResult<f32> {
    match array.getattr("dtype") {
        Ok(py_dtype) => {
            let descr = py_dtype.extract::<&PyArrayDescr>()?;
            if descr.is_equiv_to(dtype::<f32>(py)) {
                sum_32_typed::<f32>(array)
            } else if descr.is_equiv_to(dtype::<bool>(py)) {
                sum_32_typed::<bool>(array)
            } else if descr.is_equiv_to(dtype::<i8>(py)) {
                sum_32_typed::<i8>(array)
            } else if descr.is_equiv_to(dtype::<i16>(py)) {
                sum_32_typed::<i16>(array)
            } else if descr.is_equiv_to(dtype::<u8>(py)) {
                sum_32_typed::<u8>(array)
            } else if descr.is_equiv_to(dtype::<u16>(py)) {
                sum_32_typed::<u16>(array)
            } else if descr.is_equiv_to(dtype::<u16>(py)) {
                sum_32_typed::<u16>(array)
            } else {
                Err(PyValueError::new_err(format!(
                    "Cannot safely convert {} to a 32-bit float.",
                    descr,
                )))
            }
        }
        Err(_) => Err(PyValueError::new_err("No dtype provided.")),
    }
}

#[pymodule]
fn rust_fast_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_32, m)?)?;
    Ok(())
}
