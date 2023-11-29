use pyo3::{exceptions::PyValueError, prelude::*};

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};

mod expansion;
mod sum;

#[pyfunction]
fn sum_32(array: &PyAny) -> PyResult<f32> {
    let ndims = match PyReadonlyArrayDyn::<f32>::extract(array) {
        Ok(a) => a.shape().len(),
        Err(_) => return Err(PyValueError::new_err("Only NumPy arrays are supported.")),
    };

    match ndims {
        1 => Ok(sum::online_sum::<_, 7>(
            PyReadonlyArray1::extract(array)?.as_array(),
        )),
        2 => Ok(sum::online_sum::<_, 7>(
            PyReadonlyArray2::extract(array)?.as_array(),
        )),
        _ => Ok(sum::online_sum::<_, 7>(
            PyReadonlyArrayDyn::extract(array)?.as_array(),
        )),
    }
}

#[pymodule]
fn rust_fast_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_32, m)?)?;
    Ok(())
}
