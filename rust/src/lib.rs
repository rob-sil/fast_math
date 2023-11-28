use pyo3::prelude::*;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};

mod expansion;
mod sum;

use sum::online_sum;

#[pyfunction]
fn sum_32_1d(x: PyReadonlyArray1<f32>) -> PyResult<f32> {
    Ok(online_sum::<_, 7>(x.as_array()))
}

#[pyfunction]
fn sum_32_2d(x: PyReadonlyArray2<f32>) -> PyResult<f32> {
    Ok(online_sum::<_, 7>(x.as_array()))
}

#[pyfunction]
// Significantly slower due to runtime-determined dimension
fn sum_32_dyn(x: PyReadonlyArrayDyn<f32>) -> PyResult<f32> {
    Ok(online_sum::<_, 7>(x.as_array()))
}

#[pymodule]
fn rust_fast_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_32_1d, m)?)?;
    m.add_function(wrap_pyfunction!(sum_32_2d, m)?)?;
    m.add_function(wrap_pyfunction!(sum_32_dyn, m)?)?;
    Ok(())
}
