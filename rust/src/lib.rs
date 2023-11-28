use pyo3::prelude::*;

use numpy::PyReadonlyArray1;

mod expansion;
mod sum;

use sum::online_sum;

#[pyfunction]
fn sum_32(x: PyReadonlyArray1<f32>) -> PyResult<f32> {
	Ok(online_sum::<_, 7>(x.as_array()))
}

#[pymodule]
fn rust_fast_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_32, m)?)?;
    Ok(())
}
