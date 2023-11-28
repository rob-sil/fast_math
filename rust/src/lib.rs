use pyo3::prelude::*;

use numpy::PyReadonlyArray1;

mod expansion;
use expansion::Expansion;

#[pyfunction]
fn sum_32(x: PyReadonlyArray1<f32>) -> PyResult<f32> {
	let mut expansion = Expansion::new();
    for &value in x.as_array() {
        expansion.add(value)
    }
    Ok(expansion.into())
}

#[pymodule]
fn rust_fast_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_32, m)?)?;
    Ok(())
}
