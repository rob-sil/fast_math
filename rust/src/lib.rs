use pyo3::prelude::*;

use numpy::PyReadonlyArray1;

/// Add two numbers with the Fast2Sum algorithm
///
/// Returns the sum along with the rounding error.
fn fast2sum(a: f32, b: f32) -> (f32, f32) {
    if a.abs() < b.abs() {
        let high = b + a;
        let low = a - (high - b);
        (high, low)
    } else {
        let high = a + b;
        let low = b - (high - a);
        (high, low)
    }
}

/// Add a value to a list of floats.
///
/// Implements Shewchuk's Grow-Expansion algorithm.
fn grow_expansion(h: &mut Vec<f32>, value: f32) {
    let mut current = value;
    let mut j = 0;
    for i in 0..h.len() {
        let (high, low) = fast2sum(current, h[i]);
        current = high;
        if low != 0_f32 {
            h[j] = low;
            j += 1;
        }
    }
    h.truncate(j);
    if current != 0_f32 {
        h.push(current);
    }
}

#[pyfunction]
fn sum_32(x: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let mut h: Vec<f32> = vec![];
    for &value in x.as_array() {
        grow_expansion(&mut h, value);
    }
    Ok(h[h.len() - 1])
}

#[pymodule]
fn rust_fast_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_32, m)?)?;
    Ok(())
}
