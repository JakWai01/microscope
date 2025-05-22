use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod routing;
pub mod graph;

use crate::routing::micro_swap::micro_swap_boosted;

/// A Python module implemented in Rust.
#[pymodule]
fn microscope(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(micro_swap_boosted, m)?)?;
    m.add_function(wrap_pyfunction!(hello_sabre, m)?)?;
    Ok(())
}

#[pyfunction]
fn hello_sabre(_py: Python, name: PyObject) {
    println!("Hello {}", name)
}



