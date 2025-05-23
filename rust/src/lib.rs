use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod graph;
pub mod routing;

use crate::routing::micro_swap::micro_swap_boosted;

/// A Python module implemented in Rust.
#[pymodule]
fn microboost(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(micro_swap_boosted, m)?)?;
    m.add_function(wrap_pyfunction!(hello_sabre, m)?)?;
    m.add_class::<graph::dag::MicroDAG>()?;
    m.add_class::<graph::dag::MicroDAGNode>()?;
    m.add_class::<routing::sabre::MicroSABRE>()?;

    Ok(())
}

#[pyfunction]
fn hello_sabre(_py: Python, name: PyObject) {
    println!("Hello {}", name)
}
