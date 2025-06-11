use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod graph;
pub mod routing;

use crate::graph::dag::{MicroDAG, MicroDAGNode};
use crate::routing::{layout::MicroLayout, sabre::MicroSABRE};

/// A Python module implemented in Rust.
#[pymodule]
fn microboost(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_sabre, m)?)?;
    m.add_class::<MicroDAG>()?;
    m.add_class::<MicroDAGNode>()?;
    m.add_class::<MicroSABRE>()?;
    m.add_class::<MicroLayout>()?;

    Ok(())
}

#[pyfunction]
fn hello_sabre(_py: Python, name: PyObject) {
    println!("Hello {}", name)
}
