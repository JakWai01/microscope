use pyo3::prelude::*;
use pyo3::types::PyModule;

pub mod graph;
pub mod routing;

use crate::graph::dag::{MicroDAG, MicroDAGNode};
use crate::routing::multi_sabre::MultiSABRE;
use crate::routing::{layout::MicroLayout, sabre::MicroSABRE};

/// A Python module implemented in Rust.
#[pymodule]
fn microboost(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MicroDAG>()?;
    m.add_class::<MicroDAGNode>()?;
    m.add_class::<MicroSABRE>()?;
    m.add_class::<MicroLayout>()?;
    m.add_class::<MultiSABRE>()?;
    Ok(())
}