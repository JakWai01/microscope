use std::collections::HashMap;

use pyo3::{pyclass, pymethods, types::PyAnyMethods, Bound, PyAny, PyRef, PyResult, Python};

#[pyclass(module = "microboost.graph.dag")]
pub(crate) struct MicroDAG {
    nodes: HashMap<NodeIndex, MicroDAGNode>,
    edges: Vec<(VirtualQubit, VirtualQubit)>,
}

#[pymethods]
impl MicroDAG {
    #[new]
    pub fn new(nodes: HashMap<i32, PyRef<'_, MicroDAGNode >>, edges: Vec<(i32, i32)>) -> PyResult<Self> {
        // Currently working with https://pyo3.rs/v0.25.0/class.html
        println!("Creating MicroDAG in Rust");
    

        println!("Nodes: {:?}", nodes);
        println!("Edges: {:?}", edges);
        Ok(Self {
            nodes: HashMap::new(),
            edges: Vec::new()
        })
    }
}

#[derive(Debug)]
#[pyclass]
pub(crate) struct MicroDAGNode {
    id: i32,
    qubits: Vec<i32>
}

#[pymethods]
impl MicroDAGNode {
    #[new]
    pub fn new(id: i32, qubits: Vec<i32>) -> PyResult<Self> {
        Ok(Self {
            id,
            qubits
        })
    }

    #[getter]
    pub fn qubits(&self) -> PyResult<Vec<i32>> {
        Ok(self.qubits.clone())
    }

    #[getter]
    pub fn node_id(&self) -> PyResult<i32> {
        Ok(self.id)
    }
}

pub(crate) struct NodeIndex(i32);
pub(crate) struct NodeId(i32);
#[derive(Debug)]
pub(crate) struct VirtualQubit(i32);
pub(crate) struct PhysicalQubit(i32);