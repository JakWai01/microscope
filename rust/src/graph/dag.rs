use std::collections::HashMap;

use pyo3::{pyclass, pymethods, types::PyAnyMethods, Bound, PyAny, PyRef, PyResult, Python};

#[derive(Clone)]
#[pyclass(module = "microboost.graph.dag")]
pub(crate) struct MicroDAG {
    nodes: HashMap<i32, MicroDAGNode>,
    edges: Vec<(i32, i32)>,
}

#[pymethods]
impl MicroDAG {
    #[new]
    pub fn new(nodes: HashMap<i32, MicroDAGNode>, edges: Vec<(i32, i32)>) -> PyResult<Self> {
        Ok(Self { nodes, edges })
    }

    #[getter]
    pub fn nodes(&self) -> PyResult<HashMap<i32, MicroDAGNode>> {
        Ok(self.nodes.clone())
    }

    #[getter]
    pub fn edges(&self) -> PyResult<Vec<(i32, i32)>> {
        Ok(self.edges.clone())
    }

    pub fn __len__(&self) -> PyResult<usize> {
        Ok(self.nodes.len())
    }

    pub fn get(&self, node_index: i32) -> PyResult<MicroDAGNode> {
        Ok(self.nodes.get(&node_index).unwrap().clone())
    }
}

#[derive(Debug, Clone)]
#[pyclass]
pub(crate) struct MicroDAGNode {
    id: i32,
    qubits: Vec<i32>,
}

#[pymethods]
impl MicroDAGNode {
    #[new]
    pub fn new(id: i32, qubits: Vec<i32>) -> PyResult<Self> {
        Ok(Self { id, qubits })
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
