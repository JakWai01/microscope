use crate::graph::dag::{MicroDAG, NodeId, NodeIndex, PhysicalQubit, VirtualQubit};
use std::collections::{HashMap, HashSet};

use pyo3::{pyclass, pymethods, PyResult};

#[pyclass(module="microboost.routing.sabre")]
pub(crate) struct MicroSABRE {
    dag: MicroDAG,
    current_mapping: HashMap<i32, i32>,
    coupling_map: Vec<Vec<i32>>,
    out_map: HashMap<i32, Vec<(i32, i32)>>,
    gate_order: Vec<i32>,
    front_layer: HashSet<i32>,
    required_predecessors: HashMap<i32, i32>,
    adjacency_list: HashMap<i32, Vec<i32>>,
}

#[pymethods]
impl MicroSABRE {
    #[new]
    pub fn new(
        dag: MicroDAG,
        initial_mapping: HashMap<i32, i32>,
        coupling_map: Vec<Vec<i32>>,
    ) -> PyResult<Self> {
        Ok(Self {
            dag,
            current_mapping: initial_mapping.clone(),
            coupling_map,
            // TODO: This was a defaultdict
            out_map: HashMap::new(),
            gate_order: Vec::new(),
            front_layer: HashSet::new(),
            // TODO: Those below shouldn't be empty
            required_predecessors: HashMap::new(),
            adjacency_list: HashMap::new()
        })
    }

    fn run(&self, heuristic: String, critical_path: bool, extended_set_size: i32) {
        unimplemented!();
    }
}
