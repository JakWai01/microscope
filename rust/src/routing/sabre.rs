use std::collections::{HashMap, HashSet};
use crate::graph::dag::{MicroDAG, NodeId, NodeIndex, PhysicalQubit, VirtualQubit};

struct MicroSABRE {
    dag: MicroDAG,
    current_mapping: HashMap<VirtualQubit, PhysicalQubit>,
    coupling_map: Vec<(PhysicalQubit, PhysicalQubit)>,
    out_map: HashMap<NodeId, Vec<(VirtualQubit, VirtualQubit)>>,
    gate_order: Vec<NodeId>,
    front_layer: HashSet<NodeIndex>,
    required_predecessors: HashMap<NodeIndex, i32>,
    adjacency_list: HashMap<NodeIndex, Vec<NodeIndex>>
}

impl MicroSABRE {
    fn new(dag: MicroDAG, initial_mapping: HashMap<VirtualQubit, PhysicalQubit>, coupling_map: Vec<(PhysicalQubit, PhysicalQubit)>) -> Self {
        unimplemented!();
    }

    fn run(heuristic: String, critical_path: bool, extended_set_size: i32) {
        unimplemented!();
    }
}