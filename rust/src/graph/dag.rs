use std::collections::HashMap;

pub(crate) struct MicroDAG {
    nodes: HashMap<NodeIndex, MicroDAGNode>,
    edges: Vec<(VirtualQubit, VirtualQubit)>,
    last_op_on_qubit: HashMap<VirtualQubit, NodeIndex>
}

struct MicroDAGNode {
    id: i32,
    qubits: Vec<VirtualQubit>
}

pub(crate) struct NodeIndex(i32);
pub(crate) struct NodeId(i32);
pub(crate) struct VirtualQubit(i32);
pub(crate) struct PhysicalQubit(i32);