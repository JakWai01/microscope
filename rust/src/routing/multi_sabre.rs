use std::{collections::{HashSet, VecDeque}, thread::current};

use crate::{graph::dag::MicroDAG, routing::{front_layer::MicroFront, layout::MicroLayout, sabre::get_successor_map_and_critical_paths, utils::{build_adjacency_list, compute_all_pairs_shortest_paths}}};
use pyo3::{pyclass, pymethods, PyResult};
use rustc_hash::FxHashMap;

#[pyclass(module = "microboost.routing.mutlisabre")]
pub struct MultiSABRE {
    dag: MicroDAG,
    coupling_map: Vec<Vec<i32>>,
    out_map: FxHashMap<i32, Vec<(i32, i32)>>,
    gate_order: Vec<i32>,
    required_predecessors: Vec<i32>,
    adjacency_list: FxHashMap<i32, Vec<i32>>,
    distance: Vec<Vec<i32>>,
    initial_mapping: MicroLayout,
    running_mapping: MicroLayout,
    successor_map: Vec<usize>,
    front_layer: MicroFront,
}

#[pymethods]
impl MultiSABRE {
    #[new]
    pub fn new(
        dag: MicroDAG,
        initial_mapping: MicroLayout,
        coupling_map: Vec<Vec<i32>>,
        num_qubits: i32,
    ) -> PyResult<Self> {
        let (successor_map, _) = get_successor_map_and_critical_paths(&dag);

        Ok(Self {
            required_predecessors: vec![0; dag.nodes.len()],
            adjacency_list: build_adjacency_list(&dag),
            dag,
            distance: compute_all_pairs_shortest_paths(&coupling_map),
            coupling_map,
            out_map: FxHashMap::default(),
            gate_order: Vec::new(),
            running_mapping: initial_mapping.clone(),
            initial_mapping: initial_mapping,
            successor_map,
            front_layer: MicroFront::new(num_qubits)
        })
    }

    fn run(
        &mut self,
        _layers: i32,
    ) {
        // Initialize required predecessors
        self.dag.edges().unwrap().iter().for_each(|edge| self.required_predecessors[edge.1 as usize] += 1);

        // Initialize front layer
        let initial_front = self.initial_front();

        // Advance front layer to first gates that cannot be executed
        self.advance_front_layer(&initial_front);

        let mut execute_gate_list: Vec<i32> = Vec::new();

        while !self.front_layer.is_empty() {
            let mut current_swaps: Vec<(i32, i32)> = Vec::new();

            while execute_gate_list.is_empty() {
                // Precompute two swap layers up front and return multiple results that will be applied
                let swaps = self.compute_swaps(2);

                for swap in swaps {
                    let q0 = swap.0;
                    let q1 = swap.1;

                    current_swaps.push(swap);
                    self.apply_swap((q0, q1));

                    if let Some(node) = self.executable_node_on_qubit(q0) {
                        execute_gate_list.push(node);
                    }

                    if let Some(node) = self.executable_node_on_qubit(q1) {
                        execute_gate_list.push(node);
                    }
                }
            }

            let node_id = self.dag.get(execute_gate_list[0]).unwrap().id;
            self.out_map.entry(node_id).or_default().extend(current_swaps);

            for &node in &execute_gate_list {
                self.front_layer.remove(&node);
            }
            
            self.advance_front_layer(&execute_gate_list);
            execute_gate_list.clear();
        }
    }
}

impl MultiSABRE {
    fn compute_swaps(&self, layers: i32) -> Vec<(i32, i32)> {
        unimplemented!()
    }

    fn apply_swap(&mut self, swap: (i32, i32)) {
        self.front_layer.apply_swap([swap.0, swap.1]);
        self.running_mapping.swap_physical(swap.0, swap.1);
    }   

    fn executable_node_on_qubit(&self, physical_qubit: i32) -> Option<i32> {
        for [a, b] in self.front_layer.nodes.values() {
            if *a == physical_qubit || *b == physical_qubit {
                if self.distance[*a as usize][*b as usize] == 1 {
                    return Some(self.front_layer.qubits[*a as usize].unwrap().0);
                }
            }
        }
        None
    }

    fn initial_front(&self) -> Vec<i32> {
        let mut nodes_with_predecessors: HashSet<i32> = HashSet::new();
        let all_nodes: HashSet<i32> = (0..self.dag.nodes.len() as i32).collect();

        for &(_, target) in &self.dag.edges {
            nodes_with_predecessors.insert(target);
        }

        all_nodes
            .difference(&nodes_with_predecessors)
            .cloned()
            .collect()
    }

     fn advance_front_layer(&mut self, nodes: &Vec<i32>) {
        let mut node_queue: VecDeque<i32> = VecDeque::from(nodes.clone());

        while let Some(node_index) = node_queue.pop_front() {
            let node = self.dag.get(node_index).unwrap();

            if node.qubits.len() == 2 {
                let physical_q0 = self.running_mapping.virtual_to_physical(node.qubits[0]);
                let physical_q1 = self.running_mapping.virtual_to_physical(node.qubits[1]);

                if self.distance[physical_q0 as usize][physical_q1 as usize] != 1 {
                    self.front_layer
                        .insert(node_index, [physical_q0, physical_q1]);
                    continue;
                }
            }

            if !self.gate_order.contains(&node.id) {
                self.gate_order.push(node.id);
            }

            if let Some(successors) = self.adjacency_list.get(&node_index) {
                for successor in successors {
                    if let Some(count) = self.required_predecessors.get_mut(*successor as usize) {
                        *count -= 1;
                        if *count == 0 {
                            node_queue.push_back(*successor);
                        }
                    }
                }
            }
        }
    }
}