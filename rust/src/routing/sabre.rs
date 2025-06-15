use crate::routing::front_layer::MicroFront;
use crate::routing::layout::MicroLayout;
use crate::routing::utils::{
    build_coupling_neighbour_map, compute_all_pairs_shortest_paths, min_score,
};
use crate::{graph::dag::MicroDAG, routing::utils::build_adjacency_list};
use std::collections::{HashSet, VecDeque};

use pyo3::{pyclass, pymethods, PyResult};

use rustc_hash::FxHashMap;

#[pyclass(module = "microboost.routing.sabre")]
pub struct MicroSABRE {
    dag: MicroDAG,
    coupling_map: Vec<Vec<i32>>,
    out_map: FxHashMap<i32, Vec<(i32, i32)>>,
    gate_order: Vec<i32>,
    front_layer: MicroFront,
    required_predecessors: Vec<i32>,
    adjacency_list: FxHashMap<i32, Vec<i32>>,
    distance: Vec<Vec<i32>>,
    initial_mapping: MicroLayout,
    initial_dag: MicroDAG,
    initial_coupling_map: Vec<Vec<i32>>,
    neighbour_map: FxHashMap<i32, Vec<i32>>,
    layout: MicroLayout,
    num_qubits: i32,
}

#[pymethods]
impl MicroSABRE {
    #[new]
    pub fn new(
        dag: MicroDAG,
        initial_layout: MicroLayout,
        coupling_map: Vec<Vec<i32>>,
        num_qubits: i32,
    ) -> PyResult<Self> {
        Ok(Self {
            required_predecessors: vec![0; dag.nodes.len()],
            adjacency_list: build_adjacency_list(&dag),
            dag: dag.clone(),
            layout: initial_layout.clone(),
            distance: compute_all_pairs_shortest_paths(&coupling_map),
            coupling_map: coupling_map.clone(),
            out_map: FxHashMap::default(),
            gate_order: Vec::new(),
            front_layer: MicroFront::new(num_qubits),
            initial_mapping: initial_layout.clone(),
            initial_dag: dag,
            neighbour_map: build_coupling_neighbour_map(&coupling_map),
            initial_coupling_map: coupling_map,
            num_qubits,
        })
    }

    // Maybe it would make sense to also maintain an extended set and apply swaps there
    fn apply_swap(&mut self, swap: (i32, i32)) {
        self.front_layer.apply_swap([swap.0, swap.1]);
        self.layout.swap_physical(swap.0, swap.1);
    }

    fn clear_data_structures(&mut self) {
        // In theory, this should always be zero in the end (so we could skip it - only if everything goes well)
        self.required_predecessors = vec![0; self.dag.nodes.len()];
        self.adjacency_list = build_adjacency_list(&self.dag);

        self.layout = self.initial_mapping.clone();

        self.out_map.clear();
        self.gate_order.clear();
        self.front_layer = MicroFront::new(self.num_qubits);

        self.dag = self.initial_dag.clone();
        self.coupling_map = self.initial_coupling_map.clone();
    }

    fn run(
        &mut self,
        heuristic: &str,
        _critical_path: bool,
        extended_set_size: i32,
    ) -> (FxHashMap<i32, Vec<(i32, i32)>>, Vec<i32>) {
        self.clear_data_structures();
        self.dag
            .edges()
            .unwrap()
            .iter()
            .for_each(|edge| self.required_predecessors[edge.1 as usize] += 1);
        let initial_front = self.initial_front();
        self.advance_front_layer(&initial_front);

        let mut execute_gate_list: Vec<i32> = Vec::new();

        while !self.front_layer.is_empty() {
            let mut current_swaps: Vec<(i32, i32)> = Vec::new();

            while execute_gate_list.is_empty() {
                if current_swaps.len() > 1000 {
                    panic!("We are stuck!")
                }
                let best_swap = self.choose_best_swap(heuristic, extended_set_size);

                let physical_q0 = best_swap.0;
                let physical_q1 = best_swap.1;

                current_swaps.push(best_swap);
                self.apply_swap((physical_q0, physical_q1));

                if let Some(node) = self.executable_node_on_qubit(physical_q0) {
                    execute_gate_list.push(node);
                }

                if let Some(node) = self.executable_node_on_qubit(physical_q1) {
                    execute_gate_list.push(node);
                }
            }

            let node_id = self.dag.get(execute_gate_list[0]).unwrap().id;
            self.out_map
                .entry(node_id)
                .or_default()
                .extend(current_swaps);

            for &node in &execute_gate_list {
                self.front_layer.remove(&node);
            }
            self.advance_front_layer(&execute_gate_list);
            execute_gate_list.clear();
        }
        (
            std::mem::take(&mut self.out_map),
            std::mem::take(&mut self.gate_order),
        )
    }
    fn calculate_heuristic(
        &mut self,
        front_layer: Option<MicroFront>,
        heuristic: &str,
        extended_set_size: i32,
    ) -> f64 {
        match heuristic {
            "basic" => self.h_basic(front_layer, 1.0),
            "basic-scale" => self.h_basic(front_layer, 1.0),
            "lookahead" => self.h_lookahead(front_layer, 1.0, extended_set_size),
            "lookahead-0.5" => self.h_lookahead(front_layer, 0.5, extended_set_size),
            "lookahead-scaling" => self.h_lookahead(front_layer, 1.0, extended_set_size),
            "lookahead-0.5-scaling" => self.h_lookahead(front_layer, 1.0, extended_set_size),
            _ => panic!("Unknown heuristic type: {}", heuristic),
        }
    }

    fn h_lookahead(
        &mut self,
        front_layer: Option<MicroFront>,
        weight: f64,
        extended_set_size: i32,
    ) -> f64 {
        let nodes = front_layer
            .as_ref()
            .unwrap_or(&self.front_layer)
            .nodes
            .values();

        let front_len = nodes.len() as f64;
        if front_len == 0.0 {
            return 0.0;
        }

        let h_basic_result = self.h_basic(None, 1.0);
        let extended_set = self.get_extended_set(extended_set_size);
        let extended_len = extended_set.len();
        let h_basic_result_extended = self.h_basic(Some(extended_set), 1.0);

        let extended_len = extended_len.max(1) as f64;

        (1.0 / front_len) * h_basic_result + weight * (1.0 / extended_len) * h_basic_result_extended
    }

    fn h_basic(&self, front_layer: Option<MicroFront>, weight: f64) -> f64 {
        let nodes = front_layer
            .as_ref()
            .unwrap_or(&self.front_layer)
            .nodes
            .values();

        nodes.fold(0.0, |h_sum, [a, b]| {
            let distance = self.distance[*a as usize][*b as usize];
            h_sum + weight * distance as f64
        })
    }

    fn get_extended_set(&mut self, extended_set_size: i32) -> MicroFront {
        let mut to_visit: Vec<i32> = self.front_layer.nodes.keys().copied().collect();
        let mut i = 0;

        let mut extended_set: MicroFront = MicroFront::new(self.num_qubits);
        let mut visit_now: Vec<i32> = Vec::new();

        let dag_size = self.dag.nodes.len();

        let mut decremented = vec![0; dag_size];

        let mut visited = vec![false; dag_size];

        while i < to_visit.len() && extended_set.len() < extended_set_size as usize {
            visit_now.push(to_visit[i]);
            let mut j = 0;

            while j < visit_now.len() {
                let node_id = visit_now[j];

                if let Some(successors) = self.adjacency_list.get(&node_id) {
                    for &successor in successors {
                        if !visited[successor as usize] {
                            let succ = self.dag.get(successor).unwrap();
                            visited[successor as usize] = true;

                            decremented[successor as usize] += 1;
                            self.required_predecessors[successor as usize] -= 1;

                            if self.required_predecessors[successor as usize] == 0 {
                                if succ.qubits.len() == 2 {
                                    let physical_q0 =
                                        self.layout.virtual_to_physical(succ.qubits[0]);
                                    let physical_q1 =
                                        self.layout.virtual_to_physical(succ.qubits[1]);
                                    extended_set.insert(successor, [physical_q0, physical_q1]);
                                    to_visit.push(successor);
                                    continue;
                                }
                                visit_now.push(successor);
                            }

                            // Also adding the first layer of unroutable gates seems to improve results
                            // Interesting observation: It gets stuck without it
                            if succ.qubits.len() == 2 {
                                let physical_q0 = self.layout.virtual_to_physical(succ.qubits[0]);
                                let physical_q1 = self.layout.virtual_to_physical(succ.qubits[1]);
                                extended_set.insert(successor, [physical_q0, physical_q1]);
                            }
                        }
                    }
                }
                j += 1;
            }

            visit_now.clear();
            i += 1;
        }

        for (index, amount) in decremented.iter().enumerate() {
            self.required_predecessors[index] += amount;
        }
        extended_set
    }

    fn choose_best_swap(&mut self, heuristic: &str, extended_set_size: i32) -> (i32, i32) {
        let mut scores: FxHashMap<(i32, i32), f64> = FxHashMap::default();

        let swap_candidates: Vec<(i32, i32)> = self.compute_swap_candidates();

        for &(q0, q1) in &swap_candidates {
            let before = self.calculate_heuristic(None, heuristic, extended_set_size);

            self.apply_swap((q0, q1));

            let after = self.calculate_heuristic(None, heuristic, extended_set_size);

            self.apply_swap((q1, q0));

            scores.insert((q0, q1), after - before);
        }
        min_score(scores)
    }

    fn compute_swap_candidates(&self) -> Vec<(i32, i32)> {
        let mut swap_candidates: Vec<(i32, i32)> = Vec::new();

        for &phys in self.front_layer.nodes.values().flatten() {
            for neighbour in self.neighbour_map[&phys].iter() {
                if neighbour > &phys || !self.front_layer.is_active(*neighbour) {
                    swap_candidates.push((phys, *neighbour))
                }
            }
        }
        swap_candidates
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
}

impl MicroSABRE {
    fn advance_front_layer(&mut self, nodes: &Vec<i32>) {
        let mut node_queue: VecDeque<i32> = VecDeque::from(nodes.clone());

        while let Some(node_index) = node_queue.pop_front() {
            let node = self.dag.get(node_index).unwrap();

            if node.qubits.len() == 2 {
                let physical_q0 = self.layout.virtual_to_physical(node.qubits[0]);
                let physical_q1 = self.layout.virtual_to_physical(node.qubits[1]);

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
