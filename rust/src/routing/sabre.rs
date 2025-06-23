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
    extended_set_max: f64,
    successor_map: Vec<usize>,
    recent_swaps: VecDeque<(i32, i32)>,
    random_choices: i32,
    total_choices: i32,
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
            successor_map: get_successor_map(&dag),
            initial_dag: dag,
            neighbour_map: build_coupling_neighbour_map(&coupling_map),
            initial_coupling_map: coupling_map,
            num_qubits,
            extended_set_max: 0.0,
            recent_swaps: VecDeque::with_capacity(5),
            random_choices: 0,
            total_choices: 0
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
                if current_swaps.len() > 10000 {
                    panic!("We are stuck!")
                }
                let best_swap = self.choose_best_swap(heuristic, extended_set_size);

                if self.recent_swaps.len() == self.recent_swaps.capacity() {
                    self.recent_swaps.pop_back();
                }
                self.recent_swaps.push_front(best_swap);

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

            // Clearing recent swaps since we just want to prevent that swaps are reversed in a
            // single iteration. In general, there can't be two same swaps in a single pass
            self.recent_swaps.clear();
        }

        // println!("extended-set maximum: {:?}", self.extended_set_max);
        println!("Randomness: {:?}", self.random_choices as f64 / self.total_choices as f64);

        (
            std::mem::take(&mut self.out_map),
            std::mem::take(&mut self.gate_order),
        )
    }

    fn calculate_heuristic(&mut self, heuristic: &str, extended_set_size: i32) -> f64 {
        match heuristic {
            "basic" => self.h_basic("basic"),
            "basic_ln_1p" => self.h_basic("ln_1p"),
            "lookahead" => self.h_lookahead(0.5, extended_set_size, "basic", "basic"),
            "lookahead_ln_1p_basic" => self.h_lookahead(0.5, extended_set_size, "ln_1p", "basic"),
            "lookahead_basic_ln" => self.h_lookahead(0.5, extended_set_size, "basic", "ln"),
            "lookahead_ln_1p_ln" => self.h_lookahead(0.5, extended_set_size, "ln_1p", "ln"),
            _ => panic!("Unknown heuristic type: {}", heuristic),
        }
    }

    fn h_lookahead(
        &mut self,
        extended_set_weight: f64,
        extended_set_size: i32,
        critical_path_mode_basic: &str,
        critical_path_mode_extended: &str,
    ) -> f64 {
        // Compute heuristic for front layer
        let h_basic_result = self.h_basic(critical_path_mode_basic);

        // Determine extended set
        let extended_set = self.get_extended_set(extended_set_size);

        // Compute heuristic for extended set
        let h_basic_result_extended = self.h_extended(&extended_set, critical_path_mode_extended);

        let extended_len = extended_set.len().max(1) as f64;
        if self.extended_set_max < extended_len {
            self.extended_set_max = extended_len;
        }

        // Compute overall heuristic result
        h_basic_result + extended_set_weight * h_basic_result_extended
    }

    fn h_extended(&self, extended_set: &MicroFront, critical_path_mode: &str) -> f64 {
        extended_set
            .nodes
            .iter()
            .fold(0.0, |h_sum, (node_id, [a, b])| {
                let distance = self.distance[*a as usize][*b as usize];
                let num_successors = self.successor_map[*node_id as usize];

                let penalty_factor = match critical_path_mode {
                    "basic" => 1.0,
                    "ln" => 1.0 / (1.0 + (num_successors as f64).ln()),
                    _ => panic!("Invalid critical_path_mode for h_extended"),
                };

                h_sum + distance as f64 * penalty_factor
            })
    }

    fn h_basic(&self, critical_path_mode: &str) -> f64 {
        self.front_layer
            .nodes
            .iter()
            .fold(0.0, |h_sum, (node_id, [a, b])| {
                let distance = self.distance[*a as usize][*b as usize];
                let num_successors = self.successor_map[*node_id as usize];

                let penalty_factor = match critical_path_mode {
                    "basic" => 1.0,
                    "ln_1p" => 1.0 / (1.0 + (num_successors as f64).ln_1p()),
                    _ => panic!("Invalid critical_path_mode for h_basic"),
                };

                h_sum + distance as f64 * penalty_factor
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
            // while i < to_visit.len() {
            visit_now.push(to_visit[i]);
            let mut j = 0;

            while j < visit_now.len() {
                // if extended_set.len() > extended_set_size as usize {
                //     break
                // }
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

                            // if succ.qubits.len() == 2
                            //     && self.required_predecessors[successor as usize] == 1 {
                            //         let vq0 = succ.qubits[0];
                            //         let vq1 = succ.qubits[1];
                            //         let pq0 = self.layout.virtual_to_physical(vq0);
                            //         let pq1 = self.layout.virtual_to_physical(vq1);

                            //         // Only add if the physical distance is small enough
                            //         let distance = self.distance[pq0 as usize][pq1 as usize];
                            //         if distance <= 3 {
                            //             extended_set.insert(successor, [pq0, pq1]);
                            //         }
                            //     }
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

        // println!("Swap Candidates: {:?}", swap_candidates);

        for &(q0, q1) in &swap_candidates {
            let before: f64 = self.calculate_heuristic(heuristic, extended_set_size);

            self.apply_swap((q0, q1));

            let after = self.calculate_heuristic(heuristic, extended_set_size);

            self.apply_swap((q1, q0));

            // if (after - before).abs() < 1e-6 {
            //     continue // Skip neutral swaps
            // }

            if self.recent_swaps.contains(&(q0, q1)) || self.recent_swaps.contains(&(q1, q0)) {
                continue; // Skip recent swaps
            }

            scores.insert((q0, q1), after - before);
        }

        let (best_swap, random) = min_score(scores);

        // Calculate fraction of random choices
        if random {
            self.random_choices += 1;
        }
        self.total_choices += 1;

        best_swap
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

fn get_successor_map(dag: &MicroDAG) -> Vec<usize> {
    let adj = build_adjacency_list(dag);
    let mut successor_set: FxHashMap<i32, HashSet<i32>> =
        dag.nodes.keys().map(|&n| (n, HashSet::new())).collect();

    for u in (0..dag.nodes.len() as i32).rev() {
        if let Some(neighbors) = adj.get(&u) {
            for &v in neighbors {
                successor_set.get_mut(&u).unwrap().insert(v);
                if let Some(succ_v) = successor_set.get(&v) {
                    let succ_v_cloned = succ_v.clone();
                    successor_set.get_mut(&u).unwrap().extend(succ_v_cloned);
                }
            }
        }
    }

    dag.nodes.keys().map(|&n| successor_set[&n].len()).collect()
}
