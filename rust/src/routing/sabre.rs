use crate::routing::front_layer::MicroFront;
use crate::routing::layout::MicroLayout;
use crate::routing::utils::{
    build_coupling_neighbour_map, compute_all_pairs_shortest_paths, min_score,
};
use crate::{graph::dag::MicroDAG, routing::utils::build_adjacency_list};
use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};

use pyo3::{pyclass, pymethods, PyResult};

use rustc_hash::FxHashMap;
use rustworkx_core::dictmap::{DictMap, InitWithHasher};
use rustworkx_core::petgraph::csr::IndexType;
use rustworkx_core::petgraph::graph::{DiGraph, NodeIndex};
use rustworkx_core::shortest_path::dijkstra;

#[pyclass(module = "microboost.routing.sabre")]
pub struct MicroSABRE {
    dag: MicroDAG,
    coupling_map: Vec<Vec<i32>>,
    out_map: FxHashMap<i32, Vec<[i32; 2]>>,
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
    fn apply_swap(&mut self, swap: [i32; 2]) {
        self.front_layer.apply_swap(swap);
        self.layout.swap_physical(swap);
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

    fn run(&mut self) -> (FxHashMap<i32, Vec<[i32; 2]>>, Vec<i32>) {
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
            let mut current_swaps: Vec<[i32; 2]> = Vec::new();

            while execute_gate_list.is_empty() && current_swaps.len() <= 10000 {
                let swaps = self.choose_best_swaps(2);

                for swap in swaps {
                    let q0 = swap[0];
                    let q1 = swap[1];

                    current_swaps.push([q0, q1]);
                    self.apply_swap([q0, q1]);

                    if let Some(node) = self.executable_node_on_qubit(q0) {
                        execute_gate_list.push(node);
                    }

                    if let Some(node) = self.executable_node_on_qubit(q1) {
                        execute_gate_list.push(node);
                    }
                    
                    if !execute_gate_list.is_empty() {
                        let node_id = self.dag.get(execute_gate_list[0]).unwrap().id;
                        self.out_map
                            .entry(node_id)
                            .or_default()
                            .extend(current_swaps.clone());

                        for &node in &execute_gate_list {
                            self.front_layer.remove(&node);
                        }

                        self.advance_front_layer(&execute_gate_list);

                        if self.front_layer.is_empty() {
                            return (
                                std::mem::take(&mut self.out_map),
                                std::mem::take(&mut self.gate_order),
                            )
                        }
                        execute_gate_list.clear();
                        current_swaps.clear();
                    }
                }
            }

            if execute_gate_list.is_empty() {
                current_swaps
                    .drain(..)
                    .rev()
                    .for_each(|swap| self.apply_swap(swap));
                let force_routed = self.release_valve(&mut current_swaps);
                execute_gate_list.extend(force_routed);
                
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

        }

        (
            std::mem::take(&mut self.out_map),
            std::mem::take(&mut self.gate_order),
        )
    }

    fn calculate_heuristic(&mut self) -> f64 {
        let extended_set = self.get_extended_set();

        let basic = self
            .front_layer
            .nodes
            .iter()
            .fold(0.0, |h_sum, (_node_id, [a, b])| {
                h_sum + self.distance[*a as usize][*b as usize] as f64
            });

        let lookahead = extended_set
            .nodes
            .iter()
            .fold(0.0, |h_sum, (_node_id, [a, b])| {
                h_sum + self.distance[*a as usize][*b as usize] as f64
            });

        // (1. / num_executable_gates as f64) * (basic + 0.5 * lookahead)
        basic + 0.5 * lookahead
    }

    fn get_extended_set(&mut self) -> MicroFront {
        let mut required_predecessors = self.required_predecessors.clone();

        let mut to_visit: Vec<i32> = self.front_layer.nodes.keys().copied().collect();
        let mut i = 0;

        let mut extended_set: MicroFront = MicroFront::new(self.num_qubits);
        let mut visit_now: Vec<i32> = Vec::new();

        let dag_size = self.dag.nodes.len();

        let mut decremented = vec![0; dag_size];

        let mut visited = vec![false; dag_size];

        while i < to_visit.len() && extended_set.len() < 20 as usize {
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
                            required_predecessors[successor as usize] -= 1;

                            if required_predecessors[successor as usize] == 0 {
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
                        }
                    }
                }
                j += 1;
            }

            visit_now.clear();
            i += 1;
        }

        for (index, amount) in decremented.iter().enumerate() {
            required_predecessors[index] += amount;
        }
        extended_set
    }

    fn choose_best_swap(&mut self) -> Vec<[i32; 2]> {
        let mut scores: FxHashMap<Vec<[i32; 2]>, f64> = FxHashMap::default();

        let swap_candidates: Vec<[i32; 2]> = self.compute_swap_candidates();

        for &[q0, q1] in &swap_candidates {
            // TODO: Isn't before always the after from the previous iteration?
            let before = self.calculate_heuristic();

            self.apply_swap([q0, q1]);

            let after = self.calculate_heuristic();

            self.apply_swap([q1, q0]);

            scores.insert(vec![[q0, q1]], after - before);
        }

        min_score(scores)
    }

    fn choose_best_swaps(&mut self, depth: usize) -> Vec<[i32; 2]> {
        let initial_state = self.create_snapshot();

        let mut stack = Vec::new();
        let mut scores: FxHashMap<Vec<[i32; 2]>, f64> = FxHashMap::default();

        stack.push(StackItem {
            state: self.create_snapshot(),
            swap_sequence: Vec::new(),
            score: 0.0,
            current_depth: depth,
        });

        while let Some(item) = stack.pop() {
            if item.current_depth == 0 {
                scores.insert(item.swap_sequence.clone(), item.score);
                continue;
            }

            let state = item.state.clone();

            let swap_candidates = self.compute_swap_candidates();

            if swap_candidates.is_empty() {
                scores.insert(item.swap_sequence.clone(), item.score);
                continue;
            }

            for &[q0, q1] in &swap_candidates {
                self.load_snapshot(state.clone());

                self.apply_swap([q0, q1]);

                let score = self.calculate_heuristic();

                let mut execute_gate_list = Vec::new();

                if let Some(node) = self.executable_node_on_qubit(q0) {
                    execute_gate_list.push(node);
                    self.front_layer.remove(&node);
                }
                if let Some(node) = self.executable_node_on_qubit(q1) {
                    execute_gate_list.push(node);
                    self.front_layer.remove(&node);
                }

                self.advance_front_layer(&execute_gate_list);

                let mut swap_sequence = item.swap_sequence.clone();
                swap_sequence.push([q0, q1]);

                stack.push(StackItem {
                    state: self.create_snapshot(),
                    swap_sequence: swap_sequence,
                    score: item.score + score,
                    current_depth: item.current_depth - 1,
                })
            }
        }

        self.load_snapshot(initial_state);

        min_score(scores)
    }

    fn compute_swap_candidates(&self) -> Vec<[i32; 2]> {
        let mut swap_candidates: Vec<[i32; 2]> = Vec::new();

        for &phys in self.front_layer.nodes.values().flatten() {
            for neighbour in self.neighbour_map[&phys].iter() {
                if neighbour > &phys || !self.front_layer.is_active(*neighbour) {
                    swap_candidates.push([phys, *neighbour])
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

#[derive(Clone)]
struct State {
    front_layer: MicroFront,
    required_predecessors: Vec<i32>,
    layout: MicroLayout,
    gate_order: Vec<i32>,
}

#[derive(Clone)]
struct StackItem {
    state: State,
    swap_sequence: Vec<[i32; 2]>,
    score: f64,
    current_depth: usize,
}

impl MicroSABRE {
    fn create_snapshot(&self) -> State {
        State {
            front_layer: self.front_layer.clone(),
            required_predecessors: self.required_predecessors.clone(),
            layout: self.layout.clone(),
            gate_order: self.gate_order.clone(),
        }
    }

    fn load_snapshot(&mut self, state: State) {
        self.front_layer = state.front_layer;
        self.required_predecessors = state.required_predecessors;
        self.layout = state.layout;
        self.gate_order = state.gate_order;
    }

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

    fn release_valve(&mut self, current_swaps: &mut Vec<[i32; 2]>) -> Vec<i32> {
        let (&closest_node, &qubits) = {
            self.front_layer
                .nodes
                .iter()
                .min_by(|(_, qubits_a), (_, qubits_b)| {
                    self.distance[qubits_a[0] as usize][qubits_a[1] as usize]
                        .partial_cmp(&self.distance[qubits_b[0] as usize][qubits_b[1] as usize])
                        .unwrap_or(Ordering::Equal)
                })
                .unwrap()
        };

        let shortest_path = {
            let mut shortest_paths: DictMap<NodeIndex, Vec<NodeIndex>> = DictMap::new();
            (dijkstra(
                &build_digraph_from_neighbors(&self.neighbour_map),
                NodeIndex::new(qubits[0] as usize),
                Some(NodeIndex::new(qubits[1] as usize)),
                |_| Ok(1.),
                Some(&mut shortest_paths),
            ) as PyResult<Vec<Option<f64>>>)
                .unwrap();

            shortest_paths
                .get(&NodeIndex::new(qubits[1] as usize))
                .unwrap()
                .iter()
                .map(|n| n.index())
                .collect::<Vec<_>>()
        };

        let split: usize = shortest_path.len() / 2;
        current_swaps.reserve(shortest_path.len() - 2);
        for i in 0..split {
            current_swaps.push([shortest_path[i] as i32, shortest_path[i + 1] as i32]);
        }
        for i in 0..split - 1 {
            let end = shortest_path.len() - 1 - i;
            current_swaps.push([shortest_path[end] as i32, shortest_path[end - 1] as i32]);
        }
        current_swaps.iter().for_each(|&swap| self.apply_swap(swap));

        if current_swaps.len() > 1 {
            vec![closest_node]
        } else {
            // check if the closest node has neighbors that are now routable -- for that we get
            // the other physical qubit that was swapped and check whether the node on it
            // is now routable
            let mut possible_other_qubit = current_swaps[0]
                .iter()
                // check if other nodes are in the front layer that are connected by this swap
                .filter_map(|&swap_qubit| self.front_layer.qubits[swap_qubit as usize])
                // remove the closest_node, which we know we already routed
                .filter(|(node_index, _other_qubit)| *node_index != closest_node)
                .map(|(_node_index, other_qubit)| other_qubit);

            // if there is indeed another candidate, check if that gate is routable
            if let Some(other_qubit) = possible_other_qubit.next() {
                if let Some(also_routed) = self.executable_node_on_qubit(other_qubit) {
                    return vec![closest_node, also_routed];
                }
            }
            vec![closest_node]
        }
    }
}

fn build_digraph_from_neighbors(neighbor_map: &FxHashMap<i32, Vec<i32>>) -> DiGraph<(), ()> {
    let edge_list: Vec<(u32, u32)> = neighbor_map
        .iter()
        .flat_map(|(&src, targets)| targets.iter().map(move |&dst| (src as u32, dst as u32)))
        .collect();

    // `from_edges` creates a graph where node indices are inferred from edge endpoints
    DiGraph::<(), ()>::from_edges(edge_list)
}

pub fn get_successor_map_and_critical_paths(dag: &MicroDAG) -> (Vec<usize>, Vec<usize>) {
    let adj = build_adjacency_list(dag);
    let mut successor_set: FxHashMap<i32, HashSet<i32>> =
        dag.nodes.keys().map(|&n| (n, HashSet::new())).collect();
    let mut critical_path_len: FxHashMap<i32, usize> = dag.nodes.keys().map(|&n| (n, 0)).collect();

    // Reverse topological traversal: assumes nodes are 0..N and acyclic
    for u in (0..dag.nodes.len() as i32).rev() {
        if let Some(neighbors) = adj.get(&u) {
            for &v in neighbors {
                // Add v as successor of u
                successor_set.get_mut(&u).unwrap().insert(v);
                // Add all successors of v to u
                if let Some(succ_v) = successor_set.get(&v) {
                    let succ_v_cloned = succ_v.clone();
                    successor_set.get_mut(&u).unwrap().extend(succ_v_cloned);
                }

                // Update critical path length
                let cand_len = 1 + critical_path_len[&v];
                if cand_len > critical_path_len[&u] {
                    critical_path_len.insert(u, cand_len);
                }
            }
        }
    }

    let successor_counts: Vec<usize> = dag.nodes.keys().map(|&n| successor_set[&n].len()).collect();

    let critical_path_lengths: Vec<usize> =
        dag.nodes.keys().map(|&n| critical_path_len[&n]).collect();

    (successor_counts, critical_path_lengths)
}
