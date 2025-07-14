use std::{
    cmp::Ordering,
    collections::{HashSet, VecDeque},
};

use crate::{
    graph::dag::MicroDAG,
    routing::{
        front_layer::MicroFront,
        layout::MicroLayout,
        utils::{
            build_adjacency_list, build_coupling_neighbour_map, compute_all_pairs_shortest_paths,
        },
    },
};
use pyo3::{pyclass, pymethods, PyResult};
use rand::{rng, seq::IndexedRandom};
use rustc_hash::FxHashMap;

use rustworkx_core::shortest_path::dijkstra;
use rustworkx_core::{
    dictmap::{DictMap, InitWithHasher},
    petgraph::prelude::{DiGraph, NodeIndex},
};

#[pyclass(module = "microboost.routing.mutlisabre")]
pub struct MultiSABRE {
    dag: MicroDAG,
    out_map: FxHashMap<i32, Vec<[i32; 2]>>,
    gate_order: Vec<i32>,
    required_predecessors: Vec<i32>,
    adjacency_list: FxHashMap<i32, Vec<i32>>,
    distance: Vec<Vec<i32>>,
    running_mapping: MicroLayout,
    front_layer: MicroFront,
    neighbour_map: FxHashMap<i32, Vec<i32>>,
    num_qubits: i32,
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
        Ok(Self {
            required_predecessors: vec![0; dag.nodes.len()],
            adjacency_list: build_adjacency_list(&dag),
            dag,
            distance: compute_all_pairs_shortest_paths(&coupling_map),
            neighbour_map: build_coupling_neighbour_map(&coupling_map),
            out_map: FxHashMap::default(),
            gate_order: Vec::new(),
            running_mapping: initial_mapping.clone(),
            front_layer: MicroFront::new(num_qubits),
            num_qubits,
        })
    }

    fn run(
        &mut self,
        layers: i32,
    ) -> (FxHashMap<i32, Vec<[i32; 2]>>, Vec<i32>, f64, f64, f64, f64) {
        self.dag
            .edges()
            .unwrap()
            .iter()
            .for_each(|edge| self.required_predecessors[edge.1 as usize] += 1);

        let initial_front = self.initial_front();

        self.advance_front_layer(&initial_front);

        let mut execute_gate_list = Vec::new();

        println!("Front layer len: {:?}", self.front_layer.len());
        while !self.front_layer.is_empty() {
            let mut current_swaps: Vec<[i32; 2]> = Vec::new();

            println!("Are we even in here?");
            // while execute_gate_list.is_empty() && current_swaps.len() < self.num_qubits as usize * 10 {
            while execute_gate_list.is_empty() {
                if current_swaps.len() > 10 * self.num_qubits as usize {
                    panic!("We are stuck!");
                }

                let swaps = self.choose_best_swaps(layers as usize);

                // We apparently never get here
                println!("Chose swaps: {:?}", swaps);

                for swap in swaps {
                    let q0 = swap[0];
                    let q1 = swap[1];

                    current_swaps.push(swap);
                    self.apply_swap([q0, q1]);

                    if let Some(node) = self.executable_node_on_qubit(q0) {
                        execute_gate_list.push(node);
                    }
                    if let Some(node) = self.executable_node_on_qubit(q1) {
                        execute_gate_list.push(node);
                    }
                    
                    // TODO: I can probably improve this part
                    if !execute_gate_list.is_empty() {
                        for &node in &execute_gate_list {
                            self.front_layer.remove(&node);
                        }

                        let node_id = self.dag.get(execute_gate_list[0]).unwrap().id;
                        self.out_map
                            .entry(node_id)
                            .or_default()
                            .extend(&current_swaps);

                        self.advance_front_layer(&execute_gate_list);

                        if self.front_layer.is_empty() {
                            return (
                                std::mem::take(&mut self.out_map),
                                std::mem::take(&mut self.gate_order),
                                0.,
                                0.,
                                0.,
                                0.,
                            );
                        }
                        // When we end up here, we always clear the executed
                        // gate list and hence will always go into the next
                        // iteration
                        execute_gate_list.clear();
                        current_swaps.clear();
                    }
                }
            }

            // if execute_gate_list.is_empty() {
            //     current_swaps.drain(..).rev().for_each(|swap| self.apply_swap(swap));
            //     let force_routed = self.release_valve(&mut current_swaps);
            //     execute_gate_list.extend(force_routed);
            // }
        }

        (
            std::mem::take(&mut self.out_map),
            std::mem::take(&mut self.gate_order),
            0.,
            0.,
            0.,
            0.,
        )
    }
}

#[derive(Clone)]
struct RoutingState {
    front_layer: MicroFront,
    required_predecessors: Vec<i32>,
    running_mapping: MicroLayout,
    gate_order: Vec<i32>,
    num_executable_gates: usize
}

#[derive(Clone)]
struct StackState {
    mapper_state: RoutingState,
    current_sequence: Vec<[i32; 2]>,
    // current_score: f64,
    layer: usize
}

impl MultiSABRE {
    fn save_state(&self, num_executable_gates: usize) -> RoutingState {
        RoutingState {
            front_layer: self.front_layer.clone(),
            required_predecessors: self.required_predecessors.clone(),
            running_mapping: self.running_mapping.clone(),
            gate_order: self.gate_order.clone(),
            num_executable_gates 
        }
    }

    fn apply_state(&mut self, state: RoutingState) {
        self.front_layer = state.front_layer;
        self.required_predecessors = state.required_predecessors;
        self.running_mapping = state.running_mapping;
        self.gate_order = state.gate_order;
    }

    fn choose_best_swaps(&mut self, layers: usize) -> Vec<[i32; 2]> {
        let start_state = self.save_state(0);
        
        let mut stack = Vec::new();
        let mut scores: FxHashMap<Vec<[i32; 2]>, f64> = FxHashMap::default();

        stack.push(StackState {
            mapper_state: self.save_state(0),
            current_sequence: Vec::new(),
            layer: layers
        });

        while let Some(state) = stack.pop() {
            // Shouldn't we also consider if swap_candidates.is_empty()?
            if state.layer == 0 {
                self.apply_state(state.mapper_state.clone());

                for &[q0, q1] in &state.current_sequence {
                    self.apply_swap([q0, q1]);
                }

                let score = self.calculate_heuristic(state.mapper_state.num_executable_gates);
                scores.insert(state.current_sequence.clone(), score);
                continue;
            }

            let initial_state  = state.mapper_state.clone();
            let swap_candidates = self.compute_swap_candidates();

            for &[q0, q1] in &swap_candidates {
                self.apply_state(initial_state.clone());
                self.apply_swap([q0, q1]);

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

                let mut new_sequence = state.current_sequence.clone();
                new_sequence.push([q0, q1]);

                stack.push(StackState {
                    mapper_state: self.save_state(state.mapper_state.num_executable_gates + execute_gate_list.len()),
                    current_sequence: new_sequence,
                    layer: state.layer - 1
                });
            }
        }

        self.apply_state(start_state);

        min_score(scores)
    }

    

    fn compute_swap_candidates(&self) -> Vec<[i32; 2]> {
        let mut swap_candidates: Vec<[i32; 2]> = Vec::new();

        for &phys in self.front_layer.nodes.values().flatten() {
            for &neighbour in self.neighbour_map[&phys].iter() {
                if neighbour > phys || !self.front_layer.is_active(neighbour) {
                    swap_candidates.push([phys, neighbour])
                }
            }
        }
        swap_candidates
    }

    fn calculate_heuristic(
        &mut self,
        num_executable_gates: usize
    ) -> f64 {
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

        (1. / num_executable_gates as f64) * (basic + 0.5 * lookahead)
    }

    fn get_extended_set(
        &mut self,
    ) -> MicroFront {
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
                                        self.running_mapping.virtual_to_physical(succ.qubits[0]);
                                    let physical_q1 =
                                        self.running_mapping.virtual_to_physical(succ.qubits[1]);
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

    fn apply_swap(&mut self, swap: [i32; 2]) {
        self.front_layer.apply_swap(swap);
        self.running_mapping.swap_physical(swap);
    }

    // Check if any node in the front layer can be executed after a
    // physical_qubit was involved in a swap
    fn executable_node_on_qubit(&self, physical_qubit: i32) -> Option<i32> {
        for [a, b] in self.front_layer.nodes.values() {
            if (*a == physical_qubit || *b == physical_qubit)
                && self.distance[*a as usize][*b as usize] == 1
            {
                return Some(self.front_layer.qubits[*a as usize].unwrap().0);
            }
        }
        None
    }

    // Returns all nodes without any predecessors
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

    /// Advances front layer by looking at successors of the nodes that were just
    /// executed. Successors are added to the front layer if they do NOT have any
    /// required predecessors AND can't be routed without inserting any SWAPs 
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

            // If we get here, the gate can be executed
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

fn min_score(scores: FxHashMap<Vec<[i32; 2]>, f64>) -> Vec<[i32; 2]> {
    let mut best_swap_sequences = Vec::new();

    let mut iter = scores.iter();

    let (min_swap_sequence, mut min_score) = iter
        .next()
        .map(|(swap_sequence, &score)| (swap_sequence, score))
        .unwrap();

    best_swap_sequences.push(min_swap_sequence);

    // TODO: Consider introducing an epsilon threshold here
    for (swap_sequence, &score) in iter {
        if score < min_score {
            min_score = score;
            best_swap_sequences.clear();
            best_swap_sequences.push(&swap_sequence);
        } else if score == min_score {
            best_swap_sequences.push(&swap_sequence);
        }
    }

    let mut rng = rng();

    if best_swap_sequences.len() > 1 {
        println!("Actually making a random choice between: {:?}", scores);
    }

    best_swap_sequences.choose(&mut rng).unwrap().to_vec()
}

fn build_digraph_from_neighbors(neighbor_map: &FxHashMap<i32, Vec<i32>>) -> DiGraph<(), ()> {
    let edge_list: Vec<(u32, u32)> = neighbor_map
        .iter()
        .flat_map(|(&src, targets)| targets.iter().map(move |&dst| (src as u32, dst as u32)))
        .collect();

    // `from_edges` creates a graph where node indices are inferred from edge endpoints
    DiGraph::<(), ()>::from_edges(edge_list)
}
