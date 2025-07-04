use std::{
    cmp::Ordering,
    collections::{HashSet, VecDeque},
    thread::current,
};

use crate::{
    graph::dag::MicroDAG,
    routing::{
        front_layer::MicroFront,
        layout::MicroLayout,
        sabre::get_successor_map_and_critical_paths,
        utils::{
            build_adjacency_list, build_coupling_neighbour_map, compute_all_pairs_shortest_paths,
        },
    },
};
use ahash::RandomState;
use indexmap::IndexMap;
use pyo3::{pyclass, pymethods, PyResult};
use rand::{rng, seq::IndexedRandom};
use rustc_hash::FxHashMap;

use rustworkx_core::{dictmap::{DictMap, InitWithHasher}, petgraph::prelude::{DiGraph, NodeIndex}};
use rustworkx_core::shortest_path::dijkstra;

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
        let (successor_map, _) = get_successor_map_and_critical_paths(&dag);

        Ok(Self {
            required_predecessors: vec![0; dag.nodes.len()],
            adjacency_list: build_adjacency_list(&dag),
            dag,
            distance: compute_all_pairs_shortest_paths(&coupling_map),
            neighbour_map: build_coupling_neighbour_map(&coupling_map),
            coupling_map,
            out_map: FxHashMap::default(),
            gate_order: Vec::new(),
            running_mapping: initial_mapping.clone(),
            initial_mapping: initial_mapping,
            successor_map,
            front_layer: MicroFront::new(num_qubits),
            num_qubits,
        })
    }

    fn run(
        &mut self,
        _layers: i32,
    ) -> (
        FxHashMap<i32, Vec<(i32, i32)>>,
        Vec<i32>,
        f64,
        f64,
        f64,
        f64,
    ) {
        // Initialize required predecessors
        self.dag
            .edges()
            .unwrap()
            .iter()
            .for_each(|edge| self.required_predecessors[edge.1 as usize] += 1);

        // Initialize front layer
        let initial_front = self.initial_front();

        // Advance front layer to first gates that cannot be executed
        self.advance_front_layer(&initial_front);

        let mut execute_gate_list = Vec::new();

        while !self.front_layer.is_empty() {
            let mut current_swaps: Vec<(i32, i32)> = Vec::new();

            while execute_gate_list.is_empty() {
                if current_swaps.len() > 10000 {
                    panic!("We are stuck!");
                }
                // Precompute two swap layers up front and return multiple results that will be applied
                let swaps = self.compute_swaps();

                println!("2 Swaps are: {:?}", swaps);

                // TODO: This is a bold move! Check that we can actually do that
                // info: swapped [swaps.0, swaps.1] for swaps
                for swap in swaps {
                    let q0 = swap.0;
                    let q1 = swap.1;

                    current_swaps.push(swap);
                    self.apply_swap([q0, q1]);

                    println!("Current swap: {:?}", swap);
                    if let Some(node) = self.executable_node_on_qubit(q0) {
                        println!("Executable node: {:?}", node);
                        execute_gate_list.push(node);
                        let node = self.dag.get(node).unwrap();
                        let q0 = self.running_mapping.virtual_to_physical(node.qubits[0]);
                        let q1 = self.running_mapping.virtual_to_physical(node.qubits[1]);
                        println!(
                            "Node operates on virtual qubits: {:?}",
                            node.qubits().unwrap()
                        );
                        println!("Node operates on physical qubits: {:?}, {:?}", q0, q1);
                    }
                    if let Some(node) = self.executable_node_on_qubit(q1) {
                        println!("Executable node: {:?}", node);
                        execute_gate_list.push(node);
                        let node = self.dag.get(node).unwrap();
                        let q0 = self.running_mapping.virtual_to_physical(node.qubits[0]);
                        let q1 = self.running_mapping.virtual_to_physical(node.qubits[1]);
                        println!(
                            "Node operates on virtual qubits: {:?}",
                            node.qubits().unwrap()
                        );
                        println!("Node operates on physical qubits: {:?}, {:?}", q0, q1);
                    }

                    // TODO: We should remove the node from the front layer here
                    if !execute_gate_list.is_empty() {
                        for &node in &execute_gate_list {
                            println!("Trying to remove {:?}", node);
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

impl MultiSABRE {
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

    fn advance_front_in_place(
        &mut self,
        front_layer: &MicroFront,
        nodes: &Vec<i32>,
    ) -> (MicroFront, Vec<i32>) {
        let mut node_queue: VecDeque<i32> = VecDeque::from(nodes.clone());

        let mut front_layer = front_layer.clone();
        let mut required_predecessors = self.required_predecessors.clone();

        while let Some(node_index) = node_queue.pop_front() {
            let node = self.dag.get(node_index).unwrap();

            if node.qubits.len() == 2 {
                let physical_q0 = self.running_mapping.virtual_to_physical(node.qubits[0]);
                let physical_q1 = self.running_mapping.virtual_to_physical(node.qubits[1]);

                if self.distance[physical_q0 as usize][physical_q1 as usize] != 1 {
                    front_layer.insert(node_index, [physical_q0, physical_q1]);
                    continue;
                }
            }

            // if !self.gate_order.contains(&node.id) {
            //     self.gate_order.push(node.id);
            // }

            if let Some(successors) = self.adjacency_list.get(&node_index) {
                for successor in successors {
                    if let Some(count) = required_predecessors.get_mut(*successor as usize) {
                        *count -= 1;
                        if *count == 0 {
                            node_queue.push_back(*successor);
                        }
                    }
                }
            }
        }

        (front_layer, required_predecessors)
    }

    fn compute_swap_candidates(&self, front_layer: &MicroFront) -> Vec<(i32, i32)> {
        let mut swap_candidates: Vec<(i32, i32)> = Vec::new();

        for &phys in front_layer.nodes.values().flatten() {
            for neighbour in self.neighbour_map[&phys].iter() {
                if neighbour > &phys || !front_layer.is_active(*neighbour) {
                    swap_candidates.push((phys, *neighbour))
                }
            }
        }
        swap_candidates
    }

    // Returns the two heuristically best consecutive swaps. As a starting
    // point, the heuristic is computed before any change is made. Then, the
    // swap candidates are computed from the current front layer. For each of
    // those SWAP candidates, we create a temporary front_layer to track our
    // changes. We temporariliy execute the swap, check for executable gates,
    // advance the front layer, compute the swap candidates and apply those
    // second swaps. Finally, we compute the heuristic after executing both
    // swaps temporarily and add the result to the scores vector.
    fn compute_swaps(&mut self) -> Vec<(i32, i32)> {
        let mut scores: FxHashMap<Vec<(i32, i32)>, f64> = FxHashMap::default();

        let swap_candidates: Vec<(i32, i32)> = self.compute_swap_candidates(&self.front_layer);

        if swap_candidates.is_empty() {
            panic!("No swap candidates left!");
        }

        println!("Number of swap candidates: {:?}", swap_candidates.len());

        for &(q0, q1) in &swap_candidates {
            // Temporarily apply first swap and calculate heuristics
            let before_first = self.calculate_heuristic(
                &self.front_layer.clone(),
                &self.required_predecessors.clone(),
            );
            self.apply_swap([q0, q1]);
            let after_first = self.calculate_heuristic(
                &self.front_layer.clone(),
                &self.required_predecessors.clone(),
            );
            let diff_first = after_first - before_first;

            // Create temporary data structures to track changes of temporary swaps
            let mut tmp_execute_gate_list = Vec::new();

            // q0 and q1 are the qubits that were involved in a swap. Here, we check whether some
            // node on the front_layer that operates on one of those qubits that was just changed
            // is now executable. So if both of the following if-statements return the same node,
            // then this would mean that this node contains both qubits q0 and q1, which indicates
            // that the gate should have been executable before even applying the swap.
            if let Some(node) = self.executable_node_on_qubit(q0) {
                println!("First one: {:?}", node);
                tmp_execute_gate_list.push(node);
            }

            if let Some(node) = self.executable_node_on_qubit(q1) {
                println!("Second one: {:?}", node);
                tmp_execute_gate_list.push(node);
            }

            let mut tmp_front_layer_before = self.front_layer.clone();
            for node in tmp_execute_gate_list.iter() {
                println!("Removing: {:?}", node);
                tmp_front_layer_before.remove(node);
            }

            let (mut tmp_front_layer, new_required_predecessors) =
                self.advance_front_in_place(&tmp_front_layer_before, &tmp_execute_gate_list);

            // In case the inner front layer is empty, just push the current swap into the list of swaps
            if tmp_front_layer.is_empty() {
                scores.insert(vec![(q0, q1)], diff_first);
                // panic!("Inner front layer is empty");
                break;
            }
            let inner_swap_candidates = self.compute_swap_candidates(&tmp_front_layer);

            if inner_swap_candidates.is_empty() {
                panic!("No inner swap candidates left!");
            }

            for &(inner_q0, inner_q1) in &inner_swap_candidates {
                // As long as we revert this afterwards, we should be fine
                //
                let before_second =
                    self.calculate_heuristic(&tmp_front_layer, &new_required_predecessors);
                self.apply_swap([inner_q0, inner_q1]);
                tmp_front_layer.apply_swap([inner_q0, inner_q1]);

                let after_second =
                    self.calculate_heuristic(&tmp_front_layer, &new_required_predecessors);
                let diff_second = after_second - before_second;

                scores.insert(
                    vec![(q0, q1), (inner_q0, inner_q1)],
                    diff_first + diff_second,
                );

                tmp_front_layer.apply_swap([inner_q1, inner_q0]);
                self.apply_swap([inner_q1, inner_q0]);
            }

            self.apply_swap([q1, q0]);
        }

        if scores.is_empty() {
            panic!("Nothing to score!");
        }

        min_score(scores)
    }

    fn calculate_heuristic(
        &mut self,
        front_layer: &MicroFront,
        new_required_predecessors: &Vec<i32>,
    ) -> f64 {
        let extended_set = self.get_extended_set(front_layer, new_required_predecessors);

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

        basic + 0.5 * lookahead
    }

    fn get_extended_set(
        &mut self,
        front_layer: &MicroFront,
        required_predecessors: &Vec<i32>,
    ) -> MicroFront {
        let mut required_predecessors = required_predecessors.clone();

        let mut to_visit: Vec<i32> = front_layer.nodes.keys().copied().collect();
        let mut i = 0;

        let mut extended_set: MicroFront = MicroFront::new(self.num_qubits);
        let mut visit_now: Vec<i32> = Vec::new();

        let dag_size = self.dag.nodes.len();

        let mut decremented = vec![0; dag_size];

        let mut visited = vec![false; dag_size];

        while i < to_visit.len() && extended_set.len() < 20 as usize {
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

    // Advance front layer to gates that can't be executed on the hardware right away
    fn advance_front_layer(&mut self, nodes: &Vec<i32>) {
        let mut node_queue: VecDeque<i32> = VecDeque::from(nodes.clone());

        while let Some(node_index) = node_queue.pop_front() {
            let node = self.dag.get(node_index).unwrap();

            // Only two-qubit gates can potentially not be executed right away
            if node.qubits.len() == 2 {
                let physical_q0 = self.running_mapping.virtual_to_physical(node.qubits[0]);
                let physical_q1 = self.running_mapping.virtual_to_physical(node.qubits[1]);

                if self.distance[physical_q0 as usize][physical_q1 as usize] != 1 {
                    // If not executable, insert into front layer
                    self.front_layer
                        .insert(node_index, [physical_q0, physical_q1]);

                    // Skip remainder of the loop-iteration
                    continue;
                }
            }

            // If we get here, the gate can be executed
            if !self.gate_order.contains(&node.id) {
                // Add gate to the gate order
                self.gate_order.push(node.id);
            }

            // I should probably rename the adjacency list if it's actually just accounting for the
            // successors
            // Since we "executed" the gate, update successors.
            // Required predecessors only accounts for the predecessors of the actual node we are
            // looking at. If a node has two incoming edges, it has two required predecessors.
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

fn min_score(scores: FxHashMap<Vec<(i32, i32)>, f64>) -> Vec<(i32, i32)> {
    println!("Scores: {:?}", scores);
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
