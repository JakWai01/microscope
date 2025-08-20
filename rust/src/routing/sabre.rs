use crate::routing::front_layer::MicroFront;
use crate::routing::layout::MicroLayout;
use crate::routing::utils::{
    build_coupling_neighbour_map, build_digraph_from_neighbors, compute_all_pairs_shortest_paths,
};

use crate::{graph::dag::MicroDAG, routing::utils::build_adjacency_list};

use core::f64;
use std::cmp::Ordering;

use std::collections::{HashSet, VecDeque};

use indexmap::{IndexSet};
use pyo3::{pyclass, pymethods, PyResult};

use rustc_hash::FxHashMap;

use rustworkx_core::dictmap::{DictMap, InitWithHasher};
use rustworkx_core::petgraph::csr::IndexType;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::shortest_path::dijkstra;

#[pyclass(module = "microboost.routing.sabre")]
pub struct MicroSABRE {
    dag: MicroDAG,
    out_map: FxHashMap<i32, Vec<[i32; 2]>>,
    gate_order: Vec<i32>,
    front_layer: MicroFront,
    required_predecessors: Vec<i32>,
    adjacency_list: Vec<Vec<i32>>,
    distance: Vec<Vec<i32>>,
    neighbour_map: Vec<Vec<i32>>,
    layout: MicroLayout,
    num_qubits: usize,
    last_swap_on_qubit: FxHashMap<i32, [i32; 2]>,
}

#[pymethods]
impl MicroSABRE {
    #[new]
    pub fn new(
        dag: MicroDAG,
        initial_layout: MicroLayout,
        coupling_map: Vec<Vec<i32>>,
        num_qubits: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            required_predecessors: vec![0; dag.nodes.len()],
            adjacency_list: build_adjacency_list(&dag),
            dag: dag.clone(),
            layout: initial_layout.clone(),
            distance: compute_all_pairs_shortest_paths(&coupling_map),
            out_map: FxHashMap::default(),
            gate_order: Vec::new(),
            front_layer: MicroFront::new(num_qubits),
            neighbour_map: build_coupling_neighbour_map(&coupling_map),
            num_qubits,
            last_swap_on_qubit: FxHashMap::default(),
        })
    }

    fn run(&mut self, depth: usize) -> (FxHashMap<i32, Vec<[i32; 2]>>, Vec<i32>) {
        self.dag
            .edges()
            .unwrap()
            .iter()
            .for_each(|edge| self.required_predecessors[edge.1 as usize] += 1);
        let initial_front = self.initial_front();
        self.advance_front_layer(&initial_front);

        let mut execute_gate_list: Vec<i32> = Vec::new();
        let mut insertion_queue: VecDeque<[i32; 2]> = VecDeque::new();

        while !self.front_layer.is_empty() {
            let mut current_swaps: Vec<[i32; 2]> = Vec::new();

            while execute_gate_list.is_empty() && current_swaps.len() < 10 * self.num_qubits {
                if insertion_queue.is_empty() {
                    for swap in self.choose_best_swaps(depth) {
                        insertion_queue.push_back(swap);
                    }
                }

                if let Some(swap) = insertion_queue.pop_front() {
                    let q0 = swap[0];
                    let q1 = swap[1];

                    current_swaps.push([q0, q1]);
                    self.apply_swap([q0, q1]);

                    if let Some(node) = self.executable_node_on_qubit(q0) {
                        execute_gate_list.push(node);
                        self.front_layer.remove(&node);
                    }

                    if let Some(node) = self.executable_node_on_qubit(q1) {
                        execute_gate_list.push(node);
                        self.front_layer.remove(&node);
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
            }

            let node_id = self.dag.get(execute_gate_list[0]).unwrap().id;
            self.out_map
                .entry(node_id)
                .or_default()
                .extend(current_swaps.clone());

            self.advance_front_layer(&execute_gate_list);

            execute_gate_list.clear();
        }

        (
            std::mem::take(&mut self.out_map),
            std::mem::take(&mut self.gate_order),
        )
    }
}

#[derive(Clone)]
struct State {
    front_layer: MicroFront,
    required_predecessors: Vec<i32>,
    layout: MicroLayout,
    gate_order: Vec<i32>,
    last_swap_on_qubit: FxHashMap<i32, [i32; 2]>,
}

impl MicroSABRE {
    #[inline]
    fn sum_nodes_distance_with_layout(
        &self,
        layout: &MicroLayout,
        node_ids: &IndexSet<i32>,
    ) -> f64 {
        let mut acc = 0.0;
        for &nid in node_ids.iter() {
            let node = self.dag.get(nid).unwrap();
            if node.qubits.len() == 2 {
                let p0 = layout.virtual_to_physical(node.qubits[0]) as usize;
                let p1 = layout.virtual_to_physical(node.qubits[1]) as usize;
                acc += self.distance[p0][p1] as f64;
            }
        }
        acc
    }

    #[inline]
    fn union_heuristic_with_layout(
        &self,
        layout: &MicroLayout,
        u_front: &IndexSet<i32>,
        u_ext: &IndexSet<i32>,
        lambda: f64,
    ) -> f64 {
        let basic = self.sum_nodes_distance_with_layout(layout, u_front);
        let ext_sum = self.sum_nodes_distance_with_layout(layout, u_ext);
        let m = u_ext.len().max(1) as f64;
        basic + (lambda / m) * ext_sum
    }

    /// Occupancy φ(F) = |F| / floor(num_qubits/2)
    #[inline]
    fn occupancy(&self, front_len: usize) -> f64 {
        let cap = (self.num_qubits / 2).max(1) as f64;
        (front_len as f64) / cap
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

        while i < to_visit.len() && extended_set.len() < 20 {
            visit_now.push(to_visit[i]);
            let mut j = 0;

            while j < visit_now.len() {
                let node_id = visit_now[j];

                for &successor in &self.adjacency_list[node_id as usize] {
                    if !visited[successor as usize] {
                        let succ = self.dag.get(successor).unwrap();
                        visited[successor as usize] = true;

                        decremented[successor as usize] += 1;
                        required_predecessors[successor as usize] -= 1;

                        if required_predecessors[successor as usize] == 0 {
                            if succ.qubits.len() == 2 {
                                let physical_q0 = self.layout.virtual_to_physical(succ.qubits[0]);
                                let physical_q1 = self.layout.virtual_to_physical(succ.qubits[1]);
                                extended_set.insert(successor, [physical_q0, physical_q1]);
                                to_visit.push(successor);
                                continue;
                            }
                            visit_now.push(successor);
                        }
                    }
                }
                j += 1;
            }

            visit_now.clear();
            i += 1;
        }

        extended_set
    }

    /// Sum of per-node min-swaps needed (d-1)^+ over node_ids under given layout.
    #[inline]
    fn sum_min_swaps_needed_for_nodes(
        &self,
        layout: &MicroLayout,
        node_ids: &IndexSet<i32>,
    ) -> f64 {
        let mut acc = 0.0_f64;
        for &nid in node_ids.iter() {
            let node = self.dag.get(nid).unwrap();
            if node.qubits.len() == 2 {
                let p0 = layout.virtual_to_physical(node.qubits[0]) as usize;
                let p1 = layout.virtual_to_physical(node.qubits[1]) as usize;
                let d = self.distance[p0][p1];
                acc += ((d - 1).max(0)) as f64;
            }
        }
        acc
    }

    /// Heuristic on front/extended unions using (d-1)^+, with a weight lambda for extended.
    #[inline]
    fn union_heuristic_min_swaps_with_layout(
        &self,
        layout: &MicroLayout,
        u_front: &IndexSet<i32>,
        u_ext: &IndexSet<i32>,
        lambda: f64,
    ) -> f64 {
        let front_sum = self.sum_min_swaps_needed_for_nodes(layout, u_front);
        let ext_sum = self.sum_min_swaps_needed_for_nodes(layout, u_ext);
        let m = (u_ext.len().max(1)) as f64;
        front_sum + (lambda / m) * ext_sum
    }

    // ---------- Updated choose_best_swaps ----------
    fn choose_best_swaps(&mut self, depth: usize) -> Vec<[i32; 2]> {
        let initial_state = self.create_snapshot();
        let lambda: f64 = 0.5; // weight for the extended set portion
        let gamma: f64 = 0.1; // small occupancy bonus
        let eps: f64 = 1e-12;

        // φ at the very start (same for all sequences in this call)
        let phi_start = {
            let start_front_len = initial_state.front_layer.len();
            self.occupancy(start_front_len)
        };

        #[derive(Clone)]
        struct StackItem {
            swap_sequence: Vec<[i32; 2]>,
            remaining_depth: usize,
        }

        let mut stack = Vec::new();
        stack.push(StackItem {
            swap_sequence: Vec::new(),
            remaining_depth: depth,
        });

        let mut best_seq: Option<Vec<[i32; 2]>> = None;
        let mut best_exec = 0usize;
        let mut best_secondary = f64::NEG_INFINITY;
        let mut best_len = usize::MAX;

        let mut update_best = |seq: Vec<[i32; 2]>, exec: usize, secondary: f64| {
            let len = seq.len();
            let better = (exec > best_exec)
                || (exec == best_exec
                    && (secondary > best_secondary + eps
                        || ((secondary - best_secondary).abs() <= eps && len < best_len)));

            if better {
                best_exec = exec;
                best_secondary = secondary;
                best_len = len;
                best_seq = Some(seq);
            } else if exec == best_exec
                && (secondary - best_secondary).abs() <= eps
                && len == best_len
                && rand::random::<bool>()
            {
                // stochastic tie-break like SABRE
                best_exec = exec;
                best_secondary = secondary;
                best_len = len;
                best_seq = Some(seq);
            }
        };

        while let Some(item) = stack.pop() {
            // reset to baseline for this prefix
            self.load_snapshot(initial_state.clone());

            let mut execute_gate_list = Vec::new();
            let mut advanced_gates = 0usize;

            // unions of logical 2-qubit node IDs across t = 0..|s|
            let mut u_front: IndexSet<i32> = IndexSet::new();
            let mut u_ext: IndexSet<i32> = IndexSet::new();

            // Capture step t = 0
            for &nid in self.front_layer.nodes.keys() {
                u_front.insert(nid);
            }
            {
                let ext0 = self.get_extended_set();
                for &nid in ext0.nodes.keys() {
                    u_ext.insert(nid);
                }
            }

            // Apply swaps in the prefix, executing gates and collecting unions at each step
            for &swap in &item.swap_sequence {
                let [q0, q1] = swap;

                // Apply the SWAP
                self.apply_swap([q0, q1]);

                // Execute gates on q0 and q1 if adjacent after SWAP
                if let Some(node) = self.executable_node_on_qubit(q0) {
                    execute_gate_list.push(node);
                    self.front_layer.remove(&node);
                }
                if let Some(node) = self.executable_node_on_qubit(q1) {
                    execute_gate_list.push(node);
                    self.front_layer.remove(&node);
                }

                // Advance (may add successors into the front layer)
                advanced_gates += self.advance_front_layer(&execute_gate_list) as usize;
                execute_gate_list.clear();

                // Record unions after the step
                for &nid in self.front_layer.nodes.keys() {
                    u_front.insert(nid);
                }
                {
                    let ext_t = self.get_extended_set();
                    for &nid in ext_t.nodes.keys() {
                        u_ext.insert(nid);
                    }
                }
            }

            // Terminal conditions -> evaluate sequence (single before/after)
            let should_score_leaf = item.remaining_depth == 0 || self.front_layer.is_empty();

            let swap_candidates = if !should_score_leaf {
                self.compute_swap_candidates()
            } else {
                Vec::new()
            };

            if should_score_leaf || swap_candidates.is_empty() {
                // Build union set size (unique nodes)
                let mut u_union = u_front.clone();
                for &nid in u_ext.iter() {
                    u_union.insert(nid);
                }
                let union_size = u_union.len().max(1); // avoid division by zero

                // Compute min-swaps heuristic before / after on the per-sequence unions
                let h_before = self.union_heuristic_min_swaps_with_layout(
                    &initial_state.layout,
                    &u_front,
                    &u_ext,
                    lambda,
                );
                let h_after =
                    self.union_heuristic_min_swaps_with_layout(&self.layout, &u_front, &u_ext, lambda);

                let delta_h = h_before - h_after; // larger is better
                let norm_delta = delta_h / (union_size as f64); // normalization per node

                // occupancy at end
                let phi_end = self.occupancy(self.front_layer.len());
                let secondary = norm_delta + gamma * (phi_end - phi_start);

                update_best(item.swap_sequence.clone(), advanced_gates, secondary);
                continue;
            }

            let state_after_prefix = self.create_snapshot();

            for &swap in &swap_candidates {
                // For each child, start from identical state-after-prefix
                self.load_snapshot(state_after_prefix.clone());

                // Just record the extended sequence for deeper exploration.
                let mut next_seq = item.swap_sequence.clone();
                next_seq.push(swap);

                stack.push(StackItem {
                    swap_sequence: next_seq,
                    remaining_depth: item.remaining_depth - 1,
                });
            }
        }

        self.load_snapshot(initial_state);
        best_seq.unwrap_or_default()
    }

    fn compute_swap_candidates(&self) -> Vec<[i32; 2]> {
        let mut swap_candidates: Vec<[i32; 2]> = Vec::new();

        for &phys in self.front_layer.nodes.values().flatten() {
            for &neighbour in &self.neighbour_map[phys as usize] {
                if neighbour > phys || !self.front_layer.is_active(neighbour) {
                    swap_candidates.push([phys, neighbour])
                }
            }
        }
        swap_candidates
    }

    fn executable_node_on_qubit(&self, physical_qubit: i32) -> Option<i32> {
        match self.front_layer.qubits[physical_qubit as usize] {
            Some((node_id, other_qubit))
                if self.distance[physical_qubit as usize][other_qubit as usize] == 1 =>
            {
                Some(node_id)
            }
            _ => None,
        }
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
    fn create_snapshot(&self) -> State {
        State {
            front_layer: self.front_layer.clone(),
            required_predecessors: self.required_predecessors.clone(),
            layout: self.layout.clone(),
            gate_order: self.gate_order.clone(),
            last_swap_on_qubit: self.last_swap_on_qubit.clone(),
        }
    }

    fn load_snapshot(&mut self, state: State) {
        self.front_layer = state.front_layer;
        self.required_predecessors = state.required_predecessors;
        self.layout = state.layout;
        self.gate_order = state.gate_order;
        self.last_swap_on_qubit = state.last_swap_on_qubit;
    }

    fn advance_front_layer(&mut self, nodes: &[i32]) -> i32 {
        let mut node_queue: VecDeque<i32> = VecDeque::from(nodes.to_vec());

        let mut executed_gates_counter = 0;

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

            // Execute node
            // TODO: That contains really hurts performance. A executed Vec<bool> is much better - but requires additional state tracking
            if !self.gate_order.contains(&node.id) {
                self.gate_order.push(node.id);
                executed_gates_counter += 1;
            }

            for &successor in &self.adjacency_list[node_index as usize] {
                if let Some(count) = self.required_predecessors.get_mut(successor as usize) {
                    *count -= 1;
                    if *count == 0 {
                        node_queue.push_back(successor);
                    }
                }
            }
        }

        executed_gates_counter
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

    fn apply_swap(&mut self, swap: [i32; 2]) {
        self.front_layer.apply_swap(swap);
        self.layout.swap_physical(swap);
    }
}
