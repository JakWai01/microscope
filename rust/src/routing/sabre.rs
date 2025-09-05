use crate::routing::front_layer::MicroFront;
use crate::routing::layout::MicroLayout;
use crate::routing::utils::{
    build_coupling_neighbour_map, compute_all_pairs_shortest_paths,
    Best, StackItem, State,
};
use crate::{graph::dag::MicroDAG, routing::utils::build_adjacency_list};
use core::f64;
use indexmap::IndexSet;
use pyo3::{pyclass, pymethods, PyResult};
use rustc_hash::FxHashMap;
use std::collections::{HashSet, VecDeque};

#[derive(Debug)]
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
    executed: FxHashMap<i32, bool>,
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
            executed: FxHashMap::default(),
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

            while execute_gate_list.is_empty() {

                if current_swaps.len() > 10000 {
                    panic!("We are stuck!");
                }

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

impl MicroSABRE {
    #[inline]
    fn sum_min_swaps_needed_for_nodes(
        &self,
        layout: &MicroLayout,
        node_ids: &IndexSet<i32>,
    ) -> f64 {
        let mut acc = 0.0_f64;
        for &nid in node_ids.iter() {
            let node = self.dag.get(nid).unwrap();

            if self.executed.contains_key(&node.id) {
                continue;
            }

            if node.qubits.len() == 2 {
                let p0 = layout.virtual_to_physical(node.qubits[0]) as usize;
                let p1 = layout.virtual_to_physical(node.qubits[1]) as usize;
                let d = self.distance[p0][p1];
                acc += ((d - 1).max(0)) as f64;
            }
        }
        acc
    }

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

    #[inline]
    fn populate_union(&mut self) -> (IndexSet<i32>, IndexSet<i32>) {
        let mut u_front = IndexSet::new();
        for &nid in self.front_layer.nodes.keys() {
            u_front.insert(nid);
        }
        let mut u_ext = IndexSet::new();
        for &nid in self.get_extended_set().nodes.keys() {
            u_ext.insert(nid);
        }
        (u_front, u_ext)
    }

    fn apply_prefix(
        &mut self,
        prefix: &[[i32; 2]],
    ) -> (Vec<i32>, IndexSet<i32>, IndexSet<i32>) {
        let mut execute_gate_list = Vec::new();
        let mut advanced_gates: Vec<i32> = Vec::new();

        let (mut u_front, mut u_ext) = self.populate_union();

        for &[q0, q1] in prefix {
            self.apply_swap([q0, q1]);

            for &q in &[q0, q1] {
                if let Some(node_index) = self.executable_node_on_qubit(q) {
                    execute_gate_list.push(node_index);
                    self.front_layer.remove(&node_index);
                }
            }

            // NOTE: advance_front_layer now returns node *indices* (not node.id)
            let just_advanced = self.advance_front_layer(&execute_gate_list);

            for &nid in &just_advanced {
                u_front.swap_remove(&nid);
                u_ext.swap_remove(&nid);
                advanced_gates.push(nid);
            }

            execute_gate_list.clear();

            for &nid in self.front_layer.nodes.keys() {
                u_front.insert(nid);
            }
            for &nid in self.get_extended_set().nodes.keys() {
                u_ext.insert(nid);
            }
        }

        (advanced_gates, u_front, u_ext)
    }

    fn score_leaf(
        &mut self,
        initial_layout: &MicroLayout,
        u_front: &IndexSet<i32>,
        u_ext: &IndexSet<i32>,
        extended_set_weight: f64,
    ) -> f64 {
        let mut u_union = u_front.clone();
        u_union.extend(u_ext.iter().cloned());

        let union_size = u_union.len().max(1) as f64;

        let h_before = self.union_heuristic_min_swaps_with_layout(
            initial_layout,
            u_front,
            u_ext,
            extended_set_weight,
        );
        let h_after = self.union_heuristic_min_swaps_with_layout(
            &self.layout,
            u_front,
            u_ext,
            extended_set_weight,
        );

        (h_before - h_after) / union_size
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
            executed: self.executed.clone(),
        }
    }

    fn load_snapshot(&mut self, state: State) {
        self.front_layer = state.front_layer;
        self.required_predecessors = state.required_predecessors;
        self.layout = state.layout;
        self.gate_order = state.gate_order;
        self.executed = state.executed;
    }

    fn advance_front_layer(&mut self, nodes: &[i32]) -> Vec<i32> {
        let mut node_queue: VecDeque<i32> = VecDeque::from(nodes.to_vec());

        let mut advanced_gates: Vec<i32> = Vec::new();

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

            if let None = self.executed.get(&node.id) {
                self.executed.insert(node.id, true);
                self.gate_order.push(node.id);
                advanced_gates.push(node_index);
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

        advanced_gates
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

    fn choose_best_swaps(&mut self, depth: usize) -> Vec<[i32; 2]> {
        let extended_set_weight: f64 = 0.5;

        let initial_state = self.create_snapshot();

        let mut stack = vec![StackItem {
            swap_sequence: Vec::new(),
            remaining_depth: depth,
        }];
        let mut best = Best::new();

        while let Some(item) = stack.pop() {
            self.load_snapshot(initial_state.clone());

            let (advanced_gates, u_front, u_ext) =
                self.apply_prefix(&item.swap_sequence);

            let should_score_leaf = item.remaining_depth == 0 || self.front_layer.is_empty();
            let swap_candidates = if should_score_leaf {
                vec![]
            } else {
                self.compute_swap_candidates()
            };

            if should_score_leaf || swap_candidates.is_empty() {
                let score = self.score_leaf(
                    &initial_state.layout,
                    &u_front,
                    &u_ext,
                    extended_set_weight,
                );

                best.check_best(item.swap_sequence.clone(), advanced_gates.len(), score);
                continue;
            }

            for &swap in &swap_candidates {
                let mut next_seq = item.swap_sequence.clone();
                next_seq.push(swap);
                stack.push(StackItem {
                    swap_sequence: next_seq,
                    remaining_depth: item.remaining_depth - 1,
                });
            }
        }

        self.load_snapshot(initial_state);
        best.seq.unwrap_or_default()
    }

    fn apply_swap(&mut self, swap: [i32; 2]) {
        self.front_layer.apply_swap(swap);
        self.layout.swap_physical(swap);
    }
}
