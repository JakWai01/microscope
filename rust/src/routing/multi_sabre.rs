use std::{collections::{HashSet, VecDeque}, thread::current};

use crate::{graph::dag::MicroDAG, routing::{front_layer::MicroFront, layout::MicroLayout, sabre::get_successor_map_and_critical_paths, utils::{build_adjacency_list, build_coupling_neighbour_map, compute_all_pairs_shortest_paths}}};
use pyo3::{pyclass, pymethods, PyResult};
use rand::{rng, seq::IndexedRandom};
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
                let swaps = self.compute_swaps();

                for swap in [swaps.0, swaps.1] {
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
    fn advance_front_in_place(&mut self, nodes: &Vec<i32>) -> (MicroFront, Vec<i32>) {
        let mut node_queue: VecDeque<i32> = VecDeque::from(nodes.clone());

        let mut front_layer = self.front_layer.clone();
        let mut required_predecessors = self.required_predecessors.clone();

        while let Some(node_index) = node_queue.pop_front() {
            let node = self.dag.get(node_index).unwrap();

            if node.qubits.len() == 2 {
                let physical_q0 = self.running_mapping.virtual_to_physical(node.qubits[0]);
                let physical_q1 = self.running_mapping.virtual_to_physical(node.qubits[1]);

                if self.distance[physical_q0 as usize][physical_q1 as usize] != 1 {
                    front_layer
                        .insert(node_index, [physical_q0, physical_q1]);
                    continue;
                }
            }

            // if !self.gate_order.contains(&node.id) {
            //     self.gate_order.push(node.id);
            // }
            
            // Careful! We are mutating the required predecessors here
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

    fn compute_swaps(&mut self) -> ((i32, i32), (i32, i32)) {
        let mut scores: FxHashMap<((i32, i32), (i32, i32)), f64> = FxHashMap::default();

        // TODO: This after - before is something that I want to dive deeper into
        // TODO: Ideally we want to compare before to the state of having applied two swaps
        let before = self.calculate_heuristic(&self.front_layer.clone(), &self.required_predecessors.clone());

        let swap_candidates: Vec<(i32, i32)> = self.compute_swap_candidates(&self.front_layer);

        for &(q0, q1) in &swap_candidates {

            // Apply potential first swap
            // TODO: Layer, we could use a stack here to apply and remove all temporary swaps
            // later
            self.apply_swap((q0, q1));
            
            // Remove from front?!?
            // Temporary advance front layer. Do not modify object state!
            let (new_front, new_required_predecessors) = self.advance_front_in_place(&(self.front_layer.nodes.keys().copied().collect()));

            let inner_swap_candidates = self.compute_swap_candidates(&new_front);

            for &(inner_q0, inner_q1) in &inner_swap_candidates {
                self.apply_swap((inner_q0, inner_q1));

                let after = self.calculate_heuristic(&new_front, &new_required_predecessors);

                scores.insert(((q0, q1), (inner_q0, inner_q1)), after - before);

                self.apply_swap((inner_q1, inner_q0));
                self.apply_swap((q1, q0));
            }
        }
        
        min_score(scores)
    }

    fn calculate_heuristic(&mut self, front_layer: &MicroFront, new_required_predecessors: &Vec<i32>) -> f64 {
        let extended_set = self.get_extended_set(front_layer, new_required_predecessors);

        let basic = self.front_layer.nodes.iter().fold(0.0, |h_sum, (_node_id, [a, b])| {
            h_sum + self.distance[*a as usize][*b as usize] as f64
        });

        let lookahead = extended_set.nodes.iter().fold(0.0, |h_sum, (_node_id, [a, b])| {
            h_sum + self.distance[*a as usize][*b as usize] as f64
        });

        basic + 0.5 * lookahead
    }
    

    fn get_extended_set(&mut self, front_layer: &MicroFront, required_predecessors: &Vec<i32>) -> MicroFront {
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

fn min_score(scores: FxHashMap<((i32, i32), (i32, i32)), f64>) -> ((i32, i32), (i32, i32)) {
    let mut best_swap_sequences = Vec::new();

    let mut iter = scores.iter();
    let (min_swap_sequence, mut min_score) = iter.next().map(|(&swap_sequence, &score)| (swap_sequence, score)).unwrap();

    best_swap_sequences.push(min_swap_sequence);

    // TODO: Consider introducing an epsilon threshold here
    for (&swap_sequence, &score) in iter {
        if score < min_score {
            min_score = score;
            best_swap_sequences.clear();
            best_swap_sequences.push(swap_sequence);
        } else if score == min_score {
            best_swap_sequences.push(swap_sequence);
        }
    }
    
    let mut rng = rng();

    *best_swap_sequences.choose(&mut rng).unwrap()

}