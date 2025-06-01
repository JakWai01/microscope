use crate::graph::dag::MicroDAG;
use std::{collections::{HashMap, HashSet, VecDeque}, i32};

use pyo3::{pyclass, pymethods, PyResult};

#[pyclass(module = "microboost.routing.sabre")]
pub(crate) struct MicroSABRE {
    dag: MicroDAG,
    current_mapping: HashMap<i32, i32>,
    coupling_map: Vec<Vec<i32>>,
    out_map: HashMap<i32, Vec<(i32, i32)>>,
    gate_order: Vec<i32>,
    front_layer: HashSet<i32>,
    required_predecessors: Vec<i32>,
    adjacency_list: HashMap<i32, Vec<i32>>,
    distance: Vec<Vec<i32>>,
    initial_mapping: HashMap<i32, i32>
}

#[pymethods]
impl MicroSABRE {
    #[new]
    pub fn new(
        dag: MicroDAG,
        initial_mapping: HashMap<i32, i32>,
        coupling_map: Vec<Vec<i32>>,
    ) -> PyResult<Self> {
        Ok(Self {
            required_predecessors: vec![0; dag.nodes.len()],
            adjacency_list: build_adjacency_list(&dag),
            dag,
            current_mapping: initial_mapping.clone(),
            distance: compute_all_pairs_shortest_paths(&coupling_map),
            coupling_map,
            out_map: HashMap::new(),
            gate_order: Vec::new(),
            front_layer: HashSet::new(),
            initial_mapping: initial_mapping
        })
    }

    fn clear_data_structures(&mut self) {
        self.required_predecessors = vec![0; self.dag.nodes.len()];
        self.adjacency_list = build_adjacency_list(&self.dag);

        self.current_mapping = self.initial_mapping.clone();
        self.distance = compute_all_pairs_shortest_paths(&self.coupling_map);

        self.out_map.clear();
        self.gate_order.clear();
        self.front_layer.clear();

    }

    fn calculate_heuristic(
        &mut self,
        front_layer: HashSet<i32>,
        current_mapping: HashMap<i32, i32>,
        heuristic: String,
        extended_set_size: i32
    ) -> f64 {
        match heuristic.as_str() {
            "basic" => self.h_basic(front_layer, current_mapping, 1.0, false),
            "basic-scale" => self.h_basic(front_layer, current_mapping, 1.0, true),
            "lookahead" => self.h_lookahead(front_layer, current_mapping, 1.0, false, extended_set_size),
            "lookahead-0.5" => self.h_lookahead(front_layer, current_mapping, 0.5, false, extended_set_size),
            "lookahead-scaling" => self.h_lookahead(front_layer, current_mapping, 1.0, false, extended_set_size),
            "lookahead-0.5-scaling" => self.h_lookahead(front_layer, current_mapping, 1.0, true, extended_set_size),
            _ => panic!("Unknown heuristic type: {}", heuristic),
        }
    }

    // A lot of cloning in here
    fn h_lookahead(
        &mut self,
        front_layer: HashSet<i32>,
        current_mapping: HashMap<i32, i32>,
        weight: f64,
        scale: bool,
        extended_set_size: i32
    ) -> f64 {
        if front_layer.is_empty() {
            return 0.0;
        }

        let h_basic_result = self.h_basic(front_layer.clone(), current_mapping.clone(), 1.0, scale);
        let extended_set = self.get_extended_set(extended_set_size); // Returns HashSet<usize>

        let h_basic_result_extended = self.h_basic(extended_set.clone(), current_mapping, 1.0, scale);

        let adjusted_weight = if scale {
            if extended_set.is_empty() {
                0.0
            } else {
                weight / extended_set.len() as f64
            }
        } else {
            weight
        };

        let front_len = front_layer.len() as f64;
        let extended_len = extended_set.len().max(1) as f64; // Avoid division by zero

        (1.0 / front_len) * h_basic_result + adjusted_weight * (1.0 / extended_len) * h_basic_result_extended
    }

    fn h_basic(
        &self,
        front_layer: HashSet<i32>,
        current_mapping: HashMap<i32, i32>,
        weight: f64,
        scale: bool,
    ) -> f64 {
        let mut h_sum = 0.0;

        for gate in &front_layer {
            let node = self.dag.get(*gate).unwrap(); // Assumes node has a `.qubits: Vec<usize>`

            if node.qubits.len() == 1 {
                continue;
            }

            let q0 = node.qubits[0];
            let q1 = node.qubits[1];
            let physical_q0 = current_mapping[&q0];
            let physical_q1 = current_mapping[&q1];

            let actual_weight = if scale {
                if front_layer.is_empty() {
                    0.0
                } else {
                    weight / front_layer.len() as f64
                }
            } else {
                weight
            };

            let distance = self.distance[physical_q0 as usize][physical_q1 as usize];
            h_sum += actual_weight * distance as f64;
        }

        h_sum
    }

     fn get_extended_set(&mut self, extended_set_size: i32) -> HashSet<i32> {
        let mut required_predecessors = self.required_predecessors.clone();

        let mut to_visit: Vec<i32> = self.front_layer.iter().copied().collect();
        let mut i = 0;

        let mut extended_set: HashSet<i32> = HashSet::new();
        let mut visit_now: Vec<i32> = Vec::new();

        let mut decremented: HashMap<i32, i32> = HashMap::new();
        let mut visited: HashMap<i32, bool> = HashMap::new();

        while i < to_visit.len() && extended_set.len() < extended_set_size as usize {
            visit_now.push(to_visit[i]);
            let mut j = 0;

            while j < visit_now.len() {
                let node_id = visit_now[j];

                if let Some(successors) = self.adjacency_list.get(&(node_id as i32)) {
                    for &successor in successors {
                        if *visited.get(&successor).unwrap_or(&false) == false {
                            visited.insert(successor, true);

                            *decremented.entry(successor).or_insert(0) += 1;
                            required_predecessors[successor as usize] -= 1;

                            if required_predecessors[successor as usize] == 0 {
                                if self.dag.get(successor).unwrap().qubits.len() == 2 {
                                    extended_set.insert(successor);
                                    to_visit.push(successor);
                                    continue;
                                }
                                visit_now.push(successor);
                            }

                            // Also adding the first layer of unroutable gates seems to improve results
                            if self.dag.get(successor).unwrap().qubits.len() == 2 {
                                extended_set.insert(successor);
                            }
                        }
                    }
                }
                j += 1;
            }

            visit_now.clear();
            i += 1;
        }

        // Restore required_predecessors
        for (node, amount) in decremented {
            required_predecessors[node as usize] += amount as i32;
        }

        extended_set
    }

    fn choose_best_swap(&mut self, heuristic: String, extended_set_size: i32) -> (i32, i32) {
        let mut scores: HashMap<(i32, i32), f64> = HashMap::new();
        let swap_candidates: Vec<(i32, i32)> =  self.compute_swap_candidates();
        

        for &(q0, q1) in &swap_candidates {
            let physical_q0 = self.current_mapping[&(q0 as i32)];
            let physical_q1 = self.current_mapping[&(q1 as i32)];

            let before = self.calculate_heuristic(self.front_layer.clone(), self.current_mapping.clone(), heuristic.clone(), extended_set_size);

            let temporary_mapping = swap_physical_qubits(physical_q0, physical_q1, &self.current_mapping);

            let after = self.calculate_heuristic(self.front_layer.clone(), temporary_mapping, heuristic.clone(), extended_set_size);

            scores.insert((q0, q1), after - before);
        }

        self.min_score(scores)
    }

    fn compute_swap_candidates(&self) -> Vec<(i32, i32)> {
        let mut swap_candidates: Vec<(i32, i32)> = Vec::new();

        for &gate in &self.front_layer {
            let node = self.dag.get(gate).unwrap(); // Assuming node has a `qubits: Vec<usize>`
            let physical_q0 = self.current_mapping[&node.qubits[0]];
            let physical_q1 = self.current_mapping[&node.qubits[1]];

            for edge in &self.coupling_map {
                let u = edge[0] as usize;
                let v = edge[1] as usize;

                let values: Vec<i32> = self.current_mapping.values().copied().collect();
                if values.contains(&(u as i32)) && values.contains(&(v as i32)) {
                    if u == physical_q0 as usize || u == physical_q1 as usize {
                        // Find the logical qubits mapped to u and v
                        let logical_q0 = self.current_mapping.iter()
                            .find(|(_, &v_val)| v_val as usize == u)
                            .map(|(&k, _)| k)
                            .unwrap();

                        let logical_q1 = self.current_mapping.iter()
                            .find(|(_, &v_val)| v_val as usize == v)
                            .map(|(&k, _)| k)
                            .unwrap();

                        swap_candidates.push((logical_q0, logical_q1));
                    }
                }
            }
        }

        swap_candidates
    }


    // Assuming this returns the swap with the lowest score
    fn min_score(&self, scores: HashMap<(i32, i32), f64>) -> (i32, i32) {
        *scores.iter().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }

    fn executable_node_on_qubit(&self, physical_qubit: i32) -> Option<i32> {
        for &node_id in &self.front_layer {
            let node = self.dag.get(node_id).unwrap();
            if node.qubits.len() < 2 {
                continue;
            }

            let physical_q0 = self.current_mapping[&node.qubits[0]];
            let physical_q1 = self.current_mapping[&node.qubits[1]];

            if physical_q0 == physical_qubit || physical_q1 == physical_qubit {
                if self.distance[physical_q0 as usize][physical_q1 as usize] == 1 {
                    return Some(node_id);
                }
            }
        }
        None
    }


    fn run(&mut self, heuristic: String, _critical_path: bool, extended_set_size: i32) -> (HashMap<i32, Vec<(i32, i32)>>, Vec<i32>){
        self.clear_data_structures();

        self.dag
            .edges()
            .unwrap()
            .iter()
            .for_each(|edge| self.required_predecessors[edge.1 as usize] += 1);

        let initial_front = self.initial_front();

        self.advance_front_layer(initial_front);

        let mut execute_gate_list: Vec<i32> = Vec::new();

        while !self.front_layer.is_empty() {
            let mut current_swaps: Vec<(i32, i32)> = Vec::new();

            while execute_gate_list.is_empty() {
                // This clone costs a bit performance
                let best_swap = self.choose_best_swap(heuristic.clone(), extended_set_size);
                
                let physical_q0 = self.current_mapping[&(best_swap.0 as i32)];
                let physical_q1 = self.current_mapping[&(best_swap.1 as i32)];

                current_swaps.push(best_swap);
                self.current_mapping = swap_physical_qubits(
                    physical_q0,
                    physical_q1,
                    &self.current_mapping,
                );

                if let Some(node) = self.executable_node_on_qubit(physical_q0) {
                    execute_gate_list.push(node as i32);
                }

                if let Some(node) = self.executable_node_on_qubit(physical_q1) {
                    execute_gate_list.push(node as i32);
                }
            }

            let node_id = self.dag.get(execute_gate_list[0] as i32).unwrap().id;
            self.out_map
                .entry(node_id)
                .or_default()
                .extend(current_swaps.clone());

            for &node in &execute_gate_list {
                self.front_layer.remove(&(node as i32));
            }

            self.advance_front_layer(execute_gate_list.clone());
            execute_gate_list.clear();
        }

        (self.out_map.clone(), self.gate_order.clone()) 
    }

    fn advance_front_layer(&mut self, nodes: Vec<i32>) {
        // Copy input into a queue
        let mut node_queue: VecDeque<i32> = VecDeque::from(nodes);

        while let Some(node_index) = node_queue.pop_front() {
            let node = self.dag.get(node_index).unwrap(); // Assuming this returns a reference to Node

            if node.qubits.len() == 2 {
                let physical_q0 = self.current_mapping[&node.qubits[0]];
                let physical_q1 = self.current_mapping[&node.qubits[1]];

                // Check whether the node can be executed on the current mapping
                if self.distance[physical_q0 as usize][physical_q1 as usize] != 1 {
                    self.front_layer.insert(node_index);
                    continue;
                }
            }

            // Node can be executed
            if !self.gate_order.contains(&node.id) {
                self.gate_order.push(node.id);
            }

            // Check successors
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
    
    fn get_successor_map(&self) -> HashMap<i32, usize> {
        // Node index or node id?
        let mut successor_set: HashMap<i32, HashSet<i32>> = HashMap::new();

        for u in (0..self.dag.nodes.len()).rev() {
            successor_set.entry(u as i32).or_insert(HashSet::new());

            if let Some(neighbors) = self.adjacency_list.get(&(u as i32)) {
                for &v in neighbors {
                    // Add direct successor
                    successor_set.get_mut(&(u as i32)).unwrap().insert(v);

                    if let Some(successors) = successor_set.get(&v) {
                            // Clone before the mutable borrow
                        let v_successors_cloned: HashSet<_> = successors.iter().cloned().collect();
                        successor_set.get_mut(&(u as i32)).unwrap().extend(v_successors_cloned);
                    }
                }
            }
        }

        let mut successor_map: HashMap<i32, usize> = HashMap::new();

        for (index, _) in &self.dag.nodes {
            successor_map.insert(*index, successor_set.get(index).map_or(0, |s| s.len()));
        }

        successor_map
    }

}

fn build_adjacency_list(dag: &MicroDAG) -> HashMap<i32, Vec<i32>> {
    let mut adj = HashMap::new();
    for (u, v) in &dag.edges {
        adj.entry(*u).or_insert(Vec::new()).push(*v);
    }
    adj
}


fn compute_all_pairs_shortest_paths(coupling_map: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let n = coupling_map.iter().flatten().copied().max().unwrap_or(0) as usize + 1;
    let mut dist = vec![vec![i32::MAX / 2; n]; n]; // Avoid overflow

    // Distance from a node to itself is 0
    for i in 0..n {
        dist[i][i] = 0;
    }

    // Distance between directly connected nodes is 1
    for edge in coupling_map {
        let u = edge[0] as usize;
        let v = edge[1] as usize;
        dist[u][v] = 1;
    }

    // Floyd-Warshall algorithm
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if dist[i][k] + dist[k][j] < dist[i][j] {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }

    dist
}

fn swap_physical_qubits(
    physical_q0: i32,
    physical_q1: i32,
    current_mapping: &HashMap<i32, i32>,
) -> HashMap<i32, i32> {
    let mut resulting_mapping = current_mapping.clone();

    // Find logical qubits corresponding to the physical ones
    let logical_q0 = current_mapping
        .iter()
        .find_map(|(&key, &value)| if value == physical_q0 { Some(key) } else { None })
        .expect("physical_q0 not found in current_mapping");

    let logical_q1 = current_mapping
        .iter()
        .find_map(|(&key, &value)| if value == physical_q1 { Some(key) } else { None })
        .expect("physical_q1 not found in current_mapping");

    // Swap their mapped values
    let tmp = resulting_mapping[&logical_q0];
    resulting_mapping.insert(logical_q0, resulting_mapping[&logical_q1]);
    resulting_mapping.insert(logical_q1, tmp);

    resulting_mapping
}