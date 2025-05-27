use crate::graph::dag::{MicroDAG, MicroDAGNode, NodeId, NodeIndex, PhysicalQubit, VirtualQubit};
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
    distance: Vec<Vec<i32>>
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
            dag: dag,
            current_mapping: initial_mapping.clone(),
            distance: compute_all_pairs_shortest_paths(&coupling_map),
            coupling_map,
            // TODO: This was a defaultdict
            out_map: HashMap::new(),
            gate_order: Vec::new(),
            front_layer: HashSet::new(),
        })
    }

    fn choose_best_swap(&self) -> (i32, i32) {
        unimplemented!()
    }

    fn executable_node_on_qubit(&self, qubit: i32) -> Option<usize> {
        unimplemented!()
    }


    fn run(&mut self, heuristic: String, critical_path: bool, extended_set_size: i32) -> (HashMap<i32, Vec<(i32, i32)>>, Vec<i32>){
        // self.dag.edges().unwrap().iter().for_each(|edge| println!("{:?}", edge));
        self.dag
            .edges()
            .unwrap()
            .iter()
            .for_each(|edge| self.required_predecessors[edge.1 as usize] += 1);
        // println!("Required predecessors: {:?}", self.required_predecessors);

        let successor_map = self.get_successor_map();

        let initial_front = self.initial_front();

        println!("Front layer before advance: {:?}", self.front_layer);
        self.advance_front_layer(initial_front);
        println!("Front layer after advance: {:?}", self.front_layer);

        println!("{:?}", self.coupling_map);

        let mut execute_gate_list: Vec<i32> = Vec::new();

        while !self.front_layer.is_empty() {
            let mut current_swaps: Vec<(i32, i32)> = Vec::new();

            while execute_gate_list.is_empty() {
                let best_swap = self.choose_best_swap();
                
                let physical_q0 = self.current_mapping[&best_swap.0];
                let physical_q1 = self.current_mapping[&best_swap.1];

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
            // TODO: Check whether using the adjacency list is fine
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

        println!("{:?}", successor_set);

        let mut successor_map: HashMap<i32, usize> = HashMap::new();

        for (index, _) in &self.dag.nodes {
            successor_map.insert(*index, successor_set.get(index).map_or(0, |s| s.len()));
        }

        println!("{:?}", successor_map);
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

fn swap_physical_qubits(q1: i32, q2: i32, current_mapping: &HashMap<i32, i32>) -> HashMap<i32, i32> {
    unimplemented!()
}