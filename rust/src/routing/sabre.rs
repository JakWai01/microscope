use crate::graph::dag::MicroDAG;
use std::{collections::{HashMap, HashSet, VecDeque}, i32, time::Instant};

use pyo3::{pyclass, pymethods, PyResult};

use indexmap::IndexMap;

#[derive(Clone)]
#[pyclass(module = "microboost.routing.sabre")]
pub(crate) struct MicroFront {
    nodes: IndexMap<i32, [i32; 2]>,
    qubits: Vec<Option<(i32, i32)>>,
}

impl MicroFront {
    pub fn new(num_qubits: i32) -> Self {
        Self {
            // with_capacity_and_hasher
            nodes: IndexMap::with_capacity(
                num_qubits as usize / 2
            ),
            qubits: vec![None; num_qubits as usize]
        }
    }

    pub fn is_empty(self) -> bool {
        self.nodes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn insert(&mut self, index: i32, qubits: [i32; 2]) {
        let [a, b] = qubits;
        self.qubits[a as usize] = Some((index, b));
        self.qubits[b as usize] = Some((index, a));
        self.nodes.insert(index, qubits);
    }

    pub fn remove(&mut self, index: &i32) {
        // The actual order in the indexmap doesn't matter as long as it's reproducible.
        // Swap-remove is more efficient than a full shift-remove.
        let [a, b] = self
            .nodes
            .swap_remove(index)
            .expect("Tried removing index that does not exist.");
        self.qubits[a as usize] = None;
        self.qubits[b as usize] = None;
    }

    pub fn is_active(&self, qubit: i32) -> bool {
        self.qubits[qubit as usize].is_some()
    }

        /// Apply a physical swap to the current layout data structure.
    pub fn apply_swap(&mut self, swap: [i32; 2]) {
        let [a, b] = swap;
        match (self.qubits[a as usize], self.qubits[b as usize]) {
            (Some((index1, _)), Some((index2, _))) if index1 == index2 => {
                let entry = self.nodes.get_mut(&index1).unwrap();
                *entry = [entry[1], entry[0]];
                return;
            }
            _ => {}
        }
        if let Some((index, c)) = self.qubits[a as usize] {
            self.qubits[c as usize] = Some((index, b));
            let entry = self.nodes.get_mut(&index).unwrap();
            *entry = if *entry == [a, c] { [b, c] } else { [c, b] };
        }
        if let Some((index, c)) = self.qubits[b as usize] {
            self.qubits[c as usize] = Some((index, a));
            let entry = self.nodes.get_mut(&index).unwrap();
            *entry = if *entry == [b, c] { [a, c] } else { [c, a] };
        }
        self.qubits.swap(a as usize, b as usize);
    }

}

#[derive(Clone)]
#[pyclass(module = "microboost.routing.sabre")]
pub(crate) struct MicroLayout {
    virt_to_phys: Vec<i32>,
    phys_to_virt: Vec<i32>
}

#[pymethods]
impl MicroLayout {
    #[new]
    pub fn new(
        qubit_indices: HashMap<i32, i32>,
        virtual_qubits: usize,
        physical_qubits: usize
    ) -> Self {
        let mut res = MicroLayout {
            virt_to_phys: vec![i32::MAX; virtual_qubits],
            phys_to_virt: vec![i32::MAX; physical_qubits],
        };
        for (virt, phys) in qubit_indices {
            res.virt_to_phys[virt as usize] = phys;
            res.phys_to_virt[phys as usize] = virt;
        }
        res
    }

    pub fn virtual_to_physical(&self, virt: i32) -> i32 {
        self.virt_to_phys[virt as usize]
    }

    pub fn physical_to_virtual(&self, phys: i32) -> i32 {
        self.phys_to_virt[phys as usize]
    }

    pub fn swap_virtual(&mut self, bit_a: i32, bit_b: i32) {
        self.virt_to_phys.swap(bit_a as usize, bit_b as usize);
        self.phys_to_virt[self.virt_to_phys[bit_a as usize] as usize] = bit_a;
        self.phys_to_virt[self.virt_to_phys[bit_b as usize] as usize] = bit_b;
    }

    pub fn swap_physical(&mut self, bit_a: i32, bit_b: i32) {
        self.phys_to_virt.swap(bit_a as usize, bit_b as usize);
        self.virt_to_phys[self.phys_to_virt[bit_a as usize] as usize] = bit_a;
        self.virt_to_phys[self.phys_to_virt[bit_b as usize] as usize] = bit_b;
    }
}

#[pyclass(module = "microboost.routing.sabre")]
pub(crate) struct MicroSABRE {
    dag: MicroDAG,
    coupling_map: Vec<Vec<i32>>,
    out_map: HashMap<i32, Vec<(i32, i32)>>,
    gate_order: Vec<i32>,
    front_layer: MicroFront,
    required_predecessors: Vec<i32>,
    adjacency_list: HashMap<i32, Vec<i32>>,
    distance: Vec<Vec<i32>>,
    initial_mapping: MicroLayout,
    initial_dag: MicroDAG,
    initial_coupling_map: Vec<Vec<i32>>,
    neighbour_map: HashMap<i32, Vec<i32>>,
    layout: MicroLayout,
    num_qubits: i32
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
            out_map: HashMap::new(),
            gate_order: Vec::new(),
            front_layer: MicroFront::new(num_qubits),
            initial_mapping: initial_layout.clone(),
            initial_dag: dag,
            neighbour_map: build_coupling_neighbour_map(&coupling_map),
            initial_coupling_map: coupling_map,
            num_qubits
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
        self.distance = compute_all_pairs_shortest_paths(&self.coupling_map);

        self.out_map.clear();
        self.gate_order.clear();
        self.front_layer = MicroFront::new(self.num_qubits);

        self.dag = self.initial_dag.clone();
        self.coupling_map = self.initial_coupling_map.clone();

    }

    fn run(&mut self, heuristic: String, _critical_path: bool, extended_set_size: i32) -> (HashMap<i32, Vec<(i32, i32)>>, Vec<i32>){
        // let now = Instant::now();
        self.clear_data_structures();
        // let elapsed = now.elapsed();
        // println!("<clear_data_structures> took {:?}", elapsed);
        
        
        // let now = Instant::now();
        self.dag
            .edges()
            .unwrap()
            .iter()
            .for_each(|edge| self.required_predecessors[edge.1 as usize] += 1);
        // let elapsed = now.elapsed();
        // println!("<required_predecessors> took {:?}", elapsed);

        // let now = Instant::now();
        let initial_front = self.initial_front();
        // let elapsed = now.elapsed();
        // println!("<initial_front> took {:?}", elapsed);

        // let now = Instant::now();
        self.advance_front_layer(initial_front);
        // let elapsed = now.elapsed();
        // println!("<advance_front_layer> took {:?}", elapsed);

        let mut execute_gate_list: Vec<i32> = Vec::new();

        while !self.front_layer.clone().is_empty() {
            let mut current_swaps: Vec<(i32, i32)> = Vec::new();

            while execute_gate_list.is_empty() {
                // if current_swaps.len() > 100 {
                //     panic!("Ladies and gentleman, we are looping!")
                // }
                // let now = Instant::now();
                let best_swap = self.choose_best_swap(heuristic.clone(), extended_set_size);
                // println!("Best swap: {:?}", best_swap);
                // let elapsed = now.elapsed();
                // println!("<choose_best_swap> took {:?}", elapsed);
                
                let physical_q0 = best_swap.0;
                let physical_q1 = best_swap.1;
                
                current_swaps.push(best_swap);
                // self.layout.swap_physical(physical_q0, physical_q1);
                self.apply_swap((physical_q0, physical_q1));

                // let now = Instant::now();
                if let Some(node) = self.executable_node_on_qubit(physical_q0) {
                    execute_gate_list.push(node as i32);
                }

                if let Some(node) = self.executable_node_on_qubit(physical_q1) {
                    execute_gate_list.push(node as i32);
                }
                // let elapsed = now.elapsed();
                // println!("<executable_node_on_qubit> took {:?}", elapsed);
            }

            let node_id = self.dag.get(execute_gate_list[0] as i32).unwrap().id;
            self.out_map
                .entry(node_id)
                .or_default()
                .extend(current_swaps.clone());

            for &node in &execute_gate_list {
                self.front_layer.remove(&(node as i32));
            }

            // let now = Instant::now();
            self.advance_front_layer(execute_gate_list.clone());
            execute_gate_list.clear();
            // let elapsed = now.elapsed();
            // println!("<advance_front_layer> took {:?}", elapsed);
            
            // panic!("Stopping here");
        }

        (self.out_map.clone(), self.gate_order.clone()) 
    }
    fn calculate_heuristic(
        &mut self,
        front_layer: MicroFront,
        layout: &MicroLayout,
        heuristic: String,
        extended_set_size: i32
    ) -> f64 {
        match heuristic.as_str() {
            "basic" => self.h_basic(front_layer, layout, 1.0, false),
            "basic-scale" => self.h_basic(front_layer, layout, 1.0, true),
            "lookahead" => self.h_lookahead(front_layer, layout, 1.0, false, extended_set_size),
            "lookahead-0.5" => self.h_lookahead(front_layer, layout, 0.5, false, extended_set_size),
            "lookahead-scaling" => self.h_lookahead(front_layer, layout,1.0, false, extended_set_size),
            "lookahead-0.5-scaling" => self.h_lookahead(front_layer, layout, 1.0, true, extended_set_size),
            _ => panic!("Unknown heuristic type: {}", heuristic),
        }
    }

    // A lot of cloning in here
    fn h_lookahead(
        &mut self,
        front_layer: MicroFront,
        layout: &MicroLayout,
        weight: f64,
        scale: bool,
        extended_set_size: i32
    ) -> f64 {
        // let now_f = Instant::now();

        // let now = Instant::now();
        if front_layer.clone().is_empty() {
            return 0.0;
        }
        // let elapsed = now.elapsed();
        // println!("<is_empty> took {:?}", elapsed);


        // let now = Instant::now();
        let h_basic_result = self.h_basic(front_layer.clone(), layout, 1.0, scale);
        // let elapsed = now.elapsed();
        // println!("<h_basic> took {:?}", elapsed);
        
        // let now = Instant::now();
        let extended_set = self.get_extended_set(extended_set_size); // Returns HashSet<usize>
        // let elapsed = now.elapsed();
        // println!("<get_extended_set> took {:?}", elapsed);

        // let now = Instant::now();
        let h_basic_result_extended = self.h_basic(extended_set.clone(), layout, 1.0, scale);
        // let elapsed = now.elapsed();
        // println!("<h_basic> took {:?}", elapsed);

        // let now = Instant::now();
        let adjusted_weight = if scale {
            // TODO: I really don't think this cloning is necessary
            if extended_set.clone().is_empty() {
                0.0
            } else {
                weight / extended_set.len() as f64
            }
        } else {
            weight
        };
        // let elapsed = now.elapsed();
        // println!("<adjusted_weight> took {:?}", elapsed);


        // let now = Instant::now();
        let front_len = front_layer.len() as f64;
        let extended_len = extended_set.len().max(1) as f64; // Avoid division by zero
        // let elapsed = now.elapsed();
        // println!("<len> took {:?}", elapsed);

        // let elapsed = now_f.elapsed();
        // println!("<h_lookahead> took {:?}", elapsed);
        (1.0 / front_len) * h_basic_result + adjusted_weight * (1.0 / extended_len) * h_basic_result_extended
    }

    fn h_basic(
        &self,
        front_layer: MicroFront,
        layout: &MicroLayout,
        weight: f64,
        scale: bool,
    ) -> f64 {
        let mut h_sum = 0.0;

        // TODO: This could be simplified if iterating over active qubits right?
        for gate in front_layer.nodes.keys() {
            let node = self.dag.get(*gate).unwrap();
            
            // This shouldn't be possible, right?
            if node.qubits.len() == 1 {
                continue;
            }

            let q0 = node.qubits[0];
            let q1 = node.qubits[1];
            let physical_q0 = layout.virtual_to_physical(q0);
            let physical_q1 = layout.virtual_to_physical(q1);

            let actual_weight = if scale {
                if front_layer.nodes.is_empty() {
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

     fn get_extended_set(&mut self, extended_set_size: i32) -> MicroFront {

        let mut required_predecessors = self.required_predecessors.clone();

        let mut to_visit: Vec<i32> = self.front_layer.nodes.keys().copied().collect();
        let mut i = 0;

        let mut extended_set: MicroFront = MicroFront::new(self.num_qubits);
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
                        let succ = self.dag.get(successor).unwrap();
                        if *visited.get(&successor).unwrap_or(&false) == false {
                            visited.insert(successor, true);

                            *decremented.entry(successor).or_insert(0) += 1;
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

                            // Also adding the first layer of unroutable gates seems to improve results
                            if self.dag.get(successor).unwrap().qubits.len() == 2 {
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

        // Restore required_predecessors
        for (node, amount) in decremented {
            required_predecessors[node as usize] += amount as i32;
        }
        extended_set
    }

    fn choose_best_swap(&mut self, heuristic: String, extended_set_size: i32) -> (i32, i32) {
        let mut scores: HashMap<(i32, i32), f64> = HashMap::new();

        let swap_candidates: Vec<(i32, i32)> =  self.compute_swap_candidates();

        // let now = Instant::now();
        for &(q0, q1) in &swap_candidates {
            let before = self.calculate_heuristic(self.front_layer.clone(), &self.layout.clone(), heuristic.clone(), extended_set_size);

            let mut temporary_mapping = self.layout.clone();
            temporary_mapping.swap_physical(q0, q1);

            let after = self.calculate_heuristic(self.front_layer.clone(), &temporary_mapping, heuristic.clone(), extended_set_size);

            scores.insert((q0, q1), after - before);
        }
        // let elapsed = now.elapsed();
        // println!("<scoring> took {:?}", elapsed);

        self.min_score(scores)
    }

    fn compute_swap_candidates(&self) -> Vec<(i32, i32)> {
        // let mut swap_candidates: Vec<(i32, i32)> = Vec::new();
        let mut swap_candidates_new: Vec<(i32, i32)> = Vec::new();

        // They should literally be the same
        // let mut phys: Vec<i32> = Vec::new();
        let mut phys_new: Vec<i32> = Vec::new();

        // println!("Items in front_layer: {:?}", self.front_layer.len());
        // // TODO: Use only active qubits here
        // for &gate in self.front_layer.nodes.keys() {
        //     let node = self.dag.get(gate).unwrap();
        //     let physical_q0 = self.layout.virtual_to_physical(node.qubits[0]);
        //     phys.push(physical_q0);
        //     let physical_q1 = self.layout.virtual_to_physical(node.qubits[1]);
        //     phys.push(physical_q1);
         
        //     for neighbour in self.neighbour_map[&physical_q0].iter() {
        //         swap_candidates.push((physical_q0, *neighbour))
        //     }
         
        //     for neighbour in self.neighbour_map[&physical_q1].iter() {
        //         swap_candidates.push((physical_q1, *neighbour))
        //     }
        // }


        // I think the code is the same and the bug lies in the underlying data structure 
        for &phys in self.front_layer.nodes.values().flatten() {
            phys_new.push(phys);
            for neighbour in self.neighbour_map[&phys].iter() {
                // if neighbour > phys || !self.front_layer.is_active(neighbour) {
                    swap_candidates_new.push((phys, *neighbour))
                // }
            }
        }

        // println!("Phys: {:?}", phys);
        // println!("Phys_new: {:?}", phys_new);

        // println!("Length swap candidates: {:?}", swap_candidates.len());
        // println!("Length swap candidates new: {:?}", swap_candidates_new.len());
        // let difference: Vec<_> = swap_candidates.clone().into_iter().filter(|item| !swap_candidates_new.contains(item)).collect();
        // let difference_new: Vec<_> = swap_candidates_new.clone().into_iter().filter(|item| !swap_candidates.contains(item)).collect();
        // println!("Diff {:?}", difference);
        // println!("Diff2 {:?}", difference_new);

        swap_candidates_new
    }


    // Assuming this returns the swap with the lowest score
    fn min_score(&self, scores: HashMap<(i32, i32), f64>) -> (i32, i32) {
        *scores.iter().min_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }

    fn executable_node_on_qubit(&self, physical_qubit: i32) -> Option<i32> {
        // TODO: We could only use active ones here as well
        for &node_id in self.front_layer.nodes.keys() {
            let node = self.dag.get(node_id).unwrap();

            // This can't happen in the front_layer right?
            if node.qubits.len() < 2 {
                continue;
            }

            let physical_q0 = self.layout.virtual_to_physical(node.qubits[0]);
            let physical_q1 = self.layout.virtual_to_physical(node.qubits[1]);

            if physical_q0 == physical_qubit || physical_q1 == physical_qubit {
                if self.distance[physical_q0 as usize][physical_q1 as usize] == 1 {
                    return Some(node_id);
                }
            }
        }
        None
    }



    fn advance_front_layer(&mut self, nodes: Vec<i32>) {
        // Copy input into a queue
        let mut node_queue: VecDeque<i32> = VecDeque::from(nodes);

        while let Some(node_index) = node_queue.pop_front() {
            let node = self.dag.get(node_index).unwrap(); // Assuming this returns a reference to Node

            if node.qubits.len() == 2 {
                let physical_q0 = self.layout.virtual_to_physical(node.qubits[0]);
                let physical_q1 = self.layout.virtual_to_physical(node.qubits[1]);

                // Check whether the node can be executed on the current mapping
                if self.distance[physical_q0 as usize][physical_q1 as usize] != 1 {
                    self.front_layer.insert(node_index, [physical_q0, physical_q1]);
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
    
fn build_coupling_neighbour_map(coupling_map: &Vec<Vec<i32>>) -> HashMap<i32, Vec<i32>> {
    let mut neighbour_map = HashMap::new();

    for edge in coupling_map {
        let u = edge[0];
        let v = edge[1];

        neighbour_map.entry(u).or_insert(Vec::new()).push(v);
    }

    neighbour_map
}