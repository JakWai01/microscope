use rustc_hash::FxHashMap;
use crate::{
    routing::{front_layer::MicroFront, layout::MicroLayout},
    MicroDAG,
};

#[derive(Clone)]
pub struct State {
    pub front_layer: MicroFront,
    pub required_predecessors: Vec<i32>,
    pub layout: MicroLayout,
    pub gate_order: Vec<i32>,
    pub executed: FxHashMap<i32, bool>,
}

#[derive(Clone)]
pub struct StackItem {
    pub swap_sequence: Vec<[i32; 2]>,
    pub remaining_depth: usize,
}

#[derive(Clone)]
pub struct Best {
    pub seq: Option<Vec<[i32; 2]>>,
    pub exec: usize,
    pub secondary: f64,
    pub len: usize,
}

impl Best {
    pub fn new() -> Self {
        Self {
            seq: None,
            exec: 0,
            secondary: f64::NEG_INFINITY,
            len: usize::MAX,
        }
    }

    pub fn check_best(&mut self, seq: Vec<[i32; 2]>, exec: usize, secondary: f64) {
        let len = seq.len();

        let better = (exec > self.exec)
            || (exec == self.exec
                && (secondary > self.secondary + f64::EPSILON
                    || ((secondary - self.secondary).abs() <= f64::EPSILON && len < self.len)));

        let equal = exec == self.exec
            && (secondary - self.secondary).abs() <= f64::EPSILON
            && len == self.len;

        if better || (equal && rand::random::<bool>()) {
            self.exec = exec;
            self.secondary = secondary;
            self.len = len;
            self.seq = Some(seq);
        }
    }
}

pub fn build_adjacency_list(dag: &MicroDAG) -> Vec<Vec<i32>> {
    let mut adjacency = vec![Vec::new(); dag.nodes.len()];
    for &(src, dst) in dag.edges().unwrap().iter() {
        adjacency[src as usize].push(dst);
    }
    adjacency
}

pub fn compute_all_pairs_shortest_paths(coupling_map: &Vec<Vec<i32>>) -> Vec<Vec<i32>> {
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

pub fn build_coupling_neighbour_map(coupling_map: &[Vec<i32>]) -> Vec<Vec<i32>> {
    let n = coupling_map.len();
    let mut neighbours = vec![Vec::new(); n];
    for edge in coupling_map {
        let q0 = edge[0] as usize;
        let q1 = edge[1] as usize;
        neighbours[q0].push(edge[1]);
        neighbours[q1].push(edge[0]);
    }
    neighbours
}