use ahash::HashSet;
use rand::seq::IndexedRandom;
use rustc_hash::FxHashMap;
use rustworkx_core::petgraph::graph::DiGraph;

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

pub fn min_score(scores: FxHashMap<Vec<[i32; 2]>, (f64, usize)>, epsilon: f64) -> Vec<[i32; 2]> {
    let mut best_swap_sequences = Vec::new();

    let mut iter = scores.iter();

    let (min_swap_sequence, mut min_score) = iter.next().unwrap();
    best_swap_sequences.push(min_swap_sequence);

    for (swap_sequence, score) in iter {
        let diff = score.0 - min_score.0;

        if diff < -epsilon {
            min_score = score;
            best_swap_sequences.clear();
            best_swap_sequences.push(swap_sequence);
        } else if diff.abs() <= epsilon {
            best_swap_sequences.push(swap_sequence);
        }
    }

    let mut rng = rand::rng();

    best_swap_sequences.choose(&mut rng).unwrap().to_vec()
}

pub fn build_digraph_from_neighbors(neighbor_map: &Vec<Vec<i32>>) -> DiGraph<(), ()> {
    let edge_list: Vec<(u32, u32)> = neighbor_map
        .iter()
        .enumerate()
        .flat_map(|(src, targets)| targets.iter().map(move |&dst| (src as u32, dst as u32)))
        .collect();

    DiGraph::<(), ()>::from_edges(edge_list)
}

pub fn get_successor_map_and_critical_paths(dag: &MicroDAG) -> (Vec<usize>, Vec<usize>) {
    // adjacency list as Vec<Vec<i32>>
    let adj: Vec<Vec<i32>> = build_adjacency_list(dag);

    // successor sets, indexed by node ID
    let mut successor_set: Vec<HashSet<i32>> =
        (0..dag.nodes.len()).map(|_| HashSet::default()).collect();

    // critical path lengths, indexed by node ID
    let mut critical_path_len: Vec<usize> = vec![0; dag.nodes.len()];

    // Reverse topological traversal: assumes nodes are 0..N and acyclic
    for u in (0..dag.nodes.len()).rev() {
        for &v in &adj[u] {
            if u == v as usize {
                continue; // skip self-loop just in case
            }

            // split into two disjoint mutable slices
            let (low, high) = if u < v as usize {
                let (low, high) = successor_set.split_at_mut(v as usize);
                (&mut low[u], &high[0])
            } else {
                let (low, high) = successor_set.split_at_mut(u);
                (&mut high[0], &low[v as usize])
            };

            // Add v as successor of u
            low.insert(v);
            // Add all successors of v to u
            low.extend(high.iter().copied());

            // Update critical path length
            let cand_len = 1 + critical_path_len[v as usize];
            if cand_len > critical_path_len[u] {
                critical_path_len[u] = cand_len;
            }
        }
    }

    let successor_counts: Vec<usize> = successor_set.iter().map(|s| s.len()).collect();

    (successor_counts, critical_path_len)
}
