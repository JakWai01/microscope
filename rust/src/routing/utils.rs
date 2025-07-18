use rand::{rng, seq::IndexedRandom};
use rustc_hash::FxHashMap;

use crate::MicroDAG;

pub fn build_adjacency_list(dag: &MicroDAG) -> FxHashMap<i32, Vec<i32>> {
    let mut adj = FxHashMap::default();
    for (u, v) in &dag.edges {
        adj.entry(*u).or_insert(Vec::new()).push(*v);
    }
    adj
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

pub fn build_coupling_neighbour_map(coupling_map: &Vec<Vec<i32>>) -> FxHashMap<i32, Vec<i32>> {
    let mut neighbour_map = FxHashMap::default();

    for edge in coupling_map {
        let u = edge[0];
        let v = edge[1];

        neighbour_map.entry(u).or_insert(Vec::new()).push(v);
    }

    neighbour_map
}

pub fn min_score(scores: FxHashMap<Vec<[i32; 2]>, f64>) -> Vec<[i32; 2]> {
    let mut best_swap_sequences = Vec::new();

    let mut iter = scores.iter();

    let (min_swap_sequence, mut min_score) = iter.next().unwrap();

    best_swap_sequences.push(min_swap_sequence);

    for (swap_sequence, score) in iter {
        if score < min_score {
            min_score = score;
            best_swap_sequences.clear();
            best_swap_sequences.push(swap_sequence);
        } else if score == min_score {
            best_swap_sequences.push(swap_sequence);
        }
    }

    let mut rng = rng();

    best_swap_sequences.choose(&mut rng).unwrap().to_vec()
}