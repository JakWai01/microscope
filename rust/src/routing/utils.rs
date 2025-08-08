use ahash::HashSet;
use rand::seq::IndexedRandom;
use rustc_hash::FxHashMap;
use rustworkx_core::petgraph::graph::DiGraph;

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

pub fn best_progress_sequence(
    scores: FxHashMap<Vec<[i32; 2]>, (f64, usize)>,
    epsilon: f64,
) -> Vec<[i32; 2]> {
    let mut best_swap_sequences = Vec::new();
    let mut iter = scores.iter();

    let (first_seq, &(first_score, first_executed)) = iter.next().unwrap();
    let mut max_exec = first_executed;
    let mut best_score = first_score;
    let mut min_len = first_seq.len();

    best_swap_sequences.push(first_seq);

    for (swap_sequence, &(score, executed)) in iter {
        let len = swap_sequence.len();

        if executed > max_exec {
            max_exec = executed;
            best_score = score;
            min_len = len;
            best_swap_sequences.clear();
            best_swap_sequences.push(swap_sequence);
            // println!("Executed decides!");
        } else if executed == max_exec {
            let diff = score - best_score;
            // if diff < -epsilon || (diff.abs() <= epsilon && len < min_len) {
            //     best_score = score;
            //     min_len = len;
            //     best_swap_sequences.clear();
            //     best_swap_sequences.push(swap_sequence);
            //     // println!("Length decides!");
            // } else if diff.abs() <= epsilon && len == min_len {
            //     best_swap_sequences.push(swap_sequence);
            //     // println!("Score decides!");
            // }
            if len < min_len {
                println!("This case should never happen");
                println!("Best score {:?} Score {:?}", best_score, score);
                best_score = score;
                min_len = len;
                best_swap_sequences.clear();
                best_swap_sequences.push(swap_sequence);
            } else if len == min_len && diff < -epsilon {
                best_score = score;
                min_len = len;
                best_swap_sequences.clear();
                best_swap_sequences.push(swap_sequence);
            } else if len == min_len && diff.abs() <= epsilon {
                best_swap_sequences.push(swap_sequence);
            }
        }
    }

    let mut rng = rand::rng();
    best_swap_sequences.choose(&mut rng).unwrap().to_vec()
}

pub fn build_digraph_from_neighbors(neighbor_map: &FxHashMap<i32, Vec<i32>>) -> DiGraph<(), ()> {
    let edge_list: Vec<(u32, u32)> = neighbor_map
        .iter()
        .flat_map(|(&src, targets)| targets.iter().map(move |&dst| (src as u32, dst as u32)))
        .collect();

    // `from_edges` creates a graph where node indices are inferred from edge endpoints
    DiGraph::<(), ()>::from_edges(edge_list)
}

pub fn get_successor_map_and_critical_paths(dag: &MicroDAG) -> (Vec<usize>, Vec<usize>) {
    let adj = build_adjacency_list(dag);
    let mut successor_set: FxHashMap<i32, HashSet<i32>> =
        dag.nodes.keys().map(|&n| (n, HashSet::default())).collect();
    let mut critical_path_len: FxHashMap<i32, usize> = dag.nodes.keys().map(|&n| (n, 0)).collect();

    // Reverse topological traversal: assumes nodes are 0..N and acyclic
    for u in (0..dag.nodes.len() as i32).rev() {
        if let Some(neighbors) = adj.get(&u) {
            for &v in neighbors {
                // Add v as successor of u
                successor_set.get_mut(&u).unwrap().insert(v);
                // Add all successors of v to u
                if let Some(succ_v) = successor_set.get(&v) {
                    let succ_v_cloned = succ_v.clone();
                    successor_set.get_mut(&u).unwrap().extend(succ_v_cloned);
                }

                // Update critical path length
                let cand_len = 1 + critical_path_len[&v];
                if cand_len > critical_path_len[&u] {
                    critical_path_len.insert(u, cand_len);
                }
            }
        }
    }

    let successor_counts: Vec<usize> = dag.nodes.keys().map(|&n| successor_set[&n].len()).collect();

    let critical_path_lengths: Vec<usize> =
        dag.nodes.keys().map(|&n| critical_path_len[&n]).collect();

    (successor_counts, critical_path_lengths)
}