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

// pub fn min_score(scores: FxHashMap<(i32, i32), f64>) -> (i32, i32) {
//     let mut best_swaps = Vec::new();
// 
//     let mut iter = scores.iter();
//     let (min_swap, mut min_score) = iter.next().map(|(&swap, &score)| (swap, score)).unwrap();
// 
//     best_swaps.push(min_swap);
// 
//     for (&swap, &score) in iter {
//         if score < min_score {
//             min_score = score;
//             best_swaps.clear();
//             best_swaps.push(swap);
//         } else if score == min_score {
//             best_swaps.push(swap);
//         }
//     }
// 
//     let mut rng = rng();
//     *best_swaps.choose(&mut rng).unwrap()
// }

// pub fn min_score(scores: FxHashMap<(i32, i32), f64>) -> (i32, i32) {
//     scores
//         .into_iter()
//         .min_by(|a, b| {
//             a.1
//                 .partial_cmp(&b.1)
//                 .unwrap_or(std::cmp::Ordering::Equal)
//                 .then_with(|| a.0.cmp(&b.0)) // Optional: tie-break using swap tuple
//         })
//         .map(|(swap, _)| swap)
//         .unwrap()
// }

use rand::{thread_rng, seq::SliceRandom}; // <-- important!

pub fn min_score(scores: FxHashMap<(i32, i32), f64>) -> (i32, i32) {
    let mut best_swaps = Vec::new();

    let mut iter = scores.iter();
    let (min_swap, mut min_score) = iter.next().map(|(&swap, &score)| (swap, score)).unwrap();

    best_swaps.push(min_swap);

    for (&swap, &score) in iter {
        if score < min_score {
            min_score = score;
            best_swaps.clear();
            best_swaps.push(swap);
        } else if score == min_score {
            best_swaps.push(swap);
        }
    }

    let mut rng = thread_rng(); // ✅ real random, seeded from system entropy
    *best_swaps.choose(&mut rng).unwrap()
}

