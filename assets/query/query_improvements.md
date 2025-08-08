Nice — this is a great, detailed implementation and you’ve already got many of the right pieces in place. I focused on two things you asked for:

* **Practical Rust-level performance improvements** (low/medium effort → big win)
* **Algorithmic / complexity improvements** (how to avoid doing needless work, and how to make the search/heuristic better)

Below I give an ordered set of suggestions (quick wins first), explain *why* they help, and include small, targeted code sketches you can drop into your codebase. Where changes are bigger I mark them as “medium / larger refactor”.

---

# Quick wins (highest impact, simple changes)

1. **Replace `.contains(&node.id)` on `Vec` with a `Vec<bool>` or `BitVec`**

   * `gate_order.contains(&node.id)` is O(n) and is called a lot. Replace it with an `executed: Vec<bool>` indexed by node index (or use `bitvec`/`SmallVec`).
   * **Effect**: removes repeated linear scans from hot paths.

   ```rust
   // store in MicroSABRE
   executed: Vec<bool>, // length = dag.nodes.len()

   // in new():
   executed: vec![false; dag.nodes.len()],

   // in advance_front_layer when executing:
   let idx = node_index as usize;
   if !self.executed[idx] {
       self.executed[idx] = true;
       self.gate_order.push(node.id);
       executed_gates_counter += 1;
   }
   ```

2. **Replace `FxHashMap` keyed by small dense integer IDs with `Vec`**

   * You use `FxHashMap<i32, Vec<i32>>` for `neighbour_map`, `adjacency_list`, and sometimes for `out_map`. If qubit indices or node indices are in `0..N`, use `Vec<Vec<usize>>` or `Vec<Option<...>>` instead of hashmaps.
   * Hash lookups are expensive and hurt locality — contiguous `Vec` gives huge speedups.

   ```rust
   // neighbour_map: Vec<Vec<usize>>
   // adjacency_list: Vec<Vec<usize>>
   ```

3. **Avoid cloning large `State` objects per stack entry**

   * In `choose_best_swaps` you `stack.push(StackItem { state: self.create_snapshot(), ... })` and then you clone/load snapshots many times. Cloning `front_layer`, `layout` etc. is expensive.
   * **Replace with a backtracking approach**: *mutate* state in-place when you descend, and *undo* the changes when you return. This avoids deep cloning.
   * I show a concise change-log based backtrack sketch below (medium complexity).

4. **Don’t rebuild `scores` containing every possible swap sequence**

   * `scores: FxHashMap<Vec<[i32; 2]>, (f64, usize)>` stores *all* explored sequences — memory & CPU heavy. If you only need *the best* sequence (or top-k) then:

     * Use a single best variable or a bounded priority queue (beam).
     * Or use *beam search* (keep top B nodes per depth) instead of exhaustive DFS.
   * **Effect**: avoid exponential blowup of recorded sequences.

5. **Make heuristic calculations non-mutating / local-only**

   * `get_extended_set` currently mutates `self.required_predecessors` and then restores it. That’s expensive and error-prone. Instead, operate on a *local copy* of `required_predecessors` (a `Vec<i32>`) and local queue; do not touch `self`. That avoids repeated global state mutation and the cost of reverting many increments.
   * E.g. `let mut local_pre = self.required_predecessors.clone(); // mutate local only`.

6. **Pre-allocate containers (use `.reserve`)**

   * You frequently push into `Vec`s inside loops. Use `reserve` when you can estimate size (e.g., `swap_sequence.reserve(depth)`).
   * Use `SmallVec` for small vectors like swap sequences to avoid heap allocations for short sequences.

7. **Avoid repeated casting `i32 <-> usize`**

   * Use `usize` for indices internally (node indices, physical qubits) to avoid casts (and the checks they produce in debug builds).

---

# Medium changes (worth the effort)

1. **Backtracking with a small undo log instead of cloning whole state**

   * Record minimal changes caused by `apply_swap` + `advance_front_layer` (which gates moved from front to executed, predecessor decrements, layout changes). After exploring the child, *undo* using the recorded change-log.
   * **Why**: cloning front\_layer + layout every branch is major overhead; undo log gives near-constant overhead per branch.

   **Sketch of a change log (simplified):**

   ```rust
   struct ChangeLog {
       swapped: (usize, usize), // the swap we did (so we can swap back)
       executed_nodes: Vec<usize>, // nodes executed as a result of the swap
       predecessor_decrements: Vec<(usize, i32)>, // node idx -> amount decreased
       front_layer_inserted: Vec<usize>, // nodes inserted into front layer (for rollback)
   }

   // On entering branch:
   let mut log = ChangeLog::default();
   apply_swap_with_log(&mut self, swap, &mut log);
   let advanced = advance_front_layer_with_log(&mut self, &execute_list, &mut log);
   // recurse / push child
   // On returning:
   undo_with_log(&mut self, log);
   ```

   * `apply_swap_with_log` performs `layout.swap_physical(swap)` and `front_layer.apply_swap(swap)` and records the positions changed. Since a swap is its own inverse, undoing can be `layout.swap_physical(swap)` again and `front_layer.apply_swap(swap)` again — but you must also undo the effect of `advance_front_layer` (executed nodes and predecessor counts).

2. **Use beam-search instead of brute-force depth-limited full expansion**

   * If depth is > 4, the branching factor explodes. Use beam width `B` (e.g., 10–50) and at each depth keep the `B` best partial states (by `score + heuristic`). Beam search reduces complexity to `O(B * depth * branching)` and often finds good sequences. You can choose `B` configurable.
   * Implementation: use a `BinaryHeap` with partial states scored by priority; expand top-B, collect children, keep best-B, repeat.

3. **Incremental heuristic deltas (only update changed parts)**

   * `calculate_heuristic` computes `basic` by iterating front layer nodes and `lookahead` by building `extended_set` (which currently is expensive). When you swap two qubits, only gates involving those qubits change distance. Compute delta of heuristic by:

     * For any gate in front layer involving `q0` or `q1`, compute new distance and subtract old.
     * For lookahead, either:

       * compute local incremental lookahead for successors reachable within the limited extended horizon, or
       * **cheaper**: *disable lookahead* in inner search and only use lookahead for final scoring or tie-breaking.

4. **Replace repeated graph shortest-path calls with precomputed next-hop tables**

   * `release_valve` runs Dijkstra (rustworkx) to get the shortest path sequence between two qubits every time. You already have `distance` matrix, but not the path. Precompute for the coupling graph a `next_hop[src][dst] -> next_node` or `parents` from BFS for each source. Then reconstruct path in O(path length) without Dijkstra overhead.
   * For small `num_qubits` this is trivial and fast.

5. **Use smaller numeric types for dense matrices**

   * If `distance` values are small (likely < 127), store `distance: Vec<Vec<u8>>` or `Vec<Vec<u16>>` to improve cache footprint. Same for `required_predecessors` if small.

6. **Minimize API overhead in hot loops**

   * Hot calls like `executable_node_on_qubit` iterate `front_layer.nodes.values()`. Instead maintain `front_layer.qubits: Vec<Option<(node_id, other_qubit)>>` (you appear to have this already), and use direct index lookup `front_layer.qubits[physical_qubit]` which is O(1).

---

# Heuristic improvements (better quality & cheaper heuristics)

Your current heuristic:

```text
basic = sum(distance[a][b] for gates in front_layer)
lookahead = sum(distance[a][b] for extended_set)
score = basic + (0.5 / len(extended_set)) * lookahead
```

This is reasonable but has issues:

* `get_extended_set` is expensive (mutating and exploring), and you normalize `lookahead` by the extended\_set length which makes scaling inconsistent.
* If lookahead is noisy and expensive, it’s better to use a simpler, cheaper heuristic in the inner search.

**Concrete improvements / alternates:**

1. **Simpler, fast inner-heuristic (use for deep search)**

   * Use `h_inner = sum(distance[a][b] for gates in front_layer)`.
   * Use lookahead *only* at leaf nodes (or apply a cheap approximate lookahead based on successors count).

2. **Reward enabling multiple gates**

   * Instead of pure distance, prefer swaps that *enable* multiple front-layer gates.
   * `h = sum(distance) - alpha * (#front gates that would become distance==1 after swap)`
   * Choose `alpha` depending on scale (e.g., `alpha = mean(distance)`)

3. **Critical-path weighting**

   * Weight gates by their *criticality*: gates closer to the end of the DAG or on the longest path get higher weight. Precompute topological levels or distance-to-end and multiply distances by `w = 1 + beta*(level / max_level)`.

4. **Swap-cost-aware heuristic**

   * Penalize sequences that contain swaps which have been recently used (use `last_swap_on_qubit` to discourage flipping back and forth).
   * Add a small penalty `penalty = gamma * (# of swaps on this qubit in last T moves)`.

5. **Normalized lookahead**

   * If you keep lookahead, normalize it to something like `lookahead / lookahead_depth` or use exponential decay `sum_{i} distance_i * decay^{i}`, where `i` is topological distance from front (smaller i => heavier weight). That reduces large contributions from many lookahead nodes.

**Example fast heuristic (cheap + informative):**

```rust
fn fast_heuristic(&self) -> f64 {
    // basic immediate sum
    let basic: f64 = self.front_layer.nodes.values()
        .map(|[a,b]| self.distance[a as usize][b as usize] as f64)
        .sum();

    // count immediate improvements if we swapped certain qubits (used when scoring a swap)
    basic
}
```

Use the cheap heuristic during search; compute the more expensive lookahead only when evaluating the top few candidates.

---

# Algorithmic changes & pruning

1. **Beam search** — keep best `B` sequences at each depth. Very commonly used for SABRE variants and reduces exploration massively.

2. **Branch & Bound** — compute a lower-bound heuristic for remaining cost; if `current_score + bound >= best_found`, prune.

3. **Memoization (transposition table)** — key: `(layout_signature, front_layer_signature)`. If you see the same configuration again with a worse or equal cost, prune. Use a compact signature (e.g., pack virtual->physical as bytes, plus front-layer bitmask) and store best score seen.

4. **Avoid repeated exploration of symmetric swaps** — swaps are undirected; canonicalize swap pairs (q0 < q1) and avoid exploring sequences that contain the same swap repeated twice in a row (unless necessary).

5. **Limit swap candidates** — currently you generate neighbors of every physical qubit in front layer. Instead:

   * Rank candidate swaps by immediate heuristic delta and cap to top `N` (e.g., 8).
   * Or filter swaps that don’t reduce distance for at least one front-layer gate.

---

# Profiling & measurement (don’t guess — measure)

Before big refactors, profile to find the real hotspots:

* Use `cargo build --release`.
* Use `perf`, `cargo-flamegraph`, or `pprof` (or `perf + inferno`) to get flamegraphs.
* Focus optimization where the most time is spent (I’d bet: `choose_best_swaps`, `calculate_heuristic`/`get_extended_set`, and cloning in the stack).

**Checklist:**

* Build release with `RUSTFLAGS="-C target-cpu=native"` for faster codegen.
* Enable LTO in `Cargo.toml` for hot loops if library is performance-critical.

---

# Concrete code examples & patterns

## 1) Replace `Vec.contains` with `Vec<bool>`

I included that above; it’s trivial and high-impact.

## 2) Convert neighbor/adjacency maps to `Vec<Vec<usize>>`

```rust
// before: FxHashMap<i32, Vec<i32>>
let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); dag.nodes.len()];
for (src, dst) in &self.dag.edges {
    adjacency[*src as usize].push(*dst as usize);
}
self.adjacency_list = adjacency;
```

## 3) Beam-search skeleton (pseudo-Rust)

```rust
use std::collections::BinaryHeap;

struct BeamState {
    priority: f64,
    // minimal state snapshot (or use undo log)
    // we prefer storing only the swap sequence and score, and mutate a single global state with rollback
    swap_sequence: Vec<[usize;2]>,
    score: f64,
}

fn beam_search(&mut self, depth: usize, beam_width: usize) -> Vec<[i32;2]> {
    let mut beam = vec![BeamState { priority: self.calculate_heuristic(), swap_sequence: vec![], score: 0.0 }];
    for _d in 0..depth {
        let mut candidates = Vec::new();
        for state in beam.into_iter().take(beam_width) {
            // load state (via undo/redo or by reapplying swap_sequence)
            // compute swap_candidates, evaluate k best, push into candidates
        }
        // sort candidates by priority and keep top beam_width
        beam = candidates.into_iter().take(beam_width).collect();
    }
    // return best swap sequence from beam
}
```

---

# Micro-optimizations & Rust tips

* **`#[inline]`** on tiny functions (hot ones).
* **Use `SmallVec`** for small vectors (swap sequences, neighbor lists) to avoid heap allocations.
* **Store `distance` in a flattened `Vec<T>`** (`distance[src * n + dst]`) for better cache locality.
* **Use `usize` everywhere for indexing**; only use `i32` for external API.
* **Avoid temporary `Vec` allocations inside inner loops** — reuse buffers with `.clear()` and `.reserve()`.

---

# Where I expect the biggest wins (priority)

1. **Remove cloning of `State` in `choose_best_swaps`** and switch to undo-log/backtracking → big reduction in allocations & copying (10x+ often).
2. **Replace hashmaps with `Vec` indexed by `usize`** for adjacency/neighbor maps → large speedup via cache & elimination of hashing.
3. **Use beam-search / prune / memoization** rather than storing all sequences in `scores` → drastically reduces branching.
4. **Replace expensive `get_extended_set` with non-mutating or local copy approach** and/or reduce lookahead depth.
5. **Precompute next-hop for shortest-path reconstruction** so `release_valve` doesn’t run Dijkstra each time.

---

# Final checklist you can follow (concrete roadmap)

1. Add simple profiler and run a couple of problems (identify hotspots).
2. Replace `contains` with `executed: Vec<bool>`.
3. Replace small `FxHashMap` keyed by dense ints with `Vec`.
4. Replace `scores` map with beam search or maintain only best sequences.
5. Implement undo-log/backtracking for `choose_best_swaps`.
6. Make `get_extended_set` local-only (copy `required_predecessors`).
7. Precompute next-hop BFS trees for path reconstruction.
8. Add memoization / transposition table keyed by compact layout + front-layer mask.
9. Experiment with heuristic variants (fast inner heuristic + expensive leaf heuristic).
10. Rerun profiler and iterate.

---

If you want I can:

* **(A)** Show a full, concrete code example of an undo-log/backtracking replacement for `choose_best_swaps` (I’ll produce a focused patch with the small `ChangeLog` struct and recursive search).
* **(B)** Translate your `FxHashMap` adjacency/neighbor maps to `Vec<Vec<usize>>` with exact code edits.
* **(C)** Sketch a beam-search implementation and show how to plug it into the present `run()` loop.

Which of these would you like first? (I recommend starting with the undo-log patch; it usually yields the biggest single benefit.)
