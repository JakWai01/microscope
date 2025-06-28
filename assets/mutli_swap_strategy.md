1. Advance Front Layer
2. Determine Swap Candidates
3. Calculate Heuristic for Candidates
4. Choose best swap

Instead: 

1. Advance Front Layer
2. Determine Swap Candidates
3. Calculate Heuristics
4. For each swap candidate
    - Assume it was chosen
    - Advance Front & Execute Gates
    - Calculate Heuristics
    - Push result to a min-heap
5. Pop min element from heap to get min 2 layer swap
