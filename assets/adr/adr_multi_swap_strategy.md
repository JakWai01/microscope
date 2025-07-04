# Multiple SWAP Strategy

Consider more solutions by evaluating multiple consecutive SWAP gates.

## Minimum Viable Product (MVP) 

### Objective

The first goal is to heuristically evaluate and execute two swaps at a time in order to allow for more degrees of freedom.

### Procedure

#### Single SWAP Strategy (old)

1. Determine initial front and advance that front as much as possible without inserting any SWAPs
2. As long as the front layer is not empty (as long as not all gates are executed):
    1. As long as no new gate can be executed
        - Compute the next heuristically best SWAPs to insert
            - Determine SWAP candidates
            - For each SWAP candidate
                - Calculate Heuristic
                - Apply SWAP temporarily
                - Revert SWAP
                - Insert Score
            - Choose minimum score
        - While there are SWAPs to insert
            - Apply SWAP
            - Check if gates on front layer that operate on the swapped qubits (maximum 2) can be executed.
                - Remove executable gates from the front layer
                - Advance front layer to add successors of nodes that were just executed to the front layer

#### Two SWAP Strategy (new)

1. Determine initial front and advance that front as much as possible without inserting any SWAPs
2. As long as the front layer is not empty (as long as not all gates are executed):
    1. As long as no new gate can be executed
        - Compute the next heuristically best SWAPs to insert
            - Determine SWAP candidates
            - For each SWAP candidate
                - Calculate Heuristic
                - Apply SWAP temporarily
                - Check if gates on front layer that operate on the swapped qubits (maximum 2) can be executed
                    - Remove executable gates from the front layer
                    - Advance front layer to add successors of nodes that were just executed to the front layer
                - Determine SWAP candidates
                - For each SWAP candidate
                    - Calculate Heuristc
                    - Apply SWAP temporarily
                    - Insert score (of both swaps)
                    - Revert SWAP
                - Revert SWAP    
            - Choose minimum score
        - While there are SWAPs to insert
            - Apply SWAP
            - Check if gates on front layer that operate on the swapped qubits (maximum 2) can be executed.
                - Remove executable gates from the front layer
                - Advance front layer to add successors of nodes that were just executed to the front layer