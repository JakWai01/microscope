from graph.dag import DAG, DAGNode
from transpilation.helper import swap_physical_qubits, pretty_print_mapping
from transpilation.heuristics import calculate_heuristic
import random


def micro_sabre(dag, coupling_map, initial_mapping, heuristic):
    current_mapping = initial_mapping.copy()
    front_layer = set(initial_front(dag))

    new_dag = DAG()

    while front_layer:
        execute_gate_list = []
        for gate in front_layer:
            node = dag.get(gate)

            physical_q0 = current_mapping[node.control]
            physical_q1 = current_mapping[node.target]

            if coupling_map.distance(physical_q0, physical_q1) == 1:
                execute_gate_list.append(gate)

        if execute_gate_list:
            for gate in execute_gate_list:
                # Remove from front since we execute it
                front_layer.remove(gate)

                node = dag.get(gate)
                new_dag.insert(node.control, node.target, False)

                # I thought this is not necessary, but apparently this fixes the redundant swaps
                execute_gate_list.remove(gate)

                # Get successors
                successors = get_successors(dag, gate)
                for successor in successors:
                    # Check if dependencies are resolved
                    if no_dependencies(dag, front_layer, successor):
                        front_layer.add(successor)

                        node_suc = dag.get(successor)
            continue
        else:
            # Gates in front_layer cannot be executed on hardware.
            scores = dict()
            swap_candidates = compute_swap_candidates(
                dag, front_layer, current_mapping, coupling_map
            )
            for swap in swap_candidates:
                # For now, convert to physical qubits again
                # If this works, insert physical qubits in the first place
                physical_q0 = current_mapping[swap[0]]
                physical_q1 = current_mapping[swap[1]]

                # Create temporary mapping to calculate score with
                temporary_mapping = swap_physical_qubits(
                    physical_q0, physical_q1, current_mapping
                )

                # Calculate score using front_layer, DAG, temporary_mapping, distance_matrix and swap
                scores[swap] = calculate_heuristic(
                    dag, front_layer, coupling_map, temporary_mapping, heuristic
                )

            best_swap = min_score(scores)

            # Swap physical qubits
            physical_q0 = current_mapping[best_swap[0]]
            physical_q1 = current_mapping[best_swap[1]]

            new_dag.insert(physical_q0, physical_q1, True)
            current_mapping = swap_physical_qubits(
                physical_q0, physical_q1, current_mapping
            )

    return new_dag


def compute_swap_candidates(dag, front_layer, current_mapping, coupling_map):
    swap_candidates = []
    # Compute neighbours
    for gate in front_layer:
        node = dag.get(gate)
        # TODO: Check that we are always mapping "logical":"physical"
        # TODO: Check that looking for both, control and target, is fine
        physical_q0 = current_mapping[node.control]
        physical_q1 = current_mapping[node.target]

        for edge in coupling_map:
            # This is necessary since we go through each edge in the coupling_map. Only proceed if the physical qubits are mapped.
            if (
                edge[0] in current_mapping.values()
                and edge[1] in current_mapping.values()
            ):
                if edge[0] == physical_q0 or edge[0] == physical_q1:
                    logical_q0 = [
                        key
                        for key, value in current_mapping.items()
                        if value == edge[0]
                    ][0]
                    logical_q1 = [
                        key
                        for key, value in current_mapping.items()
                        if value == edge[1]
                    ][0]
                    # Important: SWAP candidates are logical qubits! Not like in micro_swap!
                    swap_candidates.append((logical_q0, logical_q1))
    return swap_candidates


def min_score(scores):
    min_swap = list(scores)[0]
    min_score = scores[min_swap]
    best_swaps = []
    for swap, score in scores.items():
        if score < min_score:
            min_score = scores[swap]
            min_swap = swap
            best_swaps = []
            best_swaps.append(swap)
        if score == min_score:
            best_swaps.append(swap)

    # TODO: Make seed optional
    random.seed(0)

    return random.choice(best_swaps)


def no_dependencies(dag, front_layer, successor):
    for gate in front_layer:
        node = dag.get(gate)
        successor_node = dag.get(successor)

        if (
            node.control == successor_node.control
            or node.target == successor_node.control
            or node.control == successor_node.target
            or node.target == successor_node.target
        ):
            return False
    return True


def get_successors(dag, node_id):
    successors = []
    for s, t in dag.edges:
        if s == node_id:
            successors.append(t)
    return successors


def initial_front(dag):
    nodes_with_predecessors = set()
    nodes = set(range(len(dag)))

    for s, t in dag.edges:
        nodes_with_predecessors.add(t)

    return list(nodes - nodes_with_predecessors)
