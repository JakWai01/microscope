from graph.dag import DAG, DAGNode
from transpilation.helper import swap_physical_qubits, pretty_print_mapping
from transpilation.heuristics import calculate_heuristic
import random
from collections import defaultdict


# Returns current front be advancing as much as possible without inserting
# SWAPs.
# Add the gates that can be executed to the execute_gate_list. Add the rest to
# the front_layer.
# Search forward from the nodes provided. They should have no predecessors.
# TODO: Check if we behave correctly when the front_layer already contains items
# TODO: Returning the gate order might become important
def advance_front_layer(nodes, dag, current_mapping, coupling_map):
    node_queue = nodes.copy()

    current_front = []

    while node_queue:
        node_id = node_queue.pop(0)

        node = dag.get(node_id)

        physical_q0 = current_mapping[node.control]
        physical_q1 = current_mapping[node.target]

        # Check whether the node can be executed on the current mapping
        if coupling_map.distance(physical_q0, physical_q1) == 1:
            # Node can be executed, check successors
            successors = get_successors(dag, node_id)
            for successor in successors:
                if no_dependencies(dag, current_front, successor):
                    node_queue.append(successor)
        else:
            # Node cannot be executed without adding SWAPs. Add to front_layer.
            current_front.append(node_id)

    return list(set(current_front))


def micro_sabre_v2(dag, coupling_map, initial_mapping, heuristic):
    current_mapping = initial_mapping.copy()

    # Initialize the front_layer by executing all gates that can be executed
    # immediately without inserting any SWAP gates. Assign the gates to the
    # front_layer that have  no dependencies but cannot be executed without any
    # SWAPs.
    front_layer = advance_front_layer(
        initial_front(dag), dag, current_mapping, coupling_map
    )

    execute_gate_list = []

    out_map = defaultdict(list)

    while front_layer:
        current_swaps = []

        while not execute_gate_list:
            best_swap = choose_best_swap(
                dag, front_layer, coupling_map, current_mapping, heuristic
            )
            print(f"Best Swap: {best_swap}")

            # Swap physical qubits
            physical_q0 = current_mapping[best_swap[0]]
            physical_q1 = current_mapping[best_swap[1]]

            current_swaps.append(best_swap)
            current_mapping = swap_physical_qubits(
                physical_q0, physical_q1, current_mapping
            )

            # Check if we can execute any gates from front_layer due to the
            # SWAP
            if (
                node := executable_node_on_qubit(
                    front_layer, physical_q0, dag, coupling_map, current_mapping
                )
            ) is not None:
                execute_gate_list.append(node)

            if (
                node := executable_node_on_qubit(
                    front_layer, physical_q1, dag, coupling_map, current_mapping
                )
            ) is not None:
                execute_gate_list.append(node)

        # We found something to execute (execute_gate_list is not empty anymore)
        # TODO: Think about decomposing this part into another function
        out_map[execute_gate_list[0]].append(*current_swaps)

        print(f"Front Layer: {front_layer}")
        print(f"Execute Gate List: {execute_gate_list}")
        for node in execute_gate_list:
            front_layer.remove(node)

        # TODO: Maybe this part leads to a failure in the following iterations
        front_layer = advance_front_layer(
            execute_gate_list, dag, current_mapping, coupling_map
        )
        print(f"Next Front Layer: {front_layer}")
        execute_gate_list.clear()

    return dict(out_map)


def executable_node_on_qubit(
    front_layer, physical_qubit, dag, coupling_map, current_mapping
):
    for node_id in front_layer:
        node = dag.get(node_id)

        physical_q0 = current_mapping[node.control]
        physical_q1 = current_mapping[node.target]

        if physical_q0 == physical_qubit or physical_q1 == physical_qubit:
            return node_id
    return None


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

                # Instead of new_dag, qiskit maintains an out_map
                new_dag.insert(node.control, node.target, False)

                # I thought this is not necessary, but apparently this fixes the redundant swaps
                execute_gate_list.remove(gate)

                # Get successors
                successors = get_successors(dag, gate)
                for successor in successors:
                    # Check if dependencies are resolved
                    if no_dependencies(dag, front_layer, successor):
                        front_layer.add(successor)
            continue
        else:
            best_swap = choose_best_swap(
                dag, front_layer, coupling_map, current_mapping, heuristic
            )
            # Swap physical qubits
            physical_q0 = current_mapping[best_swap[0]]
            physical_q1 = current_mapping[best_swap[1]]

            new_dag.insert(physical_q0, physical_q1, True)
            current_mapping = swap_physical_qubits(
                physical_q0, physical_q1, current_mapping
            )

    return new_dag


def choose_best_swap(dag, front_layer, coupling_map, current_mapping, heuristic):
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

    return min_score(scores)


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
            # This is necessary since we go through each edge in the coupling_map.edge.
            # Only proceed if the physical qubits are mapped.
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
