from transpilation.helper import get_successors


def calculate_heuristic(dag, front_layer, coupling_map, current_mapping, heuristic):
    if heuristic == "basic":
        return h_basic(dag, front_layer, coupling_map, current_mapping)
    if heuristic == "lookahead":
        return h_lookahead(dag, front_layer, coupling_map, current_mapping, 1)


# Look-Ahead Ability
def h_lookahead(dag, front_layer, coupling_map, current_mapping, weight):
    h_basic_result = h_basic(dag, front_layer, coupling_map, current_mapping)

    extended_set = get_extended_set(dag, front_layer)
    h_basic_result_extended = h_basic(dag, extended_set, coupling_map, current_mapping)
    return (
        1 / len(front_layer) * h_basic_result
        + weight * 1 / len(extended_set) * h_basic_result_extended
    )


# Nearest Neighbour Cost Function
def h_basic(dag, front_layer, coupling_map, current_mapping):
    h_sum = 0

    for gate in front_layer:
        node = dag.get(gate)

        physical_q0 = current_mapping[node.control]
        physical_q1 = current_mapping[node.target]

        h_sum += coupling_map.distance(physical_q0, physical_q1)
    return h_sum


# Returning the successors of the current front_layer
# TODO: Think about allowing multiple hops
def get_extended_set(dag, front_layer):
    extended_set = set()
    for gate in front_layer:
        successors = get_successors(dag, gate)
        for successor in successors:
            extended_set.add(successor)

    return extended_set
