from transpilation.helper import swap_physical_qubits
import random
from collections import defaultdict


class MicroSabre:
    def __init__(self, dag, initial_mapping, coupling_map, heuristic):
        self.current_mapping = initial_mapping.copy()
        self.coupling_map = coupling_map
        self.dag = dag
        self.heuristic = heuristic
        self.out_map = defaultdict(list)
        self.gate_order = []
        self.front_layer = set()
        self.required_predecessors = [0 for i in range(len(self.dag.nodes))]

    def _advance_front_layer(self, nodes):
        """Advance front layer without inserting SWAPs.
        Add the gates that can be executed to the execute_gate_list.
        Add the rest to the front_layer. Search forward from the nodes
        provided. They should have no predecessors.
        """
        node_queue = nodes.copy()

        while node_queue:
            node_index = node_queue.pop(0)
            node = self.dag.get(node_index)

            if len(node.qubits) == 2:
                physical_q0 = self.current_mapping[node.qubits[0]]
                physical_q1 = self.current_mapping[node.qubits[1]]

                # Check whether the node can be executed on the current mapping
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    self.front_layer.add(node_index)
                    continue

            # Node can be executed
            if node.node_id not in self.gate_order:
                self.gate_order.append(node.node_id)

            # Check successors
            successors = self._get_successors(node_index)
            for successor in successors:
                self.required_predecessors[successor] -= 1
                if self.required_predecessors[successor] == 0:
                    node_queue.append(successor)

    def run(self):
        execute_gate_list = []

        for edge in self.dag.edges:
            self.required_predecessors[edge[1]] += 1

        # Initialize the front_layer by executing all gates that can be executed
        # immediately without inserting any SWAP gates. Assign the gates to the
        # front_layer that have  no dependencies but cannot be executed without any
        # SWAPs.
        initial_front = self._initial_front()
        self._advance_front_layer(initial_front)

        while self.front_layer:
            current_swaps = []

            while not execute_gate_list:
                best_swap = self._choose_best_swap()
                # Swap physical qubits
                physical_q0 = self.current_mapping[best_swap[0]]
                physical_q1 = self.current_mapping[best_swap[1]]

                current_swaps.append(best_swap)
                self.current_mapping = swap_physical_qubits(
                    physical_q0, physical_q1, self.current_mapping
                )

                # Check if we can execute any gates from front_layer due to the
                # SWAP
                if (node := self._executable_node_on_qubit(physical_q0)) is not None:
                    execute_gate_list.append(node)

                if (node := self._executable_node_on_qubit(physical_q1)) is not None:
                    execute_gate_list.append(node)

            # if len(current_swaps) >= 3:
            #     print("Len of current swaps ", len(current_swaps))
            #     self.force_enable_closest_node(current_swaps)

            # We found something to execute (execute_gate_list is not empty anymore)
            self.out_map[self.dag.get(execute_gate_list[0]).node_id].extend(
                current_swaps
            )

            for node in execute_gate_list:
                self.front_layer.remove(node)

            self._advance_front_layer(execute_gate_list)
            execute_gate_list.clear()

        return (dict(self.out_map), self.gate_order)

    def force_enable_closest_node(self, current_swaps):
        """
        Add swaps to bring the closest nodes together. Prevents SABRE from getting
        stuck after too many heuristic choices.
        """

        # TODO: Change this to something more future-proof
        min_distance = 1000
        closest_node = 1000

        # Identify minimum distance qubits
        for gate in self.front_layer:
            node = self.dag.get(gate)
            physical_q0 = self.current_mapping[node.qubits[0]]
            physical_q1 = self.current_mapping[node.qubits[1]]

            distance = self.coupling_map.distance(physical_q0, physical_q1)

            if distance < min_distance:
                min_distance = distance
                closest_node = gate

        print(min_distance)

        node = self.dag.get(closest_node)
        physical_q0 = self.current_mapping[node.qubits[0]]
        physical_q1 = self.current_mapping[node.qubits[1]]

        path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)

        print("Path ", path)

        # Split to alternate moving left and right
        split = len(path) // 2

        # TODO: Check if the indices are the right id's to insert
        # Current swaps are physical bits here instead of logical before
        current_swaps = []
        for i in range(split):
            current_swaps.append((path[i], path[i + 1]))

        for i in range(split - 1):
            end = len(path) - 1 - i
            current_swaps.append((path[end], path[end - 1]))

        for swap in current_swaps:
            self.current_mapping = swap_physical_qubits(
                swap[0], swap[1], self.current_mapping
            )

        if len(current_swaps) > 1:
            return [closest_node]
        else:
            possible_other_qubit = None

            s_1 = current_swaps[0][0]
            s_2 = current_swaps[0][1]
            for gate in self.front_layer:
                node = self.dag.get(gate)
                physical_q0 = self.current_mapping[node.qubits[0]]
                physical_q1 = self.current_mapping[node.qubits[1]]

                if gate != closest_node:
                    if physical_q0 == s_1:
                        possible_other_qubit = physical_q1
                    if physical_q1 == s_1:
                        possible_other_qubit = physical_q0
                    if physical_q0 == s_2:
                        possible_other_qubit = physical_q1
                    if physical_q1 == s_2:
                        possible_other_qubit = physical_q0

            # TODO: Does this check for None?
            if possible_other_qubit:
                also_routed = self._executable_node_on_qubit(possible_other_qubit)
                if also_routed:
                    return [closest_node, also_routed]

            return [closest_node]

    def _executable_node_on_qubit(self, physical_qubit):
        for node_id in self.front_layer:
            node = self.dag.get(node_id)

            physical_q0 = self.current_mapping[node.qubits[0]]
            physical_q1 = self.current_mapping[node.qubits[1]]

            if physical_q0 == physical_qubit or physical_q1 == physical_qubit:
                # Check if they are actually routable now
                if self.coupling_map.distance(physical_q0, physical_q1) == 1:
                    return node_id
        return None

    def _choose_best_swap(self):
        # Gates in front_layer cannot be executed on hardware.
        scores = dict()
        swap_candidates = self._compute_swap_candidates()
        for swap in swap_candidates:
            # For now, convert to physical qubits again
            # If this works, insert physical qubits in the first place
            physical_q0 = self.current_mapping[swap[0]]
            physical_q1 = self.current_mapping[swap[1]]

            # Create temporary mapping to calculate score with
            temporary_mapping = swap_physical_qubits(
                physical_q0, physical_q1, self.current_mapping
            )

            # Calculate score using front_layer, DAG, temporary_mapping, distance_matrix and swap
            scores[swap] = self._calculate_heuristic(
                self.front_layer, temporary_mapping
            )
        return self._min_score(scores)

    def _compute_swap_candidates(self):
        swap_candidates = []
        # Compute neighbours
        for gate in self.front_layer:
            node = self.dag.get(gate)
            physical_q0 = self.current_mapping[node.qubits[0]]
            physical_q1 = self.current_mapping[node.qubits[1]]

            for edge in self.coupling_map:
                # This is necessary since we go through each edge in the coupling_map.edge.
                # Only proceed if the physical qubits are mapped.
                if (
                    edge[0] in self.current_mapping.values()
                    and edge[1] in self.current_mapping.values()
                ):
                    if edge[0] == physical_q0 or edge[0] == physical_q1:
                        logical_q0 = [
                            key
                            for key, value in self.current_mapping.items()
                            if value == edge[0]
                        ][0]
                        logical_q1 = [
                            key
                            for key, value in self.current_mapping.items()
                            if value == edge[1]
                        ][0]
                        # Important: SWAP candidates are logical qubits! Not like in micro_swap!
                        swap_candidates.append((logical_q0, logical_q1))
        return swap_candidates

    def _min_score(self, scores):
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
        # random.seed(0)

        return random.choice(best_swaps)

    def _no_dependencies(self, front_layer, successor):
        for gate in front_layer:
            node = self.dag.get(gate)
            successor_node = self.dag.get(successor)

            if set(node.qubits).intersection(set(successor_node.qubits)):
                return False
        return True

    def _get_successors(self, node_id):
        successors = []
        for s, t in self.dag.edges:
            if s == node_id:
                successors.append(t)
        return successors

    def _initial_front(self):
        nodes_with_predecessors = set()
        nodes = set(range(len(self.dag)))

        for s, t in self.dag.edges:
            nodes_with_predecessors.add(t)

        return list(nodes - nodes_with_predecessors)

    def _calculate_heuristic(self, front_layer, current_mapping):
        # TODO: Switch here
        if self.heuristic == "basic":
            return self._h_basic(front_layer, current_mapping)
        if self.heuristic == "lookahead":
            return self._h_lookahead(front_layer, current_mapping, 1)

    # Look-Ahead Ability
    def _h_lookahead(self, front_layer, current_mapping, weight):
        h_basic_result = self._h_basic(front_layer, current_mapping)

        extended_set = self._get_extended_set()

        h_basic_result_extended = self._h_basic(extended_set, current_mapping)
        return (
            1 / len(front_layer) * h_basic_result
            + weight * 1 / len(extended_set) * h_basic_result_extended
        )

    # Nearest Neighbour Cost Function
    def _h_basic(self, front_layer, current_mapping):
        h_sum = 0

        for gate in front_layer:
            node = self.dag.get(gate)

            if len(node.qubits) == 1:
                continue

            physical_q0 = current_mapping[node.qubits[0]]
            physical_q1 = current_mapping[node.qubits[1]]

            h_sum += self.coupling_map.distance(physical_q0, physical_q1)
        return h_sum

    # Returning the successors of the current front_layer
    # TODO: Think about allowing multiple hops
    def _get_extended_set(self):
        extended_set = set()
        for gate in self.front_layer:
            successors = self._get_successors(gate)
            for successor in successors:
                extended_set.add(successor)
        return extended_set
