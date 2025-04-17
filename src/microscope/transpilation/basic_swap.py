from graph.dag import DAG
from transpilation.helper import swap_physical_qubits

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.exceptions import TranspilerError


def micro_swap(dag, coupling_map, initial_mapping):
    # Mapping logical to physical qubits e.g. {"logic": "physical"}
    current_mapping = initial_mapping.copy()

    new_dag = DAG()

    for node_id in range(len(dag)):
        node = dag.get(node_id)

        physical_q0 = current_mapping[node.control]
        physical_q1 = current_mapping[node.target]

        # Check if SWAP is required
        if coupling_map.distance(physical_q0, physical_q1) != 1:
            # Returns the shortest undirected path between two physical qubits
            path = coupling_map.shortest_undirected_path(physical_q0, physical_q1)

            for i in range(len(path) - 2):
                connected_wire_1 = path[i]
                connected_wire_2 = path[i + 1]

                # TODO: Check if we can improve the data structure to avoid this
                # Probably just maintaining both mappings is enough...
                logical_q0 = [
                    key
                    for key, value in current_mapping.items()
                    if value == connected_wire_1
                ][0]
                logical_q1 = [
                    key
                    for key, value in current_mapping.items()
                    if value == connected_wire_2
                ][0]

                qubit_1 = current_mapping[logical_q0]
                qubit_2 = current_mapping[logical_q1]

                new_dag.insert(qubit_1, qubit_2, True)

            for i in range(len(path) - 2):
                current_mapping = swap_physical_qubits(
                    path[i], path[i + 1], current_mapping
                )

        new_dag.insert(
            current_mapping[node.control], current_mapping[node.target], False
        )

    return new_dag


def basic_swap(dag, coupling_map, initial_mapping):
    canonical_register = dag.qregs["q"]
    current_mapping = initial_mapping.copy()

    new_dag = dag.copy_empty_like()

    if coupling_map is None:
        raise TranspilerError("BasicSwap cannot run with coupling_map=None")

    if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
        raise TranspilerError("Basic swap runs on physical circuits only")

    if len(dag.qubits) > len(coupling_map.physical_qubits):
        raise TranspilerError(
            "The layout does not match the amount of qubits in the DAG"
        )

    for layer in dag.serial_layers():
        subdag = layer["graph"]
        for gate in subdag.two_qubit_ops():
            physical_q0 = current_mapping[gate.qargs[0]]
            physical_q1 = current_mapping[gate.qargs[1]]

            # Check if SWAP is required
            if coupling_map.distance(physical_q0, physical_q1) != 1:
                # Insert a new layer with the SWAP(s)
                swap_layer = DAGCircuit()
                swap_layer.add_qreg(canonical_register)

                # Find shortest SWAP path
                path = coupling_map.shortest_undirected_path(physical_q0, physical_q1)

                for i in range(len(path) - 2):
                    connected_wire_1 = path[i]
                    connected_wire_2 = path[i + 1]

                    qubit_1 = current_mapping[connected_wire_1]
                    qubit_2 = current_mapping[connected_wire_2]
                    # Create SWAP operation
                    swap_layer.apply_operation_back(
                        SwapGate(), (qubit_1, qubit_2), cargs=(), check=False
                    )

                # Layer insertion
                order = current_mapping.reorder_bits(new_dag.qubits)
                new_dag.compose(swap_layer, qubits=order)

                # Update current Mapping
                for i in range(len(path) - 2):
                    current_mapping.swap(path[i], path[i + 1])

        order = current_mapping.reorder_bits(new_dag.qubits)
        new_dag.compose(subdag, qubits=order)

    return new_dag
