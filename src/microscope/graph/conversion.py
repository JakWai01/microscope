from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import SwapGate
import matplotlib.pyplot as plt


def merge_top_swap(micro_dag, input_dag, initial_mapping, coupling_map):
    # Mapping physical to logical
    current_mapping = initial_mapping.copy()

    transpiled_qiskit_dag = input_dag.copy_empty_like()

    canonical_register = input_dag.qregs["q"]

    topological_swaps = []

    for idx, node in micro_dag.nodes.items():
        if node.is_swap:
            topological_swaps.append(node)

    # Go through graph and execute gates if possible
    for layer in input_dag.serial_layers():
        subdag = layer["graph"]

        for gate in subdag.two_qubit_ops():
            physical_q0 = current_mapping[gate.qargs[0]]
            physical_q1 = current_mapping[gate.qargs[1]]

            while coupling_map.distance(physical_q0, physical_q1) != 1:
                swap_layer = DAGCircuit()
                swap_layer.add_qreg(canonical_register)

                # Pop first swap from list
                swap = topological_swaps.pop(0)

                # Die indizes hier im swap müssen die physischen swaps sein
                swap_layer.apply_operation_back(
                    SwapGate(),
                    (
                        current_mapping[swap.control],
                        current_mapping[swap.target],
                    ),
                    cargs=(),
                    check=False,
                )

                current_mapping.swap(
                    current_mapping[swap.control], current_mapping[swap.target]
                )

                order = current_mapping.reorder_bits(transpiled_qiskit_dag.qubits)
                transpiled_qiskit_dag.compose(swap_layer, qubits=order)

                physical_q0 = current_mapping[gate.qargs[0]]
                physical_q1 = current_mapping[gate.qargs[1]]

        order = current_mapping.reorder_bits(transpiled_qiskit_dag.qubits)
        transpiled_qiskit_dag.compose(subdag, qubits=order)

    return transpiled_qiskit_dag


# Pass through the input_dag and insert swaps at all points in which the
# micro_dag has swaps.
# The algorithm iterates through the gates and inserts swaps before if
# necessary.
# TODO: The bug is inside this function!
def transpiled_micro_dag_to_transpiled_qiskit_dag(
    micro_dag, input_dag, initial_mapping
):
    # Mapping phyiscal qubits to logical qubits e.g. {"physical": "logical"}
    current_mapping = initial_mapping.copy()

    transpiled_qiskit_dag = input_dag.copy_empty_like()

    canonical_register = input_dag.qregs["q"]

    current_dag_pointer = 0

    # Iterate through each gate
    for layer in input_dag.serial_layers():
        subdag = layer["graph"]

        # If gate is two qubit operation (max. one)
        for gate in subdag.two_qubit_ops():

            micro_dag_node = micro_dag.get(current_dag_pointer)

            if micro_dag_node.is_swap:
                swap_layer = DAGCircuit()
                swap_layer.add_qreg(canonical_register)

                swaps = []

                while micro_dag_node.is_swap:
                    swap_layer.apply_operation_back(
                        SwapGate(),
                        (
                            current_mapping[micro_dag_node.control],
                            current_mapping[micro_dag_node.target],
                        ),
                        cargs=(),
                        check=False,
                    )

                    swaps.append((micro_dag_node.control, micro_dag_node.target))

                    current_dag_pointer += 1
                    micro_dag_node = micro_dag.get(current_dag_pointer)

                order = current_mapping.reorder_bits(transpiled_qiskit_dag.qubits)
                transpiled_qiskit_dag.compose(swap_layer, qubits=order)

                for swap in swaps:
                    current_mapping.swap(swap[0], swap[1])

            current_dag_pointer += 1

        order = current_mapping.reorder_bits(transpiled_qiskit_dag.qubits)
        transpiled_qiskit_dag.compose(subdag, qubits=order)

    return transpiled_qiskit_dag


def mapping_to_micro_mapping(initial_mapping):
    micro_mapping = dict()
    # important: keys are virtual qubits and values are physical qubits
    for k, v in initial_mapping.get_virtual_bits().items():
        micro_mapping[k._index] = v
    return micro_mapping
