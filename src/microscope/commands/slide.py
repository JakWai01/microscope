from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    CheckMap,
)

from graph.dag import DAG
from tqdm import tqdm
from qiskit._accelerate.nlayout import NLayout
from typing import List

import microboost

from commands.helper import (
    plot_result,
    generate_initial_mapping,
    preprocess,
    apply_swaps,
    mapping_to_micro_mapping,
    result_table,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode

def transpile_circuit(circuit):
    coupling_map = CouplingMap.from_line(circuit.num_qubits)
    dag = circuit_to_dag(circuit)

    preprocessed_circuit, preprocessed_dag = preprocess(circuit, dag, coupling_map)

    initial_mapping = generate_initial_mapping(preprocessed_dag)

    micro_dag = DAG().from_qiskit_dag(preprocessed_dag).to_micro_dag()
    micro_mapping = mapping_to_micro_mapping(initial_mapping)

    _, _, transpiled_dag, segments = microsabre(
        preprocessed_dag,
        micro_dag,
        micro_mapping,
        coupling_map,
        False,
        "lookahead",
        preprocessed_circuit.num_qubits,
        False,
        20,
    )

    return preprocessed_dag, transpiled_dag, segments

def slide():
    circuit = QuantumCircuit.from_qasm_file("examples/adder_n10.qasm")
    _, _, segments = transpile_circuit(circuit)
    sliding_window(segments)


def sliding_window(segments):
    """
    Iterate over a given transpiled quantum circuit to find possible
    improvements.

    This is achieved by splitting the circuit into mutliple segments that
    resemble circuits themselves. Then, the SABRE algorithm is executed
    on these subcircuits (SWAPs from solution removed) to find possible
    improvements. The input and output permutations of all solutions are
    then matched to find solutions that can be merged together to form the
    overall lowest cost solution where cost is defined as the lowest number
    of swaps.

    Segments are separated by one or more SWAPs e.g.:

        SEGMENT : SWAPS : SEGMENT : SWAPS : SEGMENT

    After optimizing the subcircuits, the final solution can be obtained by
    combining the segments, filling in the required SWAPs and choosing the
    solution with the least SWAPs.

    Questions:
    - How can we skip optimal subcircuits?
        Can we utilize lightcone bounds?
    - Do the qargs represent the original values or are they already the
      swapped qubits?
    """

    print(segments[0].__dict__)
    print(segments[1].__dict__)

    # segments = circuit_to_unswapped_segments(preprocessed_dag, transpiled_dag, micro_dag)
    # print(segments[0])

    # TODO: Check if we are using the original unswapped qubits

    # TODO: Combine multiple adjascent segments to a subcircuit

    # TODO: Run MicroSABRE on the subcircuits
