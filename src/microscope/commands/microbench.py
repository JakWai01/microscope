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


def single(config, show):
    path = config["single"]["path"]
    heuristics = config["single"]["heuristics"]
    trials = config["single"]["trials"]
    extended_set = config["single"]["extended-set"]

    data = []
    accumulated_rows = []
    columns = ["File", "Heuristic", "Extended Set Size", "Swaps", "Depth"]

    for heuristic in heuristics:
        for extended_set_size in range(0, 101, 10):
            (es, swaps), rows = run(path, heuristic, trials, extended_set, show, extended_set_size)
            data.append((es, swaps, heuristic))
            accumulated_rows.extend(rows)

    result_table(accumulated_rows, columns)
    plot_result(data)


def bench(config, show):
    files = config["multi"]["files"]
    heuristic = config["multi"]["heuristic"]
    num_executions = config["multi"]["trials"]
    extended_set = config["multi"]["extended-set"]

    data = []

    for file in files:
        (es, swaps), rows = run(file, heuristic, num_executions, extended_set, show)
        data.append((es, swaps, file))


def run(path: str, heuristic: str, trials: int, extended_set: str, show: bool, extended_set_size: int):
    input_circuit = QuantumCircuit.from_qasm_file(path)

    coupling_map = CouplingMap.from_line(input_circuit.num_qubits)
    preprocessing_dag = circuit_to_dag(input_circuit)

    _, input_dag = preprocess(input_circuit, preprocessing_dag, coupling_map)

    canonical_register = input_dag.qregs["q"]
    current_layout = Layout.generate_trivial_layout(canonical_register)
    qubit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
    layout_mapping = {
        qubit_indices[k]: v for k, v in current_layout.get_virtual_bits().items()
    }
    initial_layout = microboost.MicroLayout(
        layout_mapping, len(input_dag.qubits), coupling_map.size()
    )

    rust_dag = DAG().from_qiskit_dag(input_dag).to_micro_dag()

    rows = []

    test_executions = []
    
    test_executions.append((heuristic, False, extended_set_size))
    # if extended_set == "solo":
    #     test_executions.append((heuristic, False, 20))

    # if extended_set == "constant":
    #     for i in range(10, 101, 10):
    #         test_executions.append((heuristic, False, 20))

    # if extended_set == "dynamic":
    #     for i in range(10, 101, 10):
    #         test_executions.append((heuristic, False, i))

    es_size = []
    num_swaps = []

    for heuristic, critical, extended_set_size in tqdm(test_executions):
        accumulated_depth = 0
        accumulated_swaps = 0

        for _ in range(0, trials):
            depth, swaps, transpiled_dag, _ = microsabre(
                input_dag,
                rust_dag,
                initial_layout,
                coupling_map,
                show,
                heuristic,
                input_circuit.num_qubits,
                critical,
                extended_set_size,
            )

            accumulated_depth += depth
            accumulated_swaps += swaps
        
        if extended_set == "solo":
            circuit = dag_to_circuit(transpiled_dag)
            circuit.draw("mpl", fold=-1)

        avg_depth = accumulated_depth / trials 
        avg_swaps = accumulated_swaps / trials 

        # Set values in table
        rows.append(
            [path, heuristic, str(extended_set_size), str(avg_swaps), str(avg_depth)]
        )

        # Add values for plot
        es_size.append(extended_set_size)
        num_swaps.append(avg_swaps)

    return (es_size, num_swaps), rows


def microsabre(
    preprocessed_dag,
    rust_dag,
    micro_mapping,
    coupling_map,
    show,
    heuristic,
    num_qubits,
    critical=False,
    extended_set_size=20,
):
    # We don't actually need to create the object each time but we seem to do it right here.
    rust_ms = microboost.MicroSABRE(
        rust_dag, micro_mapping, coupling_map.get_edges(), num_qubits
    )

    sabre_result = rust_ms.run(heuristic, extended_set_size)

    transpiled_sabre_dag_boosted, segments_boosted = apply_sabre_result(
        preprocessed_dag.copy_empty_like(),
        preprocessed_dag,
        sabre_result,
        preprocessed_dag.qubits,
        coupling_map,
    )

    transpiled_micro_sabre_circuit_boosted = dag_to_circuit(
        transpiled_sabre_dag_boosted
    )

    if show:
        transpiled_micro_sabre_circuit_boosted.draw("mpl", fold=-1)

    cm = CheckMap(coupling_map=coupling_map)
    qiskit_pm = PassManager([cm])
    _ = qiskit_pm.run(transpiled_micro_sabre_circuit_boosted)

    if not cm.property_set.get("is_swap_mapped"):
        raise ValueError("CheckMap identified invalid mapping from DAG to coupling_map")

    depth = transpiled_micro_sabre_circuit_boosted.depth()
    num_swaps = len(transpiled_sabre_dag_boosted.op_nodes(op=SwapGate))

    return depth, num_swaps, transpiled_sabre_dag_boosted, segments_boosted


def apply_sabre_result(
    dest_dag, source_dag, sabre_result, physical_qubits, coupling_map
):
    # Qubit: index
    root_logical_map = {qbit: index for index, qbit in enumerate(source_dag.qubits)}

    # Generate Rust-space mapping of virtual indices
    canonical_register = source_dag.qregs["q"]
    current_layout = Layout.generate_trivial_layout(canonical_register)
    qubit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
    layout_mapping = {
        qubit_indices[k]: v for k, v in current_layout.get_virtual_bits().items()
    }
    initial_layout = NLayout(
        layout_mapping, len(source_dag.qubits), coupling_map.size()
    )

    swap_map, node_order = sabre_result

    segments = [DAG()]
    i = 0

    for node_id in node_order:
        node = source_dag.node(node_id)

        if node_id in swap_map:
            segments.append(DAG())
            i += 1
            apply_swaps(dest_dag, swap_map[node_id], initial_layout, physical_qubits)

        if node.op.num_qubits == 2:
            segments[i].insert(node_id, [node.qargs[0]._index, node.qargs[1]._index])
        elif node.op.num_qubits == 1:
            segments[i].insert(node_id, [node.qargs[0]._index])
        else:
            raise Exception("Error creating segments")

        qubits = [
            physical_qubits[initial_layout.virtual_to_physical(root_logical_map[q])]
            for q in node.qargs
        ]
        dest_dag._apply_op_node_back(
            DAGOpNode.from_instruction(
                node._to_circuit_instruction().replace(qubits=qubits),
            ),
            check=False,
        )
    return dest_dag, segments
