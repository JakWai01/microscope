from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import (
    CheckMap,
    Unroll3qOrMore,
    ApplyLayout,
    RemoveBarriers,
    SabreLayout,
    SabreSwap,
    FullAncillaAllocation,
    SetLayout,
)

from collections import defaultdict
from graph.dag import DAG

from rich.console import Console  # type: ignore
from rich.table import Table  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from commands.helper import apply_sabre_result, result_table, generate_initial_mapping

import microboost  # type: ignore


def coupling_line(n):
    return CouplingMap.from_line(n)


def coupling_grid(n):
    import math

    rows = math.isqrt(n)
    cols = math.ceil(n / rows)
    return CouplingMap.from_grid(rows, cols)


def ocular(config):
    files = config["ocular"]["files"]
    algorithmic_depth = config["ocular"]["depth"]

    for file in files:
        print(file)

        test_cases = [("lookahead", 20)]
        test_results = defaultdict(list)

        input_circuit = QuantumCircuit.from_qasm_file(file)

        num_qubits = input_circuit.num_qubits

        coupling_map = coupling_line(input_circuit.num_qubits)

        pm = PassManager(
            [
                Unroll3qOrMore(),
                SabreLayout(coupling_map, skip_routing=True, seed=42),
                FullAncillaAllocation(coupling_map=coupling_map),
                ApplyLayout(),
                RemoveBarriers(),
            ]
        )
        preprocessed_circuit = pm.run(input_circuit)

        preprocessed_dag = circuit_to_dag(preprocessed_circuit)

        ops_longest_path = preprocessed_dag.count_ops_longest_path()
        longest_path_len = sum(ops_longest_path.values())

        depth = preprocessed_dag.depth()

        canonical_register = preprocessed_dag.qregs["q"]
        current_layout = Layout.generate_trivial_layout(canonical_register)
        qubit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
        layout_mapping = {
            qubit_indices[k]: v for k, v in current_layout.get_virtual_bits().items()
        }
        initial_layout = microboost.MicroLayout(
            layout_mapping, num_qubits, coupling_map.size()
        )

        micro_dag = DAG().from_qiskit_dag(preprocessed_dag)

        num_dag_nodes = len(micro_dag)

        table = Table(title="Circuit Metrics")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row(
            *["Critical Path Length", str(longest_path_len)], style="bright_green"
        )
        table.add_row(*["DAG Nodes", str(num_dag_nodes)], style="bright_green")
        console = Console()
        console.print(table)

        rust_dag = micro_dag.to_micro_dag()

        for heuristic, extended_set_size in test_cases:
            ms = microboost.MicroSABRE(
                rust_dag, initial_layout, coupling_map.get_edges(), num_qubits
            )
            sabre_result = ms.run(algorithmic_depth)

            cm = CheckMap(coupling_map=coupling_map)
            qiskit_pm = PassManager(
                [SabreSwap(coupling_map, heuristic=heuristic, trials=1), cm]
            )
            transpiled_qc = qiskit_pm.run(preprocessed_circuit)
            if not cm.property_set.get("is_swap_mapped"):
                raise ValueError(
                    "CheckMap identified invalid mapping from DAG to coupling_map in qiskit implementation"
                )

            transpiled_qc_dag = circuit_to_dag(transpiled_qc)
            qiskit_depth = transpiled_qc.depth()
            qiskit_swaps = len(transpiled_qc_dag.op_nodes(op=SwapGate))

            transpiled_sabre_dag_boosted, _ = apply_sabre_result(
                preprocessed_dag.copy_empty_like(),
                preprocessed_dag,
                sabre_result,
                preprocessed_dag.qubits,
                coupling_map,
            )

            transpiled_sabre_circuit_boosted = dag_to_circuit(
                transpiled_sabre_dag_boosted
            )

            # o = transpiled_sabre_circuit_boosted.draw(output="latex_source", fold=-1)
            # print(o)

            cm = CheckMap(coupling_map=coupling_map)
            pm = PassManager([cm])
            _ = pm.run(transpiled_sabre_circuit_boosted)
            if not cm.property_set.get("is_swap_mapped"):
                raise ValueError(
                    "CheckMap identified invalid mapping from DAG to coupling_map"
                )

            depth = transpiled_sabre_circuit_boosted.depth()
            swaps = len(transpiled_sabre_dag_boosted.op_nodes(op=SwapGate))

            test_results[(heuristic, extended_set_size)].append(
                (depth, swaps, qiskit_depth, qiskit_swaps)
            )

        process_results(test_results)


def process_results(test_results):
    data = defaultdict(lambda: ([], []))

    rows = []
    columns = [
        "Heuristic",
        "Extended Set Size",
        "Swaps",
        "Depth",
        "Qiskit Swaps",
        "Qiskit Depth",
    ]

    for key, results in test_results.items():
        total_depth = sum(d for d, _, _, _ in results)
        total_swaps = sum(s for _, s, _, _ in results)
        total_qiskit_depth = sum(q_d for _, _, q_d, _ in results)
        total_qiskit_swaps = sum(q_s for _, _, _, q_s in results)

        count = len(results)

        avg_swaps = total_swaps / count
        avg_depth = total_depth / count
        avg_q_swaps = total_qiskit_swaps / count
        avg_q_depth = total_qiskit_depth / count

        heuristic = key[0]
        extended_set_size = key[1]

        rows.append(
            [
                str(heuristic),
                str(extended_set_size),
                str(avg_swaps),
                str(avg_depth),
                str(avg_q_swaps),
                str(avg_q_depth),
            ]
        )
        data[heuristic][0].append(extended_set_size)
        data[heuristic][1].append(avg_swaps)

    result_table(rows, columns)

    plt.show()


def benchpress_adapter(circuit, backend, k):
    coupling_map = backend._coupling_map

    pm = PassManager(
        [
            Unroll3qOrMore(),
            SabreLayout(coupling_map, skip_routing=True, seed=42),
            FullAncillaAllocation(coupling_map=coupling_map),
            ApplyLayout(),
            RemoveBarriers(),
        ]
    )

    preprocessed_circuit = pm.run(circuit)

    preprocessed_dag = circuit_to_dag(preprocessed_circuit)

    # qiskit_pm = PassManager(
    #     [SabreSwap(coupling_map, heuristic="lookahead", trials=1)]
    # )
    # transpiled_qc = qiskit_pm.run(preprocessed_circuit)

    num_qubits = len(preprocessed_circuit.qubits)

    canonical_register = preprocessed_dag.qregs["q"]
    current_layout = Layout.generate_trivial_layout(canonical_register)
    qubit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
    layout_mapping = {
        qubit_indices[k]: v for k, v in current_layout.get_virtual_bits().items()
    }
    initial_layout = microboost.MicroLayout(
        layout_mapping, num_qubits, coupling_map.size()
    )

    dag = DAG().from_qiskit_dag(preprocessed_dag).to_micro_dag()

    ms = microboost.MicroSABRE(
        dag, initial_layout, coupling_map.get_edges(), num_qubits
    )

    sabre_result = ms.run(k)

    transpiled_sabre_dag_boosted, _ = apply_sabre_result(
        preprocessed_dag.copy_empty_like(),
        preprocessed_dag,
        sabre_result,
        preprocessed_dag.qubits,
        coupling_map,
    )

    return dag_to_circuit(transpiled_sabre_dag_boosted)
    # return transpiled_qc
