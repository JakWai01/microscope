from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import (
    CheckMap,
    Unroll3qOrMore,
    SetLayout,
    FullAncillaAllocation,
    ApplyLayout,
    RemoveBarriers,
    SabreLayout,
    SabrePreLayout,
    EnlargeWithAncilla
)

from collections import defaultdict
from graph.dag import DAG
from tqdm import tqdm

from rich.console import Console
from rich.table import Table

import matplotlib.pyplot as plt

from commands.helper import (
    apply_sabre_result,
    generate_initial_mapping,
    result_table,
)

import microboost


# We want a class that manages all the benchmark parameters and generates runs for us
class BenchmarkSet:
    def __init__(self, heuristics, trials, extended_set_size):
        self.heuristics = heuristics
        self.trials = trials
        self.extended_set_size = extended_set_size

    def get_test_cases(self):
        test_cases = []

        for heuristic in self.heuristics:
            for i in range(0, self.extended_set_size, 10):
                for _ in range(self.trials):
                    test_cases.append((heuristic, i))

        return test_cases


def ocular(config):
    # Parse config variables
    path = config["ocular"]["path"]
    extended_set_size = config["ocular"]["extended-set-size"]
    layer = config["ocular"]["layer"]

    # Create test test cases
    # test_cases = BenchmarkSet(heuristics, trials, extended_set_size).get_test_cases()
    test_cases = [("lookahead", 20)]
    test_results = defaultdict(list)

    # Parse circuit
    input_circuit = QuantumCircuit.from_qasm_file(path)
    num_qubits = input_circuit.num_qubits

    # Generate coupling map
    coupling_map = CouplingMap.from_line(input_circuit.num_qubits)

    # import math
    # n = input_circuit.num_qubits
    # rows = math.isqrt(n)
    # cols = math.ceil(n / rows)

    # # Now create the grid-based coupling map
    # coupling_map = CouplingMap.from_grid(rows, cols)
    pm = PassManager(
        [
            Unroll3qOrMore(),
            SabreLayout(coupling_map, skip_routing=True),
            ApplyLayout(),
            RemoveBarriers(),
        ]
    )

    preprocessed_circuit = pm.run(input_circuit)

    preprocessed_dag = circuit_to_dag(preprocessed_circuit)

    # Compute Program Communication
    interactions = defaultdict(set)

    # Collect interactions from multi-qubit gates
    for node in preprocessed_dag.op_nodes():
        qubits = [q._index for q in node.qargs]
        if len(qubits) > 1:
            for i in range(len(qubits)):
                for j in range(i + 1, len(qubits)):
                    interactions[qubits[i]].add(qubits[j])
                    interactions[qubits[j]].add(qubits[i])

    # Compute degrees
    degrees = [len(neighbors) for neighbors in interactions.values()]
    num_qubits = len(preprocessed_circuit.qubits)

    # If a qubit had no interactions, include it with degree 0
    all_degrees = degrees + [0] * (num_qubits - len(degrees))
    program_communication = round(sum(all_degrees) / (num_qubits * (num_qubits - 1)), 2)

    # Compute critical depth
    ops_longest_path = preprocessed_dag.count_ops_longest_path()
    longest_path_len = sum(ops_longest_path.values())
    # num_cx_longest_path = ops_longest_path["cx"]
    # num_cx = preprocessed_dag.count_ops()["cx"]
    # critical_depth = round(num_cx_longest_path / num_cx, 2)
    critical_depth = 0

    # Compute Parallelism
    num_gates = sum(preprocessed_dag.count_ops().values())
    depth = preprocessed_dag.depth()
    parallelism = round((num_gates / depth - 1) * (1 / (num_qubits - 1)), 2)

    # Generate initial layout
    canonical_register = preprocessed_dag.qregs["q"]
    current_layout = Layout.generate_trivial_layout(canonical_register)
    qubit_indices = {bit: idx for idx, bit in enumerate(canonical_register)}
    layout_mapping = {
        qubit_indices[k]: v for k, v in current_layout.get_virtual_bits().items()
    }
    initial_layout = microboost.MicroLayout(
        layout_mapping, len(preprocessed_dag.qubits), coupling_map.size()
    )

    # Create DAG
    micro_dag = DAG().from_qiskit_dag(preprocessed_dag)

    num_dag_nodes = len(micro_dag)

    table = Table(title="Circuit Metrics")

    table.add_column("Metric")
    table.add_column("Value")

    table.add_row(
        *["Program Communication", str(program_communication)], style="bright_green"
    )
    table.add_row(*["Critical Depth", str(critical_depth)], style="bright_green")
    table.add_row(*["Paralellism", str(parallelism)], style="bright_green")
    table.add_row(
        *["Critical Path Length", str(longest_path_len)], style="bright_green"
    )
    table.add_row(*["DAG Nodes", str(num_dag_nodes)], style="bright_green")

    console = Console()
    console.print(table)

    # Convert to Rust
    rust_dag = micro_dag.to_micro_dag()

    # Loop through test cases
    for heuristic, extended_set_size in tqdm(test_cases):
        # Initialize MicroSABRE struct
        # rust_ms = microboost.MicroSABRE(
        #     rust_dag, initial_layout, coupling_map.get_edges(), num_qubits
        # )

        rust_multi = microboost.MultiSABRE(
            rust_dag, initial_layout, coupling_map.get_edges(), num_qubits
        )

        # Run single SABRE execution
        # sabre_result = rust_ms.run(heuristic, extended_set_size)
        sabre_result = rust_multi.run(layer)

        (
            out_map,
            gate_order,
            randomness,
            avg_front_size,
            avg_lookahead_size,
            max_lookahead_size,
        ) = sabre_result

        # Insert SWAPs into original DAG
        transpiled_sabre_dag_boosted, _ = apply_sabre_result(
            preprocessed_dag.copy_empty_like(),
            preprocessed_dag,
            (out_map, gate_order),
            preprocessed_dag.qubits,
            coupling_map,
        )

        # Create final result circuit
        transpiled_sabre_circuit_boosted = dag_to_circuit(transpiled_sabre_dag_boosted)

        # Print resulting circuit
        # transpiled_sabre_circuit_boosted.draw("mpl", fold=-1)
        # plt.show()

        # Initialize PassManager to check correctness of result
        cm = CheckMap(coupling_map=coupling_map)
        pm = PassManager([cm])

        # Run PassManager
        _ = pm.run(transpiled_sabre_circuit_boosted)

        if not cm.property_set.get("is_swap_mapped"):
            raise ValueError(
                "CheckMap identified invalid mapping from DAG to coupling_map"
            )

        # Gather metrics
        depth = transpiled_sabre_circuit_boosted.depth()
        swaps = len(transpiled_sabre_dag_boosted.op_nodes(op=SwapGate))

        test_results[(heuristic, extended_set_size)].append(
            (
                depth,
                swaps,
                randomness,
                avg_front_size,
                avg_lookahead_size,
                max_lookahead_size,
            )
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
        "Randomness",
        "Front Size",
        "Extended Set Size",
        "Max Elements in Extended Set",
    ]

    for key, results in test_results.items():
        total_depth = sum(d for d, s, r, f, e, m in results)
        total_swaps = sum(s for d, s, r, f, e, m in results)
        total_randomness = sum(r for d, s, r, f, e, m in results)
        total_front_size = sum(f for d, s, r, f, e, m in results)
        total_lookahead_size = sum(e for d, s, r, f, e, m in results)
        total_max_size = sum(m for d, s, r, f, e, m in results)

        count = len(results)

        avg_swaps = total_swaps / count
        avg_depth = total_depth / count
        avg_randomness = total_randomness / count
        avg_front_size = total_front_size / count
        avg_lookahead_size = total_lookahead_size / count
        avg_total_max_size = total_max_size / count

        heuristic = key[0]
        extended_set_size = key[1]

        rows.append(
            [
                str(heuristic),
                str(extended_set_size),
                str(avg_swaps),
                str(avg_depth),
                str(round(avg_randomness, 2)),
                str(round(avg_front_size, 2)),
                str(round(avg_lookahead_size, 2)),
                str(round(avg_total_max_size, 2)),
            ]
        )
        data[heuristic][0].append(extended_set_size)
        data[heuristic][1].append(avg_swaps)

    result_table(rows, columns)

    # Plot result
    # _, ax = plt.subplots()

    # for heuristic, axis_data in data.items():
    #     extended_set_size = axis_data[0]
    #     swaps = axis_data[1]

    #     ax.plot(extended_set_size, swaps, label=f"{heuristic}")

    # ax.legend()

    # ax.set(
    #     xlabel="Extended-Set Size",
    #     ylabel="Swaps",
    #     title="Extended-Set Size Scaling",
    #     xlim=(0, 8),
    #     xticks=range(0, 101, 10),
    # )
    # ax.grid()

    # plt.xlim((0, 100))
