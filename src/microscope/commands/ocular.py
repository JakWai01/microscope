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
)

from collections import defaultdict
from graph.dag import DAG
from tqdm import tqdm

import matplotlib.pyplot as plt

from commands.helper import (
    apply_sabre_result,
    generate_initial_mapping,
    plot_result,
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
    print("ocular")

    # Parse config variables
    path = config["ocular"]["path"]
    heuristics = config["ocular"]["heuristics"]
    trials = config["ocular"]["trials"]
    extended_set_size = config["ocular"]["extended-set-size"]

    # Parse circuit
    input_circuit = QuantumCircuit.from_qasm_file(path)
    num_qubits = input_circuit.num_qubits

    # Generate coupling map
    coupling_map = CouplingMap.from_line(input_circuit.num_qubits)

    # Generate DAG from circuit
    input_dag = circuit_to_dag(input_circuit)

    # Preprocess circuit
    preprocessing_layout = generate_initial_mapping(input_dag)

    pm = PassManager(
        [
            Unroll3qOrMore(),
            SetLayout(preprocessing_layout),
            FullAncillaAllocation(coupling_map),
            ApplyLayout(),
            RemoveBarriers(),
        ]
    )

    preprocessed_circuit = pm.run(input_circuit)
    preprocessed_dag = circuit_to_dag(preprocessed_circuit)

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

    # Create Rust DAG
    rust_dag = DAG().from_qiskit_dag(preprocessed_dag).to_micro_dag()

    # Create test test cases
    test_cases = BenchmarkSet(heuristics, trials, extended_set_size).get_test_cases()
    test_results = defaultdict(list)

    # Loop through test cases
    for heuristic, extended_set_size in tqdm(test_cases):
        # Initialize MicroSABRE struct
        rust_ms = microboost.MicroSABRE(
            rust_dag, initial_layout, coupling_map.get_edges(), num_qubits
        )

        # Run single SABRE execution
        sabre_result = rust_ms.run(heuristic, extended_set_size)

        # Insert SWAPs into original DAG
        transpiled_sabre_dag_boosted, _ = apply_sabre_result(
            preprocessed_dag.copy_empty_like(),
            preprocessed_dag,
            sabre_result,
            preprocessed_dag.qubits,
            coupling_map,
        )

        # Create final result circuit
        transpiled_sabre_circuit_boosted = dag_to_circuit(transpiled_sabre_dag_boosted)

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

        test_results[(heuristic, extended_set_size)].append((depth, swaps))

    process_results(test_results)


def process_results(test_results):
    data = defaultdict(lambda: ([], []))

    rows = []
    columns = ["Heuristic", "Extended Set Size", "Swaps", "Depth"]

    for key, results in test_results.items():
        total_depth = sum(d for d, s in results)
        total_swaps = sum(s for d, s in results)
        count = len(results)

        avg_swaps = total_swaps / count
        avg_depth = total_depth / count
        heuristic = key[0]
        extended_set_size = key[1]

        rows.append(
            [str(heuristic), str(extended_set_size), str(avg_swaps), str(avg_depth)]
        )
        data[heuristic][0].append(extended_set_size)
        data[heuristic][1].append(avg_swaps)

    result_table(rows, columns)

    # Plot result
    _, ax = plt.subplots()

    for heuristic, axis_data in data.items():
        extended_set_size = axis_data[0]
        swaps = axis_data[1]

        ax.plot(extended_set_size, swaps, label=f"{heuristic}")

    ax.legend()

    ax.set(
        xlabel="Extended-Set Size",
        ylabel="Swaps",
        title="Extended-Set Size Scaling",
        xlim=(0, 8),
        xticks=range(0, 101, 10),
    )
    ax.grid()

    plt.xlim((0, 100))
