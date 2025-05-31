import click
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.visualization.dag_visualization import dag_drawer
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit import warnings
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Unroll3qOrMore,
    SetLayout,
    FullAncillaAllocation,
    ApplyLayout,
    SabreSwap,
    RemoveBarriers,
    CheckMap,
)

from graph.dag import DAG

from qiskit._accelerate.nlayout import NLayout
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

import microboost
import os

@click.command()
@click.argument("command", nargs=1)
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option("--show", type=bool, help="True if circuits should be shown")
def main(
    command: str,
    files: tuple[str, ...],
    show: bool
):
    # Ignore deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    match command:
        case "hamiltonians":
            hamiltonians(show)
        case "microbench":
            microbench(files, show)
        case "slide":
            slide()
        case "baseline":
            qiskit_baseline(files[0])
        case _:
            print("Invalid command. Choose one out of [hamiltonians, microbench, slide, qiskit_baseline]")
    
    plt.show()


def slide():
    circuit = QuantumCircuit.from_qasm_file("examples/adder_n10.qasm")
    _, _, segments = transpile_circuit(circuit)
    sliding_window(segments)


def hamiltonians(show):
    path = "/home/jakob/Documents/hamiltonians"
    files = os.listdir(path)

    for file in files:
        es, swaps = run(path + "/" + file, show)
        print(f"File: {file}\nExtended Set Size: {es}\nSwaps: {swaps}")


def microbench(files, show):
    data = []

    for file in files:
        es, swaps = run(file, show)
        data.append((es, swaps, file))

    plot_result(data)


def generate_initial_mapping(dag):
    regs = []

    for _, reg in dag.qregs.items():
        regs.append(reg)

    return Layout.generate_trivial_layout(*regs)


def preprocess(circuit, dag, coupling_map):
    preprocessing_layout = generate_initial_mapping(dag)

    pm = PassManager(
        [
            Unroll3qOrMore(),
            SetLayout(preprocessing_layout),
            FullAncillaAllocation(coupling_map),
            ApplyLayout(),
            RemoveBarriers(),
        ]
    )

    preprocessed_circuit = pm.run(circuit)

    preprocessed_dag = circuit_to_dag(preprocessed_circuit)

    return preprocessed_circuit, preprocessed_dag


def transpile_circuit(circuit):
    coupling_map = CouplingMap.from_line(circuit.num_qubits)
    dag = circuit_to_dag(circuit)

    preprocessed_circuit, preprocessed_dag = preprocess(circuit, dag, coupling_map)
    preprocessed_circuit.draw("mpl", fold=-1)

    initial_mapping = generate_initial_mapping(preprocessed_dag)

    micro_dag = DAG().from_qiskit_dag(preprocessed_dag)
    micro_mapping = mapping_to_micro_mapping(initial_mapping)

    _, _, transpiled_dag, segments = microsabre(
        preprocessed_dag,
        micro_dag,
        micro_mapping,
        coupling_map,
        False,
        "lookahead",
        False,
        20,
    )

    return preprocessed_dag, transpiled_dag, segments


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

def qiskit_baseline(file):
    circuit = QuantumCircuit.from_qasm_file(file)
    coupling_map = CouplingMap.from_line(circuit.num_qubits)
    preprocessing_dag = circuit_to_dag(circuit)
    
    preprocessed_circuit, _ = preprocess(circuit, preprocessing_dag, coupling_map)
    qiskit_test_executions = ["basic", "lookahead", "decay"]
    for heuristic in qiskit_test_executions:
        depth, swaps = sabre(preprocessed_circuit, coupling_map, show, heuristic)
        print(f"Qiskit:\n\tHeuristic: {heuristic}\n\tDepth: {depth}\n\tSwaps: {swaps}")


def run(file: str, show: bool):
    input_circuit = QuantumCircuit.from_qasm_file(file)

    coupling_map = CouplingMap.from_line(input_circuit.num_qubits)
    preprocessing_dag = circuit_to_dag(input_circuit)

    _, input_dag = preprocess(input_circuit, preprocessing_dag, coupling_map)

    initial_mapping = generate_initial_mapping(input_dag)

    rust_dag = DAG().from_qiskit_dag(input_dag).to_micro_dag()
    micro_mapping = mapping_to_micro_mapping(initial_mapping)

    rows = [["Depth"], ["Swaps"]]
    columns = [""]

    test_executions = []

    for i in range(10, 1000, 10):
        test_executions.append(("lookahead-0.5-scaling", False, i))
    # test_executions.append(("lookahead-0.5-scaling", False, 20))

    es_size = []
    num_swaps = []

    for heuristic, critical, extended_set_size in tqdm(test_executions):
        depth, swaps, _, _ = microsabre(
            input_dag,
            rust_dag,
            micro_mapping,
            coupling_map,
            show,
            heuristic,
            critical,
            extended_set_size,
        )
        rows[0].append(str(depth))
        rows[1].append(str(swaps))
        es_size.append(extended_set_size)
        num_swaps.append(swaps)
        columns.append(f"{heuristic} {critical} {extended_set_size}")

    # result_table(rows, columns)

    return es_size, num_swaps


def result_table(rows, columns):
    table = Table(title="SABRE Results")

    for column in columns:
        table.add_column(column)

    for row in rows:
        table.add_row(*row, style="bright_green")
    console = Console()
    console.print(table)


def plot_result(data):
    _, ax = plt.subplots()

    for es, swaps, file in data:
        ax.plot(es, swaps, label=f"{file}")

    ax.legend()

    ax.set(
        xlabel="Extended-Set Size",
        ylabel="Swaps",
        title="Extended-Set Size Scaling",
        xlim=(0, 8),
        xticks=range(0, 1001, 100),
    )
    ax.grid()

    plt.xlim((0, 1000))


def sabre(preprocessed_circuit, coupling_map, show, heuristic):
    cm = CheckMap(coupling_map=coupling_map)
    qiskit_pm = PassManager(
        [SabreSwap(coupling_map, heuristic=heuristic, trials=1), cm]
    )
    # qiskit_pm.draw("sabre_pm.png")
    transpiled_qc = qiskit_pm.run(preprocessed_circuit)
    transpiled_qc_dag = circuit_to_dag(transpiled_qc)

    if not cm.property_set.get("is_swap_mapped"):
        raise ValueError("CheckMap identified invalid mapping from DAG to coupling_map")

    depth = transpiled_qc.depth()
    num_swaps = len(transpiled_qc_dag.op_nodes(op=SwapGate))

    if show:
        transpiled_qc.draw("mpl", fold=-1)

    return depth, num_swaps


def microsabre(
    preprocessed_dag,
    rust_dag,
    micro_mapping,
    coupling_map,
    show,
    heuristic,
    critical=False,
    extended_set_size=20,
):
    # Rust implementation
    rust_ms = microboost.MicroSABRE(rust_dag, micro_mapping, coupling_map.get_edges())
    sabre_result = rust_ms.run(heuristic, critical, extended_set_size)
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


def apply_swaps(dest_dag, swaps, layout, physical_qubits):
    for a, b in swaps:
        qubits = (
            physical_qubits[layout.virtual_to_physical(a)],
            physical_qubits[layout.virtual_to_physical(b)],
        )
        layout.swap_physical(
            layout.virtual_to_physical(a), layout.virtual_to_physical(b)
        )
        dest_dag.apply_operation_back(SwapGate(), qubits, (), check=False)


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


def mapping_to_micro_mapping(initial_mapping):
    micro_mapping = dict()
    # important: keys are virtual qubits and values are physical qubits
    for k, v in initial_mapping.get_virtual_bits().items():
        micro_mapping[k._index] = v
    return micro_mapping


if __name__ == "__main__":
    main()
