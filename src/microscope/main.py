import click

import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.visualization.dag_visualization import dag_drawer
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.passes.routing.basic_swap import BasicSwap
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


from transpilation.helper import generate_initial_mapping
from graph.dag import DAG
from transpilation.basic_swap import basic_swap
from transpilation.sabre import MicroSabre

from qiskit._accelerate.nlayout import NLayout
from rich.console import Console
from rich.table import Table


@click.command()
@click.option("-f", "--filename", type=str, required=True, help="Path to .qasm file")
@click.option("-d", "--show-dag", type=bool, help="True if DAG should be shown")
@click.option(
    "-q", "--qiskit-fallback", type=bool, help="Use qiskit algorithm implementation"
)
@click.option(
    "-p", "--plot", type=bool, help="Plot the result"
)
@click.option(
    "-t", "--table", type=bool, help="Print table of result"
)
@click.option("--show", type=bool, help="True if circuits should be shown")
def main(filename: str, show_dag: bool, qiskit_fallback: bool, show: bool, plot: bool, table: bool):
    """Read in a .qasm file and print out a syntax tree."""

    # Ignore deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    input_circuit = QuantumCircuit.from_qasm_file(filename)
    if show:
        input_circuit.draw("mpl", fold=-1)

    coupling_map = CouplingMap.from_line(28)

    preprocessing_dag = circuit_to_dag(input_circuit)
    preprocessing_layout = generate_initial_mapping(preprocessing_dag)

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
    if show:
        preprocessed_circuit.draw("mpl", fold=-1)

    input_dag = circuit_to_dag(preprocessed_circuit)
    initial_mapping = generate_initial_mapping(input_dag)

    micro_dag = DAG().from_qiskit_dag(input_dag)
    micro_mapping = mapping_to_micro_mapping(initial_mapping)

    if show_dag:
        input_dag_image = dag_drawer(input_dag)
        input_dag_image.show()

    rows = [["Depth"],["Swaps"]]
    columns = [""]

    # Qiskit SABRE
    qiskit_test_executions = ["basic", "lookahead", "decay"]
    for heuristic in qiskit_test_executions:
        depth, swaps = sabre(preprocessed_circuit, coupling_map, show, heuristic)
        rows[0].append(str(depth))
        rows[1].append(str(swaps))
        columns.append(f"{heuristic}")

    # Micro SABRE
    test_executions = []

    for i in range(5, 1000, 5):
        test_executions.append(("lookahead-0.5-scaling", False, i))

    es_size = []
    num_swaps = []
    from tqdm import tqdm

    for heuristic, critical, extended_set_size in tqdm(test_executions):
        depth, swaps = microsabre(
            input_dag,
            micro_dag,
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

    
    if plot:
        plot_result(es_size, num_swaps)
    
    if table:
        table(rows, columns)

    plt.show()

def table(rows, columns):
    table = Table(title="SABRE Results")

    for column in columns:
        table.add_column(column)

    for row in rows:
        table.add_row(*row, style="bright_green")
    console = Console()
    console.print(table)

def plot_result(es_size, num_swaps):
    fig, ax = plt.subplots()

    ax.plot(es_size, num_swaps, label="adder_n28")
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
    plt.ylim((150, 300))


def sabre(preprocessed_circuit, coupling_map, show, heuristic):
    cm = CheckMap(coupling_map=coupling_map)
    qiskit_pm = PassManager(
        [SabreSwap(coupling_map, heuristic=heuristic, trials=1), cm]
    )
    qiskit_pm.draw("sabre_pm.png")
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
    input_dag,
    micro_dag,
    micro_mapping,
    coupling_map,
    show,
    heuristic,
    critical=False,
    extended_set_size=20,
):
    ms = MicroSabre(
        micro_dag, micro_mapping, coupling_map, heuristic, critical, extended_set_size
    )
    sabre_result = ms.run()

    transpiled_sabre_dag = apply_sabre_result(
        input_dag.copy_empty_like(),
        input_dag,
        sabre_result,
        input_dag.qubits,
        coupling_map,
    )

    transpiled_micro_sabre_circuit = dag_to_circuit(transpiled_sabre_dag)
    if show:
        transpiled_micro_sabre_circuit.draw("mpl", fold=-1)

    cm = CheckMap(coupling_map=coupling_map)
    qiskit_pm = PassManager([cm])
    transpiled_qc = qiskit_pm.run(transpiled_micro_sabre_circuit)

    if not cm.property_set.get("is_swap_mapped"):
        raise ValueError("CheckMap identified invalid mapping from DAG to coupling_map")

    depth = transpiled_micro_sabre_circuit.depth()
    num_swaps = len(transpiled_sabre_dag.op_nodes(op=SwapGate))

    return depth, num_swaps


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

    for node_id in node_order:
        node = source_dag.node(node_id)
        if node_id in swap_map:
            apply_swaps(dest_dag, swap_map[node_id], initial_layout, physical_qubits)

        qubits = [
            physical_qubits[initial_layout.virtual_to_physical(root_logical_map[q])]
            for q in node.qargs
        ]
        dest_dag._apply_op_node_back(
            DAGOpNode.from_instruction(
                node._to_circuit_instruction().replace(qubits=qubits)
            ),
            check=False,
        )
    return dest_dag


def mapping_to_micro_mapping(initial_mapping):
    micro_mapping = dict()
    # important: keys are virtual qubits and values are physical qubits
    for k, v in initial_mapping.get_virtual_bits().items():
        micro_mapping[k._index] = v
    return micro_mapping


def compare_heuristic_tests():
    return [
        ("basic", False),
        ("basic", True),
        ("lookahead", False),
        ("lookahead", True),
        ("lookahead-0.5", False),
        ("lookahead-0.5", True),
        ("lookahead-scaling", False),
        ("lookahead-scaling", True),
        ("lookahead-0.5-scaling", False),
        ("lookahead-0.5-scaling", True),
    ]


def extended_set_size_exponential():
    return [
        ("lookahead-0.5-scaling", False, 5),
        ("lookahead-0.5-scaling", False, 10),
        ("lookahead-0.5-scaling", False, 20),
        ("lookahead-0.5-scaling", False, 30),
        ("lookahead-0.5-scaling", False, 40),
        ("lookahead-0.5-scaling", False, 50),
        ("lookahead-0.5-scaling", False, 60),
        ("lookahead-0.5-scaling", False, 70),
        ("lookahead-0.5-scaling", False, 80),
        ("lookahead-0.5-scaling", False, 90),
        ("lookahead-0.5-scaling", False, 100),
        ("lookahead-0.5-scaling", False, 200),
        ("lookahead-0.5-scaling", False, 300),
        ("lookahead-0.5-scaling", False, 400),
        ("lookahead-0.5-scaling", False, 500),
        ("lookahead-0.5-scaling", False, 800),
        ("lookahead-0.5-scaling", False, 1000),
    ]


if __name__ == "__main__":
    main()
