from qiskit.transpiler.layout import Layout
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    Unroll3qOrMore,
    SetLayout,
    FullAncillaAllocation,
    ApplyLayout,
    RemoveBarriers,
)

from rich.console import Console
from rich.table import Table
from qiskit.converters import circuit_to_dag

import matplotlib.pyplot as plt
from qiskit.circuit.library.standard_gates import SwapGate


def mapping_to_micro_mapping(initial_mapping):
    micro_mapping = dict()
    # important: keys are virtual qubits and values are physical qubits
    for k, v in initial_mapping.get_virtual_bits().items():
        micro_mapping[k._index] = v
    return micro_mapping


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


def apply_swaps(dest_dag, swaps, layout, physical_qubits):
    for a, b in swaps:
        qubits = (
            physical_qubits[a],
            physical_qubits[b],
        )
        layout.swap_physical(a, b)
        dest_dag.apply_operation_back(SwapGate(), qubits, (), check=False)
