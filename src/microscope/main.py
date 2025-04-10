import click
import matplotlib.pyplot as plt
import microscope

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.visualization.dag_visualization import dag_drawer
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.passes.routing.basic_swap import BasicSwap
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit import warnings
from qiskit import QuantumRegister
from qiskit.circuit import Qubit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (
    UnitarySynthesis,
    HighLevelSynthesis,
    Unroll3qOrMore,
    SetLayout,
    TrivialLayout,
    FullAncillaAllocation,
    EnlargeWithAncilla,
    ApplyLayout,
    SabreSwap,
)

from qiskit import transpile

from transpilation.helper import generate_initial_mapping
from graph.conversion import (
    mapping_to_micro_mapping,
    transpiled_micro_dag_to_transpiled_qiskit_dag,
    merge_top_swap,
)
from graph.dag import DAG, DAGNode
from transpilation.basic_swap import micro_swap, basic_swap
from transpilation.sabre import micro_sabre, micro_sabre_v2


@click.command()
@click.option("-f", "--filename", type=str, required=True, help="Path to .qasm file")
@click.option("-d", "--show-dag", type=bool, help="True if DAG should be shown")
@click.option(
    "-q", "--qiskit-fallback", type=bool, help="Use qiskit algorithm implementation"
)
def main(filename: str, show_dag: bool, qiskit_fallback: bool):
    """Read in a .qasm file and print out a syntax tree."""

    # Ignore deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    input_circuit = QuantumCircuit.from_qasm_file(filename)
    input_circuit.draw("mpl", fold=160)

    # A line with 10 physical qubits
    coupling_map = CouplingMap.from_line(10)

    preprocessing_dag = circuit_to_dag(input_circuit)
    preprocessing_layout = generate_initial_mapping(preprocessing_dag)

    pm = PassManager(
        [
            Unroll3qOrMore(),
            SetLayout(preprocessing_layout),
            FullAncillaAllocation(coupling_map),
            ApplyLayout(),
        ]
    )

    preprocessed_circuit = pm.run(input_circuit)
    preprocessed_circuit.draw("mpl", fold=160)

    input_dag = circuit_to_dag(preprocessed_circuit)

    initial_mapping = generate_initial_mapping(input_dag)

    micro_mapping = mapping_to_micro_mapping(initial_mapping)

    micro_dag = DAG().from_qiskit_dag(input_dag)

    if show_dag:
        input_dag_image = dag_drawer(input_dag)
        input_dag_image.show()

    # Execute basic swap algorithm
    if qiskit_fallback:
        bs = BasicSwap(coupling_map)
        transpiled_dag = bs.run(input_dag)
    else:
        pass
        # Rust implementation
        boosted_transpiled_micro_dag = microscope.micro_swap_boosted(
            micro_dag, coupling_map, micro_mapping
        )
        boosted_transpiled_qiskit_dag = merge_top_swap(
            boosted_transpiled_micro_dag, input_dag, initial_mapping, coupling_map
        )
        boosted_transpiled_qiskit_dag_circuit = dag_to_circuit(
            boosted_transpiled_qiskit_dag
        )
        boosted_transpiled_qiskit_dag_circuit.draw("mpl", fold=160)

        # MicroDAG implementation
        transpiled_micro_dag = micro_swap(micro_dag, coupling_map, micro_mapping)
        transpiled_qiskit_dag = merge_top_swap(
            transpiled_micro_dag, input_dag, initial_mapping, coupling_map
        )

        transpiled_qiskit_dag_circuit = dag_to_circuit(transpiled_qiskit_dag)
        transpiled_qiskit_dag_circuit.draw("mpl", fold=160)

        # Qiskit-style implementation
        transpiled_dag = basic_swap(input_dag, coupling_map, initial_mapping)

    if show_dag:
        output_dag_image = dag_drawer(transpiled_dag)
        output_dag_image.show()

    # Convert DAG to circuit
    transpiled_circuit = dag_to_circuit(transpiled_dag)
    transpiled_circuit.draw("mpl", fold=160)

    # Qiskit SABRE implementation
    qiskit_pm = PassManager([SabreSwap(coupling_map)])
    transpiled_qc = qiskit_pm.run(preprocessed_circuit)
    transpiled_qc.draw("mpl", fold=160)

    # MicroSABRE implementation
    transpiled_sabre_dag = micro_sabre_v2(
        micro_dag, coupling_map, micro_mapping, "lookahead"
    )

    # transpiled_micro_sabre_dag = merge_top_swap(
    #     transpiled_sabre_dag, input_dag, initial_mapping, coupling_map
    # )
    #
    # transpiled_micro_sabre_circuit = dag_to_circuit(transpiled_micro_sabre_dag)
    # transpiled_micro_sabre_circuit.draw("mpl", fold=160)

    plt.show()


if __name__ == "__main__":
    main()
