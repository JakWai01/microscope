import click
import matplotlib.pyplot as plt

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

@click.command()
@click.option("-f", "--filename", type=str, required=True, help="Path to .qasm file")
def main(filename: str):
    """Read in a .qasm file and print out a syntax tree."""
    circuit = QuantumCircuit.from_qasm_file(filename)
    circuit.draw('mpl') 
    # plt.show()
    
    dag = circuit_to_dag(circuit)
    # image = dag_drawer(dag)
    # image.show()
    
    # A line with 10 physical qubits
    coupling_map = CouplingMap.from_line(10)

    # Generate initial 'trivial' mapping 
    initial_mapping = generate_initial_mapping(dag)

    # Execute basic_swap
    new_dag = basic_swap(dag, coupling_map, initial_mapping)
    # bs = BasicSwap(coupling_map)
    # new_dag = bs.run(dag)
    
    # image = dag_drawer(res)
    # image.show()


    # image = dag_drawer(new_dag)
    # image.show()

    res_circuit = dag_to_circuit(new_dag)
    res_circuit.draw('mpl')
    plt.show()


def generate_initial_mapping(dag):
    canonical_register = dag.qregs["q"]
    return Layout.generate_trivial_layout(canonical_register)


def basic_swap(dag, coupling_map, initial_mapping):
    canonical_register = dag.qregs["q"]
    current_mapping = initial_mapping.copy()
    
    new_dag = dag.copy_empty_like()
    
    if coupling_map is None:
        raise TranspilerError("BasicSwap cannot run tieh coupling_map=None")

    if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
        raise TranspilerError("Basic swap runs on physical circuits only")

    if len(dag.qubits) > len(coupling_map.physical_qubits):
        raise TranspilerError("The layout does not match the amount of qubits in the DAG")
    
    for layer in dag.serial_layers():
        subdag = layer["graph"]
        
        for gate in subdag.two_qubit_ops():
            physical_q0 = current_mapping[gate.qargs[0]]
            physical_q1 = current_mapping[gate.qargs[1]]
            
            # With this example, we don't get in here
            if coupling_map.distance(physical_q0, physical_q1) != 1:
                # Insert a new layer with the SWAP(s)
                swap_layer = DAGCircuit()
                swap_layer.add_qreg(canonical_register)

                path = coupling_map.shortest_undirected_path(physical_q0, physical_q1)

                for swap in range(len(path) - 2):
                    connected_wire_1 = path[swap]
                    connected_wire_2 = path[swap + 1]

                    qubit_1 = current_mapping[connected_wire_1]
                    qubit_2 = current_mapping[connected_wire_2]

                    # Create SWAP operation
                    swap_layer.apply_operation_back(
                            SwapGate(), (qubit_1, qubit_2), cargs=(), check=False
                    )

                # Layer insertion
                order = current_mapping.reorder_bits(new_dag.qubits)
                new_dag.compose(swap_layer, qubits=order)

                # Update current Layout
                for swap in range(len(path) - 2):
                    current_mapping.swap(path[swap], path[swap + 1])

        order = current_mapping.reorder_bits(new_dag.qubits)
        new_dag.compose(subdag, qubits=order)

    return new_dag


if __name__ == "__main__":
    main()
