import click
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.visualization.dag_visualization import dag_drawer
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.layout import Layout

@click.command()
@click.option("-f", "--filename", type=str, required=True, help="Path to .qasm file")
def main(filename: str):
    """Read in a .qasm file and print out a syntax tree."""
    circuit = QuantumCircuit.from_qasm_file(filename)
    # circuit.draw('mpl') 
    # plt.show()
    
    dag = circuit_to_dag(circuit)

    coupling_map = CouplingMap.from_line(10)
    initial_mapping = generate_initial_mapping(dag)
    basic_swap(dag, coupling_map, initial_mapping)

    # image = dag_drawer(dag)
    # image.show()
        # print(layer)
    # for node in dag.topological_op_nodes():
    #     # DAGOpNode
    #     print(node.cargs)
    #     print(node.qargs)
    #     print(node.label)
    #     print(node.op)

    print(dag.qregs["q"])

def generate_initial_mapping(dag):
    canonical_register = dag.qregs["q"]
    Layout.generate_trivial_layout(canonical_register)

def basic_swap(dag, coupling_map, initial_mapping):
    for layer in dag.serial_layers():
        subdag = layer["graph"]
        # for gate in subdag.op_nodes():
        #     print(gate.name)
        for gate in subdag.two_qubit_ops():
            print(gate.name)

if __name__ == "__main__":
    main()
