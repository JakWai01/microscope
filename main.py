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
@click.option("-d", "--show-dag", type=bool, help="True if DAG should be shown")
@click.option("-q", "--qiskit-fallback", type=bool, help="Use qiskit algorithm implementation")
def main(filename: str, show_dag: bool, qiskit_fallback: bool):
    """Read in a .qasm file and print out a syntax tree."""
    circuit = QuantumCircuit.from_qasm_file(filename)
    circuit.draw('mpl') 
    
    input_dag = circuit_to_dag(circuit)

    micro_dag = DAG().from_qiskit_dag(input_dag)
    print(micro_dag.__dict__)
    
    if show_dag:
        input_dag_image = dag_drawer(input_dag)
        input_dag_image.show()
    
    # A line with 10 physical qubits
    coupling_map = CouplingMap.from_line(10)

    # Generate initial 'trivial' mapping 
    initial_mapping = generate_initial_mapping(input_dag)

    # Execute basic swap algorithm
    if qiskit_fallback:
        bs = BasicSwap(coupling_map)
        transpiled_dag = bs.run(input_dag)
    else:
        transpiled_dag = basic_swap(input_dag, coupling_map, initial_mapping)
    
    if show_dag:
        output_dag_image = dag_drawer(transpiled_dag)
        output_dag_image.show()
    
    # Convert DAG to circuit
    transpiled_circuit = dag_to_circuit(transpiled_dag)
    transpiled_circuit.draw('mpl')

    # Show circuits
    plt.show()

def generate_initial_mapping(dag):
    canonical_register = dag.qregs["q"]
    return Layout.generate_trivial_layout(canonical_register)

class DAGNode:
    def __init__(self, node_id, control, target):
        self.node_id = node_id
        self.control = control
        self.target = target

    def __repr__(self):
        return str(self.__dict__)

class DAG:
    def __init__(self):
        self.nodes = dict()
        self.edges = []
        self.last_op_on_qubit = dict()

    def insert_node(self, control, target):
        node_id = len(self.nodes)
        node = DAGNode(node_id, control, target)

        self.nodes[node_id] = node

        self._update_edges(node_id)

        return node_id

    def _update_edges(self, node_id):
        node = self.nodes[node_id]

        predecessor_node_a = self.last_op_on_qubit.get(node.control)
        predecessor_node_b = self.last_op_on_qubit.get(node.target)

        if predecessor_node_a != None:
            self.edges.append((predecessor_node_a, node_id))

        if predecessor_node_b != None and predecessor_node_a != predecessor_node_b:
            self.edges.append((predecessor_node_b, node_id))

        self.last_op_on_qubit[node.control] = node_id
        self.last_op_on_qubit[node.target] = node_id


    def from_qiskit_dag(self, dag):
        """Create DAG from qiskit DAGCircuit

        Filtering for two qubit operations manually is necessary because the
        documentation says whether `.two_qubit_ops()` is topologically ordered.
        Directives as e.g. Barriers are *not* supported.
        Nodes of the DAG represent operations, edges represent dependencies
        """

        for node in dag.topological_op_nodes():
            if node.op.num_qubits == 2:
                print(f'{node.name} -> {node.qargs[0]._index}-{node.qargs[1]._index}')
                self.insert_node(node.qargs[0]._index, node.qargs[1]._index)

        return self

    def __str__(self):
        return self.__dict__

def basic_swap(dag, coupling_map, initial_mapping):
    canonical_register = dag.qregs["q"]
    current_mapping = initial_mapping.copy()
    
    new_dag = dag.copy_empty_like()
    
    if coupling_map is None:
        raise TranspilerError("BasicSwap cannot run with coupling_map=None")

    if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
        raise TranspilerError("Basic swap runs on physical circuits only")

    if len(dag.qubits) > len(coupling_map.physical_qubits):
        raise TranspilerError("The layout does not match the amount of qubits in the DAG")
    
    for layer in dag.serial_layers():
        subdag = layer["graph"]
        
        for gate in subdag.two_qubit_ops():
            physical_q0 = current_mapping[gate.qargs[0]]
            physical_q1 = current_mapping[gate.qargs[1]]
            
            # Check if SWAP is required
            if coupling_map.distance(physical_q0, physical_q1) != 1:
                # Insert a new layer with the SWAP(s)
                swap_layer = DAGCircuit()
                swap_layer.add_qreg(canonical_register)
    
                # Find shortest SWAP path
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

                # Update current Mapping 
                for swap in range(len(path) - 2):
                    current_mapping.swap(path[swap], path[swap + 1])

        order = current_mapping.reorder_bits(new_dag.qubits)
        new_dag.compose(subdag, qubits=order)

    return new_dag


if __name__ == "__main__":
    main()
