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
from qiskit import warnings

@click.command()
@click.option("-f", "--filename", type=str, required=True, help="Path to .qasm file")
@click.option("-d", "--show-dag", type=bool, help="True if DAG should be shown")
@click.option("-q", "--qiskit-fallback", type=bool, help="Use qiskit algorithm implementation")
def main(filename: str, show_dag: bool, qiskit_fallback: bool):
    """Read in a .qasm file and print out a syntax tree."""

    import microscope

    string_sum = microscope.sum_as_string(1, 2)
    print(string_sum)

    # Ignore deprecation warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    circuit = QuantumCircuit.from_qasm_file(filename)
    circuit.draw('mpl') 
    
    input_dag = circuit_to_dag(circuit)

    # A line with 10 physical qubits
    coupling_map = CouplingMap.from_line(10)

    # Generate initial 'trivial' mapping 
    initial_mapping = generate_initial_mapping(input_dag)

    micro_mapping = mapping_to_micro_mapping(initial_mapping)

    micro_dag = DAG().from_qiskit_dag(input_dag)

    # Testing rust integration
    boosted_transpiled_micro_dag = microscope.micro_swap_boosted(micro_dag, coupling_map, micro_mapping)

    transpiled_micro_dag = micro_swap(micro_dag, coupling_map, micro_mapping)
    transpiled_qiskit_dag = transpiled_micro_dag_to_transpiled_qiskit_dag(transpiled_micro_dag, input_dag, initial_mapping)
    transpiled_qiskit_dag_circuit = dag_to_circuit(transpiled_qiskit_dag)
    transpiled_qiskit_dag_circuit.draw('mpl')

    if show_dag:
        input_dag_image = dag_drawer(input_dag)
        input_dag_image.show()
    
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

# Pass through the input_dag and insert SWAPs at all points in which the
# micro_dag has SWAPs.
# The algorithm iterates through the gates and inserts SWAPs before if
# necessary.
def transpiled_micro_dag_to_transpiled_qiskit_dag(micro_dag, input_dag, initial_mapping):
    # Mapping phyiscal qubits to logical qubits e.g. {"physical": "logical"}
    current_mapping = initial_mapping.copy()

    transpiled_qiskit_dag = input_dag.copy_empty_like()

    canonical_register = input_dag.qregs["q"]
    
    # Start at one to represent lookahead
    current_dag_pointer = 0
    
    # Iterate through each gate
    for layer in input_dag.serial_layers():
        subdag = layer["graph"]

        # If gate is two qubit operation (max. one)
        for gate in subdag.two_qubit_ops():

            micro_dag_node = micro_dag.get(current_dag_pointer)
            
            # We are in here only three times. So some basic stuff is not working
            if micro_dag_node.is_swap == True:
                swap_layer = DAGCircuit()
                swap_layer.add_qreg(canonical_register)
                
                swaps = []
                # We are in here only 4 times. This is not true
                while micro_dag_node.is_swap == True:
                    swap_layer.apply_operation_back(
                            SwapGate(), (current_mapping[micro_dag_node.control], current_mapping[micro_dag_node.target]), cargs=(), check=False
                    )
                    
                    swaps.append((micro_dag_node.control, micro_dag_node.target))

                    current_dag_pointer += 1
                    micro_dag_node = micro_dag.get(current_dag_pointer)
                
                order = current_mapping.reorder_bits(transpiled_qiskit_dag.qubits)
                transpiled_qiskit_dag.compose(swap_layer, qubits=order)
            
                for swap in swaps:
                    current_mapping.swap(swap[0], swap[1])
        
            current_dag_pointer += 1
            
        order = current_mapping.reorder_bits(transpiled_qiskit_dag.qubits)
        transpiled_qiskit_dag.compose(subdag, qubits=order)

    return transpiled_qiskit_dag


def mapping_to_micro_mapping(initial_mapping):
    micro_mapping = dict()
    # IMPORTANT: Keys are virtual qubits and values are physical qubits
    for k, v in initial_mapping.get_virtual_bits().items():
        micro_mapping[k._index] = v
    return micro_mapping

def generate_initial_mapping(dag):
    canonical_register = dag.qregs["q"]
    return Layout.generate_trivial_layout(canonical_register)

class DAGNode:
    def __init__(self, node_id, control, target, is_swap):
        self.node_id = node_id
        self.control = control
        self.target = target
        self.is_swap = is_swap

    def __repr__(self):
        return str(self.__dict__)

class DAG:
    def __init__(self):
        self.nodes = dict()
        self.edges = []
        self._last_op_on_qubit = dict()

    def insert(self, control, target, is_swap):
        node_id = len(self.nodes)
        node = DAGNode(node_id, control, target, is_swap)

        self.nodes[node_id] = node

        self._update_edges(node_id)

        return node_id
    
    def _update_edges(self, node_id):
        node = self.nodes[node_id]

        predecessor_node_a = self._last_op_on_qubit.get(node.control)
        predecessor_node_b = self._last_op_on_qubit.get(node.target)

        if predecessor_node_a != None:
            self.edges.append((predecessor_node_a, node_id))

        if predecessor_node_b != None and predecessor_node_a != predecessor_node_b:
            self.edges.append((predecessor_node_b, node_id))

        self._last_op_on_qubit[node.control] = node_id
        self._last_op_on_qubit[node.target] = node_id
    
    # Get node by id
    def get(self, node_id):
        return self.nodes[node_id]

    def from_qiskit_dag(self, dag):
        """Create DAG from qiskit DAGCircuit

        Filtering for two qubit operations manually is necessary because the
        documentation says whether `.two_qubit_ops()` is topologically ordered.
        Directives as e.g. Barriers are *not* supported.
        Nodes of the DAG represent operations, edges represent dependencies
        """

        for node in dag.topological_op_nodes():
            if node.op.num_qubits == 2:
                # SWAP boolean is false since there are no SWAP gates before the transpilation
                self.insert(node.qargs[0]._index, node.qargs[1]._index, False)

        return self

    def __str__(self):
        return self.__dict__

    def __len__(self):
        return len(self.nodes)

def micro_swap(dag, coupling_map, initial_mapping):
    # Mapping logical to physical qubits e.g. {"logic": "physical"}
    current_mapping = initial_mapping.copy()

    new_dag = DAG()
    
    # pretty_print_mapping(current_mapping)

    for node_id in range(len(dag)):
        node = dag.get(node_id)

        physical_q0 = current_mapping[node.control] 
        physical_q1 = current_mapping[node.target]
        
        # Check if SWAP is required
        if coupling_map.distance(physical_q0, physical_q1) != 1:
            # Returns the shortest undirectedpath between two physical qubits
            path = coupling_map.shortest_undirected_path(physical_q0, physical_q1)

            for swap in range(len(path) - 2):
                connected_wire_1 = path[swap]
                connected_wire_2 = path[swap + 1]
                    
                # TODO: Check if we can improve the data structure to avoid this
                # Probably just maintaining both mappings is enough...
                logical_q0 = [key for key, value in current_mapping.items() if value == connected_wire_1][0]
                logical_q1 = [key for key, value in current_mapping.items() if value == connected_wire_2][0]
                
                qubit_1 = current_mapping[logical_q0]
                qubit_2 = current_mapping[logical_q1]
                
                new_dag.insert(qubit_1, qubit_2, True)
            
            for swap in range(len(path) - 2):
                current_mapping = swap_physical_qubits(path[swap], path[swap+1], current_mapping) 
                # pretty_print_mapping(current_mapping)
        
        new_dag.insert(current_mapping[node.control], current_mapping[node.target], False)

    return new_dag

def swap_physical_qubits(physical_q0, physical_q1, current_mapping):
    logical_q0 = [key for key, value in current_mapping.items() if value == physical_q0][0]
    logical_q1 = [key for key, value in current_mapping.items() if value == physical_q1][0]
    tmp = current_mapping[logical_q0]
    current_mapping[logical_q0] = current_mapping[logical_q1]
    current_mapping[logical_q1] = tmp
    return current_mapping

def pretty_print_mapping(current_mapping):
    pretty_mapping = [None] * len(current_mapping)
    for k, v in current_mapping.items():
        pretty_mapping[v] = k
    print(pretty_mapping)

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
