import microboost

class DAG:
    def __init__(self):
        self.nodes = dict()
        self.edges = []
        self._last_op_on_qubit = dict()

    def insert(self, node_id, qubits):
        node = microboost.MicroDAGNode(node_id, qubits)
        # node = DAGNode(node_id, qubits)

        node_index = len(self.nodes)

        self.nodes[node_index] = node

        self._update_edges(node_index)

        return node_id

    def _update_edges(self, node_index):
        node = self.nodes[node_index]

        if len(node.qubits) == 2:
            predecessor_node_a = self._last_op_on_qubit.get(node.qubits[0])
            predecessor_node_b = self._last_op_on_qubit.get(node.qubits[1])

            if predecessor_node_a != None:
                self.edges.append((predecessor_node_a, node_index))

            if predecessor_node_b != None and predecessor_node_a != predecessor_node_b:
                self.edges.append((predecessor_node_b, node_index))

            self._last_op_on_qubit[node.qubits[0]] = node_index
            self._last_op_on_qubit[node.qubits[1]] = node_index
        elif len(node.qubits) == 1:
            predecessor_node_a = self._last_op_on_qubit.get(node.qubits[0])

            if predecessor_node_a != None:
                self.edges.append((predecessor_node_a, node_index))

            self._last_op_on_qubit[node.qubits[0]] = node_index
        else:
            raise Exception(
                "Cannot update edges. Operation has unsupported number of qubits"
            )

    def get(self, node_index):
        return self.nodes[node_index]

    def from_qiskit_dag(self, dag):
        """Create DAG from qiskit DAGCircuit

        Nodes of the DAG represent operations, edges represent dependencies
        """

        for node in dag.topological_op_nodes():
            if node.op.num_qubits == 2:
                self.insert(node._node_id, [node.qargs[0]._index, node.qargs[1]._index])
            elif node.op.num_qubits == 1:
                self.insert(node._node_id, [node.qargs[0]._index])
            else:
                raise Exception(
                    "Cannot create DAG from Qiskit DAG. Operation has unsupported number of qubits"
                )
        return self
    
    def to_micro_dag(self):
        # TODO: We need to have Rust DAGNode Objects as nodes
        return microboost.MicroDAG(self.nodes, self.edges)

    def __str__(self):
        return str(self.__dict__)

    def __len__(self):
        return len(self.nodes)


class DAGNode:
    def __init__(self, node_id, qubits):
        self.node_id = node_id
        self.qubits = qubits

    def __repr__(self):
        return str(self.__dict__)
