class DAG:
    def __init__(self):
        self.nodes = dict()
        self.edges = []
        self._last_op_on_qubit = dict()

    def insert(self, node_id, qubits):
        node = DAGNode(node_id, qubits)

        node_index = len(self.nodes)

        self.nodes[node_index] = node

        self._update_edges(node_index)

        return node_id

    def _update_edges(self, node_id):
        node = self.nodes[node_id]

        if len(node.qubits) == 2:
            predecessor_node_a = self._last_op_on_qubit.get(node.qubits[0])
            predecessor_node_b = self._last_op_on_qubit.get(node.qubits[1])

            if predecessor_node_a != None:
                self.edges.append((predecessor_node_a, node_id))

            if predecessor_node_b != None and predecessor_node_a != predecessor_node_b:
                self.edges.append((predecessor_node_b, node_id))

            self._last_op_on_qubit[node.qubits[0]] = node_id
            self._last_op_on_qubit[node.qubits[1]] = node_id
        elif len(node.qubits) == 1:
            predecessor_node_a = self._last_op_on_qubit.get(node.qubits[0])

            if predecessor_node_a != None:
                self.edges.append((predecessor_node_a, node_id))

            self._last_op_on_qubit[node.qubits[0]] = node_id
        else:
            raise Exception(
                "Cannot update edges. Operation has unsupported number of qubits"
            )

    # TODO: This method is wrong node
    def get(self, node_index):
        return self.nodes[node_index]

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
                self.insert(node._node_id, [node.qargs[0]._index, node.qargs[1]._index])
            elif node.op.num_qubits == 1:
                self.insert(node._node_id, [node.qargs[0]._index])
            else:
                raise Exception(
                    "Cannot create DAG from Qiskit DAG. Operation has unsupported number of qubits"
                )
        return self

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
