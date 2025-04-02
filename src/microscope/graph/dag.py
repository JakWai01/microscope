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


class DAGNode:
    def __init__(self, node_id, control, target, is_swap):
        self.node_id = node_id
        self.control = control
        self.target = target
        self.is_swap = is_swap

    def __repr__(self):
        return str(self.__dict__)
