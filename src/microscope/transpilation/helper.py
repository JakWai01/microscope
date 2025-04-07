from qiskit.transpiler.layout import Layout


def generate_initial_mapping(dag):
    regs = []

    for name, reg in dag.qregs.items():
        regs.append(reg)

    return Layout.generate_trivial_layout(*regs)


def swap_physical_qubits(physical_q0, physical_q1, current_mapping):
    resulting_mapping = current_mapping.copy()
    logical_q0 = [
        key for key, value in current_mapping.items() if value == physical_q0
    ][0]
    logical_q1 = [
        key for key, value in current_mapping.items() if value == physical_q1
    ][0]
    tmp = current_mapping[logical_q0]
    resulting_mapping[logical_q0] = current_mapping[logical_q1]
    resulting_mapping[logical_q1] = tmp
    return resulting_mapping


def pretty_print_mapping(current_mapping):
    pretty_mapping = [None] * len(current_mapping)
    for k, v in current_mapping.items():
        pretty_mapping[v] = k
    print(pretty_mapping)


def get_successors(dag, node_id):
    successors = []
    for s, t in dag.edges:
        if s == node_id:
            successors.append(t)
    return successors
