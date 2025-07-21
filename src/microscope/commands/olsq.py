from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import (
    Unroll3qOrMore,
    ApplyLayout,
    RemoveBarriers,
    SabreLayout,
)
from qiskit.qasm2 import dumps
from olsq import OLSQ
from olsq.device import qcdevice # type: ignore


def olsq(config):
    path = config["ocular"]["path"]

    input_circuit = QuantumCircuit.from_qasm_file(path)
    num_qubits = input_circuit.num_qubits

    coupling_map = CouplingMap.from_line(input_circuit.num_qubits)

    pm = PassManager(
        [
            Unroll3qOrMore(),
            SabreLayout(coupling_map, skip_routing=True),
            ApplyLayout(),
            RemoveBarriers(),
        ]
    )

    preprocessed_circuit = pm.run(input_circuit)


    basis_gates = ["cx", "u3"]  # or ['cx', 'u1', 'u2', 'u3'] depending on OLSQ

    decomposed_qc = transpile(
        preprocessed_circuit, basis_gates=basis_gates, optimization_level=0
    )

    with open("flattened.qasm", "w") as f:
        f.write(dumps(decomposed_qc))


    lsqc_solver = OLSQ("swap", "transition")  # alternatively 'normal'

    connections = [edge for edge in coupling_map.get_edges()]

    lsqc_solver.setdevice(
        qcdevice(
            name="dev", nqubits=num_qubits, connection=connections, swap_duration=1
        )
    )  # swap duration either 1 or 3

    circuit_file = open("flattened.qasm", "r").read()

    lsqc_solver.setprogram(circuit_file)

    lsqc_solver.solve()
