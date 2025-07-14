from collections import defaultdict

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import (
    SabreSwap,
    CheckMap,
    Unroll3qOrMore,
    SetLayout,
    FullAncillaAllocation,
    ApplyLayout,
    RemoveBarriers,
    SabreLayout
)
from qiskit.qasm2 import dumps

from qiskit.circuit.library.standard_gates import SwapGate

from commands.helper import generate_initial_mapping

def olsq(config):
    # Parse config variables
    path = config["ocular"]["path"]

    # Parse circuit
    input_circuit = QuantumCircuit.from_qasm_file(path)
    num_qubits = input_circuit.num_qubits

    # Generate coupling map
    coupling_map = CouplingMap.from_line(input_circuit.num_qubits)
    # Generate DAG from circuit
    input_dag = circuit_to_dag(input_circuit)

    # Preprocess circuit
    preprocessing_layout = generate_initial_mapping(input_dag)

    pm = PassManager(
        [
            Unroll3qOrMore(),
            SabreLayout(coupling_map, skip_routing=True),
            ApplyLayout(),
            RemoveBarriers(),
        ]
    )

    preprocessed_circuit = pm.run(input_circuit)

    from qiskit import transpile

    basis_gates = ['cx', 'u3']  # or ['cx', 'u1', 'u2', 'u3'] depending on OLSQ

    decomposed_qc = transpile(preprocessed_circuit, basis_gates=basis_gates, optimization_level=0)

    # Then export to QASM
    from qiskit.qasm2 import dumps
    with open("flattened.qasm", "w") as f:
        f.write(dumps(decomposed_qc))

    from olsq import OLSQ
    from olsq.device import qcdevice

    # lsqc_solver = OLSQ("swap", "transition") # alternatively 'transition'
    lsqc_solver = OLSQ("swap", "transition") # alternatively 'transition'

    connections = [edge for edge in coupling_map.get_edges()]

    # directly construct a device from properties needed by olsq
    lsqc_solver.setdevice( qcdevice(name="dev", nqubits=num_qubits, 
        connection=connections, swap_duration=1) ) # swap duration either 1 or 3

    circuit_file = open("flattened.qasm", "r").read()
    
    lsqc_solver.setprogram(circuit_file)

    result = lsqc_solver.solve()