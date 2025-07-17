from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import (
    SabreSwap,
    CheckMap,
    Unroll3qOrMore,
    FullAncillaAllocation,
    ApplyLayout,
    RemoveBarriers,
    SabreLayout
)

from qiskit.circuit.library.standard_gates import SwapGate

def qiskit(config):
    path = config["ocular"]["path"]

    test_cases = [("lookahead", 20)]

    input_circuit = QuantumCircuit.from_qasm_file(path)

    coupling_map = CouplingMap.from_line(input_circuit.num_qubits)
    
    pm = PassManager(
        [
            Unroll3qOrMore(),
            SabreLayout(coupling_map, skip_routing=True, seed=42),
            ApplyLayout(),
            FullAncillaAllocation(coupling_map),
            RemoveBarriers(),
        ]
    )

    preprocessed_circuit = pm.run(input_circuit)
    
    for heuristic, _ in test_cases:
        cm = CheckMap(coupling_map=coupling_map)

        qiskit_pm = PassManager(
            [SabreSwap(coupling_map, heuristic=heuristic, trials=1), cm]
        )

        transpiled_qc = qiskit_pm.run(preprocessed_circuit)
        transpiled_qc_dag = circuit_to_dag(transpiled_qc)

        if not cm.property_set.get("is_swap_mapped"):
            raise ValueError("CheckMap identified invalid mapping from DAG to coupling_map")
        
        depth = transpiled_qc.depth()
        num_swaps = len(transpiled_qc_dag.op_nodes(op=SwapGate))

        print(f"SWAPs: {num_swaps}, Depth: {depth}")