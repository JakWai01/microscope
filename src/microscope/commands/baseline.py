from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import (
    SabreSwap,
    CheckMap,
)
from qiskit.converters import circuit_to_dag
from commands.helper import preprocess
from qiskit.circuit.library.standard_gates import SwapGate

def qiskit_baseline(file):
    circuit = QuantumCircuit.from_qasm_file(file)
    coupling_map = CouplingMap.from_line(circuit.num_qubits)
    preprocessing_dag = circuit_to_dag(circuit)
    
    preprocessed_circuit, _ = preprocess(circuit, preprocessing_dag, coupling_map)
    qiskit_test_executions = ["basic", "lookahead", "decay"]
    for heuristic in qiskit_test_executions:
        depth, swaps = sabre(preprocessed_circuit, coupling_map, heuristic)
        print(f"Qiskit:\n\tHeuristic: {heuristic}\n\tDepth: {depth}\n\tSwaps: {swaps}")

def sabre(preprocessed_circuit, coupling_map, heuristic):
    cm = CheckMap(coupling_map=coupling_map)
    qiskit_pm = PassManager(
        [SabreSwap(coupling_map, heuristic=heuristic, trials=1), cm]
    )
    # qiskit_pm.draw("sabre_pm.png")
    transpiled_qc = qiskit_pm.run(preprocessed_circuit)
    transpiled_qc_dag = circuit_to_dag(transpiled_qc)

    if not cm.property_set.get("is_swap_mapped"):
        raise ValueError("CheckMap identified invalid mapping from DAG to coupling_map")

    depth = transpiled_qc.depth()
    num_swaps = len(transpiled_qc_dag.op_nodes(op=SwapGate))

    return depth, num_swaps