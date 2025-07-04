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
    RemoveBarriers
)

from qiskit.circuit.library.standard_gates import SwapGate

from commands.helper import generate_initial_mapping

def qiskit(config):
    # Parse config variables
    path = config["ocular"]["path"]
    heuristics = config["ocular"]["heuristics"]
    trials = config["ocular"]["trials"]
    extended_set_size = config["ocular"]["extended-set-size"]

    test_cases = [("lookahead", 20)]
    test_results = defaultdict(list)

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
            SetLayout(preprocessing_layout),
            FullAncillaAllocation(coupling_map),
            ApplyLayout(),
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