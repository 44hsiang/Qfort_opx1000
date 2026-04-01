from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary
import numpy as np

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = None


qubits = ["q0"]
multiplexed = False
reset_type_thermal_or_active = "active"

alpha_list = np.arange(1, 1.301, 0.01)

def generate_nodes(alpha_list):
    first = True
    nodes = {
        "close_other_qms": library.nodes["00_Close_other_QMs"].copy(
            name="close_other_qms",
        )
    }
    connectivity = []
    for method in range(1):
        for i, a in enumerate(alpha_list):
            if method == 0:
                name = f"quantum_process_tomo_{i:03d}"
                nodes[name] = library.nodes["110a_quantum_process_tomography_gate_err"].copy(
                    name=name,
                    qubits = qubits,
                    alpha = round(a, 2),
                    reset_type_thermal_or_active=reset_type_thermal_or_active,
                    load_data_id = None
                )
            elif method == 1:
                name = f"ellipsoid_{i:03d}"
                nodes[name] = library.nodes["100i_Random_state_ellipsoid_gate_err"].copy(
                    name=name,
                    qubits = qubits,
                    alpha = round(a, 2),
                    reset_type_thermal_or_active=reset_type_thermal_or_active,
                    load_data_id = None
                )
            if first:
                connectivity.append(
                    ("close_other_qms", name)
                )
                first = False
            else:
                connectivity.append(
                    (connectivity[-1][1], name)
                )
    return nodes, connectivity

nodes, connectivity = generate_nodes(alpha_list)

g = QualibrationGraph(
    name="qpt_ellipsoid_via_gate_err",
    parameters=Parameters(qubits=qubits),
    nodes=nodes,
    connectivity=connectivity,
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run(qubits=qubits)
