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

alpha_list = np.arange(0.9, 1.61, 0.01)

def generate_nodes(alpha_list):
    nodes = {
        "close_other_qms": library.nodes["00_Close_other_QMs"].copy(
            name="close_other_qms",
        )
    }
    connectivity = []
    for i, a in enumerate(alpha_list):
        name = f"gate_set_tomo_{i:05d}"
        nodes[name] = library.nodes["120a_Gate_Set_Tomography_imperfect"].copy(
            name=name,
            qubits = qubits,
            alpha = round(a, 2),
            max_circuit_depth_in_power = 4,
            reset_type_thermal_or_active=reset_type_thermal_or_active,
        )
        if i == 0:
            connectivity.append(
                ("close_other_qms", name)
            )
        else:
            connectivity.append(
                (connectivity[-1][1], name)
            )
    return nodes, connectivity

nodes, connectivity = generate_nodes(alpha_list)

g = QualibrationGraph(
    name="gate_set_tomo_via_gate_err",
    parameters=Parameters(qubits=qubits),
    nodes=nodes,
    connectivity=connectivity,
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run(qubits=qubits)
