from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = None


qubits = ["q0"]
multiplexed = False
reset_type_thermal_or_active = "active"


g = QualibrationGraph(
    name="quantum_memory_swap",
    parameters=Parameters(qubits=qubits),
    nodes={
        "close_other_qms": library.nodes["00_Close_other_QMs"].copy(
            name="close_other_qms",
        ),
        "ellipsoid_swap_0ns": library.nodes['101a_Random_state_ellipsoid_swap_qubit1ns'].copy(
        name='ellipsoid_swap_0ns',
        qubits = qubits,
        reset_type_thermal_or_active=reset_type_thermal_or_active,
        interaction_time_ns=0,
        repeats = 100 
        ),
        "ellipsoid_swap_1ns": library.nodes['101a_Random_state_ellipsoid_swap_qubit1ns'].copy(
        name='ellipsoid_swap_1ns',
        qubits = qubits,
        reset_type_thermal_or_active=reset_type_thermal_or_active,
        interaction_time_ns=1,
        repeats = 100 
        )

    },
    connectivity=[
        ("close_other_qms", "ellipsoid_swap_0ns"),
        ("ellipsoid_swap_0ns", "ellipsoid_swap_1ns"),
        ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run(qubits=qubits)