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
reset_type_thermal_or_active = "thermal"


g = QualibrationGraph(
    name="retuning_graph_debug",
    parameters=Parameters(qubits=qubits),
    nodes={
        "close_other_qms": library.nodes["00_Close_other_QMs"].copy(
            name="close_other_qms",
        ),
        "IQ_blob_thermal": library.nodes["07b_IQ_Blobs"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="IQ_blob_thermal",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
        ),
        "IQ_blob_active": library.nodes["07b_IQ_Blobs1"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="IQ_blob_active",
            reset_type_thermal_or_active="active",
        )
    },
    connectivity=[
        ("close_other_qms", "IQ_blob_thermal"),
        ("IQ_blob_thermal", "IQ_blob_active"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)
g.run(qubits=qubits)