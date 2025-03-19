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
    name="retuning_graph_1q_noise",
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
        ),
        "T1": library.nodes["05_T1"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="T1",
            reset_type_thermal_or_active="active",
        ),
        "ramsey_long1": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long1",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            frequency_detuning_in_mhz=0.5,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=15000,
            num_time_points=500,
            state_discrimination=True,
        ),
        "ramsey_long2": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long2",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            frequency_detuning_in_mhz=0.5,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=15000,
            num_time_points=500,
            state_discrimination=True,
        ),
        "ramdon_state_before": library.nodes["100a_Random_state"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramdon_state_before",
        ),
        "ramdon_state_phase_cor": library.nodes["100d_Random_state_phase_correction"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramdon_state_phase_cor",
        ),
        "ramdon_state_after": library.nodes["100a_Random_state"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramdon_state_after",
        ),
        "ramdon_state_ellispoid": library.nodes["100f_Random_state_ellipsoid_repeat"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramdon_state_ellispoid",
        ),


    },
    connectivity=[
        ("close_other_qms","IQ_blob_thermal"),
        ("IQ_blob_thermal", "IQ_blob_active"),
        ("IQ_blob_active", "T1"),
        ("T1", "ramsey_long1"),
        ("ramsey_long1", "ramsey_long2"),
        ("ramsey_long2", "ramdon_state_before"),
        ("ramdon_state_before", "ramdon_state_phase_cor"),
        ("ramdon_state_phase_cor", "ramdon_state_after"),
        ("ramdon_state_after","ramdon_state_ellispoid")
        ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run(qubits=qubits)