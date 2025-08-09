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
t2_max = 30000
t2_detuning = 0.4
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
        "T1_1": library.nodes["05_T1"].copy(
            qubits = qubits,
            name="T1_1",
        ),
        "T1_2": library.nodes["05_T1"].copy(
            qubits = qubits,
            name="T1_2",
        ),
        "T1_3": library.nodes["05_T1"].copy(
            qubits = qubits,
            name="T1_3",
        ),
        "T1_4": library.nodes["05_T1"].copy(
            qubits = qubits,
            name="T1_4",
        ),
        "T1_5": library.nodes["05_T1"].copy(
            qubits = qubits,
            name="T1_5",
        ),
        "ramsey_long1": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="ramsey_long1",
            frequency_detuning_in_mhz=t2_detuning,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=t2_max,
            num_time_points=500,
        ),
        "ramsey_long2": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long2",
            frequency_detuning_in_mhz=t2_detuning,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=t2_max,
            num_time_points=500,
        ),
        "ramsey_long3": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long3",
            frequency_detuning_in_mhz=t2_detuning,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=t2_max,
            num_time_points=500,
        ),
        "ramsey_long4": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long4",
            frequency_detuning_in_mhz=t2_detuning,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=t2_max,
            num_time_points=500,
        ),
        "ramsey_long5": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long5",
            frequency_detuning_in_mhz=t2_detuning,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=t2_max,
            num_time_points=500,
        ),
        "ramsey_long6": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long6",
            frequency_detuning_in_mhz=t2_detuning,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=t2_max,
            num_time_points=500,
        ),
        "ramsey_long7": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long7",
            frequency_detuning_in_mhz=t2_detuning,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=t2_max,
            num_time_points=500,
        ),
        "ramsey_long8": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long8",
            frequency_detuning_in_mhz=t2_detuning,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=t2_max,
            num_time_points=500,
        ),
        "ramsey_long9": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long9",
            frequency_detuning_in_mhz=t2_detuning,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=t2_max,
            num_time_points=500,
        ),
        "ramsey_long10": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long10",
            frequency_detuning_in_mhz=t2_detuning,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=t2_max,
            num_time_points=500,
        ),
        "ramdon_state_before": library.nodes["100a_Random_state"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="ramdon_state_before",
        ),
        "ramdon_state_phase_cor": library.nodes["100d_Random_state_phase_correction"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="ramdon_state_phase_cor",
        ),
        "ramdon_state_after": library.nodes["100a_Random_state"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="ramdon_state_after",
        ),
        "ramdon_state_ellispoid": library.nodes["100f_Random_state_ellipsoid_repeat"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="ramdon_state_ellispoid",
        ),
        "QPT": library.nodes["110a_quantum_process_tomography"].copy(
            qubits=qubits,
            flux_point_joint_or_independent="joint",
            name="QPT",
        ),  

    },
    connectivity=[
        ("close_other_qms","IQ_blob_thermal"),
        ("IQ_blob_thermal", "IQ_blob_active"),
        ("IQ_blob_active", "T1_1"),
        ("T1_1", "T1_2"),
        ("T1_2", "T1_3"),
        ("T1_3", "T1_4"),
        ("T1_4", "T1_5"),
        ("T1_5", "ramsey_long1"),
        ("ramsey_long1", "ramsey_long2"),
        ("ramsey_long2", "ramsey_long3"),
        ("ramsey_long3", "ramsey_long4"),
        ("ramsey_long4", "ramsey_long5"),
        ("ramsey_long5", "ramsey_long6"),
        ("ramsey_long6", "ramsey_long7"),
        ("ramsey_long7", "ramsey_long8"),
        ("ramsey_long8", "ramsey_long9"),
        ("ramsey_long9", "ramsey_long10"),
        ("ramsey_long10", "ramdon_state_before"),
        ("ramdon_state_before", "ramdon_state_phase_cor"),
        ("ramdon_state_phase_cor", "ramdon_state_after"),
        ("ramdon_state_after","ramdon_state_ellispoid"),
        ("ramdon_state_ellispoid", "QPT"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run(qubits=qubits)