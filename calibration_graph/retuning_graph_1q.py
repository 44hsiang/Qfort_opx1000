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
    name="retuning_graph_1q",
    parameters=Parameters(qubits=qubits),
    nodes={
        "close_other_qms": library.nodes["00_Close_other_QMs"].copy(
            name="close_other_qms",
        ),
        "qubit_fre": library.nodes["03a_Qubit_Spectroscopy"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="qubit_fre",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            operation_amplitude_factor= 0.2,
            frequency_span_in_mhz = 50,
            frequency_step_in_mhz = 0.1
        ),
        "power_rabi_x180_1": library.nodes["04_Power_Rabi"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            operation_x180_or_any_90="x180",
            name="power_rabi_x180_1",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            num_averages = 100,
            min_amp_factor = 0.01,
            max_amp_factor = 1.99,
            amp_factor_step = 0.005,
            max_number_rabi_pulses_per_sweep=1,
            update_x90=True,
            state_discrimination=False,
        ),
        "power_rabi_x180_50": library.nodes["04_Power_Rabi"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            operation_x180_or_any_90="x180",
            name="power_rabi_x180_50",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            num_averages = 100,
            min_amp_factor = 0.9,
            max_amp_factor = 1.1,
            amp_factor_step = 0.002,
            max_number_rabi_pulses_per_sweep=50,
            update_x90=True,
            state_discrimination=False,
        ),
        "ramsey_short1": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_short1",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            frequency_detuning_in_mhz=1.0,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=3000,
            num_time_points=500,
            state_discrimination=False,
        ),
        "ramsey_short2": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_short2",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            frequency_detuning_in_mhz=1.0,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=3000,
            num_time_points=500,
            state_discrimination=False,
        ),
        "Readout_fre_opt": library.nodes["07a_Readout_Frequency_Optimization"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="Readout_fre_opt",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
        ),
        "DRAG": library.nodes["09c_DRAG_Calibration_180_90"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="DRAG"
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
        "ramsey_long": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            frequency_detuning_in_mhz=0.5,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=10000,
            num_time_points=500,
            state_discrimination=True,
        ),

    },
    connectivity=[
        ("close_other_qms", "qubit_fre"),
        ("qubit_fre", "power_rabi_x180_1"),
        ("power_rabi_x180_1", "power_rabi_x180_50"),
        ("power_rabi_x180_50", "ramsey_short1"),
        ("ramsey_short1", "ramsey_short2"),
        ("ramsey_short2","Readout_fre_opt"),
        ("Readout_fre_opt", "DRAG"),
        ("DRAG", "IQ_blob_thermal"),
        ("IQ_blob_thermal", "IQ_blob_active"),
        ("IQ_blob_active", "T1"),
        ("T1", "ramsey_long")
        ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run(qubits=qubits)