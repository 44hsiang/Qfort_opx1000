from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = None


qubits = ["q0","q1", "q2", "q3", "q4"]
multiplexed = False
reset_type_thermal_or_active = "thermal"


g = QualibrationGraph(
    name="Retuning_Graph",
    parameters=Parameters(qubits=qubits),
    nodes={
        "close_other_qms": library.nodes["00_Close_other_QMs"].copy(
            name="close_other_qms",
        ),
        "qubit_spec": library.nodes["03a_Qubit_Spectroscopy"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="qubit_spec",
            frequency_span_in_mhz = 100,
            frequency_step_in_mhz= 0.25
        ),
        "power_rabi_x180_1": library.nodes["04_Power_Rabi"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            operation_x180_or_any_90="x180",
            name="power_rabi_x180_1",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            num_averages = 500,
            min_amp_factor = 0.001,
            max_amp_factor = 1.99,
            amp_factor_step = 0.005,
            max_number_rabi_pulses_per_sweep=1,
            update_x90=False,
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
            amp_factor_step = 0.005,
            max_number_rabi_pulses_per_sweep=50,
            update_x90=False,
            state_discrimination=False,
        ),
        "ramsey": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            frequency_detuning_in_mhz=1.0,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=3000,
            num_time_points=500,
            state_discrimination=False,
        ),
        "ramsey1": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey1",
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
        "DRAG": library.nodes["09c_DRAG_Calibration_180_90"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="DRAG"
        ),
        "single_qubit_randomized_benchmarking": library.nodes["10a_Single_Qubit_Randomized_Benchmarking"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=False, 
            delta_clifford=20,
            num_random_sequences=1000,
            name="Single_Qubit_Randomized_Benchmarking"
        ),
    },
    connectivity=[
        ("close_other_qms", "qubit_spec"),
        ("qubit_spec", "power_rabi_x180_1"),
        ("power_rabi_x180_1", "power_rabi_x180_50"),
        ("power_rabi_x180_50", "ramsey"),
        ("ramsey", "ramsey1"),
        ("ramsey1", "Readout_fre_opt"),
        ("Readout_fre_opt", "IQ_blob_thermal"),
        ("IQ_blob_thermal", "IQ_blob_active"),
        ("IQ_blob_active", "DRAG"),
        ("DRAG", "single_qubit_randomized_benchmarking"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run(qubits=qubits)