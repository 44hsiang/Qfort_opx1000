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
        "rabi_1": library.nodes["04_Power_Rabi"].copy(
            name="rabi_1",
            num_averages = 500,
            operation_x180_or_any_90 = "x180",
            min_amp_factor= 0.0,
            max_amp_factor = 1.99,
            amp_factor_step = 0.005,
            max_number_rabi_pulses_per_sweep = 1
        ),
        "rabi_50": library.nodes["04_Power_Rabi"].copy(
            name="rabi_50",
            num_averages = 50,
            operation_x180_or_any_90 = "x180",
            min_amp_factor= 0.9,
            max_amp_factor = 1.1,
            amp_factor_step = 0.002,
            max_number_rabi_pulses_per_sweep = 50
        ),
        "IQ_blob_thermal": library.nodes["07b_IQ_Blobs"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="IQ_blob_thermal",
            reset_type_thermal_or_active=reset_type_thermal_or_active,
            readout_scale = 1.0,
        ),
        "IQ_blob_active": library.nodes["07b_IQ_Blobs1"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="IQ_blob_active",
            reset_type_thermal_or_active="active",
            readout_scale = 1.0,
        ),
        "T1": library.nodes["05_T1"].copy(
            qubits = qubits,
            name="T1",
        ),
        "ramsey_long1": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="ramsey_long1",
            frequency_detuning_in_mhz=0.5,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=15000,
            num_time_points=500,
        ),
        "ramsey_long2": library.nodes["06_Ramsey"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="ramsey_long2",
            frequency_detuning_in_mhz=0.5,
            min_wait_time_in_ns=16,
            max_wait_time_in_ns=15000,
            num_time_points=500,
        ),
        "DRAG": library.nodes["09c_DRAG_Calibration_180_90"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="DRAG",
            operation="x180",
        ),
        "1qRB": library.nodes["10a_Single_Qubit_Randomized_Benchmarking"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="1qRB",
        ),


    },
    connectivity=[
        ("close_other_qms",'rabi_1'),
        ('rabi_1',"rabi_50"),
        ("rabi_50","IQ_blob_thermal"),
        ("IQ_blob_thermal", "IQ_blob_active"),
        ("IQ_blob_active", "T1"),
        ("T1", "ramsey_long1"),
        ("ramsey_long1", "ramsey_long2"),
        ("ramsey_long2", "DRAG"),
        ("DRAG", "1qRB"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run(qubits=qubits)