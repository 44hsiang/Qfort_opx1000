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
    name="retuning_graph_two_qubit_cal",
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
        "qubit_spec_ef": library.nodes["11a_Qubit_Spectroscopy_E_to_F"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="qubit_spec_ef"
        ),
        "ro_fre_opt_vs_amp_gef": library.nodes["11b_Readout_Frequency_Optimization_G_E_F_vs_amp"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="ro_fre_opt_vs_amp_gef"
        ),
        "power_rabi_ef": library.nodes["11c_Power_Rabi_E_to_F"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="power_rabi_ef",
        ),
        "ro_fre_opt_gef": library.nodes["11d_Readout_Frequency_Optimization_G_E_F"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="ro_fre_opt_gef",
        ),
        "IQ_blob_thermal_gef": library.nodes["11e_IQ_Blobs_G_E_F"].copy(
            qubits = qubits,
            flux_point_joint_or_independent="joint",
            name="IQ_blob_thermal",
        )
    },
    connectivity=[
        ("close_other_qms", "IQ_blob_thermal"),
        ("IQ_blob_thermal", "IQ_blob_active"),
        ("IQ_blob_active", "qubit_spec_ef"),
        ("qubit_spec_ef", "ro_fre_opt_vs_amp_gef"),
        ("ro_fre_opt_vs_amp_gef", "power_rabi_ef"),
        ("power_rabi_ef", "ro_fre_opt_gef"),
        ("ro_fre_opt_gef", "IQ_blob_thermal_gef")
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run(qubits=qubits)