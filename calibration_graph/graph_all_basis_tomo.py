from typing import List
from qualibrate.orchestration.basic_orchestrator import BasicOrchestrator
from qualibrate.parameters import GraphParameters
from qualibrate.qualibration_graph import QualibrationGraph
from qualibrate.qualibration_library import QualibrationLibrary

library = QualibrationLibrary.get_active_library()


class Parameters(GraphParameters):
    qubits: List[str] = None


qubits = ["q3"]
multiplexed = False
num_avgs = 10000


g = QualibrationGraph(
    name="retuning_graph_1q",
    parameters=Parameters(qubits=qubits),
    nodes={
        "close_other_qms": library.nodes["00_Close_other_QMs"].copy(
            name="close_other_qms",
        ),
        "IQ_blob_active": library.nodes["07b_IQ_Blobs1"].copy(
            qubits=qubits,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="IQ_blob_active",
            reset_type_thermal_or_active="active",
            readout_scale=1.0,
        ),
        "State_tomography_0": library.nodes["AA_Single_Qubit_State_Tomography"].copy(
            qubits=qubits,
            operation = None,
            num_averages = num_avgs,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="State_tomography_0",
            reset_type_thermal_or_active="active",
        ),
        "State_tomography_1": library.nodes["AA_Single_Qubit_State_Tomography"].copy(
            qubits=qubits,
            operation = "y180",
            num_averages=num_avgs,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="State_tomography_1",
            reset_type_thermal_or_active="active",
        ),
        "State_tomography_p": library.nodes["AA_Single_Qubit_State_Tomography"].copy(
            qubits=qubits,
            operation = "y90",
            num_averages=num_avgs,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="State_tomography_p",
            reset_type_thermal_or_active="active",
        ),
        "State_tomography_m": library.nodes["AA_Single_Qubit_State_Tomography"].copy(
            qubits=qubits,
            operation = "-y90",
            num_averages=num_avgs,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="State_tomography_m",
            reset_type_thermal_or_active="active",
        ),
        "State_tomography_ip": library.nodes["AA_Single_Qubit_State_Tomography"].copy(
            qubits=qubits,
            operation = "x90",
            num_averages=num_avgs,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="State_tomography_ip",
            reset_type_thermal_or_active="active",
        ),
        "State_tomography_im": library.nodes["AA_Single_Qubit_State_Tomography"].copy(
            qubits=qubits,
            operation = "-x90",
            num_averages=num_avgs,
            flux_point_joint_or_independent="joint",
            multiplexed=multiplexed,
            name="State_tomography_im",
            reset_type_thermal_or_active="active",
        ),
    },
    connectivity=[
        ("close_other_qms","IQ_blob_active"),
        ("IQ_blob_active","State_tomography_0"),
        ("State_tomography_0","State_tomography_1"),
        ("State_tomography_1","State_tomography_p"),
        ("State_tomography_p","State_tomography_m"),
        ("State_tomography_m","State_tomography_ip"),
        ("State_tomography_ip","State_tomography_im"),
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run(qubits=qubits)