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
    name="retuning_graph_debug1",
    parameters=Parameters(qubits=qubits),
    nodes={
        "close_other_qms": library.nodes["00_Close_other_QMs"].copy(
            name="close_other_qms",
        ),
        "readout_freq_opt": library.nodes["07a_Readout_Frequency_Optimization"].copy(
            name="readout_freq_opt",
        ),
    },
    connectivity=[
        ("close_other_qms", "readout_freq_opt"),
        
    ],
    orchestrator=BasicOrchestrator(skip_failed=False),
)
g.run(qubits=qubits)