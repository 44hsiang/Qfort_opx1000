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
t2_max = 50000
t2_detuning = 0.2

# ------------------------------------------------------------------
# Dynamically build 100 (T1, Ramsey) measurement pairs
nodes = {
    "close_other_qms": library.nodes["00_Close_other_QMs"].copy(
        name="close_other_qms",
    ),
}

for i in range(1, 321):  # 1 → 100  (inclusive)
    nodes[f"T1_{i}"] = library.nodes["05_T1"].copy(
        qubits=qubits,
        name=f"T1_{i}",
    )
    nodes[f"ramsey_long{i}"] = library.nodes["06_Ramsey"].copy(
        qubits=qubits,
        flux_point_joint_or_independent="joint",
        name=f"ramsey_long{i}",
        frequency_detuning_in_mhz=t2_detuning,
        min_wait_time_in_ns=16,
        max_wait_time_in_ns=t2_max,
        num_time_points=500,
    )

# Build a linear connectivity chain:
# close_other_qms → T1_1 → ramsey_long1 → T1_2 → … → ramsey_long320
connectivity = [("close_other_qms", "T1_1")]
for i in range(1, 321):
    connectivity.append((f"T1_{i}", f"ramsey_long{i}"))
    if i < 321:
        connectivity.append((f"ramsey_long{i}", f"T1_{i+1}"))
# ------------------------------------------------------------------

# Construct the graph with the generated nodes/connectivity
g = QualibrationGraph(
    name="retuning_graph_t2_repeat",
    parameters=Parameters(qubits=qubits),
    nodes=nodes,
    connectivity=connectivity,
    orchestrator=BasicOrchestrator(skip_failed=False),
)

g.run(qubits=qubits)