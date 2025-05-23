# %%
"""
Two-Qubit Readout Confusion Matrix Measurement

This sequence measures the readout error when simultaneously measuring the state of two qubits. The process involves:

1. Preparing the two qubits in all possible combinations of computational basis states (|00⟩, |01⟩, |10⟩, |11⟩)
2. Performing simultaneous readout on both qubits
3. Calculating the confusion matrix based on the measurement results

For each prepared state, we measure:
1. The readout result of the first qubit
2. The readout result of the second qubit

The measurement process involves:
1. Initializing both qubits to the ground state
2. Applying single-qubit gates to prepare the desired input state
3. Performing simultaneous readout on both qubits
4. Repeating the process multiple times to gather statistics

The outcome of this measurement will be used to:
1. Quantify the readout fidelity for two-qubit states
2. Identify and characterize crosstalk effects in the readout process
3. Provide data for readout error mitigation in two-qubit experiments

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits

Outcomes:
- 4x4 confusion matrix representing the probabilities of measuring each two-qubit state given a prepared input state
- Readout fidelity metrics for simultaneous two-qubit measurement
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_quintets: List[List[str]] = [["q2","q0","q1","q3",'q4']]# list of lists of thwe qubits making up the GHZ state
    num_shots: int = 20000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "thermal"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None


node = QualibrationNode(
    name="41c_GHZ_Zbasi_5Q", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %%

# %%

class qubit_quintet:
    def __init__(self, qubit_center, qubit_A, qubit_B, qubit_C, qubit_D):
        self.qubit_center = qubit_center
        self.qubit_A = qubit_A
        self.qubit_B = qubit_B
        self.qubit_C = qubit_C
        self.qubit_D = qubit_D

        for qp in machine.qubit_pairs:
            if machine.qubit_pairs[qp].qubit_control in [qubit_A, qubit_center] and machine.qubit_pairs[qp].qubit_target in [qubit_A, qubit_center]:
                self.qubit_pair_A = machine.qubit_pairs[qp]
            if machine.qubit_pairs[qp].qubit_control in [qubit_B, qubit_center] and machine.qubit_pairs[qp].qubit_target in [qubit_B, qubit_center]:
                self.qubit_pair_B = machine.qubit_pairs[qp]
            if machine.qubit_pairs[qp].qubit_control in [qubit_center, qubit_C] and machine.qubit_pairs[qp].qubit_target in [qubit_center, qubit_C]:
                self.qubit_pair_C = machine.qubit_pairs[qp]
            if machine.qubit_pairs[qp].qubit_control in [qubit_center, qubit_D] and machine.qubit_pairs[qp].qubit_target in [qubit_center, qubit_D]:
                self.qubit_pair_D = machine.qubit_pairs[qp]
        self.name = f"{qubit_center.name}-{qubit_A.name}-{qubit_B.name}-{qubit_C.name}-{qubit_D.name}"

qubit_quintets = [[machine.qubits[qubit] for qubit in quintets] for quintets in node.parameters.qubit_quintets]
qubit_quintets = [qubit_quintet(qubit_quintets[0], qubit_quintets[1], qubit_quintets[2], qubit_quintets[3],qubit_quintets[4]) for qubit_quintets in qubit_quintets]
num_qubit_quintets = len(qubit_quintets)

####################
# Helper functions #
####################


# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as CPhase_Oscillations:
    n = declare(int)
    n_st = declare_stream()
    state_center = [declare(int) for _ in range(num_qubit_quintets)]
    state_A = [declare(int) for _ in range(num_qubit_quintets)]
    state_B = [declare(int) for _ in range(num_qubit_quintets)]
    state_C = [declare(int) for _ in range(num_qubit_quintets)]
    state_D = [declare(int) for _ in range(num_qubit_quintets)]
    state = [declare(int) for _ in range(num_qubit_quintets)]
    state_st_center = [declare_stream() for _ in range(num_qubit_quintets)]
    state_st_A = [declare_stream() for _ in range(num_qubit_quintets)]
    state_st_B = [declare_stream() for _ in range(num_qubit_quintets)]
    state_st_C = [declare_stream() for _ in range(num_qubit_quintets)]
    state_st_D = [declare_stream() for _ in range(num_qubit_quintets)]
    state_st = [declare_stream() for _ in range(num_qubit_quintets)]
    
    for i, qubit_quintet in enumerate(qubit_quintets):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qubit_quintet.qubit_A)
        align()
        
        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)         
            # reset
            if node.parameters.reset_type == "active":
                active_reset(qubit_quintet.qubit_A)
                active_reset(qubit_quintet.qubit_B)
                active_reset(qubit_quintet.qubit_C)

            else:
                wait(5*qubit_quintet.qubit_A.thermalization_time * u.ns)
            align()
            # Bell state AB
            # qubit_quintet.qubit_A.xy.play("y90")
            # qubit_quintet.qubit_B.xy.play("y90")
            # qubit_quintet.qubit_pair_AB.gates['Cz'].execute()
            # qubit_quintet.qubit_A.xy.play("-y90")
            
            # Bell state BC
            # qubit_quintet.qubit_C.xy.play("y90")
            # qubit_quintet.qubit_B.xy.play("y90")
            # qubit_quintet.qubit_pair_BC.gates['Cz'].execute()
            # qubit_quintet.qubit_C.xy.play("-y90")
            
            # GHZ q0q2
            qubit_quintet.qubit_center.xy.play("y90")
            qubit_quintet.qubit_A.xy.play("y90")
            qubit_quintet.qubit_pair_A.gates['Cz'].execute()
            qubit_quintet.qubit_A.xy.play("-y90")
            align()
            # GHZ q1q2
            qubit_quintet.qubit_B.xy.play("y90")
            qubit_quintet.qubit_pair_B.gates['Cz'].execute()
            qubit_quintet.qubit_B.xy.play("-y90")

            # GHZ q2q3
            qubit_quintet.qubit_C.xy.play("y90")
            qubit_quintet.qubit_pair_C.gates['Cz'].execute()
            qubit_quintet.qubit_C.xy.play("-y90")
            
            # GHZ q2q4
            qubit_quintet.qubit_D.xy.play("y90")
            qubit_quintet.qubit_pair_D.gates['Cz'].execute()
            qubit_quintet.qubit_D.xy.play("-y90")
            
            align()

            readout_state(qubit_quintet.qubit_center, state_center[i])
            readout_state(qubit_quintet.qubit_A, state_A[i])
            readout_state(qubit_quintet.qubit_B, state_B[i])
            readout_state(qubit_quintet.qubit_C, state_C[i])
            readout_state(qubit_quintet.qubit_D, state_D[i])


            assign(state[i], state_center[i]*16 +state_A[i]*8 + state_B[i]*4 + state_C[i]*2+state_D[i])
            save(state_center[i], state_st_center[i])
            save(state_A[i], state_st_A[i])
            save(state_B[i], state_st_B[i])
            save(state_C[i], state_st_C[i])
            save(state_D[i], state_st_D[i])
            save(state[i], state_st[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_quintets):
            state_st_center[i].buffer(n_shots).save(f"state_center{i + 1}")
            state_st_A[i].buffer(n_shots).save(f"state_A{i + 1}")
            state_st_B[i].buffer(n_shots).save(f"state_B{i + 1}")
            state_st_C[i].buffer(n_shots).save(f"state_C{i + 1}")
            state_st_D[i].buffer(n_shots).save(f"state_D{i + 1}")
            state_st[i].buffer(n_shots).save(f"state{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(CPhase_Oscillations)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_shots, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubit_quintets, {"N": np.linspace(1, n_shots, n_shots)})
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
        
    node.results = {"ds": ds}
    
# %%
if not node.parameters.simulate:
    states = [i for i in range(32)]
    results = {}
    corrected_results = {}
    for qubit_quintet in qubit_quintets:
        results[qubit_quintet.name] = []
        for state in states:
            results[qubit_quintet.name].append((ds.sel(qubit = qubit_quintet.name).state == state).sum().values)
        results[qubit_quintet.name] = np.array(results[qubit_quintet.name])/node.parameters.num_shots
        conf_mat = np.array([[1]])
        for q in [qubit_quintet.qubit_center,qubit_quintet.qubit_A, qubit_quintet.qubit_B, qubit_quintet.qubit_C,qubit_quintet.qubit_D]:
            conf_mat = np.kron(conf_mat, q.resonator.confusion_matrix)
        # conf_mat = qp.confusion
        corrected_results[qubit_quintet.name] = np.linalg.inv(conf_mat) @ results[qubit_quintet.name]
        # corrected_results[qubit_quintet.name] = results[qubit_quintet.name]

        corrected_results[qubit_quintet.name] = np.where(corrected_results[qubit_quintet.name] < 0, 0, corrected_results[qubit_quintet.name])
        corrected_results[qubit_quintet.name] = corrected_results[qubit_quintet.name]/np.sum(corrected_results[qubit_quintet.name])
        print(f"{qubit_quintet.name}: {corrected_results[qubit_quintet.name]}")



# %%
if not node.parameters.simulate:
    if num_qubit_quintets == 1:
        f,axs = plt.subplots(1,figsize=(20,3))
    else:
        f,axs = plt.subplots(num_qubit_quintets,1,figsize=(6,3*num_qubit_quintets))
    
    for i, qubit_quintet in enumerate(qubit_quintets):
        if num_qubit_quintets == 1:
            ax = axs
        else:
            ax = axs[i]

        #ax.bar(['0000', '0001', '0010', '0011', '0100', '0101', '0110', '0111', '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111'], corrected_results[qubit_quintet.name], color='skyblue', edgecolor='navy')
        ax.bar(['00000', '00001', '00010', '00011', '00100', '00101', '00110', '00111', '01000', '01001', '01010', '01011', '01100', '01101', '01110', '01111', '10000', '10001', '10010', '10011', '10100', '10101', '10110', '10111', '11000', '11001', '11010', '11011', '11100', '11101', '11110', '11111'], corrected_results[qubit_quintet.name], color='skyblue', edgecolor='navy')
        #ax.bar(['0000', '0001', '010', '011', '100', '101', '110', '111'], corrected_results[qubit_quintet.name], color='skyblue', edgecolor='navy')
        ax.set_ylim(0, 0.5)
        for i, v in enumerate(corrected_results[qubit_quintet.name]):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        ax.set_ylabel('Probability')
        ax.set_xlabel('State')
        ax.set_title(qubit_quintet.name)
    plt.show()
    node.results["figure"] = f
    
    # grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    # grid = QubitPairGrid(grid_names, qubit_pair_names)
    # for ax, qubit_pair in grid_iter(grid):
    #     print(qubit_pair['qubit'])
    #     corrected_res = corrected_results[qubit_pair['qubit']]
    #     ax.bar(['00', '01', '10', '11'], corrected_res, color='skyblue', edgecolor='navy')
    #     ax.set_ylim(0, 1)
    #     for i, v in enumerate(corrected_res):
    #         ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    #     ax.set_ylabel('Probability')
    #     ax.set_xlabel('State')
    #     ax.set_title(qubit_pair['qubit'])
    # plt.show()
    # node.results["figure"] = grid.fig
# %%

# %% {Update_state}

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
        
# %%
