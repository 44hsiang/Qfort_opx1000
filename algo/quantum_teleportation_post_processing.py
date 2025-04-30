# %%
'''
# Quantum teleportation exp

## Experiment theoretic workflow
1. Initialization
    - prepare Alice and Bob in maximal entanglement state $|\Phi^+\rangle$
    - prepare message qubit in any state
2. Teleportation
    - Perform CNOT gate on message qubit and Alice pair, with message 
      qubit as control qubit and Alice as target qubit.
    - Perform Hadamard gate on message qubit
    - Measure message qubit and Alice
3. Post-processing
    - Use classical information to update Bob for correct solution
    - Due to our poor readout fidelity, confusion matrix correction should 
      be engaged.
    - Detail workflow will be produce later
## Implementation detail
Using 3 qubits, maintained by @qubit_set object
- single qubits
    - qubit_mess: quantum message to pass
    - qubit_A: Alice
    - qubit_B: Bob
- qubit pairs, qubit_mess must be control qubit
    - qubit_mess, qubit_A
    - qubit_A, qubit_B

To form a feasible pair, condition below must be satisfied:
- @qubit_mess must be control qubit
- one of @qubit_A or @qubit_B must be q2
Hence, @qubit_A must be q2, @qubit_mess can be q3 or q4, 
and @qubit_b can be q0 or q1.
Teleportation experiment data start from #340
'''

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
import math
import xarray as xr
import sys

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_sets: List[str] = ["q3", "q2", "q1"] # list of lists of thwe qubits making up the GHZ state
    # messages: List[tuple[complex, complex]] = [(1+0j, 0+0j)] # (alpha, beta)
    messages: List[int] = [5]
    num_shots: int = 20000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "thermal"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None

node = QualibrationNode(
    name="quantum_teleportation_post_processing", parameters=Parameters()
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

# %% {Define qubit set}
class qubit_set:
    def __init__(self, qubit_mess, qubit_A, qubit_B):
        self.qubit_mess = machine.qubits[qubit_mess]
        self.qubit_A = machine.qubits[qubit_A]
        self.qubit_B = machine.qubits[qubit_B]
        self.qubit_pair_mA = None
        self.qubit_pair_AB = None
        
        for qp in machine.qubit_pairs:
            if machine.qubit_pairs[qp].qubit_control == self.qubit_mess and machine.qubit_pairs[qp].qubit_target == self.qubit_A:
                self.qubit_pair_mA = machine.qubit_pairs[qp]
            if machine.qubit_pairs[qp].qubit_control in [self.qubit_A, self.qubit_B] and machine.qubit_pairs[qp].qubit_target in [self.qubit_A, self.qubit_B]:
                self.qubit_pair_AB = machine.qubit_pairs[qp]

        assert self.qubit_pair_mA and self.qubit_pair_AB
        self.name = f"{qubit_mess}-{qubit_A}-{qubit_B}"

qset = qubit_set(*[node.parameters.qubit_sets[i] for i in range(3)])
messages = node.parameters.messages
####################
# Helper functions #
####################

# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as quantum_teleportation_post_processing:
    n = declare(int)
    n_st = declare_stream()
    '''
    state_mess = [declare(int) for _ in range(num_qubit_sets)]
    state_A = [declare(int) for _ in range(num_qubit_sets)]
    state_st_mess = [declare_stream() for _ in range(num_qubit_sets)]
    state_st_A = [declare_stream() for _ in range(num_qubit_sets)]
    '''
    state_mess = declare(int)
    state_A = declare(int)
    state_B = declare(int)
    state = declare(int)
    state_st = [declare_stream() for _ in range(len(messages))]
    tomo_axis = declare(int)
    
    for i, message in enumerate(messages):
        # Bring the active qubits to the minimum frequency point
        # ?
        machine.set_all_fluxes(flux_point, qset.qubit_A)
        align()
        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)
            with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):
                # reset
                if node.parameters.reset_type == 'active':
                    active_reset(qset.qubit_mess)
                    active_reset(qset.qubit_A)
                    active_reset(qset.qubit_B)
                else:
                    t1 = max(qset.qubit_mess.thermalization_time,
                             qset.qubit_A.thermalization_time,
                             qset.qubit_B.thermalization_time)
                    wait(5 * t1 * u.ns)
                align()

                # Initialization
                # Prepare Bell state on Alice and Bob
                # The process produce $|\Phi^+\rangle$

                qset.qubit_A.xy.play("y90")
                qset.qubit_B.xy.play("y90")
                qset.qubit_pair_AB.gates['Cz'].execute()
                qset.qubit_A.xy.play("-y90")

                assert message in [0, 1, 2, 3, 4, 5]

                if message == 0:
                    pass
                elif message == 1:
                    qset.qubit_mess.xy.play("y180")
                elif message == 2:
                    qset.qubit_mess.xy.play("y90")
                elif message == 3:
                    qset.qubit_mess.xy.play("-y90")
                elif message == 4:
                    qset.qubit_mess.xy.play("-x90")
                elif message == 5:
                    qset.qubit_mess.xy.play("x90")

                align()
                # quantum teleportation
                # CNOT gate: control: qubit_mess, target: qubit_A

                qset.qubit_A.xy.play("y90")
                qset.qubit_pair_mA.gates['Cz'].execute()
                qset.qubit_A.xy.play("-y90")
                align()
                # Hadamard gate on message qubit
                qset.qubit_mess.xy.play("y90")
                align()


                with if_(tomo_axis == 0):   # x-axis
                    qset.qubit_B.xy.play("y90")
                with elif_(tomo_axis == 1): # y-axis
                    qset.qubit_B.xy.play("x90")
                align()

                readout_state(qset.qubit_mess, state_mess)
                readout_state(qset.qubit_A, state_A)
                readout_state(qset.qubit_B, state_B)
                assign(state, state_mess * 4 + state_A * 2 + state_B)
                save(state, state_st[i])
            align()
    with stream_processing():
        n_st.save("n")
        for i in range(len(messages)):
            state_st[i].buffer(3).buffer(n_shots).save(f"state{i + 1}")
    
# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, quantum_teleportation_post_processing, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(quantum_teleportation_post_processing)

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
        # 假設你有 3 個 tomographic axis，2000 次 shots
        data = job.result_handles.get('state1').fetch_all()
        coords = {
            "tomo_axis_bob": [0, 1, 2],   # 代表 X, Y, Z
            "n": np.arange(n_shots)
        }

        ds = xr.Dataset(
            data_vars={
                "state": (["n", "tomo_axis_bob"], data)
            },
            coords=coords
        )
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

    
    node.results = {"ds": ds}

# %% {Data_fetching_and_dataset_creation}
# Assume only one element in @qubit_sets
def data_correction(data, qset, use_conf_mat=True):
    conf_mat = np.array([[1]])
    conf_mat = np.kron(conf_mat, qset.qubit_mess.resonator.confusion_matrix)
    conf_mat = np.kron(conf_mat, qset.qubit_A.resonator.confusion_matrix)
    conf_mat = np.kron(conf_mat, qset.qubit_B.resonator.confusion_matrix)
    
    state_prob = [np.array([0 for j in range(8)]) for i in range(3)] # excited state count
    for dim in range(3):
        for state in data.sel(tomo_axis_bob=dim):
            state_prob[dim][state] += 1
        state_prob[dim] = state_prob[dim] / n_shots
        if use_conf_mat:
            print('Confusion matrix used')
            state_prob[dim] = np.linalg.inv(conf_mat) @ state_prob[dim]
            state_prob[dim] = np.where(state_prob[dim] < 0, 0, state_prob[dim])  # 將負值裁為0
            state_prob[dim] /= np.sum(state_prob[dim])  # normalize
    return state_prob

#%%
if not node.parameters.simulate:
    data = ds['state']
    
    # Calculate confusion matrix
    state_prob = data_correction(data, qset)
    '''
    eprob_group = np.zeros((4, 3))
    for result in range(4):
        for dim in range(3):
            gprob, eprob = state_prob[dim][2 * result], state_prob[dim][2 * result + 1]
            # print('(dim, result, gprob, eprob):', (dim, result, gprob, eprob))
            assert not (gprob == 0 and eprob == 0)
            eprob_group[result][dim] = eprob / (gprob + eprob)
    print('eprob_group')
    print(eprob_group)
    for result in range(4):
        rx = 1 - 2 * eprob_group[result][0]
        ry = 1 - 2 * eprob_group[result][1]
        rz = 1 - 2 * eprob_group[result][2]
        norm = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        rx /= norm
        ry /= norm
        rz /= norm

        if result == 0:
            rx, ry = -rx, -ry   # Z-gate
        if result == 1:
            rx, rz = -rx, -rz   # Z-gate than X-gate
        if result == 3:
            ry, rz = -ry, -rz   # X-gate
        norm = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        rxy_len = math.sqrt(rx ** 2 + ry ** 2)
        print(f'(result, rxy_len, rz) ({result}, {rxy_len:.2f}, {rz:.2f})')
    '''
    # %% {plot raw data}
    f,axs = plt.subplots(3,1,figsize=(6,3*3))
    for dim in range(3):
        ax = axs[dim]
        ax.bar(['000', '001', '010', '011', '100', '101', '110', '111'], state_prob[dim], color='skyblue', edgecolor='navy')
        ax.set_ylim(0, 0.5)
        for j, v in enumerate(state_prob[dim]):
            ax.text(j, v, f'{v:.2f}', ha='center', va='bottom')
        ax.set_ylabel('Probability')
        ax.set_xlabel('State')
        if dim == 0:
            name = qset.name + 'x'
        elif dim == 1:
            name = qset.name + 'y'
        elif dim == 2:
            name = qset.name + 'z' 
        ax.set_title(name)
    f.tight_layout()
    plt.show()
    node.results["figure"] = f

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
