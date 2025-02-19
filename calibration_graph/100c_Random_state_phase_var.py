"""
    Quantum Memory
"""
# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset,readout_state
from quam_libs.QI_function import *
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.analysis.discriminator import two_state_discriminator
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qiskit.result import CorrelatedReadoutMitigator
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_runs: int = 10000
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    tiral: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    desired_state: Optional[List[float]] = [np.pi/2,0] #theta,phi

#theta,phi = random_bloch_state_uniform()

node = QualibrationNode(name="100a_Random_state_phase_var", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
path = "/Users/4hsiang/Desktop/Jack/python_project/instrument_control/opx1000/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration/quam_state"
machine = QuAM.load(path)

#%%
# delete the thread when using active reset
if node.parameters.reset_type_thermal_or_active == "active":
    for i in machine.active_qubit_names:
        del machine.qubits[i].xy.thread
        del machine.qubits[i].resonator.thread

#%%
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()


# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

# %% {QUA_program}
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
#theta,phi = random_bloch_state_uniform()
theta,phi = node.parameters.desired_state[0],node.parameters.desired_state[1]
t=4
def QuantumMemory_program(qubit,theta=theta,phi=phi):
    with program() as QuantumMemory:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = [declare(int) for _ in range(1)]
        state_st = [declare_stream() for _ in range(1)]
        tomo_axis = declare(int)

        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)
            with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):

                if reset_type == "active":
                    active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                elif reset_type == "thermal":
                    qubit.wait(4 * qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")

                qubit.align()
                qubit.xy.play("y180",amplitude_scale = theta/np.pi)
                qubit.xy.frame_rotation_2pi(phi/np.pi/2-0.5)
                wait(t)          
                align()

                #tomography pulses
                with if_(tomo_axis == 0):
                    qubit.xy.play("y90")
                with elif_(tomo_axis == 1):
                    qubit.xy.play("x90")
                align()

                readout_state(qubit, state[0])
                save(state[0], state_st[0])
        
        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

        with stream_processing():
            n_st.save("n")
            state_st[0].buffer(3).save_all("state1")
        return QuantumMemory



# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, QuantumMemory, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    
elif node.parameters.load_data_id is None:
    job_ = []
    for i in range(node.parameters.tiral):
        job_.append([])
        for qubit in qubits:
            with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                job = qm.execute(QuantumMemory_program(qubit))
                results = fetching_tool(job, ["n"], mode="live")
                while results.is_processing():
                    n = results.fetch_all()[0]
                    progress_counter(n, n_runs, start_time=results.start_time)
                job_[i].append(job)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        #ds = fetch_results_as_xarray(job.result_handles, qubits, {"N": np.linspace(1, n_runs, n_runs)})
        for i in range(node.parameters.tiral):
            for j in range(num_qubits):
                ds_ = fetch_results_as_xarray(job_[i][j].result_handles, [qubits[j]], {"N": np.linspace(1, n_runs, n_runs)})
                ds_qubit = xr.concat([ds_qubit, ds_], dim="qubit") if j != 0 else ds_
            ds_ = ds_qubit
            ds = xr.concat([ds, ds_], dim="tiral") if i != 0 else ds_qubit
        extract_state = ds.state.values['value']
        if node.parameters.tiral == 1:
            ds = ds.assign_coords(axis=("axis", ['x', 'y', 'z']))
            ds['state'] = (["qubit","N", "axis"], extract_state)
        else:
            ds = ds.assign_coords(axis=("axis", ['x', 'y', 'z']),trials = ("trials",np.arange(node.parameters.tiral)+1))
            ds['state'] = (["trials","qubit","N", "axis"], extract_state)

    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    
    # %% {Data_analysis}
    node.results = {"ds": ds, "figs": {}, "results": {}}
    ds['state_avg'] = ds['state'].mean(dim='N')
    ds["Bloch_vector"] = ((1-2*ds["state_avg"].sel(axis='x'))**2+(1-2*ds["state_avg"].sel(axis='y'))**2+(1-2*ds["state_avg"].sel(axis='z'))**2 )**0.5   
    ds["Bloch_phi"] = np.rad2deg(np.arctan2(1-2*ds["state_avg"].sel(axis='y'),1-2*ds["state_avg"].sel(axis='x')))
    for q in qubits:
        print(f"qubit {q.name}")
        print(f"phi mean = {ds.sel(qubit=q.name).Bloch_phi.mean().values}")
        print(f"phi std = {ds.sel(qubit=q.name).Bloch_phi.std().values}")
    # %% plot the data
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds.sel(qubit=qubit['qubit']).Bloch_phi.plot(ax=ax)
    node.results["figure_phase_var"] = grid.fig
    # TODO  different trials and label its vector and phi

    # %% {Update_state}
    if node.parameters.reset_type_thermal_or_active == "active":
        for i,j in zip(machine.active_qubit_names,"abcde"):
            machine.qubits[i].xy.thread = j
            machine.qubits[i].resonator.thread = j

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
    

# %%
