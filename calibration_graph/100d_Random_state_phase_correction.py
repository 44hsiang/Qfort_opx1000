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
from qualang_tools.loops import from_array
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from qiskit.result import CorrelatedReadoutMitigator
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q1']
    num_runs: int = 40000
    min_wait_time_in_ns: int = 16
    max_time_in_ns: int = 200
    wait_time_step_in_ns: int = 4
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    desired_state: Optional[List[float]] = [np.pi/2,0] #theta,phi

#theta,phi = random_bloch_state_uniform()

node = QualibrationNode(name="100d_Random_state_phase_correction", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

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
times_cycles = np.arange(node.parameters.min_wait_time_in_ns//4, node.parameters.max_time_in_ns // 4,node.parameters.wait_time_step_in_ns//4)

def QuantumMemory_program(qubit,theta=theta,phi=phi):
    with program() as QuantumMemory:
        t = declare(int)
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = [declare(int) for _ in range(1)]
        state_st = [declare_stream() for _ in range(1)]
        tomo_axis = declare(int)

        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)
            with for_(*from_array(t,times_cycles)):
                with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):

                    if reset_type == "active":
                        active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                    elif reset_type == "thermal":
                        qubit.wait(4 * qubit.thermalization_time * u.ns)
                    else:
                        raise ValueError(f"Unrecognized reset type {reset_type}.")

                    qubit.align()
                    qubit.xy.play("y180",amplitude_scale = theta/np.pi)
                    #qubit.xy.frame_rotation_2pi((phi-qubit.extras["phi_offset"])/np.pi/2-0.5)
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
            state_st[0].buffer(3).buffer(len(times_cycles)).average().save("state1")
        return QuantumMemory



# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, QuantumMemory_program(qubit), simulation_config)
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
    for qubit in qubits:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(QuantumMemory_program(qubit))
            for i in range(num_qubits):
                results = fetching_tool(job, ["n"], mode="live")
                while results.is_processing():
                    n = results.fetch_all()[0]
                    progress_counter(n, n_runs, start_time=results.start_time)
            job_.append(job)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        for i in range(num_qubits):
            ds_ = fetch_results_as_xarray(job_[i].result_handles, [qubits[i]], { "axis": ['x','y','z'],"timedelay": times_cycles*4})
            ds = xr.concat([ds, ds_], dim="qubit") if i != 0 else ds_
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    

    # %% {Data_analysis}
    node.results = {"ds": ds, "figs": {}, "results": {}}
    ds["Bloch_phi"] = np.rad2deg(np.arctan2(1-2*ds["state"].sel(axis='y'),1-2*ds["state"].sel(axis='x')))
    # TODO: try to use slope to calculate the detuning
    def linear_func(x, a, b):
        return a * x + b

    fit_results = {}
    for q in qubits:
        fit_results[q.name]={}
        params, covariance = curve_fit(linear_func, ds.timedelay, ds.sel(qubit=q.name).Bloch_phi)
        fit_results[q.name]["slope"] = params[0]
        fit_results[q.name]["intercept"] = params[1]
    
    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds.sel(qubit=qubit['qubit']).Bloch_phi.plot(ax=ax)
        ax.plot(ds.timedelay, linear_func(ds.timedelay, fit_results[qubit['qubit']]['slope'],fit_results[qubit['qubit']]['intercept']), 'r--')
        ax.set_ylabel('Bloch_phi(degrees)')
        ax.set_xlabel('time_delay(ns)')

    node.results["figure_phase"] = grid.fig
    grid = QubitGrid(ds, [q.grid_location for q in qubits],is_3d=True)
    for ax, qubit in grid_iter(grid):
        bloch = Bloch(axes=ax,font_size=12)
        bloch.add_vectors([1-2*ds["state"].sel(axis='x',qubit=qubit['qubit']).values[0],1-2*ds["state"].sel(axis='y',qubit=qubit['qubit']).values[0],1-2*ds["state"].sel(axis='z',qubit=qubit['qubit']).values[0]])
        bloch.add_vectors([1-2*ds["state"].sel(axis='x',qubit=qubit['qubit']).values[-1],1-2*ds["state"].sel(axis='y',qubit=qubit['qubit']).values[-1],1-2*ds["state"].sel(axis='z',qubit=qubit['qubit']).values[-1]])
        bloch.add_vectors([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        bloch.vector_color = ['r','b','g']
        bloch.vector_labels = ['raw','mitigated','ideal']
        bloch.render(title=qubit['qubit'])
    node.results["figure_Bloch_vector"] = grid.fig
    # %% {Update_state}
    if not node.parameters.simulate:
        for qubit in qubits:
            qubit.extras["phi_offset"] = fit_results[qubit.name]["intercept"]/180*np.pi
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
