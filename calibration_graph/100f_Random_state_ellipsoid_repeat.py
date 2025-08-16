"""
    Quantum Memory
"""
# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset,readout_state
from quam_libs.QI_function import *
from quam_libs.fit_ellipsoid import *
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
from quam_libs.quantum_memory.NoiseAnalyze import *
from quam_libs.quantum_memory.marcos import *
from quam_libs.quantum_memory.EllipsoidTool import *

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit, minimize

from qiskit.result import CorrelatedReadoutMitigator
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q0'] # only support one qubit
    num_runs: int = 10000
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 10000
    timeout: int = 100
    load_data_id: Optional[int] = 4921
    multiplexed: bool = False
    number_of_points: int = 1
    repeats: int = 200

if Parameters().load_data_id is not None:
    node  = QualibrationNode(name=f"{Parameters().load_data_id}_analyze_again", parameters=Parameters())
else:
    node = QualibrationNode(name="100f_Random_state_ellipsoid_repeat", parameters=Parameters())
load_data_id = node.parameters.load_data_id

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

#%% delete the thread when using active reset
if node.parameters.load_data_id is None and node.parameters.reset_type_thermal_or_active == "active":
#if node.parameters.reset_type_thermal_or_active == "active":
    for i in machine.active_qubit_names:
        del machine.qubits[i].xy.core
        del machine.qubits[i].resonator.core

#%% Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if load_data_id is None:
    qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

theta_range = np.arange(0,np.pi,1e-4)
phi_range = np.arange(0,2*np.pi,1e-4)
def theta_phi_list(n_points):
    theta_list,phi_list = [],[]
    for i in range(n_points):
        theta,phi = np.random.choice(theta_range),np.random.choice(phi_range)
        theta_list.append(theta)
        phi_list.append(phi)
    return theta_list,phi_list
# %% {QUA_program}
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
n_points = node.parameters.number_of_points
repeats = node.parameters.repeats


theta_list = np.array([theta_phi_list(n_points)[0] for _ in range(repeats)])
phi_list = [theta_phi_list(n_points)[1] for _ in range(repeats)]
t=4
def QuantumMemory_program(qubit,repeats,theta,phi):
    with program() as QuantumMemory:
        ii = declare(int)
        #theta = declare(fixed, value = theta_list[repeats])
        #phi = declare(fixed, value = phi_list[repeats])
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = [declare(int) for _ in range(1)]
        state_st = [declare_stream() for _ in range(1)]
        tomo_axis = declare(int)

        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)
            with for_(ii, 0, ii < n_points, ii + 1):
                with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):

                    if reset_type == "active":
                        active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                    elif reset_type == "thermal":
                        qubit.wait(4 * qubit.thermalization_time * u.ns)
                    else:
                        raise ValueError(f"Unrecognized reset type {reset_type}.")

                    qubit.align()
                    qubit.xy.play("y180",amplitude_scale = theta/np.pi)
                    qubit.xy.frame_rotation_2pi((phi-qubit.extras["phi_offset"])/np.pi/2-0.5)
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
            state_st[0].buffer(3).buffer(n_points).average().save("state1")
        return QuantumMemory

# %% {Simulate_or_execute}

if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, QuantumMemory_program(qubits[0],0,theta=theta_list[0][0],phi=phi_list[0][0]), simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples,plot=True,save_path="./")

    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    
elif load_data_id is None:
    job_ = []
    for repeat in range(node.parameters.repeats):
        job_.append([])
        for qubit in qubits:
            with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                job = qm.execute(QuantumMemory_program(qubit,repeat,theta=theta_list[repeat][0],phi=phi_list[repeat][0]))
                results = fetching_tool(job, ["n"], mode="live")
                while results.is_processing():
                    n = results.fetch_all()[0]
                    progress_counter(n, n_runs, start_time=results.start_time)
                job_[repeat].append(job)
                print(f"repeat {repeat} qubit {qubit.name} is done")

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if load_data_id is None:
        import time
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        for i in range(node.parameters.repeats):
            for j in range(num_qubits):
                ds_ = fetch_results_as_xarray(job_[i][j].result_handles, [qubits[j]], { "axis": ['x','y','z'],"n_points": np.arange(n_points)})
                ds_qubit = xr.concat([ds, ds_], dim="qubit") if j != 0 else ds_
            ds_ = ds_qubit
            ds = xr.concat([ds, ds_], dim="repeat") if i != 0 else ds_qubit
        reshaped_ds = ds.stack(new_n_points=('repeat', 'n_points')).transpose('qubit', 'new_n_points', 'axis')
        reshaped_ds = reshaped_ds.assign_coords(new_n_points=np.arange(repeats*n_points))
        ds = reshaped_ds.rename({'new_n_points': 'n_points'})    
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # %% 
    #%% add the theta and phi to the dataset
    if load_data_id is None:
        theta_list = np.array(theta_list).flatten()
        phi_list = np.array(phi_list).flatten()
        ds = ds.assign_coords(theta = theta_list,phi =phi_list)    
        ds["Bloch_vector_x"] = 1-2*ds["state"].sel(axis='x')
        ds["Bloch_vector_y"] = 1-2*ds["state"].sel(axis='y')
        ds["Bloch_vector_z"] = 1-2*ds["state"].sel(axis='z')
        ds["Bloch_phi"] = np.arctan2(ds.Bloch_vector_y,ds.Bloch_vector_x)
        ds["Bloch_phi"] = ds.Bloch_phi.where(ds.Bloch_phi>0,ds.Bloch_phi+2*np.pi)
        ds["Bloch_theta"] = np.arccos(ds.Bloch_vector_z/np.sqrt(ds.Bloch_vector_x**2+ds.Bloch_vector_y**2+ds.Bloch_vector_z**2))
    else:
        pass

    # %% useful functions

    # %% {Data_analysis}
    node.results = {"ds": ds, "figs": {}, "results": {}}
    
    data_xyz = np.array([[ds.Bloch_vector_x.values[0][i], ds.Bloch_vector_y.values[0][i], ds.Bloch_vector_z.values[0][i]] for i in range(len(ds.n_points.values))])
    data_angle = np.array([[ds.theta.values[i], ds.phi.values[i]] for i in range(len(ds.n_points.values))])
    data_dm = np.array([bloch_vector_to_density_matrix(data_xyz[i]) for i in range(len(data_xyz))])
    noise_analyzer = NoiseAnalyze(data_xyz,data_angle)
    corrected_dm = noise_analyzer.corrected_dm
    corrected_bloch = noise_analyzer.corrected_bloch
    # first fit ellipsoid
    center, axes, R, volume, param = noise_analyzer.ellipsoid_fit()
    # compare the fitted ellipsoid with quadric form and find the best fit.
    axes, R = find_best_fit(center, axes, R, param)

    qm_analyze = QuantumMemory(axes,center,R)
    choi_state = qm_analyze.choi_state()

    checker = Checker(choi_state)
    choi, count = checker.choi_checker(index=[1], repeat=100, print_reason=False)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    raw_plot = EllipsoidTool(corrected_bloch).plot(ax=ax,axes=axes, center=center, R=R, title='raw data',show_points=True, show_unit_sphere=True)
    node.results["raw_plot"]= fig
    node.results['results']['raw data'] = {
        'axes': axes,
        'center': center,
        'R': R,
        'volume': volume,
        'param': param,
        'choi state': choi,
        'negativity': QuantumMemory.negativity(choi)*2,
        'memory robustness': QuantumMemory.memory_robustness(choi)
        }

    confusion_matrix = np.array(machine.qubits['q0'].resonator.confusion_matrix).T
    x = data_xyz[:,0]
    y = data_xyz[:,1]
    z = data_xyz[:,2]
    px, py, pz = (1-x)/2, (1-y)/2, (1-z)/2

    new_px_0 = np.array([MLE([1-px[j],px[j]],confusion_matrix)[0] for j in range(len(px))])
    new_py_0 = np.array([MLE([1-py[j],py[j]],confusion_matrix)[0] for j in range(len(py))])
    new_pz_0 = np.array([MLE([1-pz[j],pz[j]],confusion_matrix)[0] for j in range(len(pz))])
    
    MLE_data_xyz = np.array([2*new_px_0-1,2*new_py_0-1,2*new_pz_0-1]).T
    data_dm = np.array([bloch_vector_to_density_matrix(MLE_data_xyz[i]) for i in range(len(MLE_data_xyz))])
    noise_analyzer = NoiseAnalyze(MLE_data_xyz,data_angle)
    corrected_dm = noise_analyzer.corrected_dm
    corrected_bloch = noise_analyzer.corrected_bloch

    # first fit ellipsoid
    center, axes, R, volume, param = noise_analyzer.ellipsoid_fit()
    # compare the fitted ellipsoid with quadric form and find the best fit.
    axes, R = find_best_fit(center, axes, R, param)

    qm_analyze = QuantumMemory(axes,center,R)
    choi_state = qm_analyze.choi_state()

    checker = Checker(choi_state)
    choi, count = checker.choi_checker(index=[1], repeat=100, print_reason=False)

    node.results['results']['MLE data'] = {
        'axes': axes,
        'center': center,
        'R': R,
        'volume': volume,
        'param': param,
        'choi state': choi,
        'negativity': QuantumMemory.negativity(choi)*2,
        'memory robustness': QuantumMemory.memory_robustness(choi)
        }
    #plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    MLE_plot = EllipsoidTool(corrected_bloch).plot(ax=ax, axes=axes, center=center, R=R, title='MLE data', show_points=True, show_unit_sphere=True)
    node.results["MLE_plot"]= fig


# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
