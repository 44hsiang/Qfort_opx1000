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
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit, minimize

from qiskit.result import CorrelatedReadoutMitigator
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q0']
    num_runs: int = 10000
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    number_of_points: int = 1
    repeats: int = 50


node = QualibrationNode(name="100f_Random_state_ellipsoid_repeat", parameters=Parameters())


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

#%% helper functions

theta_range = np.arange(0,np.pi,1e-4)
phi_range = np.arange(0,2*np.pi,1e-4)
def theta_phi_list(n_points):
    theta_list,phi_list = [],[]
    for i in range(n_points):
        theta,phi = np.random.choice(theta_range),np.random.choice(phi_range)
        theta_list.append(theta)
        phi_list.append(phi)
    return theta_list,phi_list

# distance check
def distance_eq(point):
    x, y, z = point
    return (x - x0)**2 + (y - y0)**2 + (z - z0)**2

# condition function
def constraint(point):
    x, y, z = point
    return param[0]*x*x+param[1]*y*y+param[2]*z*z+param[3]*x*y+param[4]*x*z+param[5]*y*z+param[6]*x+param[7]*y+param[8]*z+param[9]

def angle_error_min(theoretical,actual):
    error = (actual - theoretical + np.pi) % 2*np.pi - np.pi  
    return np.abs(error)  

# %% {QUA_program}
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
n_points = node.parameters.number_of_points
repeats = node.parameters.repeats


theta_list = [theta_phi_list(n_points)[0] for _ in range(repeats)]
phi_list = [theta_phi_list(n_points)[1] for _ in range(repeats)]
t=4
def QuantumMemory_program(qubit,repeats):
    with program() as QuantumMemory:
        ii = declare(int)
        theta = declare(fixed, value = theta_list[repeats])
        phi = declare(fixed, value = phi_list[repeats])
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
                    qubit.xy.play("y180",amplitude_scale = theta[ii]/np.pi)
                    qubit.xy.frame_rotation_2pi((phi[ii]-qubit.extras["phi_offset"])/np.pi/2-0.5)
                    wait(250)
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
    %matplotlib qt
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, QuantumMemory_program(qubit,repeat), simulation_config)
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
    for repeat in range(node.parameters.repeats):
        job_.append([])
        for qubit in qubits:
            with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                job = qm.execute(QuantumMemory_program(qubit,repeat))
                results = fetching_tool(job, ["n"], mode="live")
                while results.is_processing():
                    n = results.fetch_all()[0]
                    progress_counter(n, n_runs, start_time=results.start_time)
                job_[repeat].append(job)
                print(f"repeat {repeat} qubit {qubit.name} is done")

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    
    if node.parameters.load_data_id is None:
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
    #%% 
    if node.parameters.load_data_id is None:
        print(True)
        theta_list = np.array(theta_list).flatten()
        phi_list = np.array(phi_list).flatten()
        ds = ds.assign_coords(theta = theta_list,phi =phi_list)    
        ds["Bloch_vector_x"] = 1-2*ds["state"].sel(axis='x')
        ds["Bloch_vector_y"] = 1-2*ds["state"].sel(axis='y')
        ds["Bloch_vector_z"] = 1-2*ds["state"].sel(axis='z')
        ds["Bloch_phi"] = np.arctan2(ds.Bloch_vector_y,ds.Bloch_vector_x)
        ds["Bloch_phi"] = ds.Bloch_phi.where(ds.Bloch_phi>0,ds.Bloch_phi+2*np.pi)
        ds["Bloch_theta"] = np.arccos(ds.Bloch_vector_z/np.sqrt(ds.Bloch_vector_x**2+ds.Bloch_vector_y**2+ds.Bloch_vector_z**2))
        

    # %% {Data_analysis}
    node.results = {"ds": ds, "figs": {}, "results": {}}
    node.results["ds"] = ds

    analyze_results = {}
        
    for q in qubits:
        # filter the data which theta and phi are far close to the ideal points
        filter_theta_index = np.where(np.abs(ds.theta-ds.sel(qubit=q.name).Bloch_theta.values)<0.05*np.pi)
        filter_phi_index = np.where(np.abs(ds.phi-ds.sel(qubit=q.name).Bloch_phi.values)<0.05*2*np.pi)
        filter_index = np.intersect1d(filter_theta_index,filter_phi_index)
        filter_index = np.arange(n_points*repeats)
        print(f"qubit {q.name} has {len(filter_index)} data points")
        theta_stats =[
            (ds.theta.values[filter_index]-ds.sel(qubit=q.name).Bloch_theta.values[filter_index]).mean(),
            (ds.theta.values[filter_index]-ds.sel(qubit=q.name).Bloch_theta.values[filter_index]).std()
            ]
        phi_stats = [
            angle_error_min(ds.phi.values[filter_index],ds.sel(qubit=q.name).Bloch_phi.values[filter_index]).mean(),
            angle_error_min(ds.phi.values[filter_index],ds.sel(qubit=q.name).Bloch_phi.values[filter_index]).std()
            ]
        
        # fit ellipsoid
        x = ds.sel(qubit=q.name).Bloch_vector_x.values[filter_index]
        y = ds.sel(qubit=q.name).Bloch_vector_y.values[filter_index]
        z = ds.sel(qubit=q.name).Bloch_vector_z.values[filter_index]    
        parameters = ls_ellipsoid(x,y,z)
        center,axes,R = polyToParams3D(parameters,False)

        # elliptcal parameters  define in the constraint function
        param = parameters
        # TODO: it can be improved by adding the sign to define the point is inside of outside the curve 
        shortest_distance = np.array([])
        for i in range(n_points*len(filter_index)):
            x0, y0, z0 = x[i], y[i], z[i]
            data_point = np.array([x0, y0, z0])
            cons = {'type': 'eq', 'fun': constraint} # constraint function
            result = minimize(distance_eq, data_point, constraints=[cons]) # minimize the distance
            shortest_distance = np.append(shortest_distance,np.sqrt(result.fun))

        print(
        f"the shortest distance average = {shortest_distance.mean():.3} "
        f"and std = {shortest_distance.std():.3}"
        )
        analyze_results[q.name] = {
            "center": center, 
            "axes": axes, 
            "rotation_matrix": R,
            "shortest_distance_stats":[shortest_distance.mean(),shortest_distance.std()],
            "theta_stats":theta_stats,
            "phi_stats":phi_stats
            }
    print(analyze_results)
    node.results["analyze_results"] = analyze_results

    # %% {Plotting}
    %matplotlib inline
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ax.plot(ds.phi-ds.sel(qubit=qubit['qubit']).Bloch_phi.values,label='phi')
        ax.plot(ds.theta-ds.sel(qubit=qubit['qubit']).Bloch_theta.values,label='theta')
        ax.legend()
        ax.set_xlabel("data points")
        ax.set_ylabel("error(rad)")
        ax.set_title(f"{qubit['qubit']} phase and theta error")

    node.results["figure_error"] = grid.fig
    # ellipsoid function
    u,v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    grid = QubitGrid(ds, [q.grid_location for q in qubits],is_3d=True)
    for ax, qubit in grid_iter(grid):
        x_ellipsoid_ = analyze_results[qubit['qubit']]['axes'][0] * np.outer(np.cos(u), np.sin(v))
        y_ellipsoid_ = analyze_results[qubit['qubit']]['axes'][1] * np.outer(np.sin(u), np.sin(v))
        z_ellipsoid_ = analyze_results[qubit['qubit']]['axes'][2] * np.outer(np.ones_like(u), np.cos(v))

        ellipsoid_points_ = np.dot(analyze_results[qubit['qubit']]['rotation_matrix'],np.array([x_ellipsoid_.ravel(), y_ellipsoid_.ravel(), z_ellipsoid_.ravel()]))
        ellipsoid_points_ += analyze_results[qubit['qubit']]['center'].reshape(-1, 1)
        x_ellipsoid_, y_ellipsoid_, z_ellipsoid_ = ellipsoid_points_.reshape(3, *x_ellipsoid_.shape)
        ax.scatter(
            ds.sel(qubit =qubit['qubit']).Bloch_vector_x.values[filter_index],
            ds.sel(qubit =qubit['qubit']).Bloch_vector_y.values[filter_index],
            ds.sel(qubit =qubit['qubit']).Bloch_vector_z.values[filter_index], 
            label="Data points", color="black", s=2
            )
        ax.plot_wireframe(x, y, z, color="blue", alpha=0.05, label=" Bloch sphere")
        ax.plot_wireframe(x_ellipsoid_, y_ellipsoid_, z_ellipsoid_, color="red", alpha=0.08, label="fitted ellipsoid")
        ax.set_title(f"{qubit['qubit']} Bloch Sphere")
    
    node.results["figure_Bloch_vector"] = grid.fig

    # %% {Update_state}
    if not node.parameters.simulate:
        
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
