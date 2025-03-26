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
from qualang_tools.bakery import baking
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
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    interaction_time_ns: int = 4
    simulate: bool = True
    simulation_duration_ns: int = 1000
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    number_of_points: int = 1
    repeats: int = 10

if Parameters().load_data_id is not None:
    node  = QualibrationNode(name=f"{Parameters().load_data_id}_analyze_again", parameters=Parameters())
else:
    node = QualibrationNode(name="101a_Random_state_ellipsoid_swap_qubit1ns", parameters=Parameters())
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
noise_qubit = machine.qubits['q2']
iswap_point = machine.qubit_pairs['q0_q2'].gates['iSWAP'].flux_pulse_control.amplitude
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
    error = (actual - theoretical + np.pi) % (2*np.pi) - np.pi  
    return error  

def data_thershold(data,n):
    avg,std = data[0],data[1]
    return avg+n*std,avg-n*std

#plotting
def generate_bins_labels(bin_step=0.2, max_value=2*np.pi):
    bin_width = bin_step * np.pi
    bins = np.arange(0, max_value + bin_width, bin_width)
    labels = [f'{bin_step * i:.1f}π' for i in range(len(bins))]
    return bins, labels

def plot_histogram(data, title, subplot_idx, bins, x_labels,ylim):
    plt.subplot(subplot_idx)
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.ylabel('Count')
    plt.ylim(0,ylim)
    plt.xticks(bins, x_labels, rotation=45)

def baked_waveform(waveform_amp, qubit,time):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    waveform = [waveform_amp] * 200 # create a 200ns waveform
    with baking(config, padding_method="left") as b:
        if time == 0:
            wf = [0.0] * 16
        else:
            wf = waveform[:time]
        b.add_op(f"flux_pulse_noise", qubit.z.name, wf)
        b.play(f"flux_pulse_noise", qubit.z.name)
    # Append the baking object in the list to call it from the QUA program
    pulse_segments.append(b)
    return pulse_segments


# %% {QUA_program}
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
n_points = node.parameters.number_of_points
repeats = node.parameters.repeats

theta_list = np.array([theta_phi_list(n_points)[0] for _ in range(repeats)])
phi_list = [theta_phi_list(n_points)[1] for _ in range(repeats)]
interaction_time = node.parameters.interaction_time_ns
baked_signals = baked_waveform(iswap_point, noise_qubit,interaction_time)

def QuantumMemory_program(qubit,repeat):
    with program() as QuantumMemory:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = [declare(int) for _ in range(1)]
        state_st = [declare_stream() for _ in range(1)]
        tomo_axis = declare(int)
        t = declare(int)
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)
            with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):
                if reset_type == "active":
                    active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                elif reset_type == "thermal":
                    #qubit.wait(4 * qubit.thermalization_time * u.ns)
                    qubit.wait(100)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")
                align()
                qubit.xy.play("y180",amplitude_scale = theta_list[repeat][0]/np.pi)
                qubit.xy.frame_rotation_2pi((phi_list[repeat][0]-qubit.extras["phi_offset"])/np.pi/2-0.5)
                align()
                baked_signals[0].run() 
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
    job = qmm.simulate(config, QuantumMemory_program(qubits[0],repeat=0), simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples,plot=True,save_path="./")
    '''
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    '''
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
                job = qm.execute(QuantumMemory_program(qubit,repeat))
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

    # %% {Data_analysis}
    node.results = {"ds": ds, "figs": {}, "results": {}}
    node.results["ds"] = ds
    filter_data = False
    analyze_results = {}
    data_results = {}
    for q in qubits:
        # filter bad data if fitting_again = True
        theta_diff = (ds.theta.values-ds.sel(qubit=q.name).Bloch_theta.values)
        phi_diff = angle_error_min(ds.phi.values,ds.sel(qubit=q.name).Bloch_phi.values)
        phi_diff_sort_index = np.argsort(np.abs(phi_diff))
        filter_index_sd = np.array([])
        # TODO : make short distance index iterable
        for j in range(2): # 1 => only phase 2 => phase and short distance
            if filter_data:
                delete_index = np.union1d(filter_index_sd,phi_diff_sort_index[100:])
                filter_index = np.delete(ds.n_points,delete_index.astype(int))
            else:
                filter_index = np.arange(n_points*repeats)

            theta_stats =[theta_diff[filter_index].mean(),theta_diff[filter_index].std()]
            phi_stats = [phi_diff[filter_index].mean(),phi_diff[filter_index].std()]
            # fit ellipsoid
            x = ds.sel(qubit=q.name).Bloch_vector_x.values[filter_index]
            y = ds.sel(qubit=q.name).Bloch_vector_y.values[filter_index]
            z = ds.sel(qubit=q.name).Bloch_vector_z.values[filter_index]
            theta_data = ds.theta.values[filter_index]
            phi_data = ds.phi.values[filter_index]    
            param = ls_ellipsoid(x,y,z)
            center,axes,R = polyToParams3D(param,False)

            # elliptcal parameters define in the constraint function
            shortest_distance = np.array([])
            for i in range(n_points*len(filter_index)):
                x0, y0, z0 = x[i], y[i], z[i]
                data_point = np.array([x0, y0, z0])
                cons = {'type': 'eq', 'fun': constraint} # constraint function
                result = minimize(distance_eq, data_point, constraints=[cons]) # minimize the distance
                shortest_distance = np.append(shortest_distance,np.abs(result.fun)*np.sign(constraint((x0, y0, z0 ))))

            fidelity_data = [QuantumStateAnalysis([x[i],y[i],z[i]],[theta_data[i],phi_data[i]]).fidelity for i in range(len(filter_index))]
            trace_distance_data = [QuantumStateAnalysis([x[i],y[i],z[i]],[theta_data[i],phi_data[i]]).trace_distance for i in range(len(filter_index))]
            filter_index_sd = np.where(np.abs(shortest_distance)>data_thershold([shortest_distance.mean(),shortest_distance.std()],2)[0])

        analyze_results[q.name] = {
            "center": center, 
            "axes": axes, 
            "rotation_matrix": R,
            "volume": 4/3*np.pi*axes[0]*axes[1]*axes[2],
            "shortest_distance_stats":[shortest_distance.mean(),shortest_distance.std()],
            "theta_stats":theta_stats,
            "phi_stats":phi_stats,
            "fidelity_stats":[np.mean(fidelity_data),np.std(fidelity_data)],
            "trace_distance_stats":[np.mean(trace_distance_data),np.std(trace_distance_data)] 
            }
        #
        data_results[q.name] = {"theta":theta_diff[filter_index],
                                "phi":phi_diff[filter_index],
                                "shortest_distance":shortest_distance,
                                "fidelity":fidelity_data
                                }
        print(f"qubit {q.name} has {len(filter_index)} data points")
        from pprint import pprint
        pprint(analyze_results)
    node.results["analyze_results"] = analyze_results
    # %% {Plotting}
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

    # plot the shortest distance
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ax.plot(data_results[qubit['qubit']]['shortest_distance'],'.',label='distance')
        ax.legend()
        ax.set_xlabel("data points")
        ax.set_ylabel("error(arb.)")
        #ax.set_ylim(-0.02,0.02)
        ax.set_title(f"{qubit['qubit']} distance")
    node.results["figure_short_distance"] = grid.fig

    # plot the theta and phi error
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ax.plot(data_results[qubit['qubit']]['phi'],'.',label='fun')
        #ax.plot(ds.sel(qubit=q.name).Bloch_phi.values-ds.phi.values,'.',label='direct')
        ax.plot(data_results[qubit['qubit']]['theta'],'.',label='theta')
        
        ax.legend()
        ax.set_xlabel("data points")
        ax.set_ylim(-np.pi,np.pi)
        #ax.set_xlim(0,50)
        ax.set_ylabel("error(rad)")
        ax.set_title(f"{qubit['qubit']} phase and theta error")

    node.results["figure_error"] = grid.fig

# %% check the random theta and phi

bins, x_labels = generate_bins_labels()
figure = plt.figure(figsize=(8, 5))
plot_histogram(theta_data, 'Theta', 221, bins, x_labels,ylim=55)
plot_histogram(ds.theta, 'Theta', 223, bins, x_labels,ylim=55)
plot_histogram(phi_data, 'Phi', 222, bins, x_labels,ylim=35)
plot_histogram(ds.phi, 'Phi', 224, bins, x_labels,ylim=35)

plt.suptitle(f"id = {load_data_id} total count = {len(theta_data)}")
plt.tight_layout()
node.results["figure_phi_theta_distrbution"] = figure

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.reset_type_thermal_or_active == "active" and load_data_id is None:
        for i,j in zip(machine.active_qubit_names,"abcde"):
            machine.qubits[i].xy.core = j
            machine.qubits[i].resonator.core = j

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

3# %%
