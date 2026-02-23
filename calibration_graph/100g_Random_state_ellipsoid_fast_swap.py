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
from qualang_tools.bakery import baking

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
    interaction_start_time_in_ns: int = 0
    interaction_end_time_in_ns: int = 153
    interaction_time_step_in_ns: int = 1
    compensation: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 10000
    timeout: int = 100
    load_data_id: Optional[int] = 6278
    multiplexed: bool = False
    number_of_points: int = 200
    repeats: int = 1

if Parameters().load_data_id is not None:
    node  = QualibrationNode(name=f"{Parameters().load_data_id}_analyze_again", parameters=Parameters())
else:
    node = QualibrationNode(name="100g_Random_state_ellipsoid_fast_swap", parameters=Parameters())
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
        theta,phi = float(np.random.choice(theta_range)), float(np.random.choice(phi_range))
        theta_list.append(theta)
        phi_list.append(phi)
    return theta_list,phi_list

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

theta_phi_total = theta_phi_list(n_points)
theta_list = theta_phi_total[0]
phi_list = theta_phi_total[1]


t=4
interaction_start_time = node.parameters.interaction_start_time_in_ns
interaction_end_time = node.parameters.interaction_end_time_in_ns
interaction_time_step = node.parameters.interaction_time_step_in_ns
interaction_time_list = np.arange(interaction_start_time,interaction_end_time+1,interaction_time_step)
compensation = node.parameters.compensation
iswap_point = machine.qubit_pairs['q0_q2'].gates['iSWAP'].flux_pulse_control.amplitude
noise_qubit = machine.qubits['q2']

def QuantumMemory_program(qubit, interaction_time, compensation, baked_signals):
    with program() as QuantumMemory:
        theta = declare(fixed)
        phi = declare(fixed)
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = [declare(int) for _ in range(1)]
        state_st = [declare_stream() for _ in range(1)]
        tomo_axis = declare(int)

        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)
            with for_each_((theta, phi),(theta_list, phi_list)):
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
                    align()

                    baked_signals[0].run() 
                    if compensation:
                        shift = machine.qubit_pairs['q0_q2'].gates['iSWAP'].compensations[0]["shift"]
                        com_qubit =machine.qubit_pairs['q0_q2'].gates['iSWAP'].compensations[0].qubit
                        com_qubit.z.play("const", amplitude_scale= shift / com_qubit.z.operations["const"].amplitude, 
                                                    duration = interaction_time // 4 + 10)
                    align()
                    wait(interaction_time//4+13+13)
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
    baked_signals = baked_waveform(iswap_point, noise_qubit, t)
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, QuantumMemory_program(qubits[0], 0, compensation, baked_signals), simulation_config)
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
    for tt in interaction_time_list:
        baked_signals = baked_waveform(iswap_point, noise_qubit, tt)
        print(f"Running for interaction time {tt} ns...")
        for qubit in qubits:
            with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
                job = qm.execute(QuantumMemory_program(qubit, tt, compensation, baked_signals))
                results = fetching_tool(job, ["n"], mode="live")
                while results.is_processing():
                    n = results.fetch_all()[0]
                    progress_counter(n, n_runs, start_time=results.start_time)
                job_.append(job)
        print(f"Finished interaction time {tt} ns.")

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if load_data_id is None:
        ds_list = []
        for i in range(len(interaction_time_list)):
            _ds = fetch_results_as_xarray(job_[i].result_handles, [qubits[0]], { "axis": ['x','y','z'],"n_points": np.arange(n_points)})
            # Add interaction_time as a coordinate
            _ds = _ds.assign_coords(interaction_time=interaction_time_list[i])
            ds_list.append(_ds)
        # Concatenate all datasets along the interaction_time dimension
        ds = xr.concat(ds_list, dim='interaction_time')

    else:
        # from pathlib import Path
        # QPT_ellipsoid_data_path = Path("/Users/jackchao/Desktop/Project/Phd_thesis/CH5_5GZdemonstration/data/Quantum_memory_referee/QPT_vs_ellipsoid_data/")

        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
        machine = node.machine
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
    fitter = EllipsoidFittingOptimizer()
    raw_robustness_list = []
    mle_robustness_list = []
    node.results['results']['raw'] = {}
    node.results['results']['MLE'] = {}
    for tt_idx, tt in enumerate(interaction_time_list):
        print(f"\n{'='*80}")
        print(f"INTERACTION TIME: {tt} ns  (Index {tt_idx}/{len(interaction_time_list)-1})")
        print(f"{'='*80}")
        
        ds_tt = ds.sel(interaction_time=tt)
        data_xyz = np.array([[ds_tt.Bloch_vector_x.values[0][i], ds_tt.Bloch_vector_y.values[0][i], ds_tt.Bloch_vector_z.values[0][i]] for i in range(len(ds_tt.n_points.values))])
        data_angle = np.array([[ds_tt.theta.values[i], ds_tt.phi.values[i]] for i in range(len(ds_tt.n_points.values))])
        data_dm = np.array([bloch_vector_to_density_matrix(data_xyz[i]) for i in range(len(data_xyz))])
        from quam_libs.quantum_memory.NoiseAnalyze import EllipsoidFitParameter
        ellipsoid_fit_parameters = EllipsoidFitParameter()
        noise_analyzer = NoiseAnalyze(data_xyz,data_angle,ellipsoid_fit_parameters)
        corrected_dm = noise_analyzer.corrected_dm
        corrected_bloch = noise_analyzer.corrected_bloch
        # first fit ellipsoid
        center, axes, R, volume, param = noise_analyzer.ellipsoid_fit()
        R, axes = fitter.align_by_max_value(R, axes)

        # compare the fitted ellipsoid with quadric form and find the best fit.
        #axes, R = find_best_fit(center, axes, R, param)

        qm_analyze = QuantumMemory(axes,center,R)
        choi_state = qm_analyze.choi_state()

        checker = Checker(choi_state)
        choi, count = checker.choi_checker(index=[1], repeat=100, print_reason=False)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        raw_plot = EllipsoidTool(corrected_bloch).plot(ax=ax,axes=axes, center=center, R=R, title=f'Raw: interaction time {tt} ns',show_points=True, show_unit_sphere=True)
        node.results[f"raw_{tt}_ns"] = fig  
        node.results['results']['raw'][f"interaction_{tt}_ns"] = {
            'axes': axes,
            'center': center,
            'R': R,
            'volume': volume,
            'param': param,
            'choi state': choi,
            'negativity': QuantumMemory.negativity(choi)*2,
            'memory robustness': QuantumMemory.memory_robustness(choi)
            }
        
        raw_robustness = QuantumMemory.memory_robustness(choi)
        raw_robustness_list.append(raw_robustness)
        
        print(f"\n{'─'*80}")
        print(f"📊 RAW DATA ANALYSIS")
        print(f"{'─'*80}")
        print(f"  Axes:          {axes}")
        print(f"  Center:              {center}")
        print(f"  Rotation matrix R:   {R}")
        print(f"  Ellipsoid volume:    {volume:.6f}")
        print(f"  Fitting parameters:  {param}")
        print(f"  ✓ Memory Robustness: {raw_robustness:.8f}")

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
        noise_analyzer = NoiseAnalyze(MLE_data_xyz,data_angle,ellipsoid_fit_parameters)
        corrected_dm = noise_analyzer.corrected_dm
        corrected_bloch = noise_analyzer.corrected_bloch

        # first fit ellipsoid
        center, axes, R, volume, param = noise_analyzer.ellipsoid_fit()
        # compare the fitted ellipsoid with quadric form and find the best fit.
        # axes, R = find_best_fit(center, axes, R, param)
        R, axes = fitter.align_by_max_value(R, axes)
        qm_analyze = QuantumMemory(axes,center,R)
        choi_state = qm_analyze.choi_state()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        raw_plot = EllipsoidTool(corrected_bloch).plot(ax=ax,axes=axes, center=center, R=R, title=f'MLE: interaction time {tt} ns',show_points=True, show_unit_sphere=True)
        node.results[f"MLE_{tt}_ns"] = fig  
        checker = Checker(choi_state)
        choi, count = checker.choi_checker(index=[1], repeat=100, print_reason=False)

        node.results['results']['MLE'][f"interaction_{tt}_ns"] = {
            'axes': axes,
            'center': center,
            'R': R,
            'volume': volume,
            'param': param,
            'choi state': choi,
            'negativity': QuantumMemory.negativity(choi)*2,
            'memory robustness': QuantumMemory.memory_robustness(choi)
            }
        
        mle_robustness = QuantumMemory.memory_robustness(choi)
        mle_robustness_list.append(mle_robustness)
        
        print(f"\n{'─'*80}")
        print(f"🔧 MLE DATA ANALYSIS")
        print(f"{'─'*80}")
        print(f"  Axes shape:          {axes}")
        print(f"  Center:              {center}")
        print(f"  Rotation matrix R:   {R}")
        print(f"  Ellipsoid volume:    {volume:.6f}")
        print(f"  Fitting parameters:  {param}")
        print(f"  ✓ Memory Robustness: {mle_robustness:.8f}")
        
        print(f"\n{'─'*80}")
        print(f"📈 COMPARISON")
        print(f"{'─'*80}")
        improvement = ((mle_robustness - raw_robustness) / raw_robustness * 100) if raw_robustness != 0 else 0
        print(f"  Raw robustness:      {raw_robustness:.8f}")
        print(f"  MLE robustness:      {mle_robustness:.8f}")
        print(f"  Improvement:         {improvement:+.2f}%")
        print()
    # %% Plot final results
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    ax.plot(interaction_time_list, raw_robustness_list, 'o-', label='Raw Data', linewidth=2.5, markersize=8, color='#FF6B6B')
    ax.plot(interaction_time_list, mle_robustness_list, 's-', label='MLE Data', linewidth=2.5, markersize=8, color='#4ECDC4')    
    # Add vertical lines at specific interaction times
    vline_times = [45, 90, 135]
    colors = ['#95E1D3', '#F38181', '#AA96DA']
    for vtime, vcolor in zip(vline_times, colors):
        ax.axvline(x=vtime, color=vcolor, linestyle='--', linewidth=2, alpha=0.7, label=f'{vtime} ns')
        ax.set_xlabel('Interaction Time (ns)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Memory Robustness', fontsize=12, fontweight='bold')
    ax.set_title('Memory Robustness vs Interaction Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    
    node.results["robustness_comparison"] = fig
    
    # Print final summary table
    print(f"\n{'='*80}")
    print(f"{'FINAL SUMMARY':^80}")
    print(f"{'='*80}")
    print(f"\n{'Interaction Time (ns)':<25} {'Raw Robustness':<25} {'MLE Robustness':<25}")
    print(f"{'-'*80}")
    for tt, raw_r, mle_r in zip(interaction_time_list, raw_robustness_list, mle_robustness_list):
        improvement = ((mle_r - raw_r) / raw_r * 100) if raw_r != 0 else 0
        print(f"{tt:<25} {raw_r:<25.8f} {mle_r:<25.8f}")
    print(f"{'='*80}\n")
#%%
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
