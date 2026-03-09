# %%
"""
Unipolar CPhase Gate Calibration

This sequence measures the time and detuning required for a unipolar CPhase gate. The process involves:

1. Preparing both qubits in their excited states.
2. Applying a flux pulse with varying amplitude and duration.
3. Measuring the resulting state populations as a function of these parameters.
4. Fitting the results to a Ramsey-Chevron pattern.

From this pattern, we extract:
- The coupling strength (J2) between the qubits.
- The optimal gate parameters (amplitude and duration) for the CPhase gate.

The Ramsey-Chevron pattern emerges due to the interplay between the qubit-qubit coupling and the flux-induced detuning, allowing us to precisely calibrate the CPhase gate.

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair.
- Calibrated readout for both qubits.
- Initial estimate of the flux pulse amplitude range.

Outcomes:
- Extracted J2 coupling strength.
- Optimal flux pulse amplitude and duration for the CPhase gate.
- Fitted Ramsey-Chevron pattern for visualization and verification.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, qua_declaration
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_oscillation
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
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse
from quam_libs.experiments.simulation import simulate_and_plot

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

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ['q0_q2']
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "thermal"
    simulate: bool = False
    timeout: int = 100
    method: Literal['coarse', 'fine'] = "fine"
    amp_range_coarse : float = 0.4
    amp_step_coarse : float = 0.01
    load_data_id: Optional[int] = None
    simulation_duration_ns: int = 1000
    use_waveform_report: bool = True

    num_averages: int = 500
    max_time_in_ns: int = 5000
    amp_range_fine : float = 0.005
    amp_step_fine : float = 0.001

    

node = QualibrationNode(
    name="30c_10_01_oscillations_4nS_target_t2", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file

#path = "/Users/4hsiang/Desktop/Jack/python_project/instrument_control/opx1000/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration/quam_state"
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
# if any([qp.q1.z is None or qp.q2.z is None for qp in qubit_pairs]):
#     warnings.warn("Found qubit pairs without a flux line. Skipping")

num_qubit_pairs = len(qubit_pairs)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %%

####################
# Helper functions #
####################

def rabi_chevron_model(ft, J, f0, a, offset,tau):
    f,t = ft
    J = J
    w = f
    w0 = f0
    g = offset+a * np.sin(2*np.pi*np.sqrt(4*J**2 + (w-w0)**2) * t)**2*np.exp(-tau*np.abs((w-w0))) 
    return g.ravel()

def fit_rabi_chevron(ds_qp, init_length, init_detuning):
    da_target = ds_qp.state_target
    exp_data = da_target.values
    detuning = da_target.detuning
    time = da_target.time*1e-9
    t,f  = np.meshgrid(time,detuning)
    initial_guess = (1e9/init_length/2,
            init_detuning,
            -1,
            1.0,
            100e-9)
    fdata = np.vstack((f.ravel(),t.ravel()))
    tdata = exp_data.ravel()
    popt, pcov = curve_fit(rabi_chevron_model, fdata, tdata, p0=initial_guess,maxfev=10000)
    J = popt[0]
    f0 = popt[1]
    a = popt[2]
    offset = popt[3]
    tau = popt[4]

    return J, f0, a, offset, tau

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# define the amplitudes for the flux pulses
pulse_amplitudes = {}
if node.parameters.method == "coarse":
    for qp in qubit_pairs:
        detuning = qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency - qp.qubit_target.anharmonicity
        pulse_amplitudes[qp.name] = float(np.sqrt(-detuning/qp.qubit_control.freq_vs_flux_01_quad_term))
        #if qp.name[-2:] == 'q2':
            #pulse_amplitudes[qp.name] *= 2
else:
    for qp in qubit_pairs:
        pulse_amplitudes[qp.name] = qp.gates["iSWAP"].flux_pulse_control.amplitude
      
# Loop parameters
if node.parameters.method == "coarse":
    amplitudes = np.arange(1-node.parameters.amp_range_coarse, 1+node.parameters.amp_range_coarse, node.parameters.amp_step_coarse)
else:
    amplitudes = np.arange(1-node.parameters.amp_range_fine, 1+node.parameters.amp_range_fine, node.parameters.amp_step_fine)
#amplitudes = pulse_amplitudes[qp.name] / qp.qubit_control.z.operations['const'].amplitude*np.arange(1-node.parameters.amp_range, 1+node.parameters.amp_range, node.parameters.amp_step)
times_cycles = np.arange(0, node.parameters.max_time_in_ns // 4,1)

with program() as CPhase_Oscillations:
    t = declare(int)  # QUA variable for the flux pulse segment index
    idx = declare(int)
    amp = declare(float)    
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(bool) for _ in range(num_qubit_pairs)]
    state_target = [declare(bool) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    I_control = [declare(fixed) for _ in range(num_qubit_pairs)]
    I_target = [declare(fixed) for _ in range(num_qubit_pairs)]
    Q_control = [declare(fixed) for _ in range(num_qubit_pairs)]
    Q_target = [declare(fixed) for _ in range(num_qubit_pairs)]
    I_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            # qp.apply_mutual_flux_point()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()
        wait(1000)

        if hasattr(qp.gates['Cz'], "compensations"):
            compensation_qubits = [compensation["qubit"] for compensation in qp.gates['Cz'].compensations]
        else:
            compensation_qubits = []

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            
            with for_(*from_array(amp, amplitudes)):                                       
                # rest of the pulse
                with for_(*from_array(t, times_cycles)):
                    # reset                    
                    if node.parameters.reset_type == "active":
                        active_reset_gef(qp.qubit_control)
                        qp.align()
                        active_reset(qp.qubit_target)
                        qp.align()
                    else:
                        wait(qp.qubit_control.thermalization_time * u.ns)
                    # set both qubits to the excited state
                    qp.qubit_target.xy.play("x90")
                    qp.qubit_target.xy.wait(5)
                    qp.align()

                    # play the flux pulse
                    with if_(t > 0):
                        qp.qubit_control.z.play('const', duration=t, amplitude_scale = pulse_amplitudes[qp.name] / qp.qubit_control.z.operations['const'].amplitude * amp)
                        #qp.qubit_control.z.play('const', duration=t, amplitude_scale = amp)

                        ## check if there are any compensations and play the relevant flux pulse
                        for comp_ind, qubit in enumerate(compensation_qubits):
                            shift = qp.gates['iSWAP'].compensations[comp_ind]["shift"]
                            qubit.z.play("const", amplitude_scale= shift / qubit.z.operations["const"].amplitude, 
                                                        duration = node.parameters.max_time_in_ns // 4 + 10)
                        qp.align()
                    # wait for the flux pulse to end and some extra time
                    for qubit in [qp.qubit_control, qp.qubit_target]:
                        qubit.xy.wait(node.parameters.max_time_in_ns // 4 + 10)
                    qp.align()
                    
                    # measure both qubits
                    qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                    qp.qubit_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                    assign(state_control[i], I_control[i] > qp.qubit_control.resonator.operations["readout"].threshold)
                    assign(state_target[i], I_target[i] > qp.qubit_target.resonator.operations["readout"].threshold)
                    save(I_control[i], I_st_control[i])
                    save(Q_control[i], Q_st_control[i])
                    save(I_target[i], I_st_target[i])
                    save(Q_target[i], Q_st_target[i])
                    save(state_control[i], state_st_control[i])
                    save(state_target[i], state_st_target[i])

        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            I_st_control[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_control{i + 1}")
            I_st_target[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_target{i + 1}")
            Q_st_control[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_control{i + 1}")
            Q_st_target[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_target{i + 1}")
            state_st_control[i].boolean_to_int().buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_control{i + 1}")
            state_st_target[i].boolean_to_int().buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_target{i + 1}")



# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=100)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples,plot=True,save_path="./")
    
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(CPhase_Oscillations)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"time": 4*times_cycles, "amp": amplitudes})
    else:
        ds, loaded_machine = load_dataset(node.parameters.load_data_id)
        if loaded_machine is not None:
            machine = loaded_machine

    node.results = {"ds": ds}

# %% {Data_analysis}
if not node.parameters.simulate:
    def abs_amp(qp, amp):
        return amp * pulse_amplitudes[qp.name]

    def detuning(qp, amp):
        return -(amp * pulse_amplitudes[qp.name])**2 * qp.qubit_control.freq_vs_flux_01_quad_term
    
    ds = ds.assign_coords(
        {"amp_full": (["qubit", "amp"], np.array([abs_amp(qp, ds.amp.data) for qp in qubit_pairs]))}
    )
    ds = ds.assign_coords(
        {"detuning": (["qubit", "amp"], np.array([detuning(qp, ds.amp) for qp in qubit_pairs]))}
    )
 

# %% {Plot with state}
'''
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        plot = ds.assign_coords(detuning_MHz = 1e-6*ds.detuning).state_control.sel(qubit=qubit_pair['qubit']).plot(ax = ax, x= 'time', y= 'detuning_MHz', add_colorbar=False)        
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        ax.set_title(qubit_pair["qubit"])
        ax.set_ylabel('Detuning [MHz]')
        ax.set_xlabel('time [nS]')

        quad = machine.qubit_pairs[qubit_pair["qubit"]].qubit_control.freq_vs_flux_01_quad_term
        print(f"qubit_pair: {qubit_pair['qubit']}, quad: {quad}")
        
        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad
        
        ax2 = ax.secondary_yaxis('right', functions=(detuning_to_flux, flux_to_detuning))
        ax2.set_ylabel('Flux amplitude [V]')
        ax.set_ylabel('Detuning [MHz]')
        
    plt.suptitle('control qubit state')
    plt.show()
    node.results["figure_control"] = grid.fig
    
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        plot = ds.assign_coords(detuning_MHz = 1e-6*ds.detuning).state_target.sel(qubit=qubit_pair['qubit']).plot(ax = ax, x= 'time', y= 'detuning_MHz', add_colorbar=False)        
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        ax.set_title(qubit_pair["qubit"])
        ax.set_ylabel('Detuning [MHz]')
        ax.set_xlabel('time [nS]')
        quad = machine.qubit_pairs[qubit_pair["qubit"]].qubit_control.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad
        
        ax2 = ax.secondary_yaxis('right', functions=(detuning_to_flux, flux_to_detuning))
        ax2.set_ylabel('Flux amplitude [V]')
        ax.set_ylabel('Detuning [MHz]')
        
        
    plt.suptitle('target qubit state')
    plt.show()
    node.results["figure_target"] = grid.fig

# %% {Plot with IQ}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        plot = ds.assign_coords(detuning_MHz = 1e-6*ds.detuning).I_control.sel(qubit=qubit_pair['qubit']).plot(ax = ax, x= 'time', y= 'detuning_MHz', add_colorbar=False)        
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        ax.set_title(qubit_pair["qubit"])
        ax.set_ylabel('Detuning [MHz]')
        ax.set_xlabel('time [nS]')

        quad = machine.qubit_pairs[qubit_pair["qubit"]].qubit_control.freq_vs_flux_01_quad_term
        print(f"qubit_pair: {qubit_pair['qubit']}, quad: {quad}")
        
        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad
        
        ax2 = ax.secondary_yaxis('right', functions=(detuning_to_flux, flux_to_detuning))
        ax2.set_ylabel('Flux amplitude [V]')
        ax.set_ylabel('Detuning [MHz]')
        
    plt.suptitle('control qubit I')
    plt.show()
    node.results["figure_control"] = grid.fig
    
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        plot = ds.assign_coords(detuning_MHz = 1e-6*ds.detuning).I_target.sel(qubit=qubit_pair['qubit']).plot(ax = ax, x= 'time', y= 'detuning_MHz', add_colorbar=False)        
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        ax.set_title(qubit_pair["qubit"])
        ax.set_ylabel('Detuning [MHz]')
        ax.set_xlabel('time [nS]')
        quad = machine.qubit_pairs[qubit_pair["qubit"]].qubit_control.freq_vs_flux_01_quad_term

        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad
        
        ax2 = ax.secondary_yaxis('right', functions=(detuning_to_flux, flux_to_detuning))
        ax2.set_ylabel('Flux amplitude [V]')
        ax.set_ylabel('Detuning [MHz]')
        
        
    plt.suptitle('target qubit I')
    plt.show()
    node.results["figure_target"] = grid.fig
'''

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
        
# %%
