"""
        READOUT OPTIMISATION: FREQUENCY
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). This is done while varying the readout frequency.
The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
determine the Signal-to-Noise Ratio (SNR). The readout frequency that yields the highest SNR is selected as the
optimal choice.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the readout frequency and dispersive shift chi in the state.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
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


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q0','q1']
    num_averages: int = 200
    frequency_span_in_mhz: float = 5
    frequency_step_in_mhz: float = 0.05
    min_amp_factor: float = 0.5
    max_amp_factor: float = 1.5
    amp_factor_step: float = 0.05
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True

node = QualibrationNode(name="07a_Readout_Frequency_Power_Optimization", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
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
n_avg = node.parameters.num_averages  # The number of averages
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)
flux_point = node.parameters.flux_point_joint_or_independent
amps = np.arange(
    node.parameters.min_amp_factor,
    node.parameters.max_amp_factor,
    node.parameters.amp_factor_step,
)
with program() as ro_freq_opt:
    n = declare(int)
    I_g = [declare(fixed) for _ in range(num_qubits)]
    Q_g = [declare(fixed) for _ in range(num_qubits)]
    I_e = [declare(fixed) for _ in range(num_qubits)]
    Q_e = [declare(fixed) for _ in range(num_qubits)]
    df = declare(int)
    a = declare(fixed)
    I_g_st = [declare_stream() for _ in range(num_qubits)]
    Q_g_st = [declare_stream() for _ in range(num_qubits)]
    I_e_st = [declare_stream() for _ in range(num_qubits)]
    Q_e_st = [declare_stream() for _ in range(num_qubits)]
    n_st = declare_stream()

    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(a, amps)):
                with for_(*from_array(df, dfs)):
                    # Update the resonator frequencies
                    update_frequency(qubit.resonator.name, df + qubit.resonator.intermediate_frequency)
                    # Wait for the qubits to decay to the ground state
                    wait(qubit.thermalization_time * u.ns)
                    align()
                    # Measure the state of the resonators
                    qubit.resonator.measure("readout", amplitude_scale=a,qua_vars=(I_g[i], Q_g[i]))

                    align()
                    # Wait for thermalization again in case of measurement induced transitions
                    wait(qubit.thermalization_time * u.ns)
                    # Play the x180 gate to put the qubits in the excited state
                    qubit.xy.play("x180")
                    # Align the elements to measure after playing the qubit pulses.
                    align()
                    # Measure the state of the resonators
                    qubit.resonator.measure("readout", amplitude_scale=a, qua_vars=(I_e[i], Q_e[i]))

                    # Derive the distance between the blobs for |g> and |e>
                    save(I_g[i], I_g_st[i])
                    save(Q_g[i], Q_g_st[i])
                    save(I_e[i], I_e_st[i])
                    save(Q_e[i], Q_e_st[i])
        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_g_st[i].buffer(len(dfs)).buffer(len(amps)).average().save(f"I_g{i + 1}")
            Q_g_st[i].buffer(len(dfs)).buffer(len(amps)).average().save(f"Q_g{i + 1}")
            I_e_st[i].buffer(len(dfs)).buffer(len(amps)).average().save(f"I_e{i + 1}")
            Q_e_st[i].buffer(len(dfs)).buffer(len(amps)).average().save(f"Q_e{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, ro_freq_opt, simulation_config)
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
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(ro_freq_opt)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs,"amps": amps})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits, ["I_g", "Q_g", "I_e", "Q_e"])
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2) for |g> and |e> as well as the distance between the two blobs D
        ds = ds.assign(
            {
                "D": np.sqrt((ds.I_g - ds.I_e) ** 2 + (ds.Q_g - ds.Q_e) ** 2),
                "IQ_abs_g": np.sqrt(ds.I_g**2 + ds.Q_g**2),
                "IQ_abs_e": np.sqrt(ds.I_e**2 + ds.Q_e**2),
            }
        )
        # Add the absolute frequency to the dataset
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([dfs + q.resonator.RF_frequency for q in qubits]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
        # Add the absolute amp to the dataset
        ds = ds.assign_coords(
            {
                "amp_full": (
                    ["qubit", "amps"],
                    np.array([amps*q.resonator.operations.readout.amplitude for q in qubits]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"

        
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # the maximum distance between two state
    results = {}
    for q in qubits:
        maximum_d_index = np.unravel_index(ds.sel(qubit=q.name).D.values.argmax(),ds.sel(qubit=q.name).D.values.shape)
        results[q.name] = {
            "max_d_frequency": ds.sel(qubit=q.name).freq_full.values[maximum_d_index[1]],
            "max_d_amp": ds.sel(qubit=q.name).amp_full.values[maximum_d_index[0]],
            "max_d_index": maximum_d_index,
        }
    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        (ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].D * 1e3).plot(
            ax=ax, x="freq_GHz", y="amp_full")
        ax.plot(results[qubit['qubit']]['max_d_frequency']/1e9,results[qubit['qubit']]['max_d_amp'], "ro")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        amp_index = results[qubit['qubit']]['max_d_index'][0]
        ax.plot(ds.freq,ds.sel(qubit = qubit['qubit'],amps=amps[amp_index]).IQ_abs_g * 1e3, label="IQ_abs_g")
        ax.plot(ds.freq,ds.sel(qubit = qubit['qubit'],amps=amps[amp_index]).IQ_abs_e * 1e3, label="IQ_abs_e")
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel("Resonator response [mV]")
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
    node.results["figure2"] = grid.fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        for q in qubits:
            with node.record_state_updates():
                q.resonator.operations.readout.amplitude = results[q.name]["max_d_amp"]
        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()


# %%
