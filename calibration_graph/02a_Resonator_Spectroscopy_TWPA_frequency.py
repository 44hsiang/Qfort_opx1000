"""
        RESONATOR SPECTROSCOPY MULTIPLEXED
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies for all resonators simultaneously.
The data is then post-processed to determine the resonator resonance frequency.
This frequency can be used to update the readout frequency in the state.

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibrate the IQ mixer connected to the readout line (whether it's an external mixer or an Octave port).
    - Define the desired readout pulse amplitude and duration in the state.
    - Specify the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the readout frequency, in the state for all resonators.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from quam_libs.lib.fit_utils import fit_resonator
from quam_libs.lib.qua_datasets import apply_angle, subtract_slope, convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.TWPA_pump import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 200
    frequency_span_in_mhz: float = 15.0
    frequency_step_in_mhz: float = 0.05
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


node = QualibrationNode(name="02a_Resonator_Spectroscopy_TWPA_frequency", parameters=Parameters())
assert not (
    node.parameters.simulate and node.parameters.load_data_id is not None
), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
path = "/Users/4hsiang/Desktop/Jack/python_project/instrument_control/opx1000/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration/quam_state"
machine = QuAM.load(path)
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
resonators = [qubit.resonator for qubit in qubits]
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

with program() as multi_res_spec:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the readout frequency

    # Bring the active qubits to the minimum frequency point
    #machine.apply_all_flux_to_min()
    machine.apply_all_flux_to_joint_idle()
    #machine.set_all_fluxes(flux_point=flux_point, target=qubit)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            for i, rr in enumerate(resonators):
                # Update the resonator frequencies for all resonators
                update_frequency(rr.name, df + rr.intermediate_frequency)
                # Measure the resonator
                rr.measure("readout", qua_vars=(I[i], Q[i]))
                # wait for the resonator to relax
                rr.wait(machine.depletion_time * u.ns)
                # save data
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
    if not node.parameters.multiplexed:
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, multi_res_spec, simulation_config)
    # Plot the simulated samples
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Update the node & save
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    # Open a quantum machine to execute the QUA program
    import time
    addr = 'TCPIP::192.168.50.12::INSTR'
    TWPA_fre = 6.155e9
    TWPA_gain = 1.4
    open_TWPA(addr=addr,power=True,pump_frequency=TWPA_fre,gain=TWPA_gain)
    f_span = 10e6
    frequency_sweep = np.linspace(TWPA_fre-f_span,TWPA_fre+f_span,9)
    job_ = []
    for i in frequency_sweep:
        open_TWPA(addr=addr,power=True,pump_frequency=i,gain=TWPA_gain)
        time.sleep(2)
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(multi_res_spec)
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                # Fetch results
                n = results.fetch_all()[0]
                # Progress bar
                progress_counter(n, n_avg, start_time=results.start_time)
        job_.append(job)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        for i in range(len(gain_sweep)):
            ds_ = fetch_results_as_xarray(job_[i].result_handles, qubits, {"freq": dfs})
            ds = xr.concat([ds, ds_], dim="TWPA_fre") if i != 0 else ds_
        ds = ds.assign_coords(TWPA_fre=frequency_sweep)
        #ds = fetch_results_as_xarray(job.result_handles, qubits, {"freq": dfs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Derive the phase IQ_abs = angle(I + j*Q)
        ds = ds.assign({"phase": subtract_slope(apply_angle(ds.I + 1j * ds.Q, dim="freq"), dim="freq")})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([dfs + q.resonator.RF_frequency for q in qubits]),
                )
            }
        )

    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        for i in range(len(frequency_sweep)):
            (ds.sel(TWPA_fre=frequency_sweep[i]).assign_coords(freq_MHz=ds.freq / 1e6).loc[qubit].IQ_abs * 1e3).plot(ax=ax, x="freq_MHz",label=f"{frequency_sweep[i]/1e9}GHz")
        ax.set_xlabel("Resonator detuning [MHz]")
        ax.set_ylabel("Trans. amp. [mV]")
        ax.set_title(qubit["qubit"])
    
    grid.fig.suptitle("Resonator spectroscopy (raw data)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    node.results["raw_amplitude"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        for i in range(len(gain_sweep)):
            (ds.sel(TWPA_fre=frequency_sweep[i]).assign_coords(freq_MHz=ds.freq / 1e6).loc[qubit].phase * 1e3).plot(ax=ax, x="freq_MHz")
        ax.set_xlabel("Resonator detuning [MHz]")
        ax.set_ylabel("Trans. phase [mrad]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Resonator spectroscopy (raw data)")
    plt.tight_layout()
    node.results["raw_phase"] = grid.fig


    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            open_TWPA(addr=addr,power=True,pump_frequency=6.165e9,gain=0.65)

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()
        print("Results saved")


# %%
