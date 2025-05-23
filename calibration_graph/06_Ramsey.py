"""
        RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing gates on resonance as opposed to the detuned Ramsey.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubits frequency and T2_ramsey in the state.
    - Save the current state
"""
from dataclasses import asdict

# %% {Imports}
from qualibrate import QualibrationNode
from quam_libs.components import QuAM
from quam_libs.experiments.ramsey.analysis.fetch_dataset import fetch_dataset
from quam_libs.experiments.ramsey.analysis.fitting import fit_frequency_detuning_and_t2_decay
from quam_libs.experiments.ramsey.parameters import Parameters, get_idle_times_in_clock_cycles
from quam_libs.experiments.ramsey.plotting import plot_ramseys_data_with_fit
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.fit import fit_oscillation_decay_exp,oscillation_decay_exp
from quam_libs.macros import qua_declaration, readout_state
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm.qua import *
import matplotlib.pyplot as plt
import numpy as np


node = QualibrationNode(
    name="06_Ramsey",
    parameters=Parameters(
        qubits= ['q0'],
        num_averages=500,
        frequency_detuning_in_mhz=0.5,
        min_wait_time_in_ns=16,
        max_wait_time_in_ns=20000,
        num_time_points=500,
        log_or_linear_sweep="log",
        use_state_discrimination=False,
        flux_point_joint_or_independent="joint",
        multiplexed=False,
        simulate=False,
        use_waveform_report=False
    )
)


# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load()

qubits = machine.get_qubits_used_in_node(node.parameters)
num_qubits = len(qubits)

config = machine.generate_config()
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages
idle_times = get_idle_times_in_clock_cycles(node.parameters)
detuning = node.parameters.frequency_detuning_in_mhz * u.MHz
flux_point = node.parameters.flux_point_joint_or_independent

detuning_signs = [-1, 1]

with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    idle_time = declare(int)
    detuning_sign = declare(int)
    virtual_detuning_phases = [declare(fixed) for _ in range(num_qubits)]

    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

    for multiplexed_qubits in qubits.batch():
        # todo: is this the right behaviour?
        for qubit in multiplexed_qubits.values():
            machine.set_all_fluxes(flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_each_(idle_time, idle_times):
                with for_(*from_array(detuning_sign, detuning_signs)):
                    for i, qubit in multiplexed_qubits.items():
                        with if_(detuning_sign == 1):
                            assign(virtual_detuning_phases[i], Cast.mul_fixed_by_int(detuning * 1e-9, 4 * idle_time))
                        with else_():
                            assign(virtual_detuning_phases[i], Cast.mul_fixed_by_int(-detuning * 1e-9, 4 * idle_time))

                    align()
                    with strict_timing_():
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x90")
                            qubit.xy.frame_rotation_2pi(virtual_detuning_phases[i])
                            qubit.xy.wait(idle_time)
                            qubit.xy.play("x90")

                    align()
                    for i, qubit in multiplexed_qubits.items():
                        if node.parameters.use_state_discrimination:
                            readout_state(qubit, state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        reset_frame(qubit.xy.name)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i] \
                    .buffer(len(detuning_signs)) \
                    .buffer(len(idle_times)) \
                    .average() \
                    .save(f"state{i + 1}")
            else:
                I_st[i] \
                    .buffer(len(detuning_signs)) \
                    .buffer(len(idle_times)) \
                    .average() \
                    .save(f"I{i + 1}")
                Q_st[i] \
                    .buffer(len(detuning_signs)) \
                    .buffer(len(idle_times)) \
                    .average() \
                    .save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    samples, fig = simulate_and_plot(qmm, config, ramsey, node.parameters)
    node.results = {"figure": fig}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(ramsey)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)


# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        ds = fetch_dataset(job, qubits, node.parameters)
        node.results = {"ds": ds}
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

# %% {Data_analysis}
    # Fit the Ramsey oscillations based on the qubit state or the 'I' quadrature
if node.parameters.use_state_discrimination:
    fit = fit_oscillation_decay_exp(ds.state, "time")
else:
    fit = fit_oscillation_decay_exp(ds.I, "time")
fit.attrs = {"long_name": "time", "units": "µs"}
fitted = oscillation_decay_exp(
    ds.time,
    fit.sel(fit_vals="a"),
    fit.sel(fit_vals="f"),
    fit.sel(fit_vals="phi"),
    fit.sel(fit_vals="offset"),
    fit.sel(fit_vals="decay"),
)
# TODO: add meaningful comments
frequency = fit.sel(fit_vals="f")
frequency.attrs = {"long_name": "frequency", "units": "MHz"}

decay = fit.sel(fit_vals="decay")
decay.attrs = {"long_name": "decay", "units": "nSec"}

frequency = frequency.where(frequency > 0, drop=True)

decay = fit.sel(fit_vals="decay")
decay.attrs = {"long_name": "decay", "units": "nSec"}

decay_res = fit.sel(fit_vals="decay_decay")
decay_res.attrs = {"long_name": "decay", "units": "nSec"}

tau = 1 / fit.sel(fit_vals="decay")
tau.attrs = {"long_name": "T2*", "units": "uSec"}

tau_error = tau * (np.sqrt(decay_res) / decay)
tau_error.attrs = {"long_name": "T2* error", "units": "uSec"}

within_detuning = (1e9 * frequency < 2 * detuning).mean(dim="sign") == 1
positive_shift = frequency.sel(sign=1) > frequency.sel(sign=-1)
freq_offset = (
    within_detuning * (frequency * fit.sign).mean(dim="sign")
    + ~within_detuning * positive_shift * frequency.mean(dim="sign")
    - ~within_detuning * ~positive_shift * frequency.mean(dim="sign")
)
decay = 1e-9 * tau.mean(dim="sign")
decay_error = 1e-9 * tau_error.mean(dim="sign")

# Save fitting results
fit_results = {
    q.name: {
        "freq_offset": 1e9 * freq_offset.loc[q.name].values,
        "decay": decay.loc[q.name].values,
        "decay_error": decay_error.loc[q.name].values,
    }
    for q in qubits
}
node.results["fit_results"] = fit_results
for q in qubits:
    print(
        f"Frequency offset for qubit {q.name} : {(fit_results[q.name]['freq_offset']/1e6):.2f} MHz "
    )
    print(f"T2* for qubit {q.name} : {1e6*fit_results[q.name]['decay']:.2f} us")

# %% {Plotting}
grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    if node.parameters.use_state_discrimination:
        ds.sel(sign=1).loc[qubit].state.plot(
            ax=ax, x="time", c="C0", marker=".", ms=5.0, ls="", label="$\Delta$ = +"
        )
        ds.sel(sign=-1).loc[qubit].state.plot(
            ax=ax, x="time", c="C1", marker=".", ms=5.0, ls="", label="$\Delta$ = -"
        )
        ax.plot(ds.time, fitted.loc[qubit].sel(sign=1), c="C0", ls="-", lw=1)
        ax.plot(ds.time, fitted.loc[qubit].sel(sign=-1), c="C1", ls="-", lw=1)
        ax.set_ylabel("State")
    else:
        (ds.sel(sign=1).loc[qubit].I * 1e3).plot(
            ax=ax, x="time", c="C0", marker=".", ms=5.0, ls="", label="$\Delta$ = +"
        )
        (ds.sel(sign=-1).loc[qubit].I * 1e3).plot(
            ax=ax, x="time", c="C1", marker=".", ms=5.0, ls="", label="$\Delta$ = -"
        )
        ax.set_ylabel("Trans. amp. I [mV]")
        ax.plot(ds.time, 1e3 * fitted.loc[qubit].sel(sign=1), c="C0", ls="-", lw=1)
        ax.plot(ds.time, 1e3 * fitted.loc[qubit].sel(sign=-1), c="C1", ls="-", lw=1)

    ax.set_xlabel("Idle time [nS]")
    ax.set_title(qubit["qubit"])
    ax.text(
        0.1,
        0.9,
        f'T2* = {1e6*fit_results[qubit["qubit"]]["decay"]:.1f} ± {1e6*fit_results[qubit["qubit"]]["decay_error"]:.1f} µs',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    ax.legend()
grid.fig.suptitle("Ramsey : I vs. idle time")
plt.tight_layout()
plt.show()
node.results["figure"] = grid.fig

# %% {Update_state}
if node.parameters.load_data_id is None:
    with node.record_state_updates():
        for q in qubits:
            q.xy.intermediate_frequency -= float(fit_results[q.name]["freq_offset"])
            q.T2ramsey = float(fit_results[qubit["qubit"]]["decay"])

# %% {Save_results}
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()

# %%
