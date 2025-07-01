# coding=utf-8
"""
Create date: 2025/6/21
0A_Single_Qubit_State_Tomography.py:
"""
# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters

from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.quantum_memory.marcos import MLE, project_to_cptp_1q
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q3"]
    num_averages: int = 60000
    operation: Literal["x180", "x90", "y90", "-x90","-y90"] = "x180"
    delay: List[int] = [0, 200, 1000, 5000, 20000]  # delay time must be larger than 4
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    mitigation: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2000
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False

node = QualibrationNode(name="AA_Single_Qubit_State_Tomography", parameters=Parameters())

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

if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

# %% {QUA_program}
operation = node.parameters.operation
if isinstance(node.parameters.delay, int):
    delay = [node.parameters.delay]
elif isinstance(node.parameters.delay, list):
    delay = []
    for d in node.parameters.delay:
        if isinstance(node.parameters.delay[0], int):
            delay.append(d)
else:
    raise TypeError("Delay must be a list of int or int")
n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
mitigation = node.parameters.mitigation

with program() as state_tomography:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    c = declare(int)  # QUA variable for switching between projections
    d = declare(int)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_each_(d, delay):
                with for_(c, 0, c<=2, c+1):
                    if reset_type == "active":
                        active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                    elif reset_type == "thermal":
                        qubit.wait(4 * qubit.thermalization_time * u.ns)
                    else:
                        raise ValueError(f"Unrecognized reset type {reset_type}.")

                    if operation is not None:
                        qubit.xy.play(operation)
                    with if_(d >= 4):
                        qubit.xy.wait(d)
                    qubit.align()
                    with switch_(c):
                        with case_(0):  # Project along X axis
                            qubit.xy.play("-y90")
                        with case_(1):  # Project along Y axis
                            qubit.xy.play("x90")
                        with case_(2):  # Project along Z axis
                            pass
                    readout_state(qubit, state[i])
                    qubit.align()
                    save(state[i], state_st[i])

            if not node.parameters.multiplexed:
                align()

    with stream_processing():
        n_st.save("iteration")
        for i in range(num_qubits):
            state_st[i].buffer(3).buffer(len(delay)).average().save(f"state{i+1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=100_000)  # in clock cycles
    job = qmm.simulate(config, state_tomography, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    node.results["figure"] = plt.gcf()
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    # Prepare data for saving
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        if not node.parameters.multiplexed:
            job = qm.execute(state_tomography)
        else:
            job = qm.execute(state_tomography)
        results = fetching_tool(job, ["iteration"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)
    ds = fetch_results_as_xarray(
        job.result_handles,
        qubits,
        {"proj": np.arange(3), "delay": delay},
    )
    node.results = {"ds": ds, "figs": {}, "results": {}}

    if mitigation:
        coef_MLE = []
        for i in range(len(delay)):
            p = []
            conf_mat = np.kron(np.array([[1]]), qubit.resonator.confusion_matrix)
            for j in range(3):
                pi = MLE(
                    np.array(
                        [1 - ds.state.values[0, i, j], ds.state.values[0, i, j]]
                    ),
                    conf_mat
                )
                p.append(pi[1])
            coef_MLE.append(p)
        coef_MLE = np.array(coef_MLE)

    ds.state.values = 1 - 2 * ds.state.values
    if mitigation:
        coef_MLE = 1 - 2 * coef_MLE

    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    density_matrix = np.empty((len(delay), 2, 2), dtype=complex)

    for i in range(len(delay)):
        density_matrix[i, :, :] = 0.5 * (
                I +
                ds.state.values[0, i, 0] * X +
                ds.state.values[0, i, 1] * Y +
                ds.state.values[0, i, 2] * Z
        )
    node.results["results"][qubits[0].name] = {}
    node.results["results"][qubits[0].name]["density_matrix"] = density_matrix

    if mitigation:
        density_matrix_mitigation = np.empty((len(delay), 2, 2), dtype=complex)
        for i in range(len(delay)):
            dmm = 0.5 * (
                I +
                coef_MLE[i, 0] * X +
                coef_MLE[i, 1] * Y +
                coef_MLE[i, 2] * Z
            )
            dmm, _ = project_to_cptp_1q([dmm])
            density_matrix_mitigation[i, :, :] = dmm
        node.results["results"][qubits[0].name]["density_matrix_mitigation"] = density_matrix_mitigation

    if __name__ == "__main__":
        import matplotlib

        font = {'weight': 'bold',
                'size': 25}
        matplotlib.rc('font', **font)

        vmin, vmax = -0.1, 1
        fig = plt.figure(figsize=(28, 10))
        ax = fig.subplots(2, 6)
        plt.suptitle(f"{qubits[0].name} Density matrix for different readout delay", fontweight='bold')

        ax[0, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
        ax[0, 0].text(0.5, 0.5, "Real", ha="center", va="center")
        ax[0, 0].axis("off")
        ax[1, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
        ax[1, 0].text(0.5, 0.5, "Imag", ha="center", va="center")
        ax[1, 0].axis("off")

        for i in range(1, 6):
            real = density_matrix[i - 1].real
            imag = density_matrix[i - 1].imag
            im = ax[0, i].imshow(real, vmin=vmin, vmax=vmax, cmap="rainbow")
            ax[0, i].set_title(f"{delay[i - 1]} ns")
            ax[0, i].set_xticks([0, 1])
            ax[0, i].set_yticks([0, 1])
            ax[1, i].imshow(imag, vmin=vmin, vmax=vmax, cmap="rainbow")
            ax[1, i].set_xticks([0, 1])
            ax[1, i].set_yticks([0, 1])
            for j in range(2):
                for k in range(2):
                    c = "k" if j == k else "w"
                    ax[0, i].text(j, k, f"{real[j, k]:.2f}", ha="center", va="center", color=c)
            for j in range(2):
                for k in range(2):
                    c = "k" if j == k else "w"
                    ax[1, i].text(j, k, f"{imag[j, k]:.2f}", ha="center", va="center", color=c)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.show()

        node.results["figs"] = fig

        if mitigation:
            fig_mitigation = plt.figure(figsize=(28, 10))
            ax_mitigation = fig_mitigation.subplots(2, 6)
            plt.suptitle(f"{qubits[0].name} Density matrix for different readout delay with Readout Mitigation", fontweight='bold')

            ax_mitigation[0, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
            ax_mitigation[0, 0].text(0.5, 0.5, "Real", ha="center", va="center")
            ax_mitigation[0, 0].axis("off")
            ax_mitigation[1, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
            ax_mitigation[1, 0].text(0.5, 0.5, "Imag", ha="center", va="center")
            ax_mitigation[1, 0].axis("off")

            for i in range(1, 6):
                real = density_matrix_mitigation[i - 1].real
                imag = density_matrix_mitigation[i - 1].imag
                im_mitigation = ax_mitigation[0, i].imshow(real, vmin=vmin, vmax=vmax, cmap="rainbow")
                ax_mitigation[0, i].set_title(f"{delay[i - 1]} ns")
                ax_mitigation[0, i].set_xticks([0, 1])
                ax_mitigation[0, i].set_yticks([0, 1])
                ax_mitigation[1, i].imshow(imag, vmin=vmin, vmax=vmax, cmap="rainbow")
                ax_mitigation[1, i].set_xticks([0, 1])
                ax_mitigation[1, i].set_yticks([0, 1])
                for j in range(2):
                    for k in range(2):
                        c = "k" if j == k else "w"
                        ax_mitigation[0, i].text(j, k, f"{real[j, k]:.2f}", ha="center", va="center", color=c)
                for j in range(2):
                    for k in range(2):
                        c = "k" if j == k else "w"
                        ax_mitigation[1, i].text(j, k, f"{imag[j, k]:.2f}", ha="center", va="center", color=c)

            fig_mitigation.subplots_adjust(right=0.8)
            cbar_ax_mitigation = fig_mitigation.add_axes([0.82, 0.15, 0.02, 0.7])
            fig_mitigation.colorbar(im_mitigation, cax=cbar_ax_mitigation)

            plt.show()

            node.results["mitigation_figs"] = fig_mitigation

        # %% {Save_results}
        if not node.parameters.simulate:
            node.outcomes = {q.name: "successful" for q in qubits}
            node.results["initial_parameters"] = node.parameters.model_dump()
            node.machine = machine
            node.save()
