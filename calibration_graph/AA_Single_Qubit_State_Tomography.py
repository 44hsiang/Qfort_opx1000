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
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
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
    qubits: Optional[List[str]] = ["q3"]
    num_averages: int = 60000
    operation: str = "x180"
    delay: int = 5000  # delay time must be larger than 4
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
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
delay = node.parameters.delay
n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active

with program() as state_tomography:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit frequency
    c = declare(int)  # QUA variable for switching between projections
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(c, 0, c<=2, c+1):
                if reset_type == "active":
                    active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                elif reset_type == "thermal":
                    qubit.wait(4 * qubit.thermalization_time * u.ns)
                else:
                    raise ValueError(f"Unrecognized reset type {reset_type}.")

                if operation is not None:
                    qubit.xy.play(operation)
                if delay > 4:
                    qubit.xy.wait(delay * u.ns)
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
            state_st[i].buffer(3).average().save(f"state{i+1}")

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
    node.results = {}
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
        {"proj": np.arange(3)},
    )

    ds.state.values = 1 - 2 * ds.state.values

    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    density_matrix = 0.5 * (I + ds.state.values[0, 0] * X + ds.state.values[0, 1] * Y + ds.state.values[0, 2] * Z)
    print(density_matrix)