"""
    Quantum Memory
"""
# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import qua_declaration, active_reset,readout_state
from quam_libs.QI_function import *
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.analysis.discriminator import two_state_discriminator
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
from qiskit.result import CorrelatedReadoutMitigator
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch
from numpy.linalg import norm
from quam_libs.quantum_memory.marcos import *
from quam_libs.quantum_memory.NoiseAnalyze import *

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit: Literal["q0", "q1", "q2", "q3", "q4"] = ['q0']
    min_alpha: float = 0.05
    max_alpha: float = 0.05
    alpha_step: float = 0.01
    use_state_discrimination: bool = True
    use_strict_timing: bool = False
    num_runs: int = 10000
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    num_random_sequences: int = 2000  # Number of random sequences
    num_averages: int = 1
    max_circuit_depth: int = 1000  # Maximum circuit depth
    delta_clifford: int = 20
    seed: int = 1001
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    desired_state: Optional[List[List[float]]] = [[0,0],[np.pi,0],[np.pi/2,0],[np.pi/2,np.pi/2]]
    operation_name: str = "id"

#theta,phi = random_bloch_state_uniform()

node = QualibrationNode(name="110c_quantum_process_tomography_sweep_gateF", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

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
qubit = machine.qubits[node.parameters.qubit]

# %% {QUA_program}
min_alpha = node.parameters.min_alpha
max_alpha = node.parameters.max_alpha
alpha_step = node.parameters.alpha_step
alpha_list = np.arange(min_alpha, max_alpha+alpha_step, alpha_step)

n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

num_of_sequences = node.parameters.num_random_sequences  # Number of random sequences
# Number of averaging loops for each random sequence
n_avg = node.parameters.num_averages
max_circuit_depth = node.parameters.max_circuit_depth  # Maximum circuit depth
if node.parameters.delta_clifford < 1:
    raise NotImplementedError("Delta clifford < 2 is not supported.")
#  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
delta_clifford = node.parameters.delta_clifford
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
num_depths = max_circuit_depth // delta_clifford + 1
seed = node.parameters.seed  # Pseudo-random number generator seed
# Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
state_discrimination = node.parameters.use_state_discrimination
strict_timing = node.parameters.use_strict_timing
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

desired_state = node.parameters.desired_state
theta,phi = np.pi,0
delay_time = 4
operation_name = node.parameters.operation_name

def generate_sequence():
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=max_circuit_depth + 1)
    inv_gate = declare(int, size=max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < max_circuit_depth, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate

def play_sequence(sequence_list, depth, qubit: Transmon, alpha: float):
    i = declare(int)
    a = declare(float, value=alpha)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                qubit.xy.wait(qubit.xy.operations["x180"].length // 4)
            with case_(1):  # x180
                qubit.xy.play("x180", amplitude_scale=a)
            with case_(2):  # y180
                qubit.xy.play("y180", amplitude_scale=a)
            with case_(3):  # Z180
                qubit.xy.play("y180", amplitude_scale=a)
                qubit.xy.play("x180", amplitude_scale=a)
            with case_(4):  # Z90 X180 Z-180
                qubit.xy.play("x90", amplitude_scale=a)
                qubit.xy.play("y90", amplitude_scale=a)
            with case_(5):  # Z-90 Y-90 Z-90
                qubit.xy.play("x90", amplitude_scale=a)
                qubit.xy.play("-y90", amplitude_scale=a)
            with case_(6):  # Z-90 X180 Z-180
                qubit.xy.play("-x90", amplitude_scale=a)
                qubit.xy.play("y90", amplitude_scale=a)
            with case_(7):  # Z-90 Y90 Z-90
                qubit.xy.play("-x90", amplitude_scale=a)
                qubit.xy.play("-y90", amplitude_scale=a)
            with case_(8):  # X90 Z90
                qubit.xy.play("y90", amplitude_scale=a)
                qubit.xy.play("x90", amplitude_scale=a)
            with case_(9):  # X-90 Z-90
                qubit.xy.play("y90", amplitude_scale=a)
                qubit.xy.play("-x90", amplitude_scale=a)
            with case_(10):  # z90 X90 Z90
                qubit.xy.play("-y90", amplitude_scale=a)
                qubit.xy.play("x90", amplitude_scale=a)
            with case_(11):  # z90 X-90 Z90
                qubit.xy.play("-y90", amplitude_scale=a)
                qubit.xy.play("-x90", amplitude_scale=a)
            with case_(12):  # x90
                qubit.xy.play("x90", amplitude_scale=a)
            with case_(13):  # -x90
                qubit.xy.play("-x90", amplitude_scale=a)
            with case_(14):  # y90
                qubit.xy.play("y90", amplitude_scale=a)
            with case_(15):  # -y90
                qubit.xy.play("-y90", amplitude_scale=a)
            with case_(16):  # Z90
                qubit.xy.play("-x90", amplitude_scale=a)
                qubit.xy.play("y90", amplitude_scale=a)
                qubit.xy.play("x90", amplitude_scale=a)
            with case_(17):  # -Z90
                qubit.xy.play("-x90", amplitude_scale=a)
                qubit.xy.play("-y90", amplitude_scale=a)
                qubit.xy.play("x90", amplitude_scale=a)
            with case_(18):  # Y-90 Z-90
                qubit.xy.play("x180", amplitude_scale=a)
                qubit.xy.play("y90", amplitude_scale=a)
            with case_(19):  # Y90 Z90
                qubit.xy.play("x180", amplitude_scale=a)
                qubit.xy.play("-y90", amplitude_scale=a)
            with case_(20):  # Y90 Z-90
                qubit.xy.play("y180", amplitude_scale=a)
                qubit.xy.play("x90", amplitude_scale=a)
            with case_(21):  # Y-90 Z90
                qubit.xy.play("y180", amplitude_scale=a)
                qubit.xy.play("-x90", amplitude_scale=a)
            with case_(22):  # x90 Z-90
                qubit.xy.play("x90", amplitude_scale=a)
                qubit.xy.play("y90", amplitude_scale=a)
                qubit.xy.play("x90", amplitude_scale=a)
            with case_(23):  # -x90 Z90
                qubit.xy.play("-x90", amplitude_scale=a)
                qubit.xy.play("y90", amplitude_scale=a)
                qubit.xy.play("-x90", amplitude_scale=a)


def RandomizedBenchmark(alpha):
    with program() as randomized_benchmarking_individual:
        depth = declare(int)  # QUA variable for the varying depth
        # QUA variable for the current depth (changes in steps of delta_clifford)
        depth_target = declare(int)
        # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
        saved_gate = declare(int)
        m = declare(int)  # QUA variable for the loop over random sequences
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = declare(int)
        # The relevant streams
        m_st = declare_stream()
        # state_st = declare_stream()
        state_st = declare_stream()

        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        # QUA for_ loop over the random sequences
        with for_(m, 0, m < num_of_sequences, m + 1):
            # Generate the random sequence of length max_circuit_depth
            sequence_list, inv_gate_list = generate_sequence()
            assign(depth_target, 0)  # Initialize the current depth to 0

            with for_(depth, 1, depth <= max_circuit_depth, depth + 1):
                # Replacing the last gate in the sequence with the sequence's inverse gate
                # The original gate is saved in 'saved_gate' and is being restored at the end
                assign(saved_gate, sequence_list[depth])
                assign(sequence_list[depth], inv_gate_list[depth - 1])
                # Only played the depth corresponding to target_depth
                with if_((depth == 1) | (depth == depth_target)):
                    with for_(n, 0, n < n_avg, n + 1):
                        # Initialize the qubits
                        if reset_type == "active":
                            active_reset(qubit, "readout", max_attempts=15, wait_time=500)
                        else:
                            qubit.resonator.wait(qubit.thermalization_time * u.ns)
                        # Align the two elements to play the sequence after qubit initialization
                        qubit.align()
                        # The strict_timing ensures that the sequence will be played without gaps
                        if strict_timing:
                            with strict_timing_():
                                # Play the random sequence of desired depth
                                play_sequence(sequence_list, depth, qubit, alpha)
                        else:
                            play_sequence(sequence_list, depth, qubit, alpha)
                        # Align the two elements to measure after playing the circuit.
                        qubit.align()
                        readout_state(qubit, state)

                        save(state, state_st)

                    # Go to the next depth
                    assign(depth_target, depth_target + delta_clifford)
                # Reset the last gate of the sequence back to the original Clifford gate
                # (that was replaced by the recovery gate at the beginning)
                assign(sequence_list[depth], saved_gate)
            # Save the counter for the progress bar
            save(m, m_st)

        with stream_processing():
            m_st.save("n")
            state_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_of_sequences).save(
                "state"
            )
    return randomized_benchmarking_individual

def Process_tomography(alpha):
    with program() as ProcessTomography:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = declare(int)
        state_st = declare_stream()
        tomo_axis = declare(int)
        initial_state = declare(int)
        a = declare(fixed, value=alpha)

        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)
            with for_(initial_state, 0, initial_state < 4, initial_state + 1):
    
                with if_(initial_state == 0):
                    #|0>
                    with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):
                        active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                        qubit.align()
                        wait(10)
                        #initial state |0|
                        if operation_name != "id": qubit.xy.play(operation_name, amplitude_scale=a)
                        qubit.align()
                        wait(delay_time)
                        #tomography pulses
                        with if_(tomo_axis == 0):
                            qubit.xy.play("-y90")
                        with elif_(tomo_axis == 1):
                            qubit.xy.play("x90")
                        align()
                        readout_state(qubit, state)
                        save(state, state_st)
                    #wait(1000)
                with if_(initial_state == 1):
                    #|1>
                    with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):
                        active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                        qubit.align()
                        wait(10)
                        #initial state |1>
                        qubit.xy.play("y180", amplitude_scale=a)
                        qubit.align()
                        wait(10) 
                        #
                        if operation_name != "id": qubit.xy.play(operation_name, amplitude_scale=a)
                        #
                        qubit.align() 
                        wait(delay_time)
                        #tomography pulses
                        with if_(tomo_axis == 0):
                            qubit.xy.play("-y90")
                        with elif_(tomo_axis == 1):
                            qubit.xy.play("x90")
                        align()
                        readout_state(qubit, state)
                        save(state, state_st)
                    #wait(1000)

                with if_(initial_state == 2):
                    #|+>
                    with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):
                        active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                        qubit.align()
                        wait(10)
                        #initial state |+>
                        qubit.xy.play("-y90", amplitude_scale=a)
                        #qubit.xy.frame_rotation_2pi((0-qubit.extras["phi_offset"])/np.pi/2-0.5)
                        qubit.align()
                        wait(10)
                        #
                        if operation_name != "id": qubit.xy.play(operation_name, amplitude_scale=a)
                        #
                        qubit.align()
                        wait(delay_time)
                        #tomography pulses
                        with if_(tomo_axis == 0):
                            qubit.xy.play("-y90")
                        with elif_(tomo_axis == 1):
                            qubit.xy.play("x90")
                        align()
                        readout_state(qubit, state)
                        save(state, state_st)
                    #wait(1000)

                with if_(initial_state == 3):
                    #i|+>
                    with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):
                        active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                        qubit.align()
                        #initial state i|+>
                        qubit.xy.play("-x90", amplitude_scale=a)
                        #qubit.xy.frame_rotation_2pi((np.pi/2-qubit.extras["phi_offset"])/np.pi/2-0.5)                        #
                        qubit.align()
                        wait(10)
                        #
                        if operation_name != "id": qubit.xy.play(operation_name, amplitude_scale=a)
                        #
                        qubit.align()
                        wait(delay_time)
                        #tomography pulses
                        with if_(tomo_axis == 0):
                            qubit.xy.play("-y90")
                        with elif_(tomo_axis == 1):
                            qubit.xy.play("x90")
                        align()
                        readout_state(qubit, state)
                        save(state, state_st)
                    #wait(1000)

        with stream_processing():
            n_st.save("n")
            state_st.buffer(3).buffer(4).save_all("state")

        return QuantumMemory



# %% {Simulate_or_execute}
if node.parameters.simulate:
    pass
    
elif node.parameters.load_data_id is None:
    _job_RB = []
    _job_QPT = []
    for a in alpha_list:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job_RB = qm.execute(RandomizedBenchmark(a))
            results_RB = fetching_tool(job_RB, ["n"], mode="live")
            while results_RB.is_processing():
                n = results_RB.fetch_all()[0]
                progress_counter(n, num_of_sequences, start_time=results_RB.start_time)
            _job_RB.append(job_RB)

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job_QPT = qm.execute(Process_tomography(a))
            results_QPT = fetching_tool(job_QPT, ["n"], mode="live")
            while results_QPT.is_processing():
                m = results_QPT.fetch_all()[0]
                progress_counter(m, n_runs, start_time=results_QPT.start_time)
            _job_QPT.append(job_QPT)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:

    if node.parameters.load_data_id is None:
        depths = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
        depths[0] = 1
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        for i in range(len(alpha_list)):
            ds_RB_ = fetch_results_as_xarray(
                _job_RB[i].result_handles,
                [qubit],
                {"depths": depths, "sequence": np.arange(num_of_sequences)}
            )
            ds_RB = xr.concat([ds_RB, ds_RB_], dim="alpha") if i != 0 else ds_RB_
        for i in range(len(alpha_list)):
            ds_QPT_ = fetch_results_as_xarray(
                _job_QPT[i].result_handles,
                [qubit],
                {"N": np.linspace(1, n_runs, n_runs)}
            )
            ds_QPT = xr.concat([ds_QPT, ds_QPT_], dim="alpha") if i != 0 else ds_QPT_
        extract_state = ds_QPT.state.values['value']
        ds_QPT = ds_QPT.assign_coords(alpha=("alpha", alpha_list))
        ds_QPT = ds_QPT.assign_coords(axis=("axis", ['x', 'y', 'z']))
        ds_QPT = ds_QPT.assign_coords(initial_state=("initial_state", ['0', '1', '+', 'i+']))
        ds_QPT['state'] = (["qubit","N", "alpha",'initial_state', "axis"], extract_state)
    else:
        pass
    
    # %% {Data_analysis}
    # TODO use the ds['Bloch_vector'] to make it simply
    alpha_list = np.array(alpha_list)
    node.results = {"alphas": alpha_list, "ds_RB": ds_RB, "ds_QPT": ds_QPT, "figs": {}, "results": {}}

    print(f"ideal Bloch vector: {np.rad2deg(theta):.3} and {np.rad2deg(phi):.3} in degree")
    data, mitigate_data = {}, {}

    data[qubit.name] = {}
    mitigate_data[qubit.name] = {}

    # Build the readout mitigator once per qubit
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        mitigator = CorrelatedReadoutMitigator(
            assignment_matrix=np.array(qubit.resonator.confusion_matrix).T,
            qubits=[0],
        )
    confusion_matrix = np.array(qubit.resonator.confusion_matrix).T
    desired_state_name = ['0', '1', '+', 'i+']

    # Create raw and mitigated results for each prepared initial state
    for idx, initial_state in enumerate(ds_QPT.initial_state.values):
        blochs = np.array([])
        rhos = np.array([])
        fidelities = np.array([])
        traces = np.array([])

        mitigate_blochs = np.array([])
        mitigate_rhos = np.array([])
        mitigate_fidelities = np.array([])
        mitigate_traces = np.array([])

        theta, phi = desired_state[idx]
        ds_ = ds_QPT.sel(initial_state=initial_state)

        for a in alpha_list:
            # Raw counts (ensure both outcomes exist)
            x_count = np.bincount(ds_.sel(alpha=a, axis='x').state.values, minlength=2)
            y_count = np.bincount(ds_.sel(alpha=a, axis='y').state.values, minlength=2)
            z_count = np.bincount(ds_.sel(alpha=a, axis='z').state.values, minlength=2)

            # Raw Bloch vector (from counts)
            bloch = [
                (x_count[0] - x_count[1]) / n_runs,
                (y_count[0] - y_count[1]) / n_runs,
                (z_count[0] - z_count[1]) / n_runs,
            ]
            res = QuantumStateAnalysis(bloch, [theta, phi])

            blochs = np.append(blochs, bloch)
            rhos = np.append(rhos, res.density_matrix()[0])
            fidelities = np.append(fidelities, res.fidelity)
            traces = np.append(traces, res.trace_distance)

            # Mitigate the readout for each measurement axis
            new_px_0 = np.array([MLE([x_count[0]/n_runs,x_count[1]/n_runs],confusion_matrix)[0]])
            new_py_0 = np.array([MLE([y_count[0]/n_runs,y_count[1]/n_runs],confusion_matrix)[0]])
            new_pz_0 = np.array([MLE([z_count[0]/n_runs,z_count[1]/n_runs],confusion_matrix)[0]])

            # Mitigated Bloch vector (already probabilities)
            m_bloch = np.array([2*new_px_0-1,2*new_py_0-1,2*new_pz_0-1], dtype=float).ravel()
            if np.linalg.norm(m_bloch) > 1:
                m_bloch = m_bloch/np.linalg.norm(m_bloch)
            m_res = QuantumStateAnalysis(m_bloch, [theta, phi])

            mitigate_blochs = np.append(mitigate_blochs, m_bloch)
            mitigate_rhos = np.append(mitigate_rhos, m_res.density_matrix()[0])
            mitigate_fidelities = np.append(mitigate_fidelities, m_res.fidelity)
            mitigate_traces = np.append(mitigate_traces, m_res.trace_distance)

        data[qubit.name][initial_state] = {
            'Bloch vector': blochs,
            'density matrix': rhos,
            'fidelity': fidelities,
            'trace_distance': traces,
        }

        mitigate_data[qubit.name][initial_state] = {
            'Bloch vector': mitigate_blochs,
            'density matrix': mitigate_rhos,
            'fidelity': mitigate_fidelities,
            'trace_distance': mitigate_traces,
        }
        # Brief per-state summary
        # print(qubit.name)
        # print("raw (mitigate) data")
        # print(f"Bloch vector: [{blochs[0][0]}({mitigate_blochs[0][0]}), {blochs[0][1]}({mitigate_blochs[0[1]}), {bloch[2]}({m_bloch[2]})]")
        # print(
        #     f"theta, phi: {np.rad2deg(res.theta):.3} ({np.rad2deg(m_res.theta):.3}), "
        #     f"{np.rad2deg(res.phi):.3} ({np.rad2deg(m_res.phi):.3})"
        # )
        # print(
        #     f"fidelity, trace distance: {res.fidelity:.3} ({m_res.fidelity:.3}), "
        #     f"{res.trace_distance:.3} ({m_res.trace_distance:.3})"
        # )

        # Pack per-qubit results for convenience
        node.results["results"][qubit.name] = {
            name: {
                'raw': data[qubit.name][name],
                'mitigated': mitigate_data[qubit.name][name],
            } for name in desired_state_name
        }

        # Build superoperators (raw and mitigated) for this qubit
        inputs = [rho_0, rho_1, rho_plus, rho_plus_i]
        outputs_raw = [
            data[qn]['0']['density matrix'],
            data[qn]['1']['density matrix'],
            data[qn]['+']['density matrix'],
            data[qn]['i+']['density matrix'],
        ]
        outputs_mit = [
            mitigate_data[qn]['0']['density matrix'],
            mitigate_data[qn]['1']['density matrix'],
            mitigate_data[qn]['+']['density matrix'],
            mitigate_data[qn]['i+']['density matrix'],
        ]

        ptm, m_ptm = build_pauli_transfer_matrix(inputs, outputs_raw), build_pauli_transfer_matrix(inputs, outputs_mit)
        superop, m_superop = ptm_to_superop(ptm), ptm_to_superop(m_ptm)
        choi, m_choi = superop_to_choi(superop, 2, 2)/2, superop_to_choi(m_superop, 2, 2)/2
        


        # Display result summaries
        np.set_printoptions(precision=3, suppress=True)
        print(f"operation name: {operation_name}")
        print("Process fidelity with respect to the target process:")
        print(f"raw: {process_fidelity(ptm, target=operation_name):.3}")
        print(f"mitigated: {process_fidelity(m_ptm, target=operation_name):.3}")

        # Save to results
        node.results["results"][qn]['quantum information'] = {
            'raw': {
                'ptm': ptm,
                'superoperator':superop,
                'choi': choi,
                'negativity':QuantumMemory.negativity(choi)*2,
                'Quantum Memory Robustness': QuantumMemory.memory_robustness(choi),
                'fidelity': process_fidelity(ptm, target=operation_name),
            },
            'mitigated': {
                'ptm': m_ptm,
                'superoperator': m_superop,
                'choi': m_choi,
                'negativity': QuantumMemory.negativity(m_choi)*2,
                'Quantum Memory Robustness': QuantumMemory.memory_robustness(m_choi),
                'fidelity': process_fidelity(m_ptm, target=operation_name),
            },
        }


 
    # %% {Plotting}
    for i,initial_state in enumerate(ds.initial_state.values):
        grid = QubitGrid(ds, [q.grid_location for q in qubits],is_3d=True)
        for ax, qubit in grid_iter(grid):
            theta,phi = desired_state[i][0],desired_state[i][1]
            bloch_vector = data[qubit['qubit']][initial_state]['Bloch vector']
            mitigate_bloch_vector = mitigate_data[qubit['qubit']][initial_state]['Bloch vector']
            ax.set_title(qubit['qubit'])
            ax.text(0.5,0.4,0.99,"raw",color='r',fontsize=8)
            ax.text(0.5,0.4,0.84,"mitigated",color='b',fontsize=8)
            ax.text(0.5,0.4,0.69,"initial state",color='g',fontsize=8)
            bloch = Bloch(axes=ax,font_size=12)
            bloch.add_vectors(bloch_vector)
            bloch.add_vectors(mitigate_bloch_vector)
            bloch.add_vectors([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
            bloch.vector_color = ['r','b','g']
            bloch.vector_labels = ['raw','mitigated','ideal']
            bloch.render(title=qubit['qubit']+'_'+initial_state)
        node.results[f"figs_{qubit['qubit']+'_'+initial_state}"] = grid.fig
    # %% {Update_state}
    if node.parameters.reset_type_thermal_or_active == "active":
        for i,j in zip(machine.active_qubit_names,"abcde"):
            machine.qubits[i].xy.core = j
            machine.qubits[i].resonator.core = j

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
    
# %%
