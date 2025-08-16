"""
    Quantum Memory
"""
# %% {Imports}
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
from qiskit.result import CorrelatedReadoutMitigator
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch
from numpy.linalg import norm
from quam_libs.quantum_memory.marcos import *
from quam_libs.quantum_memory.NoiseAnalyze import *

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
    desired_state: Optional[List[List[float]]] = [[0,0],[np.pi,0],[np.pi/2,0],[np.pi/2,np.pi/2]]
    operation_name: str = "id"

#theta,phi = random_bloch_state_uniform()

node = QualibrationNode(name="110a_quantum_process_tomography", parameters=Parameters())


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
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

# %% {QUA_program}
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

desired_state = node.parameters.desired_state
theta,phi = np.pi,0
delay_time = 4
operation_name = node.parameters.operation_name
def QuantumMemory_program(qubit):
    with program() as QuantumMemory:
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = [declare(int) for _ in range(1)]
        state_st = [declare_stream() for _ in range(1)]
        tomo_axis = declare(int)
        initial_state = declare(int)

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
                        if operation_name != "id": qubit.xy.play(operation_name)                   
                        qubit.align()
                        wait(delay_time)
                        #tomography pulses
                        with if_(tomo_axis == 0):
                            qubit.xy.play("y90")
                        with elif_(tomo_axis == 1):
                            qubit.xy.play("x90")
                        align()
                        readout_state(qubit, state[0])
                        save(state[0], state_st[0])
                    #wait(1000)
                with if_(initial_state == 1):
                    #|1>
                    with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):
                        active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                        qubit.align()
                        wait(10)
                        #initial state |1>
                        qubit.xy.play("y180")
                        qubit.align()
                        wait(10) 
                        #
                        if operation_name != "id": qubit.xy.play(operation_name)                   
                        #
                        qubit.align() 
                        wait(delay_time)
                        #tomography pulses
                        with if_(tomo_axis == 0):
                            qubit.xy.play("y90")
                        with elif_(tomo_axis == 1):
                            qubit.xy.play("x90")
                        align()
                        readout_state(qubit, state[0])
                        save(state[0], state_st[0])
                    #wait(1000)

                with if_(initial_state == 2):
                    #|+>
                    with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):
                        active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                        qubit.align()
                        wait(10)
                        #initial state |+>
                        qubit.xy.play("-y90")
                        #qubit.xy.frame_rotation_2pi((0-qubit.extras["phi_offset"])/np.pi/2-0.5)
                        qubit.align()
                        wait(10)
                        #
                        if operation_name != "id": qubit.xy.play(operation_name)                   
                        #
                        qubit.align()
                        wait(delay_time)
                        #tomography pulses
                        with if_(tomo_axis == 0):
                            qubit.xy.play("y90")
                        with elif_(tomo_axis == 1):
                            qubit.xy.play("x90")
                        align()
                        readout_state(qubit, state[0])
                        save(state[0], state_st[0])
                    #wait(1000)

                with if_(initial_state == 3):
                    #i|+>
                    with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):
                        active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                        qubit.align()
                        #initial state i|+>
                        qubit.xy.play("-x90")
                        #qubit.xy.frame_rotation_2pi((np.pi/2-qubit.extras["phi_offset"])/np.pi/2-0.5)                        #
                        qubit.align()
                        wait(10)
                        #
                        if operation_name != "id": qubit.xy.play(operation_name)                   
                        #
                        qubit.align()
                        wait(delay_time)
                        #tomography pulses
                        with if_(tomo_axis == 0):
                            qubit.xy.play("-y90")
                        with elif_(tomo_axis == 1):
                            qubit.xy.play("x90")
                        align()
                        readout_state(qubit, state[0])
                        save(state[0], state_st[0])
                    #wait(1000)

        
        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

        with stream_processing():
            n_st.save("n")
            state_st[0].buffer(3).buffer(4).save_all("state1")

        return QuantumMemory



# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, QuantumMemory_program(qubits[0]), simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples,plot=True,save_path="./")
    
elif node.parameters.load_data_id is None:
    job_ = []
    for qubit in qubits:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(QuantumMemory_program(qubit))
            for i in range(num_qubits):
                results = fetching_tool(job, ["n"], mode="live")
                while results.is_processing():
                    n = results.fetch_all()[0]
                    progress_counter(n, n_runs, start_time=results.start_time)
            job_.append(job)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        #ds = fetch_results_as_xarray(job.result_handles, qubits, {"N": np.linspace(1, n_runs, n_runs)})
        for i in range(num_qubits):
            ds_ = fetch_results_as_xarray(job_[i].result_handles, [qubits[i]], {"N": np.linspace(1, n_runs, n_runs)})
            ds = xr.concat([ds, ds_], dim="qubit") if i != 0 else ds_
            #job_[0].result_handles.get('state1').fetch_all()
        extract_state = ds.state.values['value']
        ds = ds.assign_coords(axis=("axis", ['x', 'y', 'z']))
        ds = ds.assign_coords(initial_state=("initial_state", ['0', '1', '+', 'i+']))
        ds['state'] = (["qubit","N",'initial_state', "axis"], extract_state)
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    
    # %% {Data_analysis}
    # TODO use the ds['Bloch_vector'] to make it simply
    node.results = {"ds": ds, "figs": {}, "results": {}}
    data={}
    mitigate_data = {}
    print(f"ideal Bloch vector: {np.rad2deg(theta):.3} and {np.rad2deg(phi):.3} in degree")
    data, mitigate_data = {}, {}

    for q in qubits:
        qn = q.name
        data[qn] = {}
        mitigate_data[qn] = {}

        # Build the readout mitigator once per qubit
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            mitigator = CorrelatedReadoutMitigator(
                assignment_matrix=np.array(q.resonator.confusion_matrix).T,
                qubits=[0],
            )
        confusion_matrix = np.array(q.resonator.confusion_matrix).T
        desired_state_name = ['0', '1', '+', 'i+']

        # Create raw and mitigated results for each prepared initial state
        for idx, initial_state in enumerate(ds.initial_state.values):
            theta, phi = desired_state[idx]
            ds_ = ds.sel(initial_state=initial_state)

            # Raw counts (ensure both outcomes exist)
            x_count = np.bincount(ds_.sel(qubit=qn, axis='x').state.values, minlength=2)
            y_count = np.bincount(ds_.sel(qubit=qn, axis='y').state.values, minlength=2)
            z_count = np.bincount(ds_.sel(qubit=qn, axis='z').state.values, minlength=2)

            # Raw Bloch vector (from counts)
            bloch = [
                (x_count[0] - x_count[1]) / n_runs,
                (y_count[0] - y_count[1]) / n_runs,
                (z_count[0] - z_count[1]) / n_runs,
            ]
            res = QuantumStateAnalysis(bloch, [theta, phi])

            data[qn][initial_state] = {
                'Bloch vector': bloch,
                'density matrix': res.density_matrix()[0],
                'fidelity': res.fidelity,
                'trace_distance': res.trace_distance,
            }

            # Mitigate the readout for each measurement axis
            mit_axes = {}
            new_px_0 = np.array([MLE([x_count[0]/n_runs,x_count[1]/n_runs],confusion_matrix)[0]])
            new_py_0 = np.array([MLE([y_count[0]/n_runs,y_count[1]/n_runs],confusion_matrix)[0]])
            new_pz_0 = np.array([MLE([z_count[0]/n_runs,z_count[1]/n_runs],confusion_matrix)[0]])

            # Mitigated Bloch vector (already probabilities)
            m_bloch = np.array([2*new_px_0-1,2*new_py_0-1,2*new_pz_0-1], dtype=float).ravel()
            if np.linalg.norm(m_bloch) > 1:
                m_bloch = m_bloch/np.linalg.norm(m_bloch)
            m_res = QuantumStateAnalysis(m_bloch, [theta, phi])

            mitigate_data[qn][initial_state] = {
                'Bloch vector': m_bloch,
                'density matrix': m_res.density_matrix()[0],
                'fidelity': m_res.fidelity,
                'trace_distance': m_res.trace_distance,
            }

            # Brief per-state summary
            print(qn)
            print("raw (mitigate) data")
            print(f"Bloch vector: [{bloch[0]}({m_bloch[0]}), {bloch[1]}({m_bloch[1]}), {bloch[2]}({m_bloch[2]})]")
            print(
                f"theta, phi: {np.rad2deg(res.theta):.3} ({np.rad2deg(m_res.theta):.3}), "
                f"{np.rad2deg(res.phi):.3} ({np.rad2deg(m_res.phi):.3})"
            )
            print(
                f"fidelity, trace distance: {res.fidelity:.3} ({m_res.fidelity:.3}), "
                f"{res.trace_distance:.3} ({m_res.trace_distance:.3})"
            )

        # Pack per-qubit results for convenience
        node.results["results"][qn] = {
            name: {
                'raw': data[qn][name],
                'mitigated': mitigate_data[qn][name],
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
