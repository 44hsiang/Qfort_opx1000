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


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q1']
    num_runs: int = 10000
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = 2162
    multiplexed: bool = False
    desired_state: Optional[List[List[float]]] = [[0,0],[np.pi,0],[np.pi/2,0],[np.pi/2,np.pi/2]]
    operation_name: str = "y90"

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
t=4
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
                        #initial state |0>

                        #
                        qubit.xy.play(operation_name)
                        #                        
                        qubit.align()
                        wait(10)
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
                        qubit.xy.play(operation_name)
                        #
                        qubit.align() 
                        wait(10)
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
                        qubit.xy.play(operation_name)
                        #
                        qubit.align()
                        wait(10)
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
                        qubit.xy.play(operation_name)
                        #
                        qubit.align()
                        wait(10)
                        #tomography pulses
                        with if_(tomo_axis == 0):
                            qubit.xy.play("y90")
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
        ds = ds.assign_coords(initial_state=("initial_state", ['|0>', '|1>', '|+>', 'i|+>']))
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

    for q in qubits:
        data[q.name]={}
        mitigate_data[q.name]={}
        for i,initial_state in enumerate(ds.initial_state.values):
            theta,phi = desired_state[i][0],desired_state[i][1]
            ds_ = ds.sel(initial_state=initial_state) 
            x, y, z = (
                np.bincount(ds_.sel(qubit=q.name, axis='x').state.values).tolist(),
                np.bincount(ds_.sel(qubit=q.name, axis='y').state.values).tolist(),
                np.bincount(ds_.sel(qubit=q.name, axis='z').state.values).tolist(),
            )
            Bloch_vector = [(1*x[0] - 1*x[1])/n_runs,(1*y[0] - 1*y[1])/n_runs,(1*z[0] - 1*z[1])/n_runs]
            results = QuantumStateAnalysis(Bloch_vector,[theta,phi])
            data[q.name][initial_state] = {
                    'x':x,'y':y,'z':z,
                    'Bloch vector':Bloch_vector,
                    'denstiy matrix':results.density_matrix()[0],
                    'fidelity':results.fidelity,
                    'trace_distance':results.trace_distance
                    }
            #mitigate the data
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                mitigator = CorrelatedReadoutMitigator(assignment_matrix=np.array(q.resonator.confusion_matrix).T,qubits=[0])
            mitigate_data[q.name][initial_state]={}
            for i in ['x','y','z']:
                mitigated_quasi_probs = mitigator.quasi_probabilities({'0':data[q.name][initial_state][i][0],'1':data[q.name][initial_state][i][1]})
                mitigated_probs = (mitigated_quasi_probs.nearest_probability_distribution().binary_probabilities())
                mitigate_data[q.name][initial_state].update({i:mitigated_probs})
            converted_data = {k: [v.get('0', 0), v.get('1', 0)] for k, v in mitigate_data[q.name][initial_state].items()}
            x, y, z = (converted_data['x'],converted_data['y'],converted_data['z'])
            m_Bloch_vector = [(1*x[0] - 1*x[1]),(1*y[0] - 1*y[1]),(1*z[0] - 1*z[1])]
            
            # construct denstiy matrix, fidelity and trace distance
            m_results = QuantumStateAnalysis(m_Bloch_vector,[theta,phi])
            m_Bloch_vector = [round(i,4) for i in m_Bloch_vector]
            mitigate_data[q.name][initial_state]={
                    'x':x,'y':y,'z':z,
                    'Bloch vector':m_Bloch_vector,
                    'denstiy matrix':m_results.density_matrix()[0],
                    'fidelity':m_results.fidelity,
                    'trace_distance':m_results.trace_distance
                    }   
            
            print(q.name)
            print("raw (mitigate) data")
            print(f"Bloch vector: [{Bloch_vector[0]}({m_Bloch_vector[0]}),{Bloch_vector[1]}({m_Bloch_vector[1]}),{Bloch_vector[2]}({m_Bloch_vector[2]})]")
            print(f"theta, phi: {np.rad2deg(results.theta):.3} ({np.rad2deg(m_results.theta):.3}), {np.rad2deg(results.phi):.3} ({np.rad2deg(m_results.phi):.3})")
            print(f"fidelity, trace distance: {results.fidelity:.3} ({m_results.fidelity:.3}), {results.trace_distance:.3} ({m_results.trace_distance:.3})")
            mitigate_data[q.name]['Bloch vector'] = m_results.bloch_vector
            mitigate_data[q.name]['fidelity'] = m_results.fidelity
            mitigate_data[q.name]['trace_distance'] = m_results.trace_distance
        desired_state_name = ['|0>', '|1>', '|+>', 'i|+>']
        node.results["results"][q.name] = {
            desired_state_name[i]:{
                'raw':data[q.name][desired_state_name[i]],
                'mitigated':mitigate_data[q.name][desired_state_name[i]],
            } for i in range(len(desired_state_name))
        }
    # %% process tomography

    def pauli_expansion_single_qubit(rho: np.ndarray) -> np.ndarray:
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j,  0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        P = [I, X, Y, Z]
        vec = []
        for i in range(4):
            coef = 1/2 * np.trace(rho @ P[i])
            vec.append(coef)

        return np.array(vec)

    def build_superoperator(rho_in: any, rho_out: any) -> np.ndarray:
        """
        Constructs the 4x4 superoperator S such that:
        vec(rho_out) = S @ vec(rho_in)

        rho_out/rho_in: list of 4 density matrices in Puali basis (2x2)
        """
        # Vectorize all input and output states
        R_in = np.column_stack([pauli_expansion_single_qubit(rho) for rho in rho_in])
        R_out = np.column_stack([pauli_expansion_single_qubit(rho) for rho in rho_out])
        # Solve S = R_out @ R_in^{-1}
        try:
            R_in_inv = np.linalg.inv(R_in)
        except np.linalg.LinAlgError:
            print("Warning: R_in is not invertible, using pseudo-inverse.")
            R_in_inv = np.linalg.pinv(R_in)

        S = R_out @ R_in_inv
        return S
    
    def process_fidelity(superop, target='id'):
        """
        Compute process fidelity with respect to a target quantum process.
        
        Parameters:
            superop: 4x4 numpy array (superoperator matrix in Pauli basis {I, X, Y, Z})
            target: str, either 'id' or 'x' (currently supports 'id' and 'x')
            
        Returns:
            fidelity (float): Process fidelity between the input superoperator and the target
        """
        # Normalize input superoperator
        superop = superop / norm(superop, 'fro')
        
        # Define target superoperator
        if target == 'id':
            target_superop = np.eye(4, dtype=complex)
        elif target == 'x180':
            target_superop = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1]
            ], dtype=complex)
        elif target == 'y180':
            target_superop = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]
            ], dtype=complex)        
        elif target == 'x90':
            target_superop = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1],
                [0, 0, 1, 0]
            ], dtype=complex)
        elif target == 'y90':
            target_superop = np.array([
                [1, 0, 0, 0],
                [0, 0, 0, -1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]
            ], dtype=complex)
        
        else:
            raise ValueError("Unsupported target process. Use 'id', 'x180', 'y180','x90', or 'y90'")
        
        target_superop = target_superop / norm(target_superop, 'fro')
        
        # Compute inner product (Hilbert-Schmidt)
        fidelity = np.abs(np.trace(np.conj(superop.T) @ target_superop))
        return fidelity

    # === Example ===
    # You can replace these with your own density matrices

    # Input density matrices
    rho_0 = np.array([[1, 0], [0, 0]])  # |0><0|
    rho_1 = np.array([[0, 0], [0, 1]])  # |1><1|
    rho_plus = 0.5 * np.array([[1, 1], [1, 1]])  # |+><+|
    rho_plus_i = 0.5 * np.array([[1, -1j], [1j, 1]])  # |+i><+i|
    for q in qubits:
        # Output density matrices (replace these with your actual measurements)
        rho_out_0 = data[q.name]['|0>']['denstiy matrix']
        rho_out_1 = data[q.name]['|1>']['denstiy matrix']
        rho_out_plus = data[q.name]['|+>']['denstiy matrix']
        rho_out_plus_i = data[q.name]['i|+>']['denstiy matrix']

        # Pack them into lists
        inputs = [rho_0, rho_1, rho_plus, rho_plus_i]
        outputs = [rho_out_0, rho_out_1, rho_out_plus, rho_out_plus_i]

        # Construct superoperator
        superoperator = build_superoperator(inputs, outputs)

        # Display result
        np.set_printoptions(precision=3, suppress=True)
        print(f"operation name: {operation_name}")
        print("Superoperator (4x4 matrix):")
        print(superoperator)

        rho_out_0 = mitigate_data[q.name]['|0>']['denstiy matrix']
        rho_out_1 = mitigate_data[q.name]['|1>']['denstiy matrix']
        rho_out_plus = mitigate_data[q.name]['|+>']['denstiy matrix']
        rho_out_plus_i = mitigate_data[q.name]['i|+>']['denstiy matrix']

        # Pack them into lists
        inputs = [rho_0, rho_1, rho_plus, rho_plus_i]
        outputs = [rho_out_0, rho_out_1, rho_out_plus, rho_out_plus_i]

        # Construct superoperator
        m_superoperator = build_superoperator(inputs, outputs)

        # Display result
        np.set_printoptions(precision=3, suppress=True)
        print("Mitigation Superoperator (4x4 matrix):")
        print(m_superoperator)

        print("Process fidelity with respect to the target process:")
        print(f"raw: {process_fidelity(superoperator, target=operation_name):.3}")
        print(f"mitigated: {process_fidelity(m_superoperator, target=operation_name):.3}")
    node.results["results"][q.name]['superoperator'] = superoperator
    node.results["results"][q.name]['mitigated superoperator'] = m_superoperator


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
        node.results["figs"][qubit['qubit']+'_'+initial_state] = grid.fig
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

def pauli_expansion_single_qubit(rho: np.ndarray) -> np.ndarray:
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j,  0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    P = [I, X, Y, Z]
    vec = []
    for i in range(4):
        coef = 1/2 * np.trace(rho @ P[i])
        vec.append(coef)

    return np.array(vec)

def build_superoperator(rho_in: any, rho_out: any) -> np.ndarray:
    """
    Constructs the 4x4 superoperator S such that:
    vec(rho_out) = S @ vec(rho_in)

    rho_out/rho_in: list of 4 density matrices in Puali basis (2x2)
    """
    # Vectorize all input and output states
    R_in = np.column_stack([pauli_expansion_single_qubit(rho) for rho in rho_in])
    R_out = np.column_stack([pauli_expansion_single_qubit(rho) for rho in rho_out])
    # Solve S = R_out @ R_in^{-1}
    try:
        R_in_inv = np.linalg.inv(R_in)
    except np.linalg.LinAlgError:
        print("Warning: R_in is not invertible, using pseudo-inverse.")
        R_in_inv = np.linalg.pinv(R_in)

    S = R_out @ R_in_inv
    return S

