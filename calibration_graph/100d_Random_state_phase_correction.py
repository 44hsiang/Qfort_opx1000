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
from qualang_tools.loops import from_array
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


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q0','q2']
    num_runs: int = 10000
    min_wait_time_in_ns: int = 16
    max_time_in_ns: int = 300
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    desired_state: Optional[List[float]] = [np.pi/2,0] #theta,phi

#theta,phi = random_bloch_state_uniform()

node = QualibrationNode(name="100d_Random_state_phase_correction", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
path = "/Users/4hsiang/Desktop/Jack/python_project/instrument_control/opx1000/qua-libs/Quantum-Control-Applications-QuAM/Superconducting/configuration/quam_state"
machine = QuAM.load(path)

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
#theta,phi = random_bloch_state_uniform()
theta,phi = node.parameters.desired_state[0],node.parameters.desired_state[1]
times_cycles = np.arange(node.parameters.min_wait_time_in_ns//4, node.parameters.max_time_in_ns // 4)

def QuantumMemory_program(qubit,theta=theta,phi=phi):
    with program() as QuantumMemory:
        t = declare(int)
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = [declare(int) for _ in range(1)]
        state_st = [declare_stream() for _ in range(1)]
        tomo_axis = declare(int)

        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)
            with for_(*from_array(t,times_cycles)):
                with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):

                    if reset_type == "active":
                        active_reset(qubit, "readout",max_attempts=15,wait_time=500)
                    elif reset_type == "thermal":
                        qubit.wait(4 * qubit.thermalization_time * u.ns)
                    else:
                        raise ValueError(f"Unrecognized reset type {reset_type}.")

                    qubit.align()
                    qubit.xy.play("y180",amplitude_scale = theta/np.pi)
                    qubit.xy.frame_rotation_2pi(phi/np.pi/2-0.5)
                    qubit.align()
                    qubit.z.play(
                    "const",
                    amplitude_scale=0,
                    duration=t,
                    )                          
                    align()

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
            state_st[0].buffer(3).buffer(len(times_cycles)).average().save("state1")
        return QuantumMemory



# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, QuantumMemory_program(qubit), simulation_config)
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
    job_ = []
    for qubit in qubits:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(QuantumMemory_program(qubit))
            for i in range(num_qubits):
                results = fetching_tool(job, ["n"], mode="live")
                print(f"Qubit {qubits[i].name} is running")
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
            ds_ = fetch_results_as_xarray(job_[i].result_handles, [qubits[i]], { "axis": ['x','y','z'],"timedelay": times_cycles*4})
            ds = xr.concat([ds, ds_], dim="qubit") if i != 0 else ds_
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    

    # %% {Data_analysis}
    node.results = {"ds": ds, "figs": {}, "results": {}}
    #ds["state_avg"] = ds["state"].mean(dim="N")
    ds["Bloch_vector"] = ((ds["state"].sel(axis='x'))**2+(ds["state"].sel(axis='y'))**2+(ds["state"].sel(axis='z'))**2 )**0.5 
    ds["Bloch_phi"] = np.degrees(np.arctan(ds["state"].sel(axis='y')/ds["state"].sel(axis='x')))
    results = QuantumStateAnalysis(Bloch_vector,[theta,phi])

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ds.sel(qubit=qubit['qubit']).Bloch_phi.plot.line(x='timedelay', ax=ax, marker='o', label='Bloch_vector')
        ax.set_ylabel('Bloch_phi(degrees)')

    grid = QubitGrid(ds, [q.grid_location for q in qubits],is_3d=True)
    for ax, qubit in grid_iter(grid):

        bloch = Bloch(axes=ax,font_size=12)
        bloch.add_vectors(ds.sel(qubit='q0').state.values[0])
        bloch.add_vectors(ds.sel(qubit='q0').state.values[-1])
        bloch.add_vectors([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        bloch.vector_color = ['r','b','g']
        bloch.vector_labels = ['raw','mitigated','ideal']
        bloch.render(title=qubit['qubit'])
    #%%
"""
    data={}
    mitigate_data = {}

    print(f"ideal Bloch vector: {theta} and {phi}")
    for q in qubits:
        x, y, z = (
            [np.bincount(ds.sel(qubit=q.name, axis='x',timedelay=times_cycles[i]).state.values).tolist() for i in range(len(times_cycles))],
            [np.bincount(ds.sel(qubit=q.name, axis='y',timedelay=times_cycles[i]).state.values).tolist() for i in range(len(times_cycles))],
            [np.bincount(ds.sel(qubit=q.name, axis='z',timedelay=times_cycles[i]).state.values).tolist() for i in range(len(times_cycles))],
        ) 
        data[q.name] ={}
        #check if the data is correct
        for i in range(len(times_cycles)): 
            if not len(x[i])==len(y[i])==len(z[i])==2:
                print("fail")
            Bloch_vector = [(1*x[i][0] - 1*x[i][1])/n_runs,(1*y[i][0] - 1*y[i][1])/n_runs,(1*z[i][0] - 1*z[i][1])/n_runs]
            data[q.name][i] = {'time_delay': times_cycles[i]*4,'Bloch vector':Bloch_vector}
            #data[q.name]={'x':x[i],'y':y[i],'z':z[i],'time_delay':times_cycles[i]*4,'Bloch vector':Bloch_vector}

        # construct denstiy matrix, fidelity and trace distance
        results = QuantumStateAnalysis(Bloch_vector,[theta,phi])
        print(q.name)
        print("raw data")
        print(f"Bloch vector: {Bloch_vector}")
        print(f"theta: {results.theta:.3} and phi: {results.phi:.3}")
        print(f"fidelity: {results.fidelity:.3} and trace distance: {results.trace_distance:.3}")
        data[q.name]['fidelity'] = results.fidelity
        data[q.name]['trace_distance'] = results.trace_distance        

        #mitigate the data
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            mitigator = CorrelatedReadoutMitigator(assignment_matrix=np.array(q.resonator.confusion_matrix).T,qubits=[0])
        mitigate_data[q.name] = {}
        for i in ['x','y','z']:
            mitigated_quasi_probs = mitigator.quasi_probabilities({'0':data[q.name][i][0],'1':data[q.name][i][1]})
            mitigated_probs = (mitigated_quasi_probs.nearest_probability_distribution().binary_probabilities())
            mitigate_data[q.name].update({i:mitigated_probs})
        converted_data = {k: [v.get('0', 0), v.get('1', 0)] for k, v in mitigate_data[q.name].items()}
        mitigate_data[q.name] = converted_data
        x, y, z = (mitigate_data[q.name]['x'],mitigate_data[q.name]['y'],mitigate_data[q.name]['z'])
        m_Bloch_vector = [(1*x[0] - 1*x[1]),(1*y[0] - 1*y[1]),(1*z[0] - 1*z[1])]

        # construct denstiy matrix, fidelity and trace distance
        m_results = QuantumStateAnalysis(m_Bloch_vector,[theta,phi])
        print(f"mitigated data")
        m_Bloch_vector = [round(i,4) for i in m_Bloch_vector]
        print(f"Mitigated Bloch vector: {m_Bloch_vector}")
        print(f"theta: {m_results.theta:.3} and phi: {m_results.phi:.3}")        
        print(f"fidelity: {m_results.fidelity:.3} and trace distance: {m_results.trace_distance:.3}")
        mitigate_data[q.name]['Bloch vector'] = m_results.bloch_vector
        mitigate_data[q.name]['fidelity'] = m_results.fidelity
        mitigate_data[q.name]['trace_distance'] = m_results.trace_distance
"""
    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits],is_3d=True)
    for ax, qubit in grid_iter(grid):
        bloch_vector = data[qubit['qubit']]['Bloch vector']
        mitigate_bloch_vector = mitigate_data[qubit['qubit']]['Bloch vector']
        ax.set_title(qubit['qubit'])
        ax.text(0.5,0.4,0.99,"raw",color='r',fontsize=8)
        ax.text(0.5,0.4,0.84,"mitigated",color='b',fontsize=8)
        ax.text(0.5,0.4,0.69,"ideal",color='g',fontsize=8)
        bloch = Bloch(axes=ax,font_size=12)
        bloch.add_vectors(bloch_vector)
        bloch.add_vectors(mitigate_bloch_vector)
        bloch.add_vectors([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
        bloch.vector_color = ['r','b','g']
        bloch.vector_labels = ['raw','mitigated','ideal']
        bloch.render(title=qubit['qubit'])

    grid.fig.suptitle(" Bloch Sphere")
    from matplotlib.lines import Line2D
    #legend_elements = [
    #    Line2D([0], [0], color='red', lw=2, label='raw'),
    #    Line2D([0], [0], color='blue', lw=2, label='mitigated'),
    #    Line2D([0], [0], color='green', lw=2, label='ideal')
    #]
    #plt.legend(handles=legend_elements,loc='upper right', bbox_to_anchor=(2, 0.5))
    #plt.legend(handles=legend_elements,loc='upper right')

    plt.tight_layout()
    plt.show()
    node.results["figure_Bloch"] = grid.fig
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        label =['0','1']
        width = 0.3
        x = np.arange(len(label)) 
        x_bar = ax.bar(x-width,data[qubit['qubit']]['x'],width,label='x')
        y_bar = ax.bar(x,data[qubit['qubit']]['y'],width,label='y')
        z_bar = ax.bar(x+width,data[qubit['qubit']]['z'],width,label='z')

        ax.bar_label(x_bar, padding=3,fontsize=8)
        ax.bar_label(y_bar, padding=3,fontsize=8)  
        ax.bar_label(z_bar, padding=3,fontsize=8)
        ax.set_ylabel('Counts')
        ylim = [0,n_runs]
        ax.set_ylim(ylim)
        ax.set_title(qubit['qubit'])    
        ax.set_xticks(x , label)
        ax.legend()
    grid.fig.suptitle("Random state counts")
    plt.tight_layout()
    #plt.legend(loc='upper center', bbox_to_anchor=(-0.9, -0.05),ncol=3)
    plt.show()
    node.results["figure_counts"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        x = np.arange(len(label)) 
        x_mbar = ax.bar(x-width,mitigate_data[qubit['qubit']]['x'],width,label='x_m')
        y_mbar = ax.bar(x,mitigate_data[qubit['qubit']]['y'],width,label='y_m')
        z_mbar = ax.bar(x+width,mitigate_data[qubit['qubit']]['z'],width,label='z_m')

        ax.bar_label(x_mbar, padding=3,fmt="%.3f", fontsize=8)
        ax.bar_label(y_mbar, padding=3,fmt="%.3f", fontsize=8)  
        ax.bar_label(z_mbar, padding=3,fmt="%.3f", fontsize=8)
        ax.set_ylabel('Counts')
        ylim = [0,1.1]
        ax.set_ylim(ylim)
        ax.set_title(qubit['qubit'])    
        ax.set_xticks(x , label)
        ax.legend()
    grid.fig.suptitle("Random state mitigated counts")
    plt.tight_layout()
    #plt.legend(loc='upper center', bbox_to_anchor=(-0.9, -0.05),ncol=3)
    plt.show()
    node.results["figure_mitigated_counts"] = grid.fig

    # %% {Update_state}
    if node.parameters.reset_type_thermal_or_active == "active":
        for i,j in zip(machine.active_qubit_names,"abcde"):
            machine.qubits[i].xy.thread = j
            machine.qubits[i].resonator.thread = j

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
    
# %%
