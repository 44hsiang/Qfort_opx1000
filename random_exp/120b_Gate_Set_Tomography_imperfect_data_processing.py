"""
    Gate Set Tomography
    ====================
    This example demonstrates how to perform gate set tomography (GST) using the 
    `QualibrationNode` class from the `qualibrate` library. GST is a powerful technique 
    for characterizing quantum gates and their errors, providing a comprehensive 
    understanding of the quantum system's behavior.
"""
# %% {Imports}
import sys
from qutip import Qobj
from picos import Problem
from picos.expressions.variables import HermitianVariable
from picos.expressions.algebra import trace, partial_transpose
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM, Transmon
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.macros import qua_declaration, active_reset,readout_state
from quam_libs.QI_function import *
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import re
import pygsti
from pygsti.modelpacks import smq1Q_XYI as std

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = ['q0']
    alpha: float = 1
    max_circuit_depth_in_power: int = 4
    num_runs: int = 10000
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 5000
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False

node = QualibrationNode(name=f"120b_Gate_Set_Tomography_imperfect_data_processing", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# delete the thread when using active reset
if node.parameters.reset_type_thermal_or_active == "active":
    for i in machine.active_qubit_names:
        del machine.qubits[i].xy.thread
        del machine.qubits[i].resonator.thread

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

# id_list = np.arange(545, 616) # 545~615
# alpha_list = np.arange(0.9, 1.61, 0.01)

id_list = np.arange(545, 586) # 545~615
alpha_list = np.arange(0.9, 1.31, 0.01)

# id_list = np.arange(545, 547) # 545~615
# alpha_list = np.arange(0.9, 0.91, 0.01)
alpha_list = [round(a, 2) for a in alpha_list]

# %% {Data_fetching_and_dataset_creation}
max_circuit_length = [
    2**i for i in range(max(node.parameters.max_circuit_depth_in_power, 0) + 1)
    ]

std_model = std.target_model()
exp_design = pygsti.protocols.StandardGSTDesign(
    std_model, 
    std.prep_fiducials(), 
    std.meas_fiducials(), 
    std.germs(),
    max_circuit_length
)

# %% {Data_analysis}

def transform_dataset_to_gst(ds):
    gst_ds = pygsti.data.DataSet(outcome_labels=['0', '1'])
    for i, crc in enumerate(exp_design.all_circuits_needing_data):
        gst_ds.add_count_dict(crc, {'0': ds.count0.values[0, i], '1': ds.count1.values[0, i]})
    return gst_ds

def entanglementRobustness(state, solver='mosek', **extra_options) :
    if isinstance(state, Qobj):
        state = (state).full()
    SP = Problem()
    # add variable
    gamma = HermitianVariable("gamma", (4, 4))
    rho = HermitianVariable("rho", (4, 4))
    # add constraints
    SP.add_constraint(
        (state + gamma) - rho == 0
    )
    SP.add_constraint(
        partial_transpose(rho, 0) >> 0
    )
    SP.add_constraint(
        gamma >> 0
    )
    SP.add_constraint(
        trace(rho) - 1 >> 0
    )
    # find the solution
    SP.set_objective(
        'min',
        trace(rho) - 1
    )
    # solve the problem
    SP.solve(solver=solver, **extra_options)
    # return results
    return max(SP.value, 0)

result_dict_list = []
qubit = qubits[0]
for id in id_list:
    print(f"\nRun: {id:05d}")
    node = node.load_from_id(id)
    ds_ = node.results["ds"]
    ds = xr.concat([ds, ds_], dim="alpha") if id != id_list[0] else ds_

    optimizer_params={
        'maxiter': 1000,     # Increase maximum of iterations
        'tol': 1e-4,   # Set up tollerance
    }

    # GST analysis using pyGSTi
    gst_ds = transform_dataset_to_gst(ds_)
    gst_data = pygsti.protocols.ProtocolData(exp_design, gst_ds)
    gst_protocol = pygsti.protocols.StandardGST(optimizer=optimizer_params)
    gst_results = gst_protocol.run(gst_data)

    new_results = {}
    new_results = {
        "TP": {
            "rho0": {"density_mx": None, "fidelity": None},
            "meas_op": {"0": {"povm": None, "fidelity": None}, "1": {"povm": None, "fidelity": None}},
            "gate_op": {
                "I": {"choi": None, "fidelity": None, "robustness": None}, 
                "x90": {"choi": None, "fidelity": None, "robustness": None}, 
                "y90": {"choi": None, "fidelity": None, "robustness": None}, 
                }
            },
        "CPTP": {
            "rho0": {"density_mx": None, "fidelity": None},
            "meas_op": {"0": {"povm": None, "fidelity": None}, "1": {"povm": None, "fidelity": None}},
            "gate_op": {
                "I": {"choi": None, "fidelity": None, "robustness": None}, 
                "x90": {"choi": None, "fidelity": None, "robustness": None}, 
                "y90": {"choi": None, "fidelity": None, "robustness": None}, 
                }
            },
        "Ideal": {
            "rho0": {"density_mx": None, "fidelity": None},
            "meas_op": {"0": {"povm": None, "fidelity": None}, "1": {"povm": None, "fidelity": None}},
            "gate_op": {
                "I": {"choi": None, "fidelity": None, "robustness": None}, 
                "x90": {"choi": None, "fidelity": None, "robustness": None}, 
                "y90": {"choi": None, "fidelity": None, "robustness": None}, 
                }
            },
    }
    estimate_keys = ['full TP', 'CPTPLND', 'Target']
    native_gate_keys = [(), ('Gxpi2', 0), ('Gypi2', 0)]
    np.set_printoptions(suppress=True, precision=6)
    for i, cond in enumerate(new_results.keys()):
        est_model = gst_results.estimates[estimate_keys[i]].models['stdgaugeopt']

        rho_vec = est_model.preps['rho0']
        rho_est_mat = pygsti.tools.vec_to_stdmx(rho_vec, basis='pp')
        rho_std_mat = pygsti.tools.vec_to_stdmx(std_model.preps["rho0"], basis='pp')
        state_fidelity = pygsti.tools.fidelity(rho_est_mat, rho_std_mat)

        new_results[cond]["rho0"]["density_mx"] = rho_est_mat
        new_results[cond]["rho0"]["fidelity"] = state_fidelity

        povm_obj = est_model.povms['Mdefault']
        for label, effect_vec in povm_obj.items():
            matrix_est_form = pygsti.tools.vec_to_stdmx(effect_vec, basis='pp')
            # # Enforce matrix to be Hermitian
            # matrix_est_form = (matrix_est_form + matrix_est_form.conj().T) / 2.0
            matri_std_form = pygsti.tools.vec_to_stdmx(std_model.povms['Mdefault'][str(label)], basis='pp')
            meas_fidelity = pygsti.tools.fidelity(matrix_est_form, matri_std_form)
            
            new_results[cond]["meas_op"][str(label)]["povm"] = matrix_est_form
            new_results[cond]["meas_op"][str(label)]["fidelity"] = meas_fidelity

        for j, gate in enumerate(new_results[cond]["gate_op"].keys()):
            matrix_ptm = est_model.operations[native_gate_keys[j]].to_dense()
            choi = pygsti.tools.jamiolkowski.jamiolkowski_iso(
                matrix_ptm, op_mx_basis='pp', choi_mx_basis='std'
                )
            infidelity = pygsti.tools.entanglement_infidelity(matrix_ptm, std_model.operations[native_gate_keys[j]].to_dense(), 'pp')
            robustness = entanglementRobustness(choi)
            new_results[cond]["gate_op"][gate]["choi"] = choi
            new_results[cond]["gate_op"][gate]["fidelity"] = 1 - infidelity
            new_results[cond]["gate_op"][gate]["robustness"] = robustness

    result_dict_list.append(new_results.copy())

ds = ds.assign_coords(alpha=alpha_list)
node.results = {"ds": ds, "figs": {}, "results": None}

prep_fidelity = np.array([[d[cond]["rho0"]["fidelity"] for d in result_dict_list] for cond in ["TP", "CPTP", "Ideal"]])
prep_density_mx = np.array([[d[cond]["rho0"]["density_mx"] for d in result_dict_list] for cond in ["TP", "CPTP", "Ideal"]])
meas_fidelity = np.array([[[d[cond]["meas_op"][str(b)]["fidelity"] for d in result_dict_list] for b in [0, 1]] for cond in ["TP", "CPTP", "Ideal"]])
meas_operation_mx = np.array([[[d[cond]["meas_op"][str(b)]["povm"] for d in result_dict_list] for b in [0, 1]] for cond in ["TP", "CPTP", "Ideal"]])
gate_fidelity = np.array([[[d[cond]["gate_op"][g]["fidelity"] for d in result_dict_list] for g in ["I", "x90", "y90"]] for cond in ["TP", "CPTP", "Ideal"]])
gate_robustness = np.array([[[d[cond]["gate_op"][g]["robustness"] for d in result_dict_list] for g in ["I", "x90", "y90"]] for cond in ["TP", "CPTP", "Ideal"]])
gate_choi_state = np.array([[[d[cond]["gate_op"][g]["choi"] for d in result_dict_list] for g in ["I", "x90", "y90"]] for cond in ["TP", "CPTP", "Ideal"]])

ds_results = {
        "TP": {
            "rho0": {"fidelity": prep_fidelity[0], "density_mx": prep_density_mx[0]},
            "meas_op": {"0": {"fidelity": meas_fidelity[0, 0], "povm": meas_operation_mx[0, 0]}, "1": {"fidelity": meas_fidelity[0, 1], "povm": meas_operation_mx[0, 1]}},
            "gate_op": {
                "I": {"fidelity": gate_fidelity[0, 0], "robustness": gate_robustness[0, 0], "choi": gate_choi_state[0, 0]}, 
                "x90": {"fidelity": gate_fidelity[0, 1], "robustness": gate_robustness[0, 1], "choi": gate_choi_state[0, 1]}, 
                "y90": {"fidelity": gate_fidelity[0, 2], "robustness": gate_robustness[0, 2], "choi": gate_choi_state[0, 2]}, 
                }
            },
        "CPTP": {
            "rho0": {"fidelity": prep_fidelity[1], "density_mx": prep_density_mx[1]},
            "meas_op": {"0": {"fidelity": meas_fidelity[1, 0], "povm": meas_operation_mx[1, 0]}, "1": {"fidelity": meas_fidelity[1, 1], "povm": meas_operation_mx[1, 1]}},
            "gate_op": {
                "I": {"fidelity": gate_fidelity[1, 0], "robustness": gate_robustness[1, 0], "choi": gate_choi_state[1, 0]}, 
                "x90": {"fidelity": gate_fidelity[1, 1], "robustness": gate_robustness[1, 1], "choi": gate_choi_state[1, 1]}, 
                "y90": {"fidelity": gate_fidelity[1, 2], "robustness": gate_robustness[1, 2], "choi": gate_choi_state[1, 2]}, 
                }
            },
        "Ideal": {
            "rho0": {"fidelity": prep_fidelity[2], "density_mx": prep_density_mx[2]},
            "meas_op": {"0": {"fidelity": meas_fidelity[2, 0], "povm": meas_operation_mx[2, 0]}, "1": {"fidelity": meas_fidelity[2, 1], "povm": meas_operation_mx[2, 1]}},
            "gate_op": {
                "I": {"fidelity": gate_fidelity[2, 0], "robustness": gate_robustness[2, 0], "choi": gate_choi_state[2, 0]}, 
                "x90": {"fidelity": gate_fidelity[2, 1], "robustness": gate_robustness[2, 1], "choi": gate_choi_state[2, 1]}, 
                "y90": {"fidelity": gate_fidelity[2, 2], "robustness": gate_robustness[2, 2], "choi": gate_choi_state[2, 2]}, 
                }
            },
    }

# %% {Plotting}
grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    ax.plot(alpha_list, prep_fidelity.T, label = ["TP", "CPTP", "Ideal"])
    ax.set_ylabel("fidelity")
    ax.set_xlabel(r"Gate error($\alpha$)")
    ax.set_title(qubit["qubit"])
    ax.legend()
grid.fig.suptitle("Prepration Fidelity")
plt.tight_layout()
plt.show()
node.results["figure_prep"] = grid.fig

grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    ax.plot(alpha_list, meas_fidelity[:, 0].T, label = ["TP", "CPTP", "Ideal"])
    ax.set_ylabel("fidelity")
    ax.set_xlabel(r"Gate error($\alpha$)")
    ax.set_title(qubit["qubit"])
    ax.legend()
grid.fig.suptitle("Measurement |0> Fidelity")
plt.tight_layout()
plt.show()
node.results["figure_meas_0"] = grid.fig

grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    ax.plot(alpha_list, meas_fidelity[:, 1].T, label = ["TP", "CPTP", "Ideal"])
    ax.set_ylabel("fidelity")
    ax.set_xlabel(r"Gate error($\alpha$)")
    ax.set_title(qubit["qubit"])
    ax.legend()
grid.fig.suptitle("Measurement |1> Fidelity")
plt.tight_layout()
plt.show()
node.results["figure_meas_1"] = grid.fig

grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    ax.plot(alpha_list, gate_fidelity[:, 0].T, label = ["TP", "CPTP", "Ideal"])
    ax.set_ylabel("fidelity")
    ax.set_xlabel(r"Gate error($\alpha$)")
    ax.set_title(qubit["qubit"])
    ax.legend()
grid.fig.suptitle("Gate 'I' Fidelity")
plt.tight_layout()
plt.show()
node.results["figure_gate_I"] = grid.fig

grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    ax.plot(alpha_list, gate_fidelity[:, 1].T, label = ["TP", "CPTP", "Ideal"])
    ax.set_ylabel("fidelity")
    ax.set_xlabel(r"Gate error($\alpha$)")
    ax.set_title(qubit["qubit"])
    ax.legend()
grid.fig.suptitle("Gate 'x90' Fidelity")
plt.tight_layout()
plt.show()
node.results["figure_gate_x90"] = grid.fig

grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    ax.plot(alpha_list, gate_fidelity[:, 2].T, label = ["TP", "CPTP", "Ideal"])
    ax.set_ylabel("fidelity")
    ax.set_xlabel(r"Gate error($\alpha$)")
    ax.set_title(qubit["qubit"])
    ax.legend()
grid.fig.suptitle("Gate 'y90' Fidelity")
plt.tight_layout()
plt.show()
node.results["figure_gate_y90"] = grid.fig

grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    ax.plot(alpha_list, gate_robustness[:, 0].T, label = ["TP", "CPTP", "Ideal"])
    ax.set_ylabel("fidelity")
    ax.set_xlabel(r"Gate error($\alpha$)")
    ax.set_title(qubit["qubit"])
    ax.legend()
grid.fig.suptitle("Gate 'I' Robustness")
plt.tight_layout()
plt.show()
node.results["figure_gate_I_robust"] = grid.fig

grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    ax.plot(alpha_list, gate_robustness[:, 1].T, label = ["TP", "CPTP", "Ideal"])
    ax.set_ylabel("fidelity")
    ax.set_xlabel(r"Gate error($\alpha$)")
    ax.set_title(qubit["qubit"])
    ax.legend()
grid.fig.suptitle("Gate 'x90' Robustness")
plt.tight_layout()
plt.show()
node.results["figure_gate_x90_robust"] = grid.fig

grid = QubitGrid(ds, [q.grid_location for q in qubits])
for ax, qubit in grid_iter(grid):
    ax.plot(alpha_list, gate_robustness[:, 2].T, label = ["TP", "CPTP", "Ideal"])
    ax.set_ylabel("fidelity")
    ax.set_xlabel(r"Gate error($\alpha$)")
    ax.set_title(qubit["qubit"])
    ax.legend()
grid.fig.suptitle("Gate 'y90' Robustness")
plt.tight_layout()
plt.show()
node.results["figure_gate_y90_robust"] = grid.fig

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
