# coding=utf-8
"""
Create date: 2025/6/22
mitigation.py:
"""
from quam_libs.components import QuAM
from qualang_tools.units import unit
import numpy as np
from scipy.optimize import minimize
from quam_libs.quantum_memory.marcos import project_to_cptp_1q, MLE, density_matrix_to_bloch_vector

def pauli_expansion_single_qubit(rho: np.ndarray) -> np.ndarray:
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    P = [X, Y, Z]
    vec = []
    for i in range(3):
        coef = np.trace(P[i] @ rho)
        vec.append(coef)

    return np.array(vec)

u = unit(coerce_to_integer=True)
machine = QuAM.load("./state_20250624.json")
config = machine.generate_config()

qubit = machine.qubits["q3"]

m = np.array([
    [0.12168333 + 0.j, -0.04001667 + 0.02345j],
    [-0.04001667 - 0.02345j, 0.87831667 + 0.j]
])

axis_projection = density_matrix_to_bloch_vector(m)
coef = np.real(-0.5 * (axis_projection - 1))

conf_mat = np.array([[1]])
conf_mat = np.kron(conf_mat, qubit.resonator.confusion_matrix)

corr_MLE = np.array([
    MLE(np.array([1-coef[0],coef[0]]), conf_mat),
    MLE(np.array([1-coef[1],coef[1]]), conf_mat),
    MLE(np.array([1-coef[2],coef[2]]), conf_mat)
])

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

coef_MLE = 1 - 2 * (corr_MLE[:, 1])
dens_MLE = 0.5 * (I + coef_MLE[0] * X + coef_MLE[1] * Y + coef_MLE[2] * Z)

coef_MLE_norm = coef_MLE / np.linalg.norm(coef_MLE)
dens_MLE_norm = 0.5 * (I + coef_MLE_norm[0] * X + coef_MLE_norm[1] * Y + coef_MLE_norm[2] * Z)

dens_MLE_cptp, _ = project_to_cptp_1q([dens_MLE])

print(dens_MLE)
print(dens_MLE_norm)
print(dens_MLE_cptp)

