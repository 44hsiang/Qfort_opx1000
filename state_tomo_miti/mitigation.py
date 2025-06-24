# coding=utf-8
"""
Create date: 2025/6/22
mitigation.py:
"""
from quam_libs.components import QuAM
from qualang_tools.units import unit
import numpy as np
from scipy.optimize import minimize


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

def MLE(original_P,confusion_matrix):
    """
    Maximum Likelihood Estimation of the true probabilities
    :param original_P: Original probabilities
    :param confusion_matrix: Confusion matrix
    :return: Estimated true probabilities
    """
    N_obs = original_P
    M = confusion_matrix
    def neg_log_likelihood(p_optimal):
        q_predict = M @ p_optimal
        return -np.sum(N_obs * np.log(q_predict + 1e-10))  # Avoid log(0)

    # Constraints: p0 + p1 = 1, p0 >= 0, p1 >= 0
    constraints = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1})
    bounds = [(0, 1), (0, 1)]

    # Initial guess (e.g., [0.5, 0.5])
    result = minimize(neg_log_likelihood, x0=[0.5, 0.5],
                    bounds=bounds, constraints=constraints)

    p_optimal_estimated = result.x
    if not result.success:
        raise ValueError("MLE Optimization failed: " + result.message)
    #print(f"Estimated true probabilities: {p_optimal_estimated}")
    return p_optimal_estimated

u = unit(coerce_to_integer=True)
machine = QuAM.load("./state_for_sweep_field.json")
config = machine.generate_config()

qubit = machine.qubits["q3"]

m = np.array([
    [0.32016667 + 0.j, - 0.0565 + 0.04128333j],
    [-0.0565 - 0.04128333j,  0.67983333 + 0.j]
])

axis_projection = pauli_expansion_single_qubit(m)
# axis_projection = axis_projection / np.linalg.norm(axis_projection)

coef = np.real(-0.5 * (axis_projection - 1))

conf_mat = np.array([[1]])
conf_mat = np.kron(conf_mat, qubit.resonator.confusion_matrix)
corr  = np.array([
    np.linalg.inv(conf_mat) @ np.array([1 - coef[0],coef[0]]),
    np.linalg.inv(conf_mat) @ np.array([1 - coef[1],coef[1]]),
    np.linalg.inv(conf_mat) @ np.array([1 - coef[2],coef[2]]),
])

corr_MLE = np.array([
    MLE(np.array([1-coef[0],coef[0]]), conf_mat),
    MLE(np.array([1-coef[1],coef[1]]), conf_mat),
    MLE(np.array([1-coef[2],coef[2]]), conf_mat)
])

coef_nor = 1 - 2 * (corr[:, 1])
coef_MLE = 1 - 2 * (corr_MLE[:, 1])
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

dens_nor = 0.5 * (I + coef_nor[0] * X + coef_nor[1] * Y + coef_nor[2] * Z)
dens_MLE = 0.5 * (I + coef_MLE[0] * X + coef_MLE[1] * Y + coef_MLE[2] * Z)

print(dens_nor)
print(dens_MLE)
