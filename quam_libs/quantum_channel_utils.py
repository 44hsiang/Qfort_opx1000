"""Utility functions for quantum channel analysis."""

import numpy as np
from .QI_utils import CorrectChoi, entanglementRobustness
from qutip import Qobj
from picos.expressions.algebra import partial_transpose

# simulation utility functions
def error_gate(rho, readout_error):

    gamma = readout_error

    if isinstance(rho, Qobj):
        rho_mat = rho.full()           
    else:
        rho_mat = np.asarray(rho, dtype=complex)

    rho_x = np.array([[0, 1],
                   [1, 0]], dtype=complex)

    rho_y = np.array([[0, -1j],
                   [1j, 0]], dtype=complex)
    
    rho_z = np.array([[1, 0],
                   [0, -1]], dtype=complex)

    err = (gamma / 3) * (rho_x @ rho_mat @ rho_x + rho_y @ rho_mat @ rho_y +rho_z @ rho_mat @ rho_z ) \
          + (1 - gamma) * rho_mat      

    return err

# quantum memory analysis utility functions
def compute_choi_state_raw(center, axes, R):
    """Compute the Choi state from ellipsoid parameters WITHOUT correction (Legacy behavior).
    
    Args:
        center: Center of the ellipsoid (Bloch vector)
        axes: Semi-axes of the ellipsoid
        R: Rotation matrix of the ellipsoid
        
    Returns:
        Uncorrected Choi matrix (matching Legacy NoiseAnalyze behavior)
    """
    pauli_matrices = [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex)
    ]
    B = np.array(center)
    radii = np.array(axes)
    eigvecs = np.array(R)
    # T maps Bloch vector components; convention kept from your original code.
    T = np.diag(radii) @ eigvecs.T

    chi = np.zeros((4, 4), dtype=complex)
    chi[0, 0] = 1
    chi[0, 1:4] = B
    chi[1:4, 0] = B.conj()
    chi[1:4, 1:4] = T
    chi = (chi + chi.conj().T) / 2  # Hermitian

    choi = np.zeros((4, 4), dtype=complex)
    for i, Pi in enumerate(pauli_matrices):
        for j, Pj in enumerate(pauli_matrices):
            choi += chi[i, j] * np.kron(Pi, Pj)

    choi = (choi + choi.conj().T) / 2
    choi /= np.trace(choi)

    choi = partial_transpose(choi, 0).np
    # Return uncorrected choi (correction is optional via correct_choi() method)
    return choi

def compute_choi_state(center, axes, R):
    """Compute the Choi state from ellipsoid parameters with optional correction.
    
    For compatibility with any code still using the old interface.
    Note: Use compute_choi_state_raw() for fast uncorrected computation matching Legacy.
    
    Args:
        center: Center of the ellipsoid (Bloch vector)
        axes: Semi-axes of the ellipsoid
        R: Rotation matrix of the ellipsoid
        
    Returns:
        Corrected Choi matrix
    """
    choi = compute_choi_state_raw(center, axes, R)
    # Only correct if explicitly needed (optional post-processing)
    cc = CorrectChoi(choi)
    corrected, count = cc.choi_checker(repeat=100, tol=1e-4, print_reason=False)
    return corrected

def compute_memory_robustness(choi):
    """Return the entanglement robustness of `choi`.
    
    Args:
        choi: Choi matrix
        
    Returns:
        Entanglement robustness value
    """
    return float(entanglementRobustness(choi))
