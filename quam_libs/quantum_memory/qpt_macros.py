"""
Quantum Process Tomography (QPT) Macros and Utilities

This module provides reusable functions for quantum process tomography experiments,
including state preparation, tomography measurements, and data analysis.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from qm.qua import *
from quam_libs.quantum_memory.marcos import (
    QuantumStateAnalysis,
    QuantumMemory,
    build_pauli_transfer_matrix,
    ptm_to_superop,
    superop_to_choi,
    process_fidelity,
    MLE,
    rho_0,
    rho_1,
    rho_plus,
    rho_plus_i,
)


# ============================================================================
# State Preparation Functions
# ============================================================================

def prepare_basis_state_0(qubit):
    """
    Prepare the |0⟩ state.
    Assumes qubit is already in ground state (or reset).
    """
    pass  # Already in |0⟩


def prepare_basis_state_1(qubit):
    """Prepare the |1⟩ state from |0⟩."""
    qubit.xy.play("y180")


def prepare_basis_state_plus(qubit):
    """Prepare the |+⟩ = (|0⟩ + |1⟩)/√2 state."""
    qubit.xy.play("-y90")


def prepare_basis_state_i_plus(qubit):
    """Prepare the i|+⟩ = (|0⟩ + i|1⟩)/√2 state."""
    qubit.xy.play("-x90")


# ============================================================================
# Tomography Measurement Functions
# ============================================================================

def apply_tomography_basis_x(qubit):
    """Apply X-basis tomography pulse (measure along X)."""
    qubit.xy.play("y90")


def apply_tomography_basis_y(qubit):
    """Apply Y-basis tomography pulse (measure along Y)."""
    qubit.xy.play("x90")


def apply_tomography_basis_z(qubit):
    """Apply Z-basis tomography pulse (no rotation)."""
    pass  # Already measuring Z


# ============================================================================
# QPT Program Generation
# ============================================================================

def generate_qpt_program(
    qubit,
    n_runs: int,
    operation_name: str,
    delay_time: int = 4,
    reset_max_attempts: int = 15,
    reset_wait_time: int = 500,
    flux_point: str = "joint",
    use_machine_flux_setter: Optional[callable] = None,
):
    """
    Generate a QUA program for quantum process tomography.
    
    Parameters
    ----------
    qubit : Transmon
        The qubit to perform QPT on.
    n_runs : int
        Number of measurement runs.
    operation_name : str
        Name of the operation/gate to characterize.
    delay_time : int
        Wait time between operation and tomography measurement (in ns).
    reset_max_attempts : int
        Maximum attempts for active reset.
    reset_wait_time : int
        Wait time for active reset (in ns).
    flux_point : str
        Either 'joint' or 'independent' for flux setting.
    use_machine_flux_setter : callable, optional
        Function to set flux points. If provided, will be called with
        (flux_point=flux_point, target=qubit).
    
    Returns
    -------
    program
        QUA program object.
    """
    from quam_libs.macros import qua_declaration, active_reset, readout_state
    
    with program() as qpt:
        # Declare variables
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=1)
        state = [declare(int) for _ in range(1)]
        state_st = [declare_stream() for _ in range(1)]
        tomo_axis = declare(int)
        initial_state_idx = declare(int)
        
        # Set flux point
        if use_machine_flux_setter:
            use_machine_flux_setter(flux_point=flux_point, target=qubit)
        
        # Main loop over runs
        with for_(n, 0, n < n_runs, n + 1):
            save(n, n_st)
            
            # Loop over 4 initial basis states: |0⟩, |1⟩, |+⟩, i|+⟩
            with for_(initial_state_idx, 0, initial_state_idx < 4, initial_state_idx + 1):
                
                # Loop over 3 tomography measurement bases: X, Y, Z
                with for_(tomo_axis, 0, tomo_axis < 3, tomo_axis + 1):
                    
                    # Reset qubit
                    active_reset(qubit, "readout", max_attempts=reset_max_attempts, wait_time=reset_wait_time)
                    qubit.align()
                    wait(10)
                    
                    # Prepare initial state
                    with if_(initial_state_idx == 0):
                        # |0⟩ - no preparation needed
                        pass
                    with elif_(initial_state_idx == 1):
                        prepare_basis_state_1(qubit)
                    with elif_(initial_state_idx == 2):
                        prepare_basis_state_plus(qubit)
                    with elif_(initial_state_idx == 3):
                        prepare_basis_state_i_plus(qubit)
                    
                    qubit.align()
                    wait(10)
                    
                    # Apply the operation to characterize
                    if operation_name != "id":
                        qubit.xy.play(operation_name)
                    
                    qubit.align()
                    wait(delay_time)
                    
                    # Apply tomography measurement basis
                    with if_(tomo_axis == 0):
                        apply_tomography_basis_x(qubit)
                    with elif_(tomo_axis == 1):
                        apply_tomography_basis_y(qubit)
                    with elif_(tomo_axis == 2):
                        apply_tomography_basis_z(qubit)
                    
                    qubit.align()
                    
                    # Readout
                    readout_state(qubit, state[0])
                    save(state[0], state_st[0])
        
        # Stream processing
        with stream_processing():
            n_st.save("n")
            state_st[0].buffer(3).buffer(4).save_all("state1")
    
    return qpt


# ============================================================================
# Data Analysis Utilities
# ============================================================================

class QPTDataAnalyzer:
    """Helper class for analyzing quantum process tomography data."""
    
    def __init__(self, n_runs: int):
        """
        Initialize the analyzer.
        
        Parameters
        ----------
        n_runs : int
            Total number of measurement runs.
        """
        self.n_runs = n_runs
        self.basis_names = ['0', '1', '+', 'i+']
        self.measurement_bases = ['x', 'y', 'z']
    
    def analyze_single_state(
        self,
        x_count: np.ndarray,
        y_count: np.ndarray,
        z_count: np.ndarray,
        theta: float,
        phi: float,
        confusion_matrix: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Analyze a single prepared state with optional readout error mitigation.
        
        Parameters
        ----------
        x_count : np.ndarray
            Measurement counts along X-basis.
        y_count : np.ndarray
            Measurement counts along Y-basis.
        z_count : np.ndarray
            Measurement counts along Z-basis.
        theta : float
            Ideal theta angle for the state.
        phi : float
            Ideal phi angle for the state.
        confusion_matrix : np.ndarray, optional
            Readout confusion matrix for error mitigation.
        
        Returns
        -------
        dict
            Dictionary with 'raw' and 'mitigated' (if confusion_matrix provided) results.
        """
        results = {}
        
        # Raw analysis
        bloch = [
            (x_count[0] - x_count[1]) / self.n_runs,
            (y_count[0] - y_count[1]) / self.n_runs,
            (z_count[0] - z_count[1]) / self.n_runs,
        ]
        res = QuantumStateAnalysis(bloch, [theta, phi])
        
        results['raw'] = {
            'Bloch vector': bloch,
            'density matrix': res.density_matrix()[0],
            'fidelity': res.fidelity,
            'trace_distance': res.trace_distance,
            'theta': res.theta,
            'phi': res.phi,
        }
        
        # Mitigated analysis (if confusion matrix provided)
        if confusion_matrix is not None:
            new_px_0 = np.array([MLE([x_count[0]/self.n_runs, x_count[1]/self.n_runs], confusion_matrix)[0]])
            new_py_0 = np.array([MLE([y_count[0]/self.n_runs, y_count[1]/self.n_runs], confusion_matrix)[0]])
            new_pz_0 = np.array([MLE([z_count[0]/self.n_runs, z_count[1]/self.n_runs], confusion_matrix)[0]])
            
            m_bloch = np.array([2*new_px_0-1, 2*new_py_0-1, 2*new_pz_0-1], dtype=float).ravel()
            if np.linalg.norm(m_bloch) > 1:
                m_bloch = m_bloch / np.linalg.norm(m_bloch)
            
            m_res = QuantumStateAnalysis(m_bloch, [theta, phi])
            
            results['mitigated'] = {
                'Bloch vector': m_bloch,
                'density matrix': m_res.density_matrix()[0],
                'fidelity': m_res.fidelity,
                'trace_distance': m_res.trace_distance,
                'theta': m_res.theta,
                'phi': m_res.phi,
            }
        
        return results
    
    def build_process_characterization(
        self,
        data: Dict,
        operation_name: str,
        qname: str,
    ) -> Dict:
        """
        Build complete process characterization (PTM, superoperator, Choi) for a qubit.
        
        Parameters
        ----------
        data : dict
            Dictionary with keys ['0', '1', '+', 'i+'] containing density matrices.
        operation_name : str
            Name of the operation being characterized.
        qname : str
            Qubit name for reporting.
        
        Returns
        -------
        dict
            Dictionary with PTM, superoperator, Choi, fidelity, and other metrics.
        """
        # Build input and output states
        inputs = [rho_0, rho_1, rho_plus, rho_plus_i]
        outputs = [
            data['0']['density matrix'],
            data['1']['density matrix'],
            data['+']['density matrix'],
            data['i+']['density matrix'],
        ]
        
        # Build process matrices
        ptm = build_pauli_transfer_matrix(inputs, outputs)
        superop = ptm_to_superop(ptm)
        choi = superop_to_choi(superop, 2, 2) / 2
        
        # Calculate quantum information metrics
        result = {
            'ptm': ptm,
            'superoperator': superop,
            'choi': choi,
            'negativity': QuantumMemory.negativity(choi) * 2,
            'memory_robustness': QuantumMemory.memory_robustness(choi),
            'fidelity': process_fidelity(ptm, target=operation_name),
        }
        
        return result
    
    def print_state_analysis_summary(
        self,
        qname: str,
        initial_state: str,
        theta_ideal: float,
        phi_ideal: float,
        raw_results: Dict,
        mitigated_results: Optional[Dict] = None,
    ):
        """Print a summary of analysis for a single prepared state."""
        print(f"\n{qname} - {initial_state} state")
        print("=" * 60)
        
        bloch_raw = raw_results['Bloch vector']
        theta_raw = np.rad2deg(raw_results['theta'])
        phi_raw = np.rad2deg(raw_results['phi'])
        fid_raw = raw_results['fidelity']
        dist_raw = raw_results['trace_distance']
        
        print(f"Bloch vector (raw): [{bloch_raw[0]:.3f}, {bloch_raw[1]:.3f}, {bloch_raw[2]:.3f}]")
        print(f"Angles (raw): θ={theta_raw:.1f}°, φ={phi_raw:.1f}°")
        print(f"Metrics (raw): Fidelity={fid_raw:.3f}, Trace distance={dist_raw:.3f}")
        
        if mitigated_results:
            bloch_mit = mitigated_results['Bloch vector']
            theta_mit = np.rad2deg(mitigated_results['theta'])
            phi_mit = np.rad2deg(mitigated_results['phi'])
            fid_mit = mitigated_results['fidelity']
            dist_mit = mitigated_results['trace_distance']
            
            print(f"Bloch vector (mit): [{bloch_mit[0]:.3f}, {bloch_mit[1]:.3f}, {bloch_mit[2]:.3f}]")
            print(f"Angles (mit): θ={theta_mit:.1f}°, φ={phi_mit:.1f}°")
            print(f"Metrics (mit): Fidelity={fid_mit:.3f}, Trace distance={dist_mit:.3f}")
        
        print(f"Ideal: θ={np.rad2deg(theta_ideal):.1f}°, φ={np.rad2deg(phi_ideal):.1f}°")
