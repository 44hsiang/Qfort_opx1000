import numpy as np

import cvxpy as cp
from scipy.linalg import eigvalsh
from quam_libs.quantum_memory.marcos import density_matrix_to_bloch_vector, dm_checker, bloch_vector_to_density_matrix
from quam_libs.fit_ellipsoid import ls_ellipsoid, polyToParams3D
from quam_libs.quantum_memory.EllipsoidTool import EllipsoidTool
from quam_libs.quantum_memory.CorrectBloch import CorrectBlochSphere,CorrectChoi
from quam_libs.quantum_memory.entanglement_robustness import entanglementRobustness
#matplotlib.use('TkAgg')

class NoiseAnalyze:
    
    def __init__(self, measurement_data, ideal_data):
        self.measurement_data = measurement_data
        self.ideal_data = ideal_data
        self.corrected_dm, self.corrected_bloch = self.valid_data()
    
    def valid_data(self,method="manual"):
        """
        project measurement data to density matrix by using MLE,
        and make sure density matrix is
        1. Hermitian => U = U^dagger
        2. Positive semi-definite => all eigenvalues are non-negative
        3. Trace = 1 => sum of diagonal elements is 1
        Returns:
            valid density matrix
            valid Bloch vector
        """ 
        corrector = CorrectBlochSphere(self.measurement_data)
        corrected_dm, corrected_bloch = corrector.project_to_cptp(method=method)
        return corrected_dm, corrected_bloch

    def ellipsoid_fit(self,do_convex=True):
        """
        Fit the ellipsoid from input data(Bloch vector)
        """
        return EllipsoidTool(self.corrected_bloch,convex=do_convex).fit()

    def ellipsoid_plot(self, ax=None,title=None,do_convex=True):
        """
        Plot the ellipsoid
        """   
        return EllipsoidTool(self.corrected_bloch,convex=do_convex).plot(ax=ax,title=title)


class QuantumMemory:

    def __init__(self,axes,center,R):
        """
        Initialize the Quantum Memory with ellipsoid parameters.
        axes: Lengths of the ellipsoid axes.
        center: Center of the ellipsoid.
        R: Rotation matrix for the ellipsoid.
        """
        self.axes = axes
        self.center = center
        self.R = R
        self.choi = self.choi_state()
        self.negativity = self.negativity()
        self.memory_robustness = self.memory_robustness()
    
    def choi_state(self):
        pauli_matrices = [
            np.eye(2, dtype=complex),
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex)
        ]
        B = np.array(self.center)
        radii = np.array(self.axes)
        eigvecs = np.array(self.R)
        #T = eigvecs @ np.diag(radii)
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
        return choi

    def partial_transpose(self, sys=1):
        choi_tensor = self.choi.reshape(2, 2, 2,2)
        if sys == 0:
            # partial transpose on system A
            choi_PT = choi_tensor.swapaxes(0,2)
        elif sys == 1:
            # partial transpose on system B
            choi_PT = choi_tensor.swapaxes(1,3)
        else:
            raise ValueError("sys must be 0 or 1")
        return choi_PT.reshape(4,4)

    def negativity(self):
        rho_PT = self.partial_transpose(sys=1)
        eigvals = eigvalsh(rho_PT)
        return -eigvals[eigvals < 0].sum() 

    def memory_robustness(self):
        return entanglementRobustness(self.choi)

class Checker:
    """
    Check if the density matrix is valid.
    """
    def __init__(self, dm):
        self.dm = dm
    
    def dm_checker(self, tol=1e-8,print_reason=True):
        """
        Check if the density matrix is Hermitian, positive semi-definite, and has a trace of 1.
        :param dm: Density matrix (2^nx2^n numpy array)
        :param tol: Tolerance for numerical errors
        :return: True if the density matrix is valid, False otherwise
        """
        is_hermitian = np.allclose(self.dm, self.dm.conj().T, atol=tol)
        eigenvalues = eigvalsh(self.dm)
        is_psd = np.all(eigenvalues >= -tol)
        trace_is_one = abs(np.trace(self.dm) - 1) < tol

        if is_hermitian and is_psd and trace_is_one:
            return True
        else:
            if print_reason:
                print("Density matrix is invalid:")
                if not is_hermitian:
                    print("❌ Density matrix is not Hermitian.")
                if not is_psd:
                    print("❌ Density matrix is not positive semi-definite. Eigenvalues:", eigenvalues)
                if not trace_is_one:
                    print(f"❌ Trace of the density matrix is not 1. Got trace = {np.trace(self.dm)}")
            return False

    def choi_checker(self,index=[1],repeat = 100, tol=1e-8, print_reason=True):
        """
        Check if the Choi state is valid.
        :param tol: Tolerance for numerical errors
        :return: True if the Choi state is valid, False otherwise
        """
        def trace_norm(rho,sigama):
            dif = rho - sigama
            evals = eigvalsh(dif)
            return 0.5 * np.sum(np.abs(evals))
        choi = self.dm.reshape(4, 4)
        count = 0
        for i in range(repeat):
            count += 1
            from pennylane.math import partial_trace
            is_hermitian = np.allclose(choi, choi.conj().T, atol=tol)
            eigenvalues = eigvalsh(choi)
            is_psd = np.all(eigenvalues >= -tol)
            trace_is_one = abs(np.trace(choi) - 1) < tol
            pt = partial_trace(choi, indices=index)
            pt_trace_is_one = trace_norm(pt,0.5*np.eye(2)) < tol
            #pt_trace_is_one = trace_norm(pt,np.eye(2)) < tol

            print(trace_norm(pt,0.5*np.eye(2)))

            if is_hermitian and is_psd and trace_is_one and pt_trace_is_one:
                break
            else:
                print_reason = True if i == repeat - 1 else print_reason
                if print_reason:
                    print("Choi state is invalid:")
                    if not is_hermitian:
                        print("❌ Choi state is not Hermitian.")
                    if not is_psd:
                        print("❌ Choi state is not positive semi-definite. Eigenvalues:", eigenvalues)
                    if not trace_is_one:
                        print(f"❌ Trace of the Choi state is not 1. Got trace = {np.trace(choi)}")
                    if not pt_trace_is_one:
                        print(f"❌ Partial trace of the Choi state is not 1. Got trace = {trace_norm(pt,0.5*np.eye(2))}")
            choi = CorrectChoi(choi).project_to_cp_tp(method='Trace distance')
            print(trace_norm(pt,0.5*np.eye(2)))
        return choi,count



if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    from pennylane.math import partial_trace
    test_dm = np.array([[ 0.441+0.j   ,  0.014-0.029j,  0.014-0.029j,  0.21 +0.226j],
       [ 0.014+0.029j,  0.058+0.j   , -0.008+0.j   , -0.002+0.017j],
       [ 0.014+0.029j, -0.008+0.j   ,  0.058+0.j   , -0.002+0.017j],
       [ 0.21 -0.226j, -0.002-0.017j, -0.002-0.017j,  0.443+0.j   ]], dtype=complex)
    print("Test density matrix:\n", test_dm)
    checker = Checker(test_dm)
    choi, count = checker.choi_checker(index=[1], repeat=100, tol=1e-8, print_reason=True)
    print(f"After {count} iterations, the Choi state is:\n{choi}")
