from scipy.linalg import eigvalsh, svd
import numpy as np
import cvxpy as cp

import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="cvxpy.atoms.affine.reshape")

identity = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])


# Bloch vector to density matrix and vice versa
def bloch_vector_to_density_matrix(bloch_vector):
    return 0.5 * (identity + bloch_vector[0] * sigma_x + bloch_vector[1] * sigma_y + bloch_vector[2] * sigma_z)

def density_matrix_to_bloch_vector(rho):
    r_x = np.real(np.trace(rho @ sigma_x))
    r_y = np.real(np.trace(rho @ sigma_y))
    r_z = np.real(np.trace(rho @ sigma_z))
    return np.array([r_x, r_y, r_z])

def angle_to_density_matrix(theta,phi):
    c = np.cos(theta/2)
    s = np.sin(theta/2)
    e = np.exp(1j*phi)
    return np.array([[c**2,c*s*np.conj(e)],[c*s*e,s**2]])

def BR_density_matrix(theta, phi, T1,T2,t,detuning=0):
    alpha = np.cos(theta/2)
    beta = np.sin(theta/2)*np.exp(1j*phi)
    return np.array([[1+(alpha**2-1)*np.exp(-t/T1),alpha*np.conj(beta)*np.exp(1j*detuning*t)*np.exp(-t/T2)],
                    [np.conj(alpha)*beta*np.exp(-1j*detuning*t)*np.exp(-t/T2),beta*np.conj(beta)*np.exp(-t/T1)]])

class QuantumStateAnalysis:
    
    """
    Define a class for analyzing quantum states. 
    Rho and sigma are experimental and ideal density matrices, respectively.

    Parameters:
        measurement_data (list): [x, y, z] - Experimental Bloch vector components.
        ideal_data (list): [theta, phi] - Ideal Bloch vector parameters.
    """

    def __init__(self, measurement_data, ideal_data):
        self.x, self.y, self.z = measurement_data
        self.ideal_theta, self.ideal_phi = ideal_data

        self.theta, self.phi = self.theta_phi()
        self.bloch_vector = [self.x, self.y, self.z]
        self.ideal_bloch_vector = [
            np.sin(self.ideal_theta) * np.cos(self.ideal_phi),
            np.sin(self.ideal_theta) * np.sin(self.ideal_phi),
            np.cos(self.ideal_theta)
        ]

        self.rho = bloch_vector_to_density_matrix(np.array(self.bloch_vector))
        self.sigma = bloch_vector_to_density_matrix(np.array(self.ideal_bloch_vector))
        self.measured_dm = self.rho
        self.ideal_dm = self.sigma

        self.fidelity = self.fidelity()
        self.trace_distance = self.trace_distance()
        

    def theta_phi(self):
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        theta = np.arccos(self.z / 1)
        phi = np.arctan2(self.y, self.x)
        return theta, phi



    def trace_distance(self):
        # Difference of the matrices
        delta = self.rho - self.sigma
        
        # Singular value decomposition
        singular_values = svd(delta, compute_uv=False)
        
        # Trace norm is the sum of the singular values
        trace_norm = np.sum(singular_values)
        
        # Trace distance
        return 0.5 * trace_norm

    def fidelity(self):
        # rho is the measured density matrix, so it is mixed state
        # sigma is the ideal density matrix, so it is pure state
        fidelity = np.trace(self.rho @ self.sigma)
        # Fidelity
        return np.abs(fidelity)

    def get_results(self):
        return {
            'theta': self.theta,
            'phi': self.phi,
            'bloch_vector': self.bloch_vector,
            'ideal_bloch_vector': self.ideal_bloch_vector,
            'ideal_dm': self.sigma,
            'measured_dm': self.rho,
            'fidelity': self.fidelity(),
            'trace_distance': self.trace_distance()
        }

    



#quantum memory 
from qutip import Qobj
from picos import Problem, value
from picos.expressions.variables import HermitianVariable
from picos.expressions.algebra import trace, partial_transpose

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


class CorrectBloch:
    """
    Correct the 2*2 density matrix to ensure it is Hermitian, positive semi-definite, and has a trace of 1.
    """
    def __init__(self, uncorrect_bloch_vector):
        "uncorrect_bloch_vector: List"
        self.bloch_vector = np.array(uncorrect_bloch_vector).reshape(-1,3)
        self.dm = self.density_matrix()

    def density_matrix(self):
        experimental_dm = []
        for i in range(len(self.bloch_vector)):
            experimental_dm.append(bloch_vector_to_density_matrix(self.bloch_vector[i]))
        experimental_dm = np.array(experimental_dm).reshape(-1, 2, 2)
        return np.array(experimental_dm)

    def project_to_cptp(self,dims=(2, 2)):
        corrected_dm = []
        for dm_i in self.dm:
            X = cp.Variable(dims, hermitian=True)
            obj = cp.Minimize(cp.norm(X - dm_i, "fro"))
            constraints = [
                X >> 0,
                cp.trace(X) == 1
            ]
            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.SCS)   
            if prob.status not in ("optimal", "optimal_inaccurate"):
                raise RuntimeError(f"Projection fail: {prob.status}")
            corrected_dm.append(X.value)
        corrected_dm = np.array(corrected_dm).reshape(-1, dims[0], dims[1])
        corrected_bloch = np.array([density_matrix_to_bloch_vector(dm) for dm in corrected_dm])
        return corrected_dm, corrected_bloch

class CorrectChoi:
    """Correct and validate a 4x4 Choi state via iterative CP/TP projection.
    
    Usage
    -----
    cc = CorrectChoi(choi)
    corrected = cc.correct(max_iterations=100, tol=1e-8)
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
            # from pennylane.math import partial_trace
            # is_hermitian = np.allclose(choi, choi.conj().T, atol=tol)
            # eigenvalues = eigvalsh(choi)
            # is_psd = np.all(eigenvalues >= -tol)
            # trace_is_one = abs(np.trace(choi) - 1) < tol
            # pt = partial_trace(choi, indices=index)
            # pt_trace_is_one = trace_norm(pt,0.5*np.eye(2)) < tol
            from picos import partial_trace
            is_hermitian = np.allclose(choi, choi.conj().T, atol=tol)
            eigenvalues = eigvalsh(choi)
            is_psd = np.all(eigenvalues >= -tol)
            trace_is_one = abs(np.trace(choi) - 1) < tol
            pt = partial_trace(choi, subsystems=index[0], dimensions=(2, 2))
            # pt = partial_trace(choi, indices=index)
            pt_trace_is_one = trace_norm(pt,0.5*np.eye(2)) < tol

            if is_hermitian and is_psd and trace_is_one and pt_trace_is_one:
                print(f"After {count} iterations, the Choi state is valid.")
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
            choi = CPTPProjector(choi).project_to_cp_tp()
        return choi,count



class CPTPProjector:
    """
    Correct the Choi state to ensure it follow
    1. Hermitian
    2. positive semi-definite
    3. trace of 1.
    4. partial trace out should be identity
    """
    def __init__(self, choi_state):
        "uncorrect_choi_state: List"
        self.choi_state = np.array(choi_state).reshape(-1,4,4)

    def project_to_cp_tp(self, d=2):
        X = cp.Variable((d*d, d*d), hermitian=True)                         # Hermitian variable
        # put in marcos in future
        def ptrace_out_cp(X, d_in=2, d_out=2):
            X4 = cp.reshape(X, (d_out, d_in, d_out, d_in))   # [i,k,j,l]
            rho_in = 0
            for i in range(d_out):
                rho_in += X4[i, :, i, :]
            return rho_in
        # find matrix X that minimizes the trace distance to choi_state
        objective = cp.Minimize(cp.norm(X - self.choi_state, 'fro'))
        constraints = [
            X >> 1e-7,                                                  # positive semi-definite
            ]
        constraints += [ptrace_out_cp(X, d_in=d, d_out=d) == 0.5*np.eye(d)]  # Tr_out(X) = I_d

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)   # 或 'CVXOPT' / 'MOSEK'

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("Projection failed:", prob.status)

        return X.value