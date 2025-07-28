import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib
from quam_libs.quantum_memory.marcos import density_matrix_to_bloch_vector, bloch_vector_to_density_matrix
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"cvxpy\.atoms\.affine\.reshape"
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"cvxpy\.reductions\.solvers\.solving_chain_utils"
)

class CorrectBlochSphere:
    """
    Correct the density matrix to ensure it is Hermitian, positive semi-definite, and has a trace of 1.
    """
    def __init__(self, uncorrect_bloch_vector):
        "uncorrect_bloch_vector: List"
        self.bloch_vector = np.array(uncorrect_bloch_vector).reshape(-1,3)
        self.dm = self.density_matrix()

    def density_matrix(self):

        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.array([[1, 0], [0, 1]], dtype=complex)

        experimental_dm = []
        for i in range(len(self.bloch_vector)):
            r_x, r_y, r_z = self.bloch_vector[i]
            rho = 0.5 * (identity + r_x * sigma_x + r_y * sigma_y + r_z * sigma_z)
            experimental_dm.append(rho)
        experimental_dm = np.array(experimental_dm).reshape(-1, 2, 2)
        return np.array(experimental_dm)


    def project_to_cptp(self,dims=(2, 2),method="cvxpy"):
        corrected_dm = []
        if method == "qiskit":
            from qiskit.quantum_info.states.utils import closest_density_matrix
            corrected_dm = [closest_density_matrix(dm_i, norm="fro") for dm_i in self.dm]
        elif method == "manual":
            for dm_i in self.dm:
                bloch_vector = density_matrix_to_bloch_vector(dm_i)
                if np.linalg.norm(bloch_vector) > 1:
                    # Normalize the Bloch vector if it exceeds the unit sphere
                    bloch_vector = bloch_vector / np.linalg.norm(bloch_vector)
                else:
                    pass
                corrected_dm.append(bloch_vector_to_density_matrix(bloch_vector))
        elif method == "cvxpy":
            import cvxpy as cp
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

    def project_to_cp_tp(self, d=2, method='Trace distnce'):
        X = cp.Variable((d*d, d*d), hermitian=True)                         # Hermitian variable
        mu = cp.Variable((1))
        # put in marcos in future
        def ptrace_out_cp(X, d_in=2, d_out=2):
            X4 = cp.reshape(X, (d_out, d_in, d_out, d_in))   # [i,k,j,l]
            rho_in = 0
            for i in range(d_out):
                rho_in += X4[i, :, i, :]
            return rho_in

        if method == 'Trace distance':
            # find matrix X that minimizes the trace distance to choi_state
            objective = cp.Minimize(cp.norm(X - self.choi_state, 'fro'))
            constraints = [
                X >> 1e-7,                                                  # positive semi-definite
                ]
        elif method == 'mu':
            # find mu that minimizes the trace distance to choi_state
            objective = cp.Minimize(mu)
            constraints = [
                X >> 1e-7, 
                mu >= 0,
                mu*np.eye(d*d) >> X-choi_raw
                -mu*np.eye(d*d) << X-choi_raw
                ]  
        constraints += [ptrace_out_cp(X, d_in=d, d_out=d) == 0.5*np.eye(d)]  # Tr_out(X) = I_d

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)   # 或 'CVXOPT' / 'MOSEK'

        if prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("Projection failed:", prob.status)

        return X.value