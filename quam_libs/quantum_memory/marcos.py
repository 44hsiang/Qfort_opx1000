from numpy.linalg import eigvalsh, norm
import numpy as np
import pandas as pd
import cvxpy as cp
import pennylane as qml
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from scipy.optimize import minimize
import numpy as np
from numpy.linalg import eigvalsh
from qutip import sigmax, sigmay, Qobj   # 只用來偵測 Qobj 與取得 Pauli
import io
from typing import List, Union

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PIL import Image
import itertools

from pathlib import Path
data_path = Path("/Users/jackchao/Desktop/Project/Phd_thesis/CH5_5GZdemonstration/data")

# fitting tool
def T1(node,index):
    T1=[]
    std = []
    for i in index:
        node_qm = node.load_from_id(i,base_path=data_path)
        ds = node_qm.results['ds']
        fit_data = fit_decay_exp(ds.state, "idle_time")
        fit_data.attrs = {"long_name": "time", "units": "µs"}
        # Fitted decay
        fitted = decay_exp(
            ds.idle_time,
            fit_data.sel(fit_vals="a"),
            fit_data.sel(fit_vals="offset"),
            fit_data.sel(fit_vals="decay"),
        )
        # Decay rate and its uncertainty
        decay = fit_data.sel(fit_vals="decay")
        decay.attrs = {"long_name": "decay", "units": "ns"}
        decay_res = fit_data.sel(fit_vals="decay_decay")
        decay_res.attrs = {"long_name": "decay", "units": "ns"}
        # T1 and its uncertainty
        tau = -1 / fit_data.sel(fit_vals="decay")
        tau.attrs = {"long_name": "T1", "units": "µs"}
        tau_error = -tau * (np.sqrt(decay_res) / decay)
        tau_error.attrs = {"long_name": "T1 error", "units": "µs"}
        T1.append(tau.values)
        std.append(tau_error.values)
    return np.array(T1).reshape(-1), np.array(std).reshape(-1)


def T2(node,index):
    val= []
    std = []
    for i in index:
        node_qm = node.load_from_id(i,base_path = data_path)
        val.append(node_qm.results['fit_results']['q0']['decay'])
        std.append(node_qm.results['fit_results']['q0']['decay_error'])
    return np.array(val).reshape(-1), np.array(std).reshape(-1)

def dephasing_errorbar(T1, T2, sT1, sT2, rho=0.0, pure=False):
    T1, T2  = np.asarray(T1, float), np.asarray(T2, float)
    sT1, sT2 = np.asarray(sT1, float), np.asarray(sT2, float)
    sign = 1.0 
    val = 1.0/T2 - sign*(1.0/(2.0*T1))
    dT2 = -1.0/(T2**2)
    dT1 = -sign/(2.0*T1**2)
    var = (dT1*sT1)**2 + (dT2*sT2)**2 + 2.0*rho*(dT1*sT1)*(dT2*sT2)
    var = np.maximum(var, 0.0)
    return val, np.sqrt(var)


# density matrix transformation related functions
def density_state(theta,phi):
    c = np.cos(theta/2)
    s = np.sin(theta/2)
    e = np.exp(1j*phi)
    return np.array([[c**2,c*s*np.conj(e)],[c*s*e,s**2]])

def BR_density_state(theta, phi, T1,T2,t,detuning=0):
    alpha = np.cos(theta/2)
    beta = np.sin(theta/2)*np.exp(1j*phi)
    return np.array([[1+(alpha**2-1)*np.exp(-t/T1),alpha*np.conj(beta)*np.exp(1j*detuning*t)*np.exp(-t/T2)],
                    [np.conj(alpha)*beta*np.exp(-1j*detuning*t)*np.exp(-t/T2),beta*np.conj(beta)*np.exp(-t/T1)]])

def theta_phi_list(n_points):
    theta_range = np.arange(0,np.pi,1e-4)
    phi_range = np.arange(0,2*np.pi,1e-4)
    theta_list,phi_list = [],[]
    for i in range(n_points):
        theta,phi = np.random.choice(theta_range),np.random.choice(phi_range)
        theta_list.append(theta)
        phi_list.append(phi)
    return theta_list,phi_list

def generate_uniform_sphere_angles(n_points):

    indices = np.arange(0, n_points)
    golden_angle = np.pi * (3 - np.sqrt(5))  

    z = 1 - 2 * (indices + 0.5) / n_points     
    theta = np.arccos(z)                      
    phi = (indices * golden_angle) % (2 * np.pi)  

    theta_list = theta.tolist()
    phi_list = phi.tolist()

    return theta_list, phi_list

def density_matrix_to_bloch_vector(rho):

    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    r_x = np.real(np.trace(rho @ sigma_x))
    r_y = np.real(np.trace(rho @ sigma_y))
    r_z = np.real(np.trace(rho @ sigma_z))

    return np.array([r_x, r_y, r_z])

def bloch_vector_to_density_matrix(bloch_vector):
    r_x, r_y, r_z = bloch_vector
    identity = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    rho = 0.5 * (identity + r_x * sigma_x + r_y * sigma_y + r_z * sigma_z)
    return rho

def BR_state_vector(alpha,beta,T1,T2,t):
    return np.array(
        [[1+(alpha**2-1)*np.exp(-t/T1),alpha*np.conj(beta)*np.exp(-t/T2)],
        [np.conj(alpha)*beta*np.exp(-t/T2),beta*np.conj(beta)*np.exp(-t/T1)]])
    
def theta_phi_to_alpha_beta(theta, phi):
    alpha = np.cos(theta/2)
    beta = np.sin(theta/2)*np.exp(1j*phi)
    return alpha, beta

# Quantum information functions
def partial_transpose(rho, sys=0):
    """
    ρ  (4x4) -> ρ^{T_sys}  (4x4)
    sys = 0 : transpose first qubit
    sys = 1 : transpose second qubit (常用)
    """
    rho_t = rho.reshape(2, 2, 2, 2)          # (a,b,c,d) 對應 |a b><c d|
    if sys == 0:
        rho_t = rho_t.swapaxes(0, 2)         # transpose on first qubit
    elif sys == 1:
        rho_t = rho_t.swapaxes(1, 3)         # transpose on second qubit
    else:
        raise ValueError("sys must be 0 or 1")
    return rho_t.reshape(4, 4)

def negativity_(rho):
    """直接用本徵值定義計算 negativity"""
    rho_pt  = partial_transpose(rho, sys=1)
    eigvals = np.linalg.eigvalsh(rho_pt)     # Hermitian eigs (real)
    return -eigvals[eigvals < 0].sum()    

def dm_checker_dict(data_dict,name = 'data_dm', tolerance=1e-8, print_details=False):
    bad_dict = {}
    for key in data_dict.keys():
        count = 0
        index_list = []
        for i in range(len(data_dict[key][name])):
            data_check = dm_checker(data_dict[key][name][i],tol = tolerance,print_reason=print_details)
            if data_check:
                pass
            else:
                print(f"Density matrix[{i}] is not valid for {key}") if print_details else None
                count += 1
                index_list.append(i)
        if print_details:
            if count > 0 :
                print(f"Total {count} valid point in Density matrix for {key}") 
            if count == 0:
                print(f"No errors in {name} and for {key}") 
            print('-'*75)
        bad_dict[key] = index_list
    return bad_dict

def dm_checker(dm, tol=1e-8,print_reason=True):
    """
    Check if the density matrix is Hermitian, positive semi-definite, and has a trace of 1.
    :param dm: Density matrix (2x2 numpy array)
    :param tol: Tolerance for numerical errors
    :return: True if the density matrix is valid, False otherwise
    """
    is_hermitian = np.allclose(dm, dm.conj().T, atol=tol)
    eigenvalues = eigvalsh(dm)
    is_psd = np.all(eigenvalues >= -tol)
    trace_is_one = abs(np.trace(dm) - 1) < tol

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
                print(f"❌ Trace of the density matrix is not 1. Got trace = {np.trace(dm)}")
        return False

def diagnose_choi_states(choi_list,index, tol_pos=1e-10, tol_tp=1e-3):
    rows = []
    def trace_norm(rho,sigama):
        dif = rho - sigama
        evals = eigvalsh(dif)
        return 0.5 * np.sum(np.abs(evals))
    for k, j in enumerate(choi_list):
        lam_min   = eigvalsh(j).min()
        trace     = j.trace().real
        tp_dev    = trace_norm(qml.math.partial_trace(j,indices=[1]),0.5*np.eye(2))
        is_Hermitian = np.allclose(j, j.conj().T)
        rows.append({
            "index":   index[k],
            "λ_min":   lam_min,
            "trace":   trace,
            "TP dev":  tp_dev,
            "Hermitian_OK": is_Hermitian,
            "CP_OK":   lam_min > -tol_pos,
            "TP_OK":   tp_dev  < tol_tp,
            "all_OK":  (lam_min > -tol_pos) and (tp_dev < tol_tp) and abs(trace-1)<1e-6 and is_Hermitian
        })
    return pd.DataFrame(rows)

def error_gate(rho, readout_error):

    gamma = readout_error

    # --- 1. 把輸入統一轉成 numpy array ----------------------------------------
    if isinstance(rho, Qobj):
        rho_mat = rho.full()            # 轉 dense numpy
    else:
        rho_mat = np.asarray(rho, dtype=complex)

    # --- 2. 準備 Pauli X、Y -------------------------------------------------
    rho_x = np.array([[0, 1],
                   [1, 0]], dtype=complex)

    rho_y = np.array([[0, -1j],
                   [1j, 0]], dtype=complex)
    
    rho_z = np.array([[1, 0],
                   [0, -1]], dtype=complex)

    # --- 3. 套用通道 ---------------------------------------------------------
    err = (gamma / 3) * (rho_x @ rho_mat @ rho_x + rho_y @ rho_mat @ rho_y +rho_z @ rho_mat @ rho_z ) \
          + (1 - gamma) * rho_mat      # I ρ I = ρ，本身就是矩陣乘積

    return err

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

# ellipsoid related functions

def ellipsoid_equation(r,param):
    x, y, z = r
    return (param[0] * x**2 + param[1] * y**2 + param[2] * z**2 +
            param[3] * x * y + param[4] * x * z + param[5] * y * z +
            param[6] * x + param[7] * y + param[8] * z + param[9])

def ellipsoid_to_quadric(center, axes, R):

    c = np.asarray(center, dtype=float).reshape(3)
    a, b, c_len = np.asarray(axes, dtype=float).reshape(3)
    R = np.asarray(R, dtype=float).reshape(3, 3)

    # 1. 形狀矩陣：Q = R · diag(1/a², 1/b², 1/c²) · Rᵀ
    Q_local = np.diag([1/a**2, 1/b**2, 1/c_len**2])
    Q = R @ Q_local @ R.T            # 對稱 3×3

    # 2. 線性項向量：ℓ = −2 Q c
    linear = -2 * Q @ c              # (G, H, I)

    # 3. 常數項：J = cᵀ Q c − 1
    J = float(c @ Q @ c - 1.0)

    # 4. 從 Q 擷取二次項係數
    A, B, C = Q[0, 0], Q[1, 1], Q[2, 2]
    # D = 2 * Q[0, 1]
    # E = 2 * Q[0, 2]
    # F = 2 * Q[1, 2]
    D = Q[0, 1]+Q[1, 0]
    E = Q[0, 2]+Q[2, 0]
    F = Q[1, 2]+Q[2, 1]
    G, H, I = linear

    return np.array([A, B, C, D, E, F, G, H, I, J])

def find_best_fit(center,axes,R,param):
    perms = list(itertools.permutations([0,1,2]))
    perm_diff_dict = {}
    for perm in np.array(perms):
        original_param = param/np.abs(param[-1])
        fitted_param = ellipsoid_to_quadric(center, axes[perm], R[perm,:])
        difference = np.abs(original_param - fitted_param)
        perm_diff_dict[tuple(perm)] = difference[0:6].sum()
    min_perm = list(min(perm_diff_dict, key=perm_diff_dict.get))
    R = R[min_perm,:]
    axes = axes[min_perm]
    if np.linalg.det(R) > 0:
        R[0,:]*=-1 
    return axes,R

# Quantum Process Tomography

# Input density matrices
rho_0 = np.array([[1, 0], [0, 0]])  # |0><0|
rho_1 = np.array([[0, 0], [0, 1]])  # |1><1|
rho_plus = 0.5 * np.array([[1, 1], [1, 1]])  # |+><+|
rho_plus_i = 0.5 * np.array([[1, -1j], [1j, 1]])  # |+i><+i|

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

def build_pauli_transfer_matrix(rho_in: any, rho_out: any) -> np.ndarray:
    """
    Constructs the 4x4 superoperator S such that:
    vec(rho_out) = S @ vec(rho_in)

    rho_out/rho_in: list of 4 density matrices in Pauli basis (2x2)
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

def pauli_matrices():
    I = np.array([[1,0],[0,1]], dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    return [I,X,Y,Z]

def vec_col(A): return A.reshape(-1, order='F')

def build_N_M(n=1):
    pals = pauli_matrices()
    N = np.column_stack([vec_col(P) for P in pals])
    M = np.vstack([vec_col(P).conj().T for P in pals])
    return N,M

def ptm_to_superop(R, n=1):
    N,M = build_N_M(n)
    return N @ R @ (M / (2**n))

def superop_to_choi(S, din, dout):
    S4 = S.reshape(dout,dout,din,din)
    L4 = np.transpose(S4, (0,2,1,3))
    return L4.reshape(din*dout, din*dout)


def process_fidelity(ptm, target='id'):
    """
    Compute process fidelity with respect to a target quantum process.
    
    Parameters:
        ptm: 4x4 numpy array (pauli transfer matrix in Pauli basis {I, X, Y, Z})
        target: str, either 'id' or 'x' (currently supports 'id' and 'x')
        
    Returns:
        fidelity (float): Process fidelity between the input pauli transfer matrix and the target
    """
    # Normalize input pauli transfer matrix
    ptm = ptm / norm(ptm, 'fro')

    # Define target pauli transfer matrix
    if target == 'id':
        target_ptm = np.eye(4, dtype=complex)
    elif target == 'x180':
        target_ptm = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
    elif target == 'y180':
        target_ptm = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)        
    elif target == 'x90':
        target_ptm = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1],
            [0, 0, 1, 0]
        ], dtype=complex)
    elif target == 'y90':
        target_ptm = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ], dtype=complex)
    
    else:
        raise ValueError("Unsupported target process. Use 'id', 'x180', 'y180','x90', or 'y90'")

    target_ptm = target_ptm / norm(target_ptm, 'fro')

    # Compute inner product (Hilbert-Schmidt)
    fidelity = np.abs(np.trace(np.conj(ptm.T) @ target_ptm))
    return fidelity

# gif generator

def figlist_to_gif(
    figure_list: List[Union[Figure, Axes]],
    outfile: str = "anim.gif",
    fps: int = 3,
    loop: int = 0,
):

    frames = []
    for obj in figure_list:
        fig: Figure = obj.figure if isinstance(obj, Axes) else obj  
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=fig.dpi)  
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGBA"))

    if not frames:
        raise ValueError("figure_list can't be empty.")

    duration = int(round(1000 / fps))  
    frames[0].save(
        outfile,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )
    print(
        f"output {outfile} fps≈{round(1000/duration, 2)}，"
        f"loop={'infinite' if loop == 0 else loop}）"
    )





# not use bu just in case
"""
def ptrace_out_cp(X, d_in=2, d_out=2):
    X4 = cp.reshape(X, (d_out, d_in, d_out, d_in))   # [i,k,j,l]
    rho_in = 0
    for i in range(d_out):
        rho_in += X4[i, :, i, :]
    return rho_in


def project_to_cp_tp(choi_raw, d=2):
    X = cp.Variable((d*d, d*d), hermitian=True)
    mu = cp.Variable((1))
    #objective = cp.Minimize(mu)
    objective = cp.Minimize(cp.norm(X - choi_raw, 'fro'))
    constraints = [X >> 1e-4] 
    #constraints += [mu >= 0]
    #constraints += [mu*np.eye(d*d) >> X-choi_raw]  
    #constraints += [-mu*np.eye(d*d) << X-choi_raw]  
    # TP: Tr_out(X) = I_d
    constraints += [ptrace_out_cp(X, d_in=d, d_out=d) == 0.5*np.eye(d)]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)   # 或 'CVXOPT' / 'MOSEK'

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError("Projection failed:", prob.status)

    return X.value

def project_to_cp_tp_count(choi_list,iterations=100):
    new_choi_list = []
    for choi in choi_list:
        # 投影回 CP（positive semidefinite）
        count = 0
        r_psd = project_to_cp_tp(choi)
        while eigvalsh(r_psd).min() < 0 :
            r_psd = project_to_cp_tp(r_psd)
            count += 1
            if count > iterations:
                print("Warning: Projection took too many iterations, may not converge.")
                break
        print(f"Iteration {count}: min eigenvalue = {eigvalsh(r_psd).min()}")
        new_choi_list.append(r_psd)
    return new_choi_list

def project_to_cptp_1q(dm_list,dims=(2, 2),method="cvxpy"):
        corrected_dm = []
        if method == "qiskit":
            from qiskit.quantum_info.states.utils import closest_density_matrix
            corrected_dm = [closest_density_matrix(dm_i, norm="fro") for dm_i in dm_list]
        elif method == "qutip":
            from qutip.tomography import maximum_likelihood_estimate
            corrected_dm = [maximum_likelihood_estimate(dm_i, basis="pauli") for dm_i in dm_list]
        elif method == "cvxpy":
            for dm_i in dm_list:
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

"""