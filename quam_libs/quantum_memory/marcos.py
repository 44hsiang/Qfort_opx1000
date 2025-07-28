from numpy.linalg import eigvalsh, norm
import numpy as np
import pandas as pd
import cvxpy as cp
import pennylane as qml
from quam_libs.lib.fit import fit_decay_exp, decay_exp
from scipy.optimize import minimize
import numpy as np
from numpy.linalg import eigvalsh

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

theta_range = np.arange(0,np.pi,1e-4)
phi_range = np.arange(0,2*np.pi,1e-4)
def theta_phi_list(n_points):
    theta_list,phi_list = [],[]
    for i in range(n_points):
        theta,phi = np.random.choice(theta_range),np.random.choice(phi_range)
        theta_list.append(theta)
        phi_list.append(phi)
    return theta_list,phi_list

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
def BR_state_vecotr(alpha,beta,T1,T2,t):
    return np.array(
        [[1+(alpha**2-1)*np.exp(-t/T1),alpha*np.conj(beta)*np.exp(-t/T2)],
        [np.conj(alpha)*beta*np.exp(-t/T2),beta*np.conj(beta)*np.exp(-t/T1)]])
        
def theta_phi_to_alpha_beta(theta, phi):
    alpha = np.cos(theta/2)
    beta = np.sin(theta/2)*np.exp(1j*phi)
    return alpha, beta

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

def ptrace_out_cp(X, d_in=2, d_out=2):
    X4 = cp.reshape(X, (d_out, d_in, d_out, d_in))   # [i,k,j,l]
    rho_in = 0
    for i in range(d_out):
        rho_in += X4[i, :, i, :]
    return rho_in
'''
def project_to_cp_tp(choi_raw, d=2):
    X = cp.Variable((d*d, d*d), hermitian=True)

    objective = cp.Minimize(cp.norm(X - choi_raw, 'fro'))
    constraints = [X >> 1e-6]                            # CP

    # TP: Tr_out(X) = I_d
    constraints += [ptrace_out_cp(X, d_in=d, d_out=d) == 0.5*np.eye(d)]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)   # 或 'CVXOPT' / 'MOSEK'

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError("Projection failed:", prob.status)

    return X.value
'''

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

def T1(node,index):
    T1=[]
    for i in index:
        node_qm = node.load_from_id(i)
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

    return np.array(T1).reshape(-1)

def T2(node,index):
    T2= []
    for i in index:
        node_qm = node.load_from_id(i)
        T2.append(node_qm.results['fit_results']['q0']['decay'])
    return T2

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

import imageio.v3 as iio  # imageio 第 3 版 API；也可改用 imageio.mimsave
from pathlib import Path

def make_gif(image_paths, output_path, duration=0.5, loop=0):

    """
    將 image_paths 裡的圖片依序合成 GIF。

    Parameters
    ----------
    image_paths : list[str | Path]
        圖片檔案路徑（完整路徑或相對路徑都可），順序就是播放順序。
    output_path : str | Path
        產出的 GIF 檔案路徑，副檔名請給 .gif。
    duration : float | list[float], optional
        每一張圖片顯示的秒數（或一個與 image_paths 等長的 list，逐張指定），預設 0.5 s。
    loop : int, optional
        動畫迴圈次數；0 代表無限迴圈，1 代表播完一次就停，依此類推，預設 0。
    """
    # 轉成 Path 物件方便檢查
    image_paths = [Path(p) for p in image_paths]
    assert all(p.exists() for p in image_paths), "有找不到的圖片路徑！"

    # 讀取所有 frame（支援各種格式：png/jpg/pdf…）
    frames = [iio.imread(p) for p in image_paths]

    # 寫出 gif
    iio.imwrite(
        output_path,
        frames,
        format="GIF",
        duration=duration,
        loop=loop,
    )

    print(f"✅ GIF 已生成：{output_path}")

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
    D = 2 * Q[0, 1]
    E = 2 * Q[0, 2]
    F = 2 * Q[1, 2]
    G, H, I = linear

    return np.array([A, B, C, D, E, F, G, H, I, J])


def ellipsoid_equation(r,param):
    x, y, z = r
    return (param[0] * x**2 + param[1] * y**2 + param[2] * z**2 +
            param[3] * x * y + param[4] * x * z + param[5] * y * z +
            param[6] * x + param[7] * y + param[8] * z + param[9])

def generate_uniform_sphere_angles(n_points):

    indices = np.arange(0, n_points)
    golden_angle = np.pi * (3 - np.sqrt(5))  

    z = 1 - 2 * (indices + 0.5) / n_points     
    theta = np.arccos(z)                      
    phi = (indices * golden_angle) % (2 * np.pi)  

    theta_list = theta.tolist()
    phi_list = phi.tolist()

    return theta_list, phi_list

import io
from typing import List, Union

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PIL import Image


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
    