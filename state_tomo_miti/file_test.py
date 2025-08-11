# coding=utf-8
"""
Create date: 2025/6/26
file_test.py:
"""
import xarray as xr
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from quam_libs.quantum_memory.marcos import dm_checker, project_to_cptp_1q


def cptp(dm, dims):
    X = cp.Variable(dims, hermitian=True)
    obj = cp.Minimize(cp.norm(X - dm, "fro"))
    constraints = [
        X >> 0,
        cp.trace(X) == 1
    ]
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Projection fail: {prob.status}")
    return X.value

a = np.load("./BellState.npz")
print(list(a.keys()))

q0q2_conf_mat = a["results.q0_q2.confusion matrix"]
q1q2_conf_mat = a["results.q1_q2.confusion matrix"]
q2q3_conf_mat = a["results.q2_q3.confusion matrix"]
q2q4_conf_mat = a["results.q2_q4.confusion matrix"]

q0q2_miti_dm = a["results.q0_q2.MLE"]
q1q2_miti_dm = a["results.q1_q2.MLE"]
q2q3_miti_dm = a["results.q2_q3.MLE"]
q2q4_miti_dm = a["results.q2_q4.MLE"]

q0q2_dm = q0q2_conf_mat @ q0q2_miti_dm
q1q2_dm = q1q2_conf_mat @ q1q2_miti_dm
q2q3_dm = q2q3_conf_mat @ q2q3_miti_dm
q2q4_dm = q2q4_conf_mat @ q2q4_miti_dm

q0q2_cptp_dm = cptp(q0q2_miti_dm, dims=(4, 4))
q1q2_cptp_dm = cptp(q1q2_miti_dm, dims=(4, 4))
q2q3_cptp_dm = cptp(q2q3_miti_dm, dims=(4, 4))
q2q4_cptp_dm = cptp(q2q4_miti_dm, dims=(4, 4))
print(q0q2_cptp_dm.shape)

fig = plt.figure(figsize=(6,6))
ax = plt.subplot(projection='3d')

x = np.linspace(-1, 1, 4)
y = np.linspace(-1, 1, 4)
x, y = np.meshgrid(x, y)
z = 0

ax.bar3d(x.ravel(),y.ravel(),z,dx=0.5,dy=0.5,dz=q0q2_cptp_dm.ravel().real)

plt.show()

np.savez(
    "./bells.npz",
    conf_q0q2=q0q2_conf_mat,conf_q1q2=q1q2_conf_mat,conf_q2q3=q2q3_conf_mat,conf_q2q4=q2q4_conf_mat,
    raw_q0q2=q0q2_dm, raw_q1q2=q1q2_dm, raw_q2q3=q2q3_dm, raw_q2q4=q2q4_dm,
    miti_q0q2=q0q2_cptp_dm, miti_q1q2=q1q2_cptp_dm, miti_q2q3=q2q3_cptp_dm, miti_q2q4=q2q4_cptp_dm,
)
