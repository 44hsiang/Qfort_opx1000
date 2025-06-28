# coding=utf-8
"""
Create date: 2025/6/22
CityPlot.py:
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from quam_libs.components import QuAM
from quam_libs.quantum_memory.marcos import density_matrix_to_bloch_vector, MLE, project_to_cptp_1q


machine = QuAM.load("./state_20250624.json")
config = machine.generate_config()
qubit = machine.qubits["q3"]

def mitigation(rho):
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    axis_projection = density_matrix_to_bloch_vector(rho)
    coef = np.real(-0.5 * (axis_projection - 1))
    conf_mat = np.kron(np.array([1]), qubit.resonator.confusion_matrix)
    corr_MLE = np.array([
        MLE(np.array([1 - coef[0], coef[0]]), conf_mat),
        MLE(np.array([1 - coef[1], coef[1]]), conf_mat),
        MLE(np.array([1 - coef[2], coef[2]]), conf_mat)
    ])
    coef_MLE = 1 - 2 * (corr_MLE[:, 1])
    dens_MLE = 0.5 * (I + coef_MLE[0] * X + coef_MLE[1] * Y + coef_MLE[2] * Z)
    dens_MLE_cptp, _ = project_to_cptp_1q([dens_MLE])
    return dens_MLE_cptp.reshape(2, 2)

font = {'weight' : 'bold',
        'size'   : 25}

matplotlib.rc('font', **font)

A = "-i"
B = "+i"
data_type = "mitigation"
density_matrix_0 = np.array(
    [
        [
            [0.47931667 + 0.j, 0.00238333 - 0.3242j],
            [0.00238333 + 0.3242j, 0.52068333 + 0.j]
        ],
        [
            [0.48215 + 0.j, 0.07035 - 0.27986667j],
            [0.07035 + 0.27986667j, 0.51785 + 0.j]
        ],
        [
            [0.49826667 + 0.j, 0.0367 - 0.10715j],
            [0.0367 + 0.10715j, 0.50173333 + 0.j]
        ],
        [
            [0.548 + 0.j, 0.07455 + 0.01791667j],
            [0.07455 - 0.01791667j, 0.452 + 0.j]
        ],
        [
            [0.65315 + 0.j, -0.04295 + 0.0376j],
            [-0.04295 - 0.0376j, 0.34685 + 0.j]
        ]
    ]
)

density_matrix_1 = np.array(
    [
        [
            [0.46058333 + 0.j, -0.04951667 + 0.3761j],
            [-0.04951667 - 0.3761j, 0.53941667 + 0.j]
        ],
        [
            [0.46563333 + 0.j, -0.12868333 + 0.33486667j],
            [-0.12868333 - 0.33486667j, 0.53436667 + 0.j]
        ],
        [
            [0.48376667 + 0.j, -0.09911667 + 0.15445j],
            [-0.09911667 - 0.15445j, 0.51623333 + 0.j]
        ],
        [
            [0.5555 + 0.j, -0.1346 + 0.0229j],
            [-0.1346 - 0.0229j, 0.4445 + 0.j]
        ],
        [
            [0.66041667 + 0.j, -0.03265 + 0.02646667j],
            [-0.03265 - 0.02646667j, 0.33958333 + 0.j]
        ]
    ]
)

common_x = np.arange(0.5, 1.6, 1)
common_y = np.arange(0.5, 1.6, 1)

common_X, common_Y = np.meshgrid(common_x, common_y)
delay_list = [0, 200, 1000, 5000, 20000]

vmin, vmax = -0.1, 1
fig1 = plt.figure(figsize=(28, 10))
ax = fig1.subplots(2, 6)
plt.suptitle(f"Density matrix of |{A}> {data_type}", fontweight='bold')

ax[0, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
ax[0, 0].text(0.5, 0.5, "Real", ha="center", va="center")
ax[0, 0].axis("off")
ax[1, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
ax[1, 0].text(0.5, 0.5, "Imag", ha="center", va="center")
ax[1, 0].axis("off")

for i in range(1, 6):
    dm = density_matrix_0[i - 1]
    if data_type == "mitigation":
        dm = mitigation(dm)
    real = dm.real
    imag = dm.imag
    im = ax[0, i].imshow(real, vmin=vmin, vmax=vmax, cmap="rainbow")
    ax[0, i].set_title(f"{delay_list[i-1]} ns")
    ax[0, i].set_xticks([0, 1])
    ax[0, i].set_yticks([0, 1])
    ax[1, i].imshow(imag, vmin=vmin, vmax=vmax, cmap="rainbow")
    ax[1, i].set_xticks([0, 1])
    ax[1, i].set_yticks([0, 1])
    for j in range(2):
        for k in range(2):
            c = "k" if j == k else "w"
            ax[0, i].text(j, k, f"{real[j, k]:.2f}", ha="center", va="center", color=c)
    for j in range(2):
        for k in range(2):
            c = "k" if j == k else "w"
            ax[1, i].text(j, k, f"{imag[j, k]:.2f}", ha="center", va="center", color=c)

fig1.subplots_adjust(right=0.8)
cbar_ax = fig1.add_axes([0.82, 0.15, 0.02, 0.7])
fig1.colorbar(im, cax=cbar_ax)
plt.show()

fig2 = plt.figure(figsize=(28, 10))
ax = fig2.subplots(2, 6)
plt.suptitle(f"Density matrix of |{B}> {data_type}", fontweight='bold')

ax[0, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
ax[0, 0].text(0.5, 0.5, "Real", ha="center", va="center")
ax[0, 0].axis("off")
ax[1, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
ax[1, 0].text(0.5, 0.5, "Imag", ha="center", va="center")
ax[1, 0].axis("off")

for i in range(1, 6):
    dm = density_matrix_1[i - 1]
    if data_type == "mitigation":
        dm = mitigation(dm)
    real = dm.real
    imag = dm.imag
    im = ax[0, i].imshow(real, vmin=vmin, vmax=vmax, cmap="rainbow")
    ax[0, i].set_title(f"{delay_list[i-1]} ns")
    ax[0, i].set_xticks([0, 1])
    ax[0, i].set_yticks([0, 1])
    ax[1, i].imshow(imag, vmin=vmin, vmax=vmax, cmap="rainbow")
    ax[1, i].set_xticks([0, 1])
    ax[1, i].set_yticks([0, 1])
    for j in range(2):
        for k in range(2):
            c = "k" if j == k else "w"
            ax[0, i].text(j, k, f"{real[j, k]:.2f}", ha="center", va="center", color=c)
    for j in range(2):
        for k in range(2):
            c = "k" if j == k else "w"
            ax[1, i].text(j, k, f"{imag[j, k]:.2f}", ha="center", va="center", color=c)

fig2.subplots_adjust(right=0.8)
cbar_ax = fig2.add_axes([0.82, 0.15, 0.02, 0.7])
fig2.colorbar(im, cax=cbar_ax)

plt.show()
