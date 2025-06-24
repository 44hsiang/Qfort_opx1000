# coding=utf-8
"""
Create date: 2025/6/22
CityPlot.py:
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'weight' : 'bold',
        'size'   : 25}

matplotlib.rc('font', **font)

density_matrix_fieldless = np.array(
    [
        [
            [0.12988333 + 0.j, - 0.02243333 - 0.0042j],
            [-0.02243333 + 0.0042j,  0.87011667 + 0.j]
        ],
        [
            [0.13511667 + 0.j, - 0.02815 - 0.00505j],
            [-0.02815 + 0.00505j,  0.86488333 + 0.j]
        ],
        [
            [ 0.1507    +0.j,         -0.00371667+0.00041667j],
            [-0.00371667-0.00041667j,  0.8493    +0.j        ]
        ],
        [
            [ 0.2097    +0.j,         -0.02828333+0.00601667j],
            [-0.02828333-0.00601667j,  0.7903    +0.j        ]
        ],
        [
            [ 0.31203333+0.j,         -0.0332    +0.02963333j],
            [-0.0332    -0.02963333j,  0.68796667+0.j        ]
        ]
    ]
)

density_matrix_sweepfield = np.array(
    [
        [
            [0.133 + 0.j, - 0.04013333 + 0.10578333j],
            [-0.04013333 - 0.10578333j,  0.867 + 0.j]
        ],
        [
            [ 0.13756667+0.j,         -0.0563    +0.10346667j],
            [-0.0563    -0.10346667j,  0.86243333+0.j        ]
        ],
        [
            [0.15576667 + 0.j, - 0.06923333 + 0.06956667j],
            [-0.06923333 - 0.06956667j,  0.84423333 + 0.j]
        ],
        [
            [0.22315 + 0.j, - 0.07573333 + 0.06321667j],
            [-0.07573333 - 0.06321667j,  0.77685 + 0.j]
        ],
        [
            [0.32016667 + 0.j, - 0.0565 + 0.04128333j],
            [-0.0565 - 0.04128333j,  0.67983333 + 0.j]
        ]
    ]
)

common_x = np.arange(0.5, 1.6, 1)
common_y = np.arange(0.5, 1.6, 1)

common_X, common_Y = np.meshgrid(common_x, common_y)
delay_list = [0, 100, 500, 2000, 5000]

vmin, vmax = -0.1, 1
fig1 = plt.figure(figsize=(28, 10))
ax = fig1.subplots(2, 6)
plt.suptitle("Density matrix under zero fields", fontweight='bold')

ax[0, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
ax[0, 0].text(0.5, 0.5, "Real", ha="center", va="center")
ax[0, 0].axis("off")
ax[1, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
ax[1, 0].text(0.5, 0.5, "Imag", ha="center", va="center")
ax[1, 0].axis("off")

for i in range(1, 6):
    real = density_matrix_fieldless[i-1].real
    imag = density_matrix_fieldless[i-1].imag
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

fig2 = plt.figure(figsize=(28, 10))
ax = fig2.subplots(2, 6)
plt.suptitle("Density matrix when sweeping fields", fontweight='bold')

ax[0, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
ax[0, 0].text(0.5, 0.5, "Real", ha="center", va="center")
ax[0, 0].axis("off")
ax[1, 0].imshow(np.zeros((2, 2)), vmin=-1, vmax=1, cmap="bwr")
ax[1, 0].text(0.5, 0.5, "Imag", ha="center", va="center")
ax[1, 0].axis("off")

for i in range(1, 6):
    real = density_matrix_sweepfield[i-1].real
    imag = density_matrix_sweepfield[i-1].imag
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
