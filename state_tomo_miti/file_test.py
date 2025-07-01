# coding=utf-8
"""
Create date: 2025/6/26
file_test.py:
"""
import xarray as xr
import numpy as np
from quam_libs.quantum_memory.marcos import dm_checker

a = np.load("./arrays_0628.npz")
# print(a["results.q3.density_matrix"])
# print(a["results.q3.density_matrix_mitigation"])

dmm = a["results.q3.density_matrix_mitigation"]
print(dmm[0])
print(dm_checker(dmm[0]))
