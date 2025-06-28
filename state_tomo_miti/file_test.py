# coding=utf-8
"""
Create date: 2025/6/26
file_test.py:
"""
import xarray as xr
import numpy as np

a = np.load("./arrays.npz")
print(a["results.q3.density_matrix"])

b = xr.open_dataset("./ds.h5")
print(b.state.shape)
