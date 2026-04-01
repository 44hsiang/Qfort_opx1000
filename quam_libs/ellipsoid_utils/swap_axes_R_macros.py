import numpy as np

def align_max_axis_to_z(R, axes):
    """
    Force the largest axis in axes to the Z-axis (index 2),
    and repair the left-handed system with det(R) = -1 to the right-handed system (det = 1)
    through a single swap operation.
    """
    # Copy arrays to avoid modifying the original data
    R_new = R.copy()
    axes_new = axes.copy()

    # Find the index of the largest axis length
    max_idx = np.argmax(axes_new)

    if max_idx == 2:
        # Case 1: Maximum value is already on Z-axis (index 2)
        # Operation: Swap X (0) and Y (1) to fix the determinant
        
        # Swap R's columns
        R_new[:, [0, 1]] = R_new[:, [1, 0]]
        # Swap axes
        axes_new[0], axes_new[1] = axes_new[1], axes_new[0]
        
    else:
        # Case 2: Maximum value is at X (0) or Y (1)
        # Operation: Swap the maximum value's axis with Z-axis (2)
        
        # Swap R's columns
        R_new[:, [max_idx, 2]] = R_new[:, [2, max_idx]]
        # Swap axes
        axes_new[max_idx], axes_new[2] = axes_new[2], axes_new[max_idx]

    return R_new, axes_new


def swap_axes_by_index(R,axes, idx1, idx2):
    """
    Swap the specified two coordinate axis labels and rows of the rotation matrix using indices (0, 1, 2).
    0: X-axis, 1: Y-axis, 2: Z-axis
    """
    # 1. Copy data to avoid polluting the original variables
    new_axes = list(axes)
    new_R = np.array(R, copy=True)
    
    # 2. Simply swap the axes label strings
    new_axes[idx1], new_axes[idx2] = new_axes[idx2], new_axes[idx1]
    
    # 3. Simply swap the two rows of R matrix
    new_R[:,[idx1, idx2]] = new_R[:,[idx2, idx1]]
    
    return new_R, new_axes


def sort_axes_ascending(R, axes):
    """
    Sort axes in ascending order (smallest to largest: x_min, y_mid, z_max).
    Correspondingly reorder the columns of rotation matrix R (eigenvectors).
    
    The rotation matrix columns represent the eigenvectors of the ellipsoid,
    so they must be reordered to match the axes ordering.
    
    Parameters:
    -----------
    R : np.ndarray
        Rotation matrix (3x3), where each column is an eigenvector
    axes : np.ndarray or list
        Ellipsoid semi-axes lengths [a, b, c]
        
    Returns:
    --------
    R_sorted : np.ndarray
        Rotation matrix with columns reordered according to ascending axes
    axes_sorted : np.ndarray
        Axes sorted in ascending order
    """
    # Get the indices that would sort axes in ascending order
    sort_indices = np.argsort(axes)
    
    # Sort axes
    axes_sorted = np.array(axes)[sort_indices]
    
    # Reorder columns of R according to the sorting indices
    # Each column of R is an eigenvector corresponding to an axis
    R_sorted = R[:, sort_indices].copy()
    if np.linalg.det(R_sorted) < 0:
        # If the determinant is negative, we have a left-handed system.
        # We can fix this by flipping the sign of one column (e.g., the first column).
        R_sorted[:, 2] *= -1
    return R_sorted, axes_sorted

