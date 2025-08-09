import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from quam_libs.fit_ellipsoid import ls_ellipsoid, polyToParams3D

class EllipsoidTool:
    """
    Fit the ellipsoid from the Bloch vector data.
    """
    def __init__(self, bloch_vector,convex=True):
        if convex:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(bloch_vector, qhull_options='QJ')
            self.bloch_vector = np.array(bloch_vector[hull.vertices]).reshape(-1, 3)
        else:
            self.bloch_vector = np.array(bloch_vector).reshape(-1, 3)

    def fit(self):
        x = self.bloch_vector[:,0]
        y = self.bloch_vector[:,1]
        z = self.bloch_vector[:,2]

        param = ls_ellipsoid(x,y,z)
        param = param / np.linalg.norm(param) 
        center,axes,R = polyToParams3D(param,False)
        volume = (4/3)*np.pi*axes[0]*axes[1]*axes[2]
        return center,axes,R,volume,param

    def plot(self, ax=None, title=None, *, axes=None, center=None, R=None,
             show_points=True, show_unit_sphere=True):
        """
        Plot an ellipsoid on a 3D axis.

        Default behavior (backward compatible):
            - If `axes`, `center`, `R` are not provided, fit from
              `self.bloch_vector` and plot the fitted ellipsoid together with
              the data points and the unit Bloch sphere.

        New behavior:
            - If `axes`, `center`, and `R` are provided, plot that ellipsoid
              directly (optionally overlaying points/unit sphere).

        Args:
            ax: existing matplotlib 3D axis. If None, one will be created.
            title (str): plot title.
            axes (array-like of length 3): semi-axes (a, b, c).
            center (array-like of length 3): ellipsoid center (x0, y0, z0).
            R (3x3 array-like): rotation matrix (columns are principal axes).
            show_points (bool): scatter the Bloch vectors.
            show_unit_sphere (bool): draw the unit Bloch sphere wireframe.
        Returns:
            The matplotlib 3D axis used.
        """
        # Prepare axis if needed
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # Parameterization grid
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones_like(u), np.cos(v))

        # Determine which ellipsoid to draw
        if axes is None or center is None or R is None:
            # Backward-compatible path: fit from data
            if show_points and self.bloch_vector is not None:
                x = self.bloch_vector[:, 0]
                y = self.bloch_vector[:, 1]
                z = self.bloch_vector[:, 2]
                ax.scatter(x, y, z, color='blue', s=8, label='Bloch Vector')

            center, axes, R, volume, param = self.fit()
        else:
            # Use provided parameters
            axes = np.asarray(axes, dtype=float).reshape(3)
            center = np.asarray(center, dtype=float).reshape(3)
            R = np.asarray(R, dtype=float).reshape(3, 3)
            if show_points and self.bloch_vector is not None:
                x = self.bloch_vector[:, 0]
                y = self.bloch_vector[:, 1]
                z = self.bloch_vector[:, 2]
                ax.scatter(x, y, z, color='blue', s=8, label='Bloch Vector')

        # Build ellipsoid surface from (axes, center, R)
        x_ell = axes[0] * sphere_x
        y_ell = axes[1] * sphere_y
        z_ell = axes[2] * sphere_z

        pts = np.vstack([x_ell.ravel(), y_ell.ravel(), z_ell.ravel()])
        pts_rot = R @ pts
        pts_rot += center[:, None]
        x_ell, y_ell, z_ell = pts_rot.reshape(3, *sphere_x.shape)

        # Unit Bloch sphere
        if show_unit_sphere:
            ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color='blue', alpha=0.05, label='Bloch Sphere')

        # Ellipsoid wireframe
        ax.plot_wireframe(x_ell, y_ell, z_ell, color='red', alpha=0.08, label='Fitted Ellipsoid')

        # Cosmetics
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.set_title('Bloch Sphere Ellipsoid Fit' if title is None else title)
        try:
            ax.legend()
        except Exception:
            pass
        return ax
