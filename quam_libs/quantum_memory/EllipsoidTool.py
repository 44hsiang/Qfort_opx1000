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

    def plot(self, ax=None,title=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        u,v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
        ideal_x, ideal_y, ideal_z = (
            np.outer(np.cos(u), np.sin(v)), 
            np.outer(np.sin(u), np.sin(v)), 
            np.outer(np.ones_like(u), np.cos(v))
            )
        x = self.bloch_vector[:, 0]
        y = self.bloch_vector[:, 1]
        z = self.bloch_vector[:, 2]

        ax.scatter(x, y, z, color='blue', label='Bloch Vector')

        center, axes, R, volume, param = self.fit()
        x_ellipsoid = axes[0] * ideal_x
        y_ellipsoid = axes[1] * ideal_y
        z_ellipsoid = axes[2] * ideal_z

        ellipsoid_points_ = np.dot(R, np.array([x_ellipsoid.ravel(), y_ellipsoid.ravel(), z_ellipsoid.ravel()]))
        ellipsoid_points_ += center.reshape(-1, 1)
        x_ellipsoid, y_ellipsoid, z_ellipsoid = ellipsoid_points_.reshape(3, *x_ellipsoid.shape)

        ax.plot_wireframe(ideal_x, ideal_y, ideal_z, color="blue", alpha=0.05, label="Bloch Sphere")
        ax.plot_wireframe(x_ellipsoid, y_ellipsoid, z_ellipsoid, color="red", alpha=0.08, label="Fitted Ellipsoid")

        ax.set_title('Bloch Sphere Ellipsoid Fit' if title is None else title)
        ax.legend()
        return ax
