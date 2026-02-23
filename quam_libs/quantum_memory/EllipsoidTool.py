import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from quam_libs.fit_ellipsoid import ls_ellipsoid, polyToParams3D



class EllipsoidTool:
    """
    Fit the ellipsoid from the Bloch vector data.
    
    Args:
        bloch_vector: (N, 3) array of Bloch vector coordinates.
        convex (bool): If True, use convex hull filtering (original method).
                       Deprecated: use filter_method='convex' instead.
        filter_method (str): Method for filtering outliers. Options:
            - 'convex': Use ConvexHull to keep only hull vertices (default, original behavior).
            - 'ransac': Use RANSAC-based quadric fitting to filter outliers.
            - 'none' or None: No filtering, use all points.
        ransac_threshold (float): Error threshold for RANSAC inlier detection (default=0.05).
        ransac_iterations (int): Number of RANSAC iterations (default=1000).
    """
    def __init__(self, bloch_vector, convex=True, filter_method=None, 
                 ransac_threshold=0.05, ransac_iterations=1000):
        # Handle backward compatibility: if filter_method is not specified, use convex parameter
        if filter_method is None:
            filter_method = 'convex' if convex else 'none'
        
        bloch_vector = np.array(bloch_vector).reshape(-1, 3)
        
        if filter_method == 'convex':
            from scipy.spatial import ConvexHull
            hull = ConvexHull(bloch_vector, qhull_options='QJ')
            self.bloch_vector = bloch_vector[hull.vertices]
            self.outliers = None
            self.ransac_coeff = None
        elif filter_method == 'ransac':
            inliers, outliers, coeff = self.filter_ellipsoid_outliers(
                bloch_vector, 
                threshold=ransac_threshold, 
                n_iterations=ransac_iterations
            )
            self.bloch_vector = inliers
            self.outliers = outliers
            self.ransac_coeff = coeff
        else:  # 'none' or any other value
            self.bloch_vector = bloch_vector
            self.outliers = None
            self.ransac_coeff = None
        
        self.filter_method = filter_method

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

    @staticmethod
    def fit_general_quadric(points):
        """
        Fits a general quadric surface using Least Squares (SVD).
        Equation: Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
        
        Args:
            points: (N, 3) numpy array of x, y, z coordinates.
            
        Returns:
            10 coefficients [A, B, C, D, E, F, G, H, I, J]
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Construct the design matrix D (N x 10)
        # Order: x^2, y^2, z^2, xy, xz, yz, x, y, z, 1
        D = np.column_stack([
            x**2, y**2, z**2,
            x*y, x*z, y*z,
            x, y, z,
            np.ones_like(x)
        ])

        # Solve D * coeff = 0 using SVD
        # This finds the coefficient vector that minimizes the algebraic error
        # subject to ||coeff|| = 1
        U, S, Vt = np.linalg.svd(D)
        coeff = Vt[-1, :]  # The eigenvector corresponding to the smallest eigenvalue
        
        return coeff

    @staticmethod
    def get_quadric_error(points, coeff):
        """
        Calculates the algebraic error for the general quadric equation.
        Error = | Ax^2 + ... + J |
        
        Args:
            points: (N, 3) numpy array of x, y, z coordinates.
            coeff: 10 coefficients [A, B, C, D, E, F, G, H, I, J]
            
        Returns:
            Absolute algebraic error for each point.
        """
        A, B, C, D, E, F, G, H, I, J = coeff
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # Calculate value of the implicit function
        equation_val = (A*x**2 + B*y**2 + C*z**2 + 
                        D*x*y + E*x*z + F*y*z + 
                        G*x + H*y + I*z + J)
        
        return np.abs(equation_val)

    @classmethod
    def filter_ellipsoid_outliers(cls, points, threshold=0.05, n_iterations=1000):
        """
        Uses RANSAC to filter outliers from a general ellipsoid.
        
        Args:
            points: (N, 3) numpy array of x, y, z coordinates.
            threshold: The error threshold to consider a point an inlier.
            n_iterations: Number of RANSAC iterations.
            
        Returns:
            inliers: The filtered points (numpy array).
            outliers: The points identified as outliers (numpy array).
            best_coeff: The 10 coefficients of the fitted ellipsoid.
        """
        n_points = points.shape[0]
        # We need at least 10 points to fit 10 coefficients, 
        # but using slightly more (e.g., 12-15) in sampling is more stable.
        sample_size = 12 
        
        best_inliers_count = 0
        best_inliers_mask = np.zeros(n_points, dtype=bool)
        best_coeff = None

        for _ in range(n_iterations):
            # 1. Random Sampling
            sample_indices = np.random.choice(n_points, sample_size, replace=False)
            sample_points = points[sample_indices]

            # 2. Fit Model (General Quadric)
            try:
                coeff = cls.fit_general_quadric(sample_points)
            except Exception:
                continue  # Skip if SVD fails (rare)

            # 3. Calculate Errors
            errors = cls.get_quadric_error(points, coeff)

            # 4. Normalization (Important!)
            # Since the coefficients are normalized (sum of squares = 1), 
            # the algebraic error scale depends on the data spread.
            # We roughly normalize by the gradient magnitude approximation or just use raw if threshold is tuned.
            # Here we stick to simple algebraic distance for speed, 
            # but you might need to tune 'threshold' based on your data scale.
            
            # Identify inliers
            current_inliers_mask = errors < threshold
            current_inliers_count = np.sum(current_inliers_mask)

            # 5. Update Best Model
            if current_inliers_count > best_inliers_count:
                best_inliers_count = current_inliers_count
                best_inliers_mask = current_inliers_mask
                best_coeff = coeff

        # Final Step: Refit model using ALL best inliers to get the tightest fit
        if np.sum(best_inliers_mask) > sample_size:
            final_inliers = points[best_inliers_mask]
            best_coeff = cls.fit_general_quadric(final_inliers)
            # Re-evaluate mask with the refined model
            final_errors = cls.get_quadric_error(points, best_coeff)
            best_inliers_mask = final_errors < threshold

        inliers = points[best_inliers_mask]
        outliers = points[~best_inliers_mask]

        return inliers, outliers, best_coeff


# Keep standalone functions for backward compatibility
def get_quadric_error(points, coeff):
    """
    Calculates the algebraic error for the general quadric equation.
    Error = | Ax^2 + ... + J |
    
    Note: This is a standalone function for backward compatibility.
          Consider using EllipsoidTool.get_quadric_error() instead.
    """
    return EllipsoidTool.get_quadric_error(points, coeff)


def filter_ellipsoid_outliers(points, threshold=0.05, n_iterations=1000):
    """
    Uses RANSAC to filter outliers from a general ellipsoid.
    
    Note: This is a standalone function for backward compatibility.
          Consider using EllipsoidTool.filter_ellipsoid_outliers() instead.
    """
    return EllipsoidTool.filter_ellipsoid_outliers(points, threshold, n_iterations)


def fit_general_quadric(points):
    """
    Fits a general quadric surface using Least Squares (SVD).
    
    Note: This is a standalone function for backward compatibility.
          Consider using EllipsoidTool.fit_general_quadric() instead.
    """
    return EllipsoidTool.fit_general_quadric(points)

