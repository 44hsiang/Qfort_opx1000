import numpy as np
from .fit_ellipsoid import ls_ellipsoid, polyToParams3D
from quam_libs.quantum_channel_utils import compute_choi_state, compute_memory_robustness


class EllipsoidTool:
    """Ellipsoid fitting utility with optional pre-filtering and rotation correction.

    Parameters
    ----------
    bloch_vector : array-like, shape (N, 3)
        Input Bloch vectors.
    filter_method : {'none', 'convex', 'ransac'}, default='none'
        - 'none': direct fit with all points.
        - 'convex': fit using convex-hull vertices.
        - 'ransac': RANSAC-like outlier rejection before fitting.
    ransac_threshold : float, default=0.05
        Inlier threshold for `filter_method='ransac'`.
    ransac_iterations : int, default=1000
        Iteration count for `filter_method='ransac'`.
    random_state : int or None, default=None
        Random seed used by RANSAC sampling.
    correct_rotation_orientation : bool, default=False
        If True, correct rotation matrix such that x and y components align with
        the corresponding axes. Ensures determinant is preserved.
    """

    VALID_FILTER_METHODS = {"none", "convex", "ransac"}

    def __init__(
        self,
        bloch_vector,
        filter_method="none",
        ransac_threshold=0.05,
        ransac_iterations=1000,
        random_state=None,
        correct_rotation_orientation=False,
        find_best_order=False
    ):
        points = np.asarray(bloch_vector, dtype=float).reshape(-1, 3)
        if points.size == 0:
            raise ValueError("`bloch_vector` is empty.")

        method = str(filter_method).lower().strip()
        self.original_bloch_vector = points
        self.filter_method = method
        self.ransac_threshold = float(ransac_threshold)
        self.ransac_iterations = int(ransac_iterations)
        self.random_state = random_state
        self.correct_rotation_orientation = bool(correct_rotation_orientation)
        self.find_best_order = bool(find_best_order)
        self.bloch_vector = points
        self.outliers = None
        self.ransac_coeff = None

        self._apply_filter()
        self.results = self.fit_results

    def _apply_filter(self):
        method = self.filter_method

        if method == "none":
            return

        elif method == "convex":
            from scipy.spatial import ConvexHull

            hull = ConvexHull(self.original_bloch_vector, qhull_options="QJ")
            self.bloch_vector = self.original_bloch_vector[hull.vertices]
            return

        elif method == "ransac":
            inliers, outliers, coeff = self.filter_ellipsoid_outliers(
                self.original_bloch_vector,
                threshold=self.ransac_threshold,
                n_iterations=self.ransac_iterations,
                random_state=self.random_state,
            )
            self.bloch_vector = inliers
            self.outliers = outliers
            self.ransac_coeff = coeff
            return

        raise ValueError(
            f"Unsupported filter_method='{method}'. "
            "Only these are allowed: 'none', 'convex', 'ransac'."
        )

    def fit(self):
        if self.bloch_vector.shape[0] < 9:
            raise ValueError("Need at least 9 points to fit an ellipsoid.")

        x = self.bloch_vector[:, 0]
        y = self.bloch_vector[:, 1]
        z = self.bloch_vector[:, 2]

        param = ls_ellipsoid(x, y, z)
        param = param / np.linalg.norm(param)
        center, axes, R = polyToParams3D(param, False)
        volume = (4.0 / 3.0) * np.pi * axes[0] * axes[1] * axes[2]
        return center, axes, R, volume, param

    @staticmethod
    def _correct_rotation_orientation(R, axes):
        """Correct rotation matrix so that x and y components align with axes.
        
        If the rotation matrix's x-y components don't align properly with the
        corresponding axes, this function swaps and negates them while preserving
        the determinant.
        
        Parameters
        ----------
        R : ndarray, shape (3, 3)
            Rotation matrix where columns are the axes directions
        axes : ndarray, shape (3,)
            Semi-axes lengths in order [a, b, c]
            
        Returns
        -------
        R_corrected : ndarray, shape (3, 3)
            Corrected rotation matrix
        axes_corrected : ndarray, shape (3,)
            Possibly reordered axes
        """
        R = np.array(R, dtype=float)
        axes = np.array(axes, dtype=float)
        
        # Check if rotation matrix determinant is +1 (proper rotation)
        det_original = np.linalg.det(R)
        
        # Get the current x and y column vectors (first two columns of R)
        x_col = R[:, 0]
        y_col = R[:, 1]
        z_col = R[:, 2]
        
        # Check if x and y are aligned properly by checking the sign of their
        # cross product against z
        cross_xy = np.cross(x_col, y_col)
        alignment = np.dot(cross_xy, z_col)
        
        # If misaligned (negative dot product), need to flip
        if alignment < 0:
            # Swap x and y, and negate y to preserve determinant
            R_corrected = R.copy()
            R_corrected[:, 0] = y_col
            R_corrected[:, 1] = -x_col
            
            # Also swap the corresponding axes
            axes_corrected = axes.copy()
            axes_corrected[0], axes_corrected[1] = axes_corrected[1], axes_corrected[0]
            
            # Verify determinant is preserved
            det_corrected = np.linalg.det(R_corrected)
            if not np.isclose(abs(det_original), abs(det_corrected), atol=1e-10):
                # Try alternative: negate x instead
                R_corrected = R.copy()
                R_corrected[:, 0] = -x_col
                R_corrected[:, 1] = y_col
                axes_corrected = axes.copy()
                
            return R_corrected, axes_corrected
        else:
            return R, axes

    
    @staticmethod
    def find_optimal_ellipsoid_configuration(eigenvalues, eigenvectors, center):
        """
        Iterates through all permutations of ellipsoid axes to find the configuration 
        that yields the maximum quantum memory robustness.
        
        Args:
            eigenvalues: 1D array of 3 eigenvalues (axes lengths).
            eigenvectors: 3x3 matrix where columns are eigenvectors (rotation matrix).
            center: 1D array of the ellipsoid center coordinates.
            calc_robustness_func: A function that calculates and returns the robustness score.
            
        Returns:
            best_rotation_matrix: The 3x3 eigenvector matrix corresponding to the best permutation.
            best_axes: The 1D array of 3 eigenvalues corresponding to the best permutation.
        """
        import itertools
        # Initialize variables to record the optimal results
        best_robustness = -float('inf')
        best_axes = None
        best_rotation_matrix = None

        # Generate all permutations of indices [0, 1, 2] (3! = 6 combinations)
        indices = [0, 1, 2]
        for p in itertools.permutations(indices):
            p_list = list(p) 
            
            # 1. Reorder axes (eigenvalues)
            current_axes = eigenvalues[p_list]
            
            # 2. Reorder rotation matrix (eigenvectors columns)
            current_rotation_matrix = eigenvectors[:, p_list] 
            if np.linalg.det(current_rotation_matrix) > 0:
                current_rotation_matrix[:, 0] = -current_rotation_matrix[:, 0]
            # 3. Calculate Robustness
            choi = compute_choi_state(center, current_axes, current_rotation_matrix)
            robustness = compute_memory_robustness(choi)
            
            # 4. Update the optimal configuration
            if robustness > best_robustness:
                best_robustness = robustness
                best_axes = current_axes
                best_rotation_matrix = current_rotation_matrix

        return best_rotation_matrix, best_axes
        
    @property
    def fit_results(self):
        """Compute fit results and print a readable summary.

        Returns
        -------
        dict
            {
                'filter_method',
                'fit_param',
                'center',
                'axes',
                'rotation_matrix',
                'volume',
                'n_input_points',
                'n_fit_points',
                'n_outliers',
                'rotation_corrected' (if correct_rotation_orientation=True)
            }
        """
        center, axes, R, volume, param = self.fit()
        
        # Apply rotation correction if enabled
        if self.correct_rotation_orientation:
            R, axes = self._correct_rotation_orientation(R, axes)
            # Recalculate volume with corrected axes
            volume = (4.0 / 3.0) * np.pi * axes[0] * axes[1] * axes[2]
        
        if self.find_best_order:
            R, axes = self.find_optimal_ellipsoid_configuration(axes,R,center)
            # Recalculate volume with corrected axes
            volume = (4.0 / 3.0) * np.pi * axes[0] * axes[1] * axes[2]
        
        all_points_residual = self.get_quadric_error(self.original_bloch_vector, param)
        residual_stats = {
            "mean": float(np.mean(all_points_residual)),
            "median": float(np.median(all_points_residual)),
            "std": float(np.std(all_points_residual)),
            "p95": float(np.percentile(all_points_residual, 95)),
            "max": float(np.max(all_points_residual)),
        }
        result = {
            "filter_method": self.filter_method,
            "fit_param": param,
            "center": center,
            "axes": axes,
            "rotation_matrix": R,
            "volume": volume,
            "n_input_points": int(self.original_bloch_vector.shape[0]),
            "n_fit_points": int(self.bloch_vector.shape[0]),
            "n_outliers": 0 if self.outliers is None else int(self.outliers.shape[0]),
            "rotation_corrected": self.correct_rotation_orientation,
            #"all_points_abs_residual": all_points_residual,
            "all_points_abs_residual_stats": residual_stats,
        }
        #self._print_result(result)
        return result

    @staticmethod
    def _print_result(result):
        print("=== Ellipsoid Fit Result ===")
        print(f"filter_method: {result['filter_method']}")
        print(
            f"points: input={result['n_input_points']}, "
            f"fit={result['n_fit_points']}, outliers={result['n_outliers']}"
        )
        print("fit_param:")
        print(np.array2string(result["fit_param"], precision=6, suppress_small=True))
        print("center:")
        print(np.array2string(result["center"], precision=6, suppress_small=True))
        print("axes:")
        print(np.array2string(result["axes"], precision=6, suppress_small=True))
        print("rotation_matrix:")
        print(np.array2string(result["rotation_matrix"], precision=6, suppress_small=True))
        print(f"volume: {result['volume']:.6f}")
        if "all_points_abs_residual_stats" in result:
            stats = result["all_points_abs_residual_stats"]
            print("all_points_abs_residual_stats:")
            print(
                "  mean={mean:.6e}, median={median:.6e}, std={std:.6e}, p95={p95:.6e}, max={max:.6e}".format(
                    **stats
                )
            )

    def plot(
        self,
        ax=None,
        title=None,
        *,
        axes=None,
        center=None,
        R=None,
        show_points=True,
        show_unit_sphere=True,
    ):
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones_like(u), np.cos(v))

        if axes is None or center is None or R is None:
            center, axes, R, _, _ = self.fit()
        else:
            axes = np.asarray(axes, dtype=float).reshape(3)
            center = np.asarray(center, dtype=float).reshape(3)
            R = np.asarray(R, dtype=float).reshape(3, 3)

        if show_points and self.bloch_vector is not None:
            x = self.bloch_vector[:, 0]
            y = self.bloch_vector[:, 1]
            z = self.bloch_vector[:, 2]
            ax.scatter(x, y, z, color="blue", s=8, label="Bloch Vector")

        x_ell = axes[0] * sphere_x
        y_ell = axes[1] * sphere_y
        z_ell = axes[2] * sphere_z

        pts = np.vstack([x_ell.ravel(), y_ell.ravel(), z_ell.ravel()])
        pts_rot = R @ pts
        pts_rot += center[:, None]
        x_ell, y_ell, z_ell = pts_rot.reshape(3, *sphere_x.shape)

        if show_unit_sphere:
            ax.plot_wireframe(
                sphere_x,
                sphere_y,
                sphere_z,
                color="blue",
                alpha=0.05,
                label="Bloch Sphere",
            )

        ax.plot_wireframe(
            x_ell,
            y_ell,
            z_ell,
            color="red",
            alpha=0.08,
            label="Fitted Ellipsoid",
        )

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title("Bloch Sphere Ellipsoid Fit" if title is None else title)
        try:
            ax.legend()
        except Exception:
            pass
        return ax

    @staticmethod
    def fit_general_quadric(points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        D = np.column_stack(
            [
                x**2,
                y**2,
                z**2,
                x * y,
                x * z,
                y * z,
                x,
                y,
                z,
                np.ones_like(x),
            ]
        )
        _, _, Vt = np.linalg.svd(D)
        return Vt[-1, :]

    @staticmethod
    def get_quadric_error(points, coeff):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        A, B, C, D, E, F, G, H, I, J = coeff
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        equation_val = (
            A * x**2
            + B * y**2
            + C * z**2
            + D * x * y
            + E * x * z
            + F * y * z
            + G * x
            + H * y
            + I * z
            + J
        )
        return np.abs(equation_val)

    @classmethod
    def filter_ellipsoid_outliers(
        cls,
        points,
        threshold=0.05,
        n_iterations=1000,
        random_state=None,
    ):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        n_points = points.shape[0]
        if n_points < 10:
            return points, np.empty((0, 3)), None

        rng = np.random.default_rng(random_state)
        sample_size = min(15, n_points)

        best_inliers_mask = np.zeros(n_points, dtype=bool)
        best_inliers_count = 0
        best_coeff = None

        for _ in range(int(n_iterations)):
            sample_indices = rng.choice(n_points, sample_size, replace=False)
            sample_points = points[sample_indices]

            try:
                coeff = cls.fit_general_quadric(sample_points)
            except Exception:
                continue

            errors = cls.get_quadric_error(points, coeff)
            current_inliers_mask = errors < threshold
            current_inliers_count = int(np.sum(current_inliers_mask))

            if current_inliers_count > best_inliers_count:
                best_inliers_count = current_inliers_count
                best_inliers_mask = current_inliers_mask
                best_coeff = coeff

        if best_coeff is None:
            return points, np.empty((0, 3)), None

        if np.sum(best_inliers_mask) >= 10:
            final_inliers = points[best_inliers_mask]
            best_coeff = cls.fit_general_quadric(final_inliers)
            final_errors = cls.get_quadric_error(points, best_coeff)
            best_inliers_mask = final_errors < threshold

        inliers = points[best_inliers_mask]
        outliers = points[~best_inliers_mask]

        if inliers.shape[0] < 9:
            return points, np.empty((0, 3)), None

        return inliers, outliers, best_coeff


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    rng = np.random.default_rng(42)

    # Build synthetic ellipsoid-like data
    n_points = 500
    theta = rng.uniform(0, 2 * np.pi, n_points)
    phi = np.arccos(rng.uniform(-1, 1, n_points))

    unit = np.column_stack(
        [
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ]
    )

    true_axes = np.array([0.80, 0.55, 0.35])
    true_center = np.array([0.08, -0.06, 0.03])
    points = unit * true_axes + true_center
    points += 0.01 * rng.normal(size=points.shape)

    # Add some outliers for RANSAC demonstration
    n_outliers = 40
    outliers = rng.uniform(low=-1.2, high=1.2, size=(n_outliers, 3))
    test_points = np.vstack([points, outliers])

    print("=== Run EllipsoidTool test ===")
    for method in ("none", "convex", "ransac"):
        print(f"\n>>> filter_method = {method}")
        try:
            tool = EllipsoidTool(
                test_points,
                filter_method=method,
                ransac_threshold=0.03,
                ransac_iterations=1500,
                random_state=42,
            )
            _ = tool.result
        except Exception as exc:
            print(f"[ERROR] {method}: {exc}")

