from dataclasses import dataclass

from .QI_utils import CorrectBloch, CorrectChoi, entanglementRobustness
from .ellipsoid_utils.ellipsoid_utils import EllipsoidTool
from .quantum_channel_utils import compute_choi_state, compute_memory_robustness
import numpy as np

@dataclass
class EllipsoidFitParameters:
    filter_method: str = "ransac"
    ransac_threshold: float = 0.05
    ransac_iterations: int = 1000
    random_state: int | None = None
    correct_rotation_orientation: bool = False
    find_best_order: bool = False

class QuantumMemoryAnalyze:
    
    def __init__(self, measurement_data, ideal_data, ellipsoid_fit_parameters=None):
        self.measurement_data = measurement_data
        self.ideal_data = ideal_data
        self.ellipsoid_fit_parameters = self._normalize_ellipsoid_fit_parameters(
            ellipsoid_fit_parameters
        )
        self.corrected_dm, self.corrected_bloch = self.valid_data()
        self.ellipsoid_result = self.ellipsoid_fit_results()
        center, axes, R = self.ellipsoid_result['center'], self.ellipsoid_result['axes'], self.ellipsoid_result['rotation_matrix']
        # Store parameters for lazy evaluation (matching Legacy behavior)
        self._center = center
        self._axes = axes
        self._R = R
        self._choi = None
        self._memory_robustness = None
    def valid_data(self,method="manual"):
        """
        project measurement data to density matrix by using MLE,
        and make sure density matrix is
        1. Hermitian => U = U^dagger
        2. Positive semi-definite => all eigenvalues are non-negative
        3. Trace = 1 => sum of diagonal elements is 1
        Returns:
            valid density matrix
            valid Bloch vector
        """ 
        corrector = CorrectBloch(self.measurement_data)
        corrected_dm, corrected_bloch = corrector.project_to_cptp()
        return corrected_dm, corrected_bloch


# ellipsoid fitting and plotting
    @staticmethod
    def _normalize_ellipsoid_fit_parameters(ellipsoid_fit_parameters):
        if ellipsoid_fit_parameters is None:
            return EllipsoidFitParameters()

        if isinstance(ellipsoid_fit_parameters, EllipsoidFitParameters):
            return ellipsoid_fit_parameters

        if isinstance(ellipsoid_fit_parameters, dict):
            return EllipsoidFitParameters(**ellipsoid_fit_parameters)

        raise TypeError(
            "ellipsoid_fit_parameters must be None, EllipsoidFitParameters, or dict."
        )
    
    def _build_ellipsoid_tool(self):
        p = self.ellipsoid_fit_parameters
        return EllipsoidTool(
            self.corrected_bloch,
            filter_method=p.filter_method,
            ransac_threshold=p.ransac_threshold,
            ransac_iterations=p.ransac_iterations,
            random_state=p.random_state,
            correct_rotation_orientation=p.correct_rotation_orientation,
        )

    def ellipsoid_fit_results(self):
        """
        Fit the ellipsoid from input data(Bloch vector)
        """
        return self._build_ellipsoid_tool().results

    def ellipsoid_plot(self, ax=None,title=None):
        """
        Plot the ellipsoid
        """   
        return self._build_ellipsoid_tool().plot(ax=ax, title=title)

    # quantum channel analysis (lazy evaluation)
    @property
    def choi(self):
        """Lazy-loaded Choi state (computed on first access, uncorrected like Legacy)."""
        if self._choi is None:
            # Import here to avoid circular imports at module level
            from .quantum_channel_utils import compute_choi_state_raw
            self._choi = compute_choi_state_raw(self._center, self._axes, self._R)
        return self._choi
    
    @choi.setter
    def choi(self, value):
        """Allow setting choi state directly."""
        self._choi = value
    
    @property
    def memory_robustness_property(self):
        """Lazy-loaded memory robustness (computed on first access)."""
        if self._memory_robustness is None:
            self._memory_robustness = compute_memory_robustness(self.choi)
        return self._memory_robustness
    
    @staticmethod
    def memory_robustness(choi):
        """Compute memory robustness from a Choi state (static method).
        
        Args:
            choi: Choi matrix
            
        Returns:
            Memory robustness value
        """
        return compute_memory_robustness(choi)
    
    @staticmethod
    def correct_choi(choi, repeat=100, tol=1e-4,print_reason=False):
        """Correct the Choi state using CorrectChoi (optional post-processing).
        
        Args:
            choi: Choi matrix to correct
            repeat: Number of iterations for correction
            tol: Tolerance for correction
            print_reason: Whether to print the reason for correction
        Returns:
            Corrected choi matrix
        """
        cc = CorrectChoi(choi)
        corrected_choi, count = cc.choi_checker(repeat=repeat, tol=tol,print_reason=print_reason)
        return corrected_choi
    
    @staticmethod
    def choi_state(center, axes, R):
        """Compute the Choi state from ellipsoid parameters (static method, uncorrected).
        
        Args:
            center: Center of the ellipsoid (Bloch vector)
            axes: Semi-axes of the ellipsoid
            R: Rotation matrix of the ellipsoid
            
        Returns:
            Uncorrected Choi matrix (matching Legacy behavior)
        """
        from .quantum_channel_utils import compute_choi_state_raw
        return compute_choi_state_raw(center, axes, R)

