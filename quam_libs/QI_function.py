
"""
This module provides functions for working with quantum states and density matrices, 
including generating random Bloch states, converting between representations, and 
calculating distances between density matrices.
Functions:
    random_bloch_state_uniform():
        Generate a random Bloch state uniformly distributed on the Bloch sphere.
    theta_phi(x, y, z):
        Convert Cartesian coordinates to spherical coordinates (theta, phi).
    bloch_to_density_matrix(bloch_vector):
        Convert a Bloch vector to a density matrix.
    trace_distance(rho, sigma):
        Calculate the trace distance between two density matrices.
"""
import numpy as np
from scipy.linalg import svd, sqrtm

def mitigation(conf_mat,x,y,z):
    """
    Mitigate the measurement error using the calibration matrix.
    Parameters:
        qubit (int): Qubit index.
        x (float): bincounts average in x np.array([state0,state1]).
        y (float): bincounts average in y.
        z (float): bincounts average in z.
    Returns:
        x_mitigated (float): Mitigated x component of the Bloch vector.
        y_mitigated (float): Mitigated y component of the Bloch vector.
        z_mitigated (float): Mitigated z component of the Bloch vector.
    """
    
    x_vector = np.array([1-x,x])
    y_vector = np.array([1-y,y])
    z_vector = np.array([1-z,z])
    x_mitigated = np.linalg.inv(conf_mat) @ np.array([1-x,x])
    y_mitigated = np.linalg.inv(conf_mat) @ np.array([1-y,y])
    z_mitigated = np.linalg.inv(conf_mat) @ np.array([1-z,z])
    x_mitigated = x_mitigated[1]
    y_mitigated = y_mitigated[1]
    z_mitigated = z_mitigated[1]

    return x_mitigated, y_mitigated, z_mitigated

def random_bloch_state_uniform():
    # Random phi in [0, 2π]
    phi = np.random.uniform(0, 2 * np.pi)
    # Random cos(theta) in [-1, 1]
    cos_theta = np.random.uniform(-1, 1)
    # Compute theta
    theta = np.arccos(cos_theta)
    
    # Construct the state
    psi = np.array([
        np.cos(theta / 2),
        np.exp(1j * phi) * np.sin(theta / 2)
    ])
    return theta,phi

class QuantumStateAnalysis:
    
    """
    Define a class for analyzing quantum states. 
    Rho and sigma are experimental and ideal density matrices, respectively.

    Parameters:
        measurement_data (list): [x, y, z] - Experimental Bloch vector components.
        ideal_data (list): [theta, phi] - Ideal Bloch vector parameters.
    """

    def __init__(self, measurement_data, ideal_data):
        self.x, self.y, self.z = measurement_data
        self.ideal_theta, self.ideal_phi = ideal_data

        self.theta, self.phi = self.theta_phi()
        self.bloch_vector = [self.x, self.y, self.z]
        self.ideal_bloch_vector = [
            np.sin(self.ideal_theta) * np.cos(self.ideal_phi),
            np.sin(self.ideal_theta) * np.sin(self.ideal_phi),
            np.cos(self.ideal_theta)
        ]

        self.rho, self.sigma = self.density_matrix()
        self.measured_dm,self.ideal_dm = self.density_matrix()

        self.fidelity = self.fidelity()
        self.trace_distance = self.trace_distance()
        

    def theta_phi(self):
        r = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        theta = np.arccos(self.z / 1)
        phi = np.arctan2(self.y, self.x)
        return theta, phi

    def density_matrix(self):
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Identity matrix
        identity = np.array([[1, 0], [0, 1]], dtype=complex)

        # Bloch vector components
        r_x, r_y, r_z = self.bloch_vector
        s_x, s_y, s_z = self.ideal_bloch_vector

        # Construct density matrices
        rho = 0.5 * (identity + r_x * sigma_x + r_y * sigma_y + r_z * sigma_z)
        sigma = 0.5 * (identity + s_x * sigma_x + s_y * sigma_y + s_z * sigma_z)

        return rho, sigma

    def trace_distance(self):
        # Difference of the matrices
        delta = self.rho - self.sigma
        
        # Singular value decomposition
        singular_values = svd(delta, compute_uv=False)
        
        # Trace norm is the sum of the singular values
        trace_norm = np.sum(singular_values)
        
        # Trace distance
        return 0.5 * trace_norm

    def fidelity(self):
        # rho is the measured density matrix, so it is mixed state
        # sigma is the ideal density matrix, so it is pure state
        fidelity = np.trace(self.rho @ self.sigma)
        # Fidelity
        return np.abs(fidelity)

    def get_results(self):
        return {
            'theta': self.theta,
            'phi': self.phi,
            'bloch_vector': self.bloch_vector,
            'ideal_bloch_vector': self.ideal_bloch_vector,
            'ideal_dm': self.sigma,
            'measured_dm': self.rho,
            'fidelity': self.fidelity,
            'trace_distance': self.trace_distance
        }

    
    
