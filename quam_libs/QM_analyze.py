import numpy as np
import xarray as xr
import json
import matplotlib.pyplot as plt
import qutip as qt
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, cKDTree
from quam_libs.fit_ellipsoid import ls_ellipsoid, polyToParams3D

## Helper functions
def MLE(original_P,confusion_matrix):
    """
    Maximum Likelihood Estimation of the true probabilities
    :param original_P: Original probabilities
    :param confusion_matrix: Confusion matrix
    :return: Estimated true probabilities
    """
    N_obs = original_P
    M = confusion_matrix
    def neg_log_likelihood(p_optimal):
        q_predict = M @ p_optimal
        return -np.sum(N_obs * np.log(q_predict + 1e-10))  # Avoid log(0)

    # Constraints: p0 + p1 = 1, p0 >= 0, p1 >= 0
    constraints = ({'type': 'eq', 'fun': lambda p: np.sum(p) - 1})
    bounds = [(0, 1), (0, 1)]

    # Initial guess (e.g., [0.5, 0.5])
    result = minimize(neg_log_likelihood, x0=[0.5, 0.5], 
                    bounds=bounds, constraints=constraints)

    p_optimal_estimated = result.x
    if not result.success:
        raise ValueError("MLE Optimization failed: " + result.message)
    #print(f"Estimated true probabilities: {p_optimal_estimated}")
    return p_optimal_estimated

def plot_results(title, axes, volume, fitting_error, fidelity, T2, T1, noise_voltage):
    fig, axs = plt.subplots(3, 2, figsize=(8, 6))

    plt.suptitle(title)
    axs_axes =axs[0,0]
    axs_volume =axs[1,0]
    axs_error =axs[0,1]
    axs_fidelity =axs[1,1]
    axs_T2 =axs[2,0]
    axs_T1 =axs[2,1]

    axs_axes.plot(noise_voltage,axes[:,0],'k',label='x')
    axs_axes.plot(noise_voltage,axes[:,1],'b',label='y')
    axs_axes.plot(noise_voltage,axes[:,2],'r',label='z')
    axs_axes.set_ylabel('Axes')
    axs_axes.legend()

    axs_volume.plot(noise_voltage,volume,'k')
    axs_volume.plot(noise_voltage,volume,'.r')
    axs_volume.set_ylabel('Volume')

    axs_error.errorbar(noise_voltage, fitting_error[:,0], yerr=fitting_error[:,1], fmt='o', capsize=5, label="Mean with Std Dev")
    axs_error.set_ylabel('fitting error')

    axs_fidelity.errorbar(noise_voltage, fidelity[:,0], yerr=fidelity[:,1], fmt='o', capsize=5, label="Mean with Std Dev")
    axs_fidelity.set_ylabel('Fidelity')

    axs_T2.plot(noise_voltage,T2,'k')
    axs_T2.plot(noise_voltage,T2,'.r')
    axs_T2.set_xlabel('Noise amp(mV)')
    axs_T2.set_ylabel('T2(us)')

    axs_T1.plot(noise_voltage,T1,'k')
    axs_T1.plot(noise_voltage,T1,'.r')
    axs_T1.set_xlabel('Noise amp(mV)')
    axs_T1.set_ylabel('T1(us)')
    plt.tight_layout()
    plt.show()
    return fig

def plot_avfe(title,xx,axes,volume,fidelity,fitting_error,negativity,y_limit=False):
    """
    Plot the average fidelity, fitting error and volume of the ellipsoid
    :param title: title of the plot, string
    :param xx: x axis data, dict{'x_title':np.array(xx)}
    :param data_x: x axis data, np.array
    :param data_y: y axis data, np.array
    :param data_z: z axis data, np.array
    :param volme: volume of the ellipsoid, np.array
    :param fidelity: fidelity of the ellipsoid, np.array
    :param fitting_error: fitting error of the ellipsoid, np.array
    :return: plot
    """
    x_title = list(xx.keys())[0]
    x_values = list(xx.values())[0]
    fig = plt.figure(figsize=(4, 8))
    ax_name = ['axes', 'volume', 'fidelity', 'fitting error', 'negativity']
    ax_axes = fig.add_subplot(len(ax_name), 1, 1)
    ax_volume = fig.add_subplot(len(ax_name), 1, 2)
    ax_fidelity = fig.add_subplot(len(ax_name), 1, 3)
    ax_fitting_error = fig.add_subplot(len(ax_name), 1, 4)
    ax_negativity = fig.add_subplot(len(ax_name), 1, 5)

    ax_axes.set_title(title)
    ax_axes.plot(x_values,axes[:,0],'k',label='x')
    ax_axes.plot(x_values,axes[:,1],'r',label='y')
    ax_axes.plot(x_values,axes[:,2],'b',label='z')
    if y_limit:
        ax_axes.set_ylim(0,1)
    ax_axes.set_ylabel('Axes')
    ax_axes.legend(loc='lower right')

    ax_volume.plot(x_values,volume,'k')
    ax_volume.plot(x_values,volume,'.r')
    if y_limit:
        ax_volume.set_ylim(0,4*np.pi/3)
    ax_volume.set_ylabel('Volume')

    ax_fidelity.errorbar(x_values,fidelity[:,0],yerr=fidelity[:,1],fmt='o')
    ax_fidelity.set_ylabel('Fidelity')

    ax_fitting_error.errorbar(x_values,fitting_error[:,0],yerr=fitting_error[:,1],fmt='o')
    ax_fitting_error.set_ylabel('Fitting error')
    #ax_fitting_error.set_xlabel(x_title)

    #ax_negativity.errorbar(x_values,negativity[:,0],yerr=negativity[:,1],fmt='o')
    ax_negativity.plot(x_values,negativity)
    ax_negativity.set_ylabel('Negativity')
    if y_limit:
        ax_negativity.set_ylim(0,1)
    ax_negativity.set_xlabel(x_title)

    plt.tight_layout()
    return fig

class QM_analyze:
    """
    Class for analyzing quantum measurement data.
    
    Parameters:
        measurement_data (np.array): [[x, y, z],[x, y, z],...] - Experimental Bloch vector components.
        ideal_data (np.array): [[theta, phi],[theta, phi],...] - Ideal Bloch vector parameters.
        the shpae is (n,3) and (n,2) respectively.
    """

    def __init__(self, measurement_data, ideal_data,do_convex_hull=False):

        if not isinstance(measurement_data, np.ndarray) or not isinstance(ideal_data, np.ndarray):
            if isinstance(measurement_data, list) and isinstance(ideal_data, list):
                measurement_data = np.array(measurement_data)
                ideal_data = np.array(ideal_data)
            else:
                raise ValueError("measurement_data and ideal_data should be numpy arrays or lists of numpy arrays.")

        self.measurement_data = measurement_data
        self.ideal_data = ideal_data
        self.do_convex_hull = do_convex_hull
        self.convex_hull = self.convex_hull()


        self.x, self.y, self.z = self.measurement_data[:,0], self.measurement_data[:,1], self.measurement_data[:,2]
        self.px, self.py, self.pz = (1-self.x)/2, (1-self.y)/2, (1-self.z)/2 # probability of 1
        self.ideal_theta, self.ideal_phi = self.ideal_data[:,0], self.ideal_data[:,1]
        self.exp_bloch_vector = np.array(measurement_data)
        self.ideal_bloch_vector = np.array([[
            np.sin(self.ideal_theta[i]) * np.cos(self.ideal_phi[i]),
            np.sin(self.ideal_theta[i]) * np.sin(self.ideal_phi[i]),
            np.cos(self.ideal_theta[i])] for i in range(len(self.ideal_theta))])
        self.exp_dm,self.ideal_dm = self.density_matrix()

        self.fidelity = np.array([np.abs(np.trace(self.exp_dm[i] @ self.ideal_dm[i])**2) for i in range(len(self.x))])
        self.trace_distance = np.array([0.5 * np.sum(np.abs(self.exp_dm[i] - self.ideal_dm[i])) for i in range(len(self.x))])        

        self.ellipsoid_center, self.ellipsoid_axes, self.ellipsoid_R,self.volume, self.ellipsoid_param = self.ellipsoid_fit()
        self.shortest_distance = self.shortest_distance()
        self.fitting_error = self.fit_error()
        
        self.choi = self.choi_state()
        self.rho_PT = self.partial_transpose_input()
        self.negativity = self.negativity()
        self.negativity_qt = self.negativity_qt()
    def ellipsoid_fit(self):
        x = self.measurement_data[:,0]
        y = self.measurement_data[:,1]
        z = self.measurement_data[:,2]
        if self.do_convex_hull == True:
            # Use the convex hull to fit the ellipsoid
            hull = ConvexHull(self.measurement_data)
            points = self.measurement_data[hull.vertices]
            x = points[:,0]
            y = points[:,1]
            z = points[:,2]
        param = ls_ellipsoid(x,y,z)
        param = param / np.linalg.norm(param) 
        center,axes,R = polyToParams3D(param,False)
        volume = (4/3)*np.pi*axes[0]*axes[1]*axes[2]
        return center,axes,R,volume,param
    
    def shortest_distance(self):
        '''
        TODO: the shortest distance is not correct, need to be fixed.
        The constraint function should be set to zero, but if we do this the minimum values will be 0
        no matter what the initial guess is.
        '''
        def distance_eq(point):
            x, y, z = point
            return ((x - x0)**2 + (y - y0)**2 + (z - z0)**2)

        shortest_distance = np.array([])
        for i in range(len(self.x)):
            x0, y0, z0 = self.x[i], self.y[i], self.z[i]
            param = self.ellipsoid_param
            guess_point = np.array([x0, y0, z0])
            cons = {'type': 'eq', 'fun': (lambda xyz:param[0]*xyz[0]*xyz[0]+param[1]*xyz[1]*xyz[1]+param[2]*xyz[2]*xyz[2]+
                                                    param[3]*xyz[0]*xyz[1]+param[4]*xyz[0]*xyz[2]+param[5]*xyz[1]*xyz[2]+
                                                    param[6]*xyz[0]+param[7]*xyz[1]+param[8]*xyz[2]+param[9])} # constraint function
            bounds = [(-1,1), (-1, 1), (-1, 1)]
            result = minimize(distance_eq, x0 = guess_point, constraints=cons,bounds=bounds) # minimize the distance

            if result.success:
                pass
            else:
                print(f"Shortest distance optimization failed: {result.message}")
            
            #determin the sign of the distance - inside or outside the ellipsoid
            sign = 1 if np.sum(np.array([x0,y0,z0])**2) > np.sum(result.x**2) else -1
            shortest_distance = np.append(shortest_distance,np.abs(result.fun)*sign)
        
        return shortest_distance

    def fit_error(self):
        """
        Calculate the fitting error of the ellipsoid.
        """
        # Calculate the fitting error
        error = np.array([])
        for i in range(len(self.x)):
            x0, y0, z0 = self.x[i], self.y[i], self.z[i]
            param = self.ellipsoid_param
            a = param[0]*x0*x0 + param[1]*y0*y0 + param[2]*z0*z0 + param[3]*x0*y0 + param[4]*x0*z0 + param[5]*y0*z0 + param[6]*x0 + param[7]*y0 + param[8]*z0 + param[9]
            error = np.append(error, a)
        return error

    def convex_hull(self):
        points = np.array(self.measurement_data)
        hull = ConvexHull(points,qhull_options='QJ')
        volume = hull.volume
        vertices = points[hull.vertices]
        return hull

    def density_matrix(self):
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Identity matrix
        identity = np.array([[1, 0], [0, 1]], dtype=complex)
        experimental_dm = []
        ideal_dm = []
        for i in range(len(self.x)):
            # Bloch vector components
            r_x, r_y, r_z = self.exp_bloch_vector[i]
            s_x, s_y, s_z = self.ideal_bloch_vector[i]

            # Construct density matrices
            rho = 0.5 * (identity + r_x * sigma_x + r_y * sigma_y + r_z * sigma_z)
            sigma = 0.5 * (identity + s_x * sigma_x + s_y * sigma_y + s_z * sigma_z)
            experimental_dm.append(rho)
            ideal_dm.append(sigma)
        
        return np.array(experimental_dm), np.array(ideal_dm)

    def choi_state(self):
        pauli_matrices = [
            np.eye(2, dtype=complex),
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex)
        ]
        B = np.array(self.ellipsoid_center)
        radii = np.array(self.ellipsoid_axes)
        eigvecs = np.array(self.ellipsoid_R)
        T = eigvecs @ np.diag(radii)

        chi = np.zeros((4, 4), dtype=complex)
        chi[0, 0] = 1
        chi[0, 1:4] = B
        chi[1:4, 0] = B.conj()
        chi[1:4, 1:4] = T
        chi = (chi + chi.conj().T) / 2  # Hermitian

        choi = np.zeros((4, 4), dtype=complex)
        for i, Pi in enumerate(pauli_matrices):
            for j, Pj in enumerate(pauli_matrices):
                choi += chi[i, j] * np.kron(Pi, Pj)

        choi = (choi + choi.conj().T) / 2
        choi /= np.trace(choi)
        return choi

    def results(self):
        return {
            "fidelity": [self.fidelity.mean(),self.fidelity.std()],
            "trace_distance": [self.trace_distance.mean(),self.trace_distance.std()],
            "ellipsoid":{
                "center": self.ellipsoid_center.tolist(),
                "axes": self.ellipsoid_axes.tolist(),
                "R": self.ellipsoid_R.tolist(),
                "volume": self.volume,
                "convex_hull_volume": self.convex_hull.volume,
                "convex_hull_vertices": self.convex_hull.vertices.tolist(),
                "fitting_param": self.ellipsoid_param.tolist(),
                "fitting_error": [self.fitting_error.mean(),self.fitting_error.std()],
                "shortest_distance": [self.shortest_distance.mean(),self.shortest_distance.std()],
                },
            "Quantum_channel":{
                "choi": self.choi.tolist(),
                "trace": np.trace(self.choi),
                "Hermitian?":np.allclose(self.choi, self.choi.conj().T),
                "Positive semi-definite?":np.all(np.linalg.eigvalsh(self.choi) >= -1e-10),
                "negativity": self.negativity,
                #"negativity_qt": self.negativity_qt,
            }           
        }

    def ellipsoid_plot(self, title= "Ellipsoid fit"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        u,v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
        ideal_x, ideal_y, ideal_z = (np.outer(np.cos(u), np.sin(v)), np.outer(np.sin(u), np.sin(v)), np.outer(np.ones_like(u), np.cos(v)))
        
        x_ellipsoid = self.ellipsoid_axes[0] * ideal_x
        y_ellipsoid = self.ellipsoid_axes[1] * ideal_y
        z_ellipsoid = self.ellipsoid_axes[2] * ideal_z

        ellipsoid_points_ = np.dot(self.ellipsoid_R,np.array([x_ellipsoid.ravel(), y_ellipsoid.ravel(), z_ellipsoid.ravel()]))
        ellipsoid_points_ += self.ellipsoid_center.reshape(-1, 1)
        x_ellipsoid, y_ellipsoid, z_ellipsoid = ellipsoid_points_.reshape(3, *x_ellipsoid.shape)

        ax.plot_wireframe(ideal_x, ideal_y, ideal_z, color="blue", alpha=0.05, label=" Bloch sphere")
        ax.plot_wireframe(x_ellipsoid, y_ellipsoid, z_ellipsoid, color="red", alpha=0.08, label="fitted ellipsoid")
        #ax.scatter(self.x, self.y, self.z, color="black", label="Experimental data",marker='x')
        points = np.array(self.measurement_data)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="black", alpha=0.5, marker='x', label="Experimental data")
        if self.do_convex_hull == True:
            hull = self.convex_hull
            ax.scatter(points[hull.vertices, 0], points[hull.vertices, 1], points[hull.vertices, 2], s=50)

            # -------------------- Convex-hull edges -----------------
            edges = set()
            for tri in hull.simplices:     # each 'tri' is a triangle face
                for i in range(3):
                    e = tuple(sorted((tri[i], tri[(i + 1) % 3])))
                    edges.add(e)

            for i, j in edges:
                ax.plot([points[i, 0], points[j, 0]],
                        [points[i, 1], points[j, 1]],
                    [points[i, 2], points[j, 2]],'k')
        ax.set_title(title)
        plt.show()
        return fig, ax

    # TODO...
    """
    def negativiy(self):
        
        Calculate the negativity of the density matrix.
        
        negativity = np.array([])
        for i in range(len(self.x)):
            
            trace_norm = np.linalg.svd(self.exp_dm[i], compute_uv=False).sum()
            negativity = np.append(negativity, 0.5 * (trace_norm - 1.0))
            
            lambdas = np.linalg.eigvalsh(self.exp_dm[i])
            negativity = np.append(negativity, -lambdas[lambdas < 0].sum())
        return negativity
    """

    def partial_transpose_input(self, d_in=2, d_out=2):
        """
        PT over the *input* side of the Choi state Φ_E.
        choi : array of shape (d_in*d_out, d_in*d_out)
        """
        choi = self.choi
        return choi.reshape(d_in, d_out, d_in, d_out).swapaxes(0, 2).reshape(d_in*d_out, d_in*d_out)

    def negativity(self):
        lam = np.linalg.eigvalsh(self.rho_PT)
        return -lam[lam < 0].sum()

    def negativity_qt(self):
        """
        Calculate the negativity of the choi state using qutip.
        """
        choi_mat = self.choi
        choi = qt.Qobj(choi_mat,dims=[[2, 2], [2, 2]])
        N = qt.negativity(choi,subsys=[0])
        return N

    def quatnum_memory():
        """
        Calculate the quantum memory of the ellipsoid.
        """
