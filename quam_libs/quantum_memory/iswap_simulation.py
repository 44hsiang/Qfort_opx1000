import numpy as np
from qutip import *
from dataclasses import dataclass, field


class Math():

    def theta_phi_to_alpha_beta(theta, phi):
        """
        Convert spherical coordinates (theta, phi) to alpha and beta coefficients.
        """
        alpha = np.cos(theta / 2)
        beta = np.exp(1j * phi) * np.sin(theta / 2)
        return alpha, beta

    def alpha_beta_to_qutip(alpha, beta):
        """
        Convert alpha and beta coefficients to a qubit state.
        """
        norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
        return (alpha / norm) * basis(2, 0) + (beta / norm) * basis(2, 1)
    def generate_uniform_sphere_angles(n_points):

        indices = np.arange(0, n_points)
        golden_angle = np.pi * (3 - np.sqrt(5))  

        z = 1 - 2 * (indices + 0.5) / n_points     
        theta = np.arccos(z)                      
        phi = (indices * golden_angle) % (2 * np.pi)  

        theta_list = theta.tolist()
        phi_list = phi.tolist()

        return theta_list, phi_list

@dataclass
class StateEvolution():

    t1: float = 16e-6        
    t2: float =  6e-6        
    detuning: float = 0.0     
    error_rate: float = 0.16   

    sx: Qobj = field(default_factory=lambda: sigmax(), init=False)
    sy: Qobj = field(default_factory=lambda: sigmay(), init=False)
    sz: Qobj = field(default_factory=lambda: sigmaz(), init=False)
    sm: Qobj = field(default_factory=lambda: sigmam(), init=False)
    I:  Qobj = field(default_factory=lambda: qeye(2), init=False)

    def _collapse_ops(self, T1: float, T2: float):
        gamma1  = 1.0 / T1
        gamma_phi  = 1.0 / T2 - gamma1 / 2
        return [
            np.sqrt(gamma1) * tensor(self.I, self.sm),
            np.sqrt(gamma_phi) * tensor(self.I, self.sz)
        ]

    def _solve(self, H: Qobj, rho0: Qobj, tlist):
        return mesolve(
            H, rho0, tlist,
            c_ops=self._collapse_ops(self.t1, self.t2),
            e_ops=[], options=Options(store_states=True)
        ).states

    def wait_dm(self, theta, phi, delay=np.arange(0,16e-9,2e-9)) -> Qobj:
        H0 = 2*np.pi*self.detuning * (
            tensor(self.I, self.sx) + tensor(self.sx, self.I)
        )
        env  = basis(2, 0)
        targ = Math.alpha_beta_to_qutip(*Math.theta_phi_to_alpha_beta(theta, phi))
        rho0 = tensor(env, targ)
        return self._solve(H0, rho0, delay)[-1]  

    def br_swap_dm(self, rho0: Qobj, g=40e6/(2*np.pi), tlist=np.arange(0,110e-9,2e-9)):

        H_couple = (g/2) * (tensor(self.sx, self.sx) + tensor(self.sy, self.sy))
        H_detune = 2*np.pi*self.detuning * (
            tensor(self.I, self.sx) + tensor(self.sx, self.I)
        )
        return self._solve(H_couple + H_detune, rho0, tlist)

    @staticmethod
    def error_gate(rho: Qobj, gamma: float):
        I, sx, sy = qeye(2), sigmax(), sigmay()
        return gamma/2 * (sx*rho*sx + sy*rho*sy) + (1-gamma)*I*rho*I


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    SE = StateEvolution()
    rho0 = SE.wait_dm(theta=np.pi, phi=0)
    print("Final state density matrix:")
    print(rho0.full())

    swap_rho = SE.br_swap_dm(rho0, g=40e6/(2*np.pi), tlist=np.arange(0, 110e-9, 2e-9))
    print("After swap gate density matrix:")
    print(swap_rho[0].ptrace(1).full())  # Get the state of the second qubit