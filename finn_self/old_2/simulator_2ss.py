# @File          :   simulator_2ss.py
# @Last modified :   2022/08/19 16:29:24
# @Author        :   Matthias Gueltig

from sys import maxsize
import numpy as np
from scipy.integrate import solve_ivp
from scipy.misc import derivative

class Simulator(object):
    """Solves Advection-Diffusion equation for 2 side sorption.
    """
    def __init__(self, d_e:float, n_e:float, rho_s:float, eta:float, 
        beta:float, f:float, k_d:float, cw_0:np.ndarray, sk_0:np.ndarray,
        t_max:float, x_right:float, x_steps:int, v:float, a_k:float):
        """Constructor method initializing the parameters for the diffusion
        sorption problem.

        Args:
            d_e (float): effective diffusion coeff. [L/T]
            n_e (float): effective porosity [-]
            rho_s (float): dry bulk density for c_s = rho_s * [M/L]
            eta (float): parameter for sorption isotherms
            beta (float): parameter for sorption isotherms
            f (float): coupling kinetic and instantaneous sorption
            k_d (float): parameter of sorption isotherm
            cw_0 (np.ndarray): initial dissolved concentration [M/L]
            sk_0 (np.ndarray): initial kinetic sorbed concentration [M/L]
            t_max (float): end time of simulation [T]
            x_right (float): right end of the 1D simulation field
            x_steps (int): number of spatial steps between 0 and x_right 
            v (float): advective velocity
            a_k (float): first order rate constant of eq 17c Simunek et al.
        """
        
        # set class parameters
        self.d_e = d_e
        self.n_e = n_e
        self.rho_s = rho_s
        self.eta = eta
        self.beta = beta
        self.f = f
        self.k_d = k_d
        self.cw_0 = cw_0
        self.sk_0 = sk_0
        self.t_max = t_max
        self.x_right = x_right
        self.x_steps = x_steps
        self.v = v
        self.cw_0 = cw_0
        self.sk_0 = sk_0
        self.a_k = a_k
        
        
        # deriveted variables
        self.x = np.linspace(1, self.x_right, self.x_steps)
        self.dx = self.x[1] - self.x[0]
        self.sorpt_isotherm = self.linear
        self.sorpt_derivat = self.d_linear


    def linear(self, c):
        """implements linear sorption isotherm with K_d [M/M], c[M/L^3]"""
        return self.k_d*c
    
    def d_linear(self, c):
        """returns derivation of linear sorption isotherm [M/TL^3]"""
        return self.k_d

    def equilib(self, c):
        """implements equilibrium sorption isotherm"""
        return self.k_d*c**self.beta/(1+self.eta*c**self.beta)
    
    def d_equilib(self, c):
        """returns derivation of equilibrium sorption isotherm [M/TL^3]"""
        return derivative(self.equilib, c, dx=1e-6)

    def generate_sample(self):
        """Function that generates solution for PDE problem.
        """
        
        # Laplacian matrix for diffusion term
        nx = np.diag(-2*np.ones(self.x_steps), k=0)
        nx_minus_1 = np.diag(np.ones(self.x_steps-1), k=-1)
        nx_plus_1 = np.diag(np.ones(self.x_steps-1), k=1)

        self.lap = nx + nx_minus_1 + nx_plus_1
        self.lap /= self.dx**2

        # symmetric differences for advection term
        nx_fd = np.diag(np.ones(self.x_steps)*(-1), k=0)
        nx_fd_plus_1 = np.diag(np.ones(self.x_steps-1), k=1)
        self.fd = nx_fd + nx_fd_plus_1
        self.fd /= self.dx

        u = np.concatenate((self.cw_0, self.sk_0))

        sol = solve_ivp(self.ad_ode, (0, self.t_max), u, method="RK23", max_step=self.dx/self.v)

        time = np.array([sol.t])
        cws = sol.y[:self.x_steps]
        
        # solution array: first 40 rows c_w data, last row time
        sol = np.concatenate((cws, time), axis=0)
        
        return sol    

    def ad_ode(self, t:float, u:np.ndarray):
        """function that should be integrated over time

        Args:
            t (time): _description_
            u (np.ndarray): concatenated c_w and c_s
        """
        
        c_w = u[:self.x_steps]
        s_k = u[self.x_steps:]

        # solve sk_ode
        s_k_new = self.a_k*((1-self.f)*self.sorpt_isotherm(c_w)-s_k)

        # solve cw_ode
        top_BC = (c_w[0] - c_w[1])/self.dx * self.d_e
        bottom_BC = (c_w[-2] - c_w[1])/self.dx * self.d_e

        # setups boundarys for c_w which are not accesed by fd and lap
        # in case nothing else is needed put zeros in the array
        dif_bound = np.zeros(self.x_steps)
        dif_bound[0] = self.d_e/(self.dx**2)*top_BC
        dif_bound[-1] = self.d_e/(self.dx**2)*bottom_BC

        adv_bound = np.zeros(self.x_steps)
        adv_bound[-1] = self.v/(self.dx) * bottom_BC

        inhomog = (1-self.n_e)/self.n_e * self.rho_s * (self.sorpt_derivat(c_w)\
            *self.f + self.a_k*((1-self.f)*self.sorpt_isotherm(c_w) - s_k))
        
        # sum
        cw_new = self.d_e*np.matmul(self.lap, c_w) + dif_bound \
            + self.v*np.matmul(self.fd, c_w) + adv_bound - inhomog

        return np.concatenate((cw_new, s_k_new))

        