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
        beta:float, f:float, k_d:float, cw_0:np.ndarray, t_max:float,
        x_right:float, x_steps:int, t_steps:int, v:float, a_k:float, 
        s_end:float, m_soil:float, m_leached:float, dia:float, length:float):
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
            t_max (float): end time of simulation [T]
            t_steps (int): number of steps
            x_right (float): right end of the 1D simulation field
            x_steps (int): number of spatial steps between 0 and x_right 
            v (float): advective velocity
            a_k (float): first order rate constant of eq 17c Simunek et al.
            s_end(float): concentration of PFOS at the end of exp [mug/kg]
            m_soil(float): mass of soil [kg]
            m_leached(float): cumulated mass [mug]
            dia(float): diameter of cylindric vessel
            length(float): height of cylindric vessel 
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
        self.t_max = t_max
        self.x_right = x_right
        self.x_steps = x_steps
        self.v = v
        self.a_k = a_k
        self.t_steps = t_steps
        self.s_end = s_end
        self.m_soil = m_soil
        self.m_leached = m_leached
        self.dia = dia
        self.length = length
        
        
        # deriveted variables
        self.x = np.linspace(1, self.x_right, self.x_steps)
        self.dx = self.x[1] - self.x[0]
        self.t = np.linspace(0, self.t_max,self.t_steps)
        self.dt = self.t[1] -self.t[0]
        self.sorpt_isotherm = self.equilib
        self.sorpt_derivat = self.d_equilib
        
        # total sorbed mass concentration
        self.s_0 = self.m_leached + self.s_end
        self.vol_wat = self.n_e*np.pi*(self.dia/4)**2*self.length


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
        #return derivative(self.equilib, c, dx=1e-6)
        return (self.beta*self.k_d*c**(self.beta -1))/((self.eta*c**self.beta)+1)**2
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

        sol = self.solve_ivp_euler()
        return sol    

    def solve_ivp_euler(self):
        """simple explicit euler to integrate ode"""
        sol_size = len(self.t)
        sol = np.ndarray((self.x_steps, sol_size))
    
        sol[:,0] = self.cw_0
        
        for i in range(sol_size-1):
            sol[:,i+1] = sol[:,i] + self.dt*self.ad_ode(sol[:,i])
        
        return sol
        
    def ad_ode(self, c_w:np.ndarray):
        """function that should be integrated over time

        Args:
            t (time): _description_
            cw (np.ndarray): c_w
        """

        # setups boundarys for c_w which are not accesed by fd and lap
        # in case nothing else is needed put zeros in the array
        dif_bound = np.zeros(self.x_steps)
        dif_bound[0] = self.d_e/(self.dx**2)*c_w[0]

        # dif_bound[-1]
        # not needed since c[bot] = 0
        # no adv bound needed since c[top] not needed in [[-1, 1, 0, 0]] and 
        # c[bot] = 0

        # [mug/kg]
        s0 = np.ones(self.x_steps)*self.s_0
        # [L/kg]
        ratio_wat_msoil = self.vol_wat/self.m_soil
        inhomog = self.a_k*self.rho_s*(((1-self.f)*self.sorpt_isotherm(c_w))-s0+self.f*self.sorpt_isotherm(c_w)+c_w*ratio_wat_msoil)
        
        # sums
        #print(inhomog[0])
        print((self.d_e*np.matmul(self.lap, c_w) + dif_bound + self.v*np.matmul(self.fd, c_w) - inhomog)[0])
        cw_new = (self.d_e*np.matmul(self.lap, c_w) + dif_bound \
            + self.v*np.matmul(self.fd, c_w) - inhomog)/(1+self.rho_s*self.f*self.sorpt_derivat(c_w))
        #print(cw_new[0])
        return cw_new

        