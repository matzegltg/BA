import numpy as np
from scipy.integrate import solve_ivp


class Simulator:

    def __init__(self, diffusion_coefficient, porosity, rho_s, eta, beta, f, k_d, 
    init_conc, t_max, t_steps, x_left, x_right, x_steps, v):
        """
        Constructor method initializing the parameters for the diffusion
        sorption problem.
        :param diffusion_coefficient: The diffusion coefficient [m^2/day]
        :param porosity: The porosity of the medium [-]
        :param rho_s: Dry bulk density [kg/m^3]
        :param kd: Partitioning coefficient [m^3/kg]
        :param t_max: Stop time of the simulation
        :param t_steps: Number of simulation steps
        :param x_left: Left end of the 1D simulation field
        :param x_right: Right end of the 1D simulation field
        :param x_steps: Number of spatial steps between x_left and x_right
        """
    
        # Set class parameters
        self.D = diffusion_coefficient
        self.por = porosity
        self.rho_s = rho_s
        self.k_d = k_d
        self.beta = beta
        self.eta = eta
        self.f = f
        self.T = t_max
        self.X0 = x_left
        self.X1 = x_right
        self.v = v
        self.Nx = x_steps
        self.Nt = t_steps
        self.init_conc = init_conc
        self.x = np.linspace(1, self.X1, self.Nx)
        self.t = np.linspace(0, self.T, self.Nt)
        self.dx = self.x[1]-self.x[0]
        self.retardation = self.retardation_two_sorption_equaltimesteps
        self.ts_sorp = [0]
        self.s_k_diff = np.zeros(self.Nx)
        
    def retardation_equilib_sorption(self, c, t):
        """implements equilibrium sorption, as described on slide 25 of Thomas
        Bierbaum
        """
        derived_sorp_top = self.k_d*self.beta*c**(self.beta-1)*\
        (1+self.eta*c**self.beta) - self.beta*self.eta*c**(self.beta-1)\
        *self.k_d*c**self.beta

        derived_sorp_bot = (1 + self.eta*c**self.beta)**2
        ret =  1+(1-self.por)/self.por * self.rho_s *\
            derived_sorp_top/derived_sorp_bot
        
        return ret

    def retardation_two_sorption(self, c, t):
        """implements two site sorption, as described on slide 25 of Thomas
        Bierbaum"""

        
        s_e_diff = np.ones(self.Nx)*self.f*self.k_d
        self.ts_sorp.append(t)

        sk_sol_diff = solve_ivp(self.sk_ode, (self.ts_sorp[0], self.ts_sorp[1]),\
             self.s_k_diff, args=(c, ), method="BDF")
        self.s_k_diff = sk_sol_diff.y[:,-1]
        s_diff = s_e_diff + sk_sol_diff.y[:,-1]
        ret = 1 + (1-self.por)/self.por * self.rho_s*s_diff
        self.ts_sorp.pop(0)
        
        return ret

    def retardation_two_sorption_equaltimesteps(self, c):
        """implements tow site sorption, as described on slide 25 of Thomas
        Bierbaum with equal time step length for both sorption and desorption"""

        # implements linear sorption
        s_e_diff = np.ones(self.Nx)*self.f*self.k_d

        # zaehler of kinetic sorption term
        sk_sol_diff_top = self.k_d*self.beta*c**(self.beta-1)*\
        (1+self.eta*c**self.beta) - self.beta*self.eta*c**(self.beta-1)\
        *self.k_d*c**self.beta

        # nenner of kinetic sorption term
        sk_sol_diff_bot = (1 + self.eta*c**self.beta)**2
        s_diff = s_e_diff * sk_sol_diff_top/sk_sol_diff_bot

        ret = 1 + (1-self.por)/self.por*self.rho_s*s_diff

        return ret

    def generate_sample(self):
        """
        Single sample generation using the parameters of this simulator.
        :return: The generated sample as numpy array(t, x)
        """

        # Initialize the simulation field
        # equally distibuted at t = 0, apart from ghost cells
        c_0 = np.ones(self.Nx)*self.init_conc
        #c_0 = np.zeros(self.Nx)
        #c_0[19] = self.init_conc
        #c_0[20] = self.init_conc
        #c_0[21] = self.init_conc
        # Laplacian matrix
        nx = np.diag(-2*np.ones(self.Nx), k=0)
        nx_minus_1 = np.diag(np.ones(self.Nx-1), k=-1)
        nx_plus_1  = np.diag(np.ones(self.Nx-1), k=1)
        
        self.lap = nx + nx_minus_1 + nx_plus_1
        self.lap /= self.dx**2

        # symmm diff for advection Term
        nx_FD = np.diag(np.ones(self.Nx)*(-1), k=0)
        #nx_FD_minus_1 = np.diag(np.ones(self.Nx-1)*(-1), k=-1)
        nx_FD_plus_1 = np.diag(np.ones(self.Nx-1), k=1)

        self.FD = nx_FD + nx_FD_plus_1
        self.FD /= (self.dx*2)
        # Solve the diffusion sorption problem
        # zwischendrin solve_ivp(self.rc_ode))
        prob = solve_ivp(self.rc_ode, (0, self.T), c_0, method="BDF")

        ode_data = prob.y
        
        return ode_data

    def sk_ode(self, t, s_k, c):
        derived_sorp_top = self.k_d*self.beta*c**(self.beta-1)*\
        (1+self.eta*c**self.beta) - self.beta*self.eta*c**(self.beta-1)\
        *self.k_d*c**self.beta

        derived_sorp_bot = (1 + self.eta*c**self.beta)**2
        omega = 1000
        return omega*(1-self.f)*self.k_d*(derived_sorp_top/derived_sorp_bot) - s_k

    def rc_ode(self, t, u):
        self.D = 1
        self.v = 1
        
        top_BC = (u[0] - u[1])/self.dx * self.D
        bottom_BC = (u[-2]-u[-1])/self.dx * self.D

        # TODO: Take care! Time dependent sortion model has timesteps as input
        # for calculating retardation factor
        ret = self.retardation(u)
        dif_bound = np.zeros(self.Nx)
        dif_bound[0]  = self.D/(ret[0]*(self.dx**2))*top_BC
        dif_bound[-1] = self.D/(ret[-1]*(self.dx**2))*bottom_BC

        adv_bound = np.zeros(self.Nx)
        adv_bound[0] = -self.v/(ret[0]*(self.dx)) * top_BC
        adv_bound[-1] = self.v/(ret[-1]*(self.dx)) * bottom_BC
        return self.D/ret*np.matmul(self.lap, u) + dif_bound\
             + self.v/ret*np.matmul(self.FD, u) + adv_bound
