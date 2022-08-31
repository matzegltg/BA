"""
This script is a modified version of DS solver. To solve PDE that describes PFAS
transport through soil
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from simulator_2ss import Simulator


##############
# PARAMETERS #
##############

# vessel sizes
DIA = 0.9
LENGTH = 4

# [dm^2/d]
DIFFUSION_COEFFICIENT = 0.025

POROSITY = 0.4

# kg/dm^3
RHO_S = 1.58

# kg
M_SOIL = 4.022

# mug/kg
M_LEACHED = 155
S_END = 1.9

# taken from slide 36 of Thomas Bierbaum
# dm^3/kg
k_d = 4.5
beta = 1
eta = 0

# dm/d
v = 3

# time
T = 200
T_STEPS = 10000

X_RIGHT = 40
X_STEPS = 80

# first day 30 mug/L PFOS outflow = 30 ng/cm^3 PFOS soluted, 
init_conc = np.ones(X_STEPS)*30
init_conc[0] = 0
# ratio instantaneous vs. kinetic sorption (f = 0) -> kinetic sorption
f = 0.4

# first order rate consant [1/d], taken from Thomas Bierbaum
a_k = 0.005
#############
# FUNCTIONS #
#############

def generate_sample(simulator):
    """
    This function generates a data sample, visualizes it if desired and saves
    the data to file if desired.
    :param simulator: The simulator object for data creation
    """

    print("Generating data...")

    # Generate a data sample
    sample_c = simulator.generate_sample()
    
    print(sample_c)
    df = pd.DataFrame(sample_c)
    df.to_csv("sim_c.csv")
    
    visualize_sample(sample=sample_c, simulator=simulator)

def visualize(sample, simulator):
    fig, ax = plt.subplots(1,1, figsize=(10,4))
    h = ax.imshow(sample, cmap='rainbow', interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax)    
    ax.set_ylim(simulator.x.min(), simulator.x.max())
    plt.tight_layout()
    plt.show()

def visualize_sample(sample, simulator):
    """
    Method to visualize a single sample. Code taken and modified from
    https://github.com/maziarraissi/PINNs
    :param sample: The actual data sample for visualization
    :param simulator: The simulator used for data generation
    :param idcs_init: The indices of the initial points
    :param idcs_bound: The indices of the boundary points
    """

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # u(t, x) over space
    h = ax[0].imshow(sample, interpolation='nearest', cmap='rainbow', 
                  extent=[simulator.t.min(), simulator.t.max(),
                          simulator.x.min(), simulator.x.max()],
                  origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
        
    ax[0].set_xlim(0, simulator.t.max())
    ax[0].set_ylim(simulator.x.min(), simulator.x.max())
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel('$t$')
    ax[0].set_ylabel('$x$')
    ax[0].set_title('$u(t,x)$', fontsize = 10)
    
    # u(t, x) over time
    line, = ax[1].plot(simulator.x, sample[:, 0], 'b-', linewidth=2, label='Exact')
    ax[1].set_xlabel('$x$')
    ax[1].set_ylabel('$u(t,x)$')    
    ax[1].set_xlim([simulator.x.min(), simulator.x.max()])
    ax[1].set_ylim([0, 1.1])

    anim = animation.FuncAnimation(fig,
                                   animate,
                                   frames=len(simulator.t),
                                   fargs=(line, sample),
                                   interval=20)
    plt.tight_layout()
    plt.draw()
    plt.show()


def animate(t, axis, field):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    axis.set_ydata(field[:, t])
  

def main():
    """
    Main method used to create the datasets.
    """


    # Create a wave generator using the parameters from the configuration file
    simulator = Simulator(d_e=DIFFUSION_COEFFICIENT,
    n_e=POROSITY,
    rho_s=RHO_S,
    eta=eta,
    beta=beta,
    f=f,
    k_d=k_d,
    cw_0=init_conc,
    t_max=T,
    t_steps=T_STEPS,
    x_right=X_RIGHT,
    x_steps=X_STEPS,
    v=v,
    a_k=a_k,
    s_end=S_END,
    m_soil=M_SOIL,
    m_leached=M_LEACHED,
    dia=DIA,
    length=LENGTH)
    # Create train, validation and test data
    generate_sample(simulator=simulator)


if __name__ == "__main__":
    main()

    print("Done.")
