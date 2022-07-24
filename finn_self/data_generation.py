"""
This script is a modified version of DS solver. To solve PDE that describes PFAS
transport through soil
"""

from doctest import FAIL_FAST
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from simulator import Simulator


##############
# PARAMETERS #
##############

# cm^2/d
DIFFUSION_COEFFICIENT = 0.46656
POROSITY = 0.4

# kg/cm^3
RHO_S = 0.00158

# taken from slide 36 of Thomas Bierbaum
# cm^3/kg
K_d = 4.5/1000

#omega = 0.005
beta = 2
eta = 5

# total mass pfos: 156,9 mug/kg
init_conc = 150

# cm/d
v = 33

# time
T = 400
T_STEPS = 401
    
X_LEFT = 1
X_RIGHT = 40
X_STEPS = 40

# ratio instantaneous vs. kinetic sorption (f = 0) -> kinetic sorption
f = 0.5

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
    
    df = pd.DataFrame(sample_c)
    df.to_csv("sim_c.csv")

    visualize_sample(sample=sample_c,
                    simulator=simulator)
        

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
    simulator = Simulator(
        diffusion_coefficient=DIFFUSION_COEFFICIENT,
        porosity=POROSITY,
        rho_s=RHO_S,
        k_d = K_d,
        init_conc = init_conc,
        beta = beta,
        eta = eta,
        f = f,
        t_max=T,
        t_steps=T_STEPS,
        x_left=X_LEFT,
        x_right=X_RIGHT,
        x_steps=X_STEPS,
        v=v
    )

    # Create train, validation and test data
    generate_sample(simulator=simulator)


if __name__ == "__main__":
    main()

    print("Done.")
