#! env/bin/python3

"""
Main file for testing (evaluating) a FINN model
"""

import numpy as np
import torch as th
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import time
import pickle
import pathlib as pth
import pandas as pd

sys.path.append("..")
from utils.configuration import Configuration
from finn import *

def run_testing(print_progress=False, visualize=False, model_number=None):

    # Load the user configurations
    config = Configuration("config.json")
    
    # Append the model number to the name of the model
    if model_number is None:
        model_number = config.model.number
    config.model.name = config.model.name + "_" + str(model_number).zfill(2)

    # Print some information to console
    print("Model name:", config.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config
    # file
    if config.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    root_path = os.path.abspath("../../data")
    data_path = os.path.join(root_path, config.data.type, config.data.name)
    
    # Set device on GPU if specified in the configuration file, else CPU
    device = th.device(config.general.device)
    
    
    if config.data.type == "burger":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        dx = x[1]-x[0]
    
        # Initialize and set up the model
        model = FINN_Burger(
            u = u,
            D = np.array([1.0]),
            BC = np.array([[0.0], [0.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="test",
            learn_coeff=True
        ).to(device=device)
    
    elif config.data.type == "diffusion_ad2ss":
        # Load samples, together with x, y, and t series
        params = Configuration(f"results/{config.model.number}/init_params.json")

        
        # Interpolation of experimental data
        if config.data.name == "data_exp":
            df = pd.read_excel("../../220613_ColumnExperiments_Data_N1.xlsx", "N1", skiprows=9, nrows=40, usecols="B:U")

            # select experimental PFOS row
            exp_conc = df.iloc[[20]].to_numpy(dtype=np.float32).squeeze()

            # ng/L -> mug/L -> mug/cm^3
            exp_conc = exp_conc/1000000

            # c(t=0, x) = 0 \forall x in \Omega_x 
            exp_conc = np.insert(exp_conc, 0, 0)

            # select experimental measure points
            exp_t = df.iloc[[35]].to_numpy(dtype=np.float32).squeeze()
 
            # insert t=0
            exp_t = np.insert(exp_t, 0, 0)

            # average concentrations -> shift to middle value of times
            exp_mean_t = []
            for i in range(0,len(exp_t)):
                if i == 0:
                    exp_mean_t.append(exp_t[i])
                else:
                    exp_mean_t.append((exp_t[i] + exp_t[i-1])/2) 
            
            # create time discretization vector
            t = np.linspace(exp_mean_t[0], params.T_MAX, num=params.T_STEPS, dtype=np.float32)
            
            # Interpolation of experimental data points
            #sample_exp = np.interp(t, exp_mean_t, exp_conc)
            
            # get indices of time where experimental data is available
            loss_indices = []
            for meas_point in exp_mean_t:
                for i in range(len(t)):
                    if np.abs(t[i]-meas_point) <= 0.01:
                        loss_indices.append(i)
                        break
            
            # visualize exp data
            #fig, ax = plt.subplots()
            #ax.plot(new_t, new_exp, label = "Interpolation")
            #ax.scatter(exp_t, exp_conc, color="y", label="Original times")
            #ax.scatter(exp_mean_t, exp_conc, color="r", label="Averaged times")
            #print(type(exp_conc))
            #np.set_printoptions(suppress=True)
            #print(exp_conc)
            #ax.set_xlabel(r'$t [d]$', fontsize=17)
            #ax.set_ylabel(r'conc PFOS $\left[\frac{\mu g}{cm^3}\right]$', fontsize=17)
            #ax.set_title('Experimental BTC', fontsize=17)
            #ax.tick_params(axis='x', labelsize=17)
            #ax.tick_params(axis='y', labelsize=17)
            #ax.set_yscale("log")
            #ax.legend()
            #plt.savefig("exp_data")

            # "upscale" to sizes required by FINN
            sample_c = th.zeros((params.X_STEPS, params.T_STEPS), dtype=th.float32)
            sample_sk = th.zeros((params.X_STEPS, params.T_STEPS), dtype=th.float32)
            # add initial conditions
            init_conc = th.zeros(params.X_STEPS)
            init_conc[params.sand.top:params.sand.bot] = params.init_conc
            init_sk = th.zeros(params.X_STEPS)
            init_sk[params.sand.top:params.sand.bot] = params.kin_sorb
            sample_c[:,0] = init_conc
            sample_sk[:,0] = init_sk

            # add measured btc points
            if config.learn.c_out:
                for i, index in enumerate(loss_indices):
                    sample_c[-1,index] = exp_conc[i]
            #sample_c[-1,:] = th.tensor(sample_exp, dtype=th.float).to(device=device)
            # add measured interp. kin. sorbed
            
            last_sk = th.zeros(params.X_STEPS)
            last_sk[params.sand.top:params.sand.bot] = params.sorb_end
            sample_sk[:,-1]= last_sk
            

            u = th.stack((sample_c, sample_sk), dim=len(sample_c.shape))
            t = th.tensor(t, dtype=th.float).to(device=device)
            dx = params.X_LENGTH/(params.X_STEPS -1)
            
        else:
            u = th.tensor(np.load(f"results/{config.model.number}/u_FD.npy"),
                             dtype=th.float).to(device=device)  
            t = th.tensor(np.load(f"results/{config.model.number}/t_series.npy"),
                      dtype=th.float).to(device=device)
            x = th.tensor(np.load(f"results/{config.model.number}/x_series.npy"),
                      dtype=th.float).to(device=device)
           
            # #adds noice with mu = 0, std = data.noise
            # #for all rows apart from the first one
            # # 1 to last in all dimensions
            u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)

            # same dx over all x
            dx = x[1]-x[0]  
            
        # Initialize and set up the model
        model = FINN_DiffAD2ss(
            u = u,
            D = np.array(params.alpha_l*params.v_e+params.D_e),
            BC = np.array([0.0, 0.0]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="test",
            learn_coeff=False,
            learn_f=False,
            learn_f_hyd=True,
            learn_g_hyd=True,
            learn_r_hyd=True,
            learn_ve=False,
            learn_k_d=False,
            learn_beta=False,
            learn_sk=False,
            learn_alpha=False,
            t_steps=len(t),
            rho_s = np.array(params.rho_s),
            f = np.array(params.f),
            k_d = np.array(params.k_d),
            beta = np.array(params.beta),
            n_e = np.array(params.porosity),
            alpha = np.array(params.a_k),
            v_e = np.array(params.v_e),
            sand=params.sandbool,
            D_sand = np.array(params.sand.alpha_l*params.sand.v_e),
            n_e_sand = np.array(params.sand.porosity),
            x_start_soil = np.array(params.sand.top),
            x_stop_soil = np.array(params.sand.bot),
            x_steps_soil = np.array(params.X_STEPS),
            alpha_l_sand = np.array(params.sand.alpha_l),
            v_e_sand = np.array(params.sand.v_e),
            config=None,
            learn_stencil=False,
            bias=True,
            sigmoid=True
        ).to(device=device)
        
    elif config.data.type == "diffusion_sorption":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)

        x = np.load(os.path.join(data_path, "x_series.npy"))
        sample_c = th.tensor(np.load(os.path.join(data_path, "sample_c.npy")),
                             dtype=th.float).to(device=device)
        sample_ct = th.tensor(np.load(os.path.join(data_path, "sample_ct.npy")),
                             dtype=th.float).to(device=device)
        print(sample_c.shape)
        dx = x[1]-x[0]
        u = th.stack((sample_c, sample_ct), dim=len(sample_c.shape))
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        # Initialize and set up the model
        if "test" in config.data.name:
            bc = np.array([[0.7, 0.7], [0.0, 0.0]])
        else:
            bc = np.array([[1.0, 1.0], [0.0, 0.0]])
            
        model = FINN_DiffSorp(
            u = u,
            D = np.array([0.0005, 0.000145]),
            BC = bc,
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="test",
            learn_coeff=False
        ).to(device=device)
    
    elif config.data.type == "diffusion_reaction":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        y = np.load(os.path.join(data_path, "y_series.npy"))
        sample_u = th.tensor(np.load(os.path.join(data_path, "sample_u.npy")),
                             dtype=th.float).to(device=device)
        sample_v = th.tensor(np.load(os.path.join(data_path, "sample_v.npy")),
                             dtype=th.float).to(device=device)
        
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        
        u = th.stack((sample_u, sample_v), dim=len(sample_u.shape))
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
    
        # Initialize and set up the model
        model = FINN_DiffReact(
            u = u,
            D = np.array([5E-4/(dx**2), 1E-3/(dx**2)]),
            BC = np.zeros((4,2)),
            dx = dx,
            dy = dy,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="test",
            learn_coeff=True
        ).to(device=device)
        
    elif config.data.type == "allen_cahn":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        dx = x[1]-x[0]
    
        # Initialize and set up the model
        model = FINN_AllenCahn(
            u = u,
            D = np.array([0.3]),
            BC = np.array([[0.0], [0.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=True
        ).to(device=device)

    elif config.data.type == "burger_2d":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        y = np.load(os.path.join(data_path, "y_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        dx = x[1]-x[0]
        dy = y[1]-y[0]
                         
        # Initialize and set up the model
        model = FINN_Burger2D(
            u = u,
            D = np.array([1.75]),
            BC = np.zeros((4,1)),
            dx = dx,
            dy = dy,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=True
        ).to(device=device)
    

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable model parameters: {pytorch_total_params}\n")

    # Load the trained weights from the checkpoints into the model
    ##x = th.load(os.path.join(os.path.abspath(""),
    ##                                          "checkpoints",
    ##                                          config.model.name,
    ##                                          config.model.name+".pt"))
    ##print(x)
    model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                              "checkpoints",
                                              config.model.name,
                                              config.model.name+".pt")))
    
    model.eval()
    print(model.__dict__)
    # Forward data through the model
    time_start = time.time()
    with th.no_grad():
        u_hat = model(t=t, u=u)
    if print_progress:
        print(f"Forward pass took: {time.time() - time_start} seconds.")
    loss_1 = 0
    loss_2 = 0
    loss_3 = 0
    neg_u = nn.ReLU()(-u_hat)
                    
    ref_u = th.zeros((neg_u.size(dim=0), neg_u.size(dim=1), neg_u.size(dim=2)), device=model.device, requires_grad=True)
    loss_3 = nn.MSELoss(reduction="mean")(ref_u, neg_u)
    
    for eval_index in loss_indices:
        if eval_index == 0:
            pass
        else:
            loss_1 += th.abs(u_hat[-1,eval_index,0]-u[-1,eval_index,0])/u[-1,eval_index,0]
    loss_1 = loss_1/(len(loss_indices)-1)

    loss_2 = nn.MSELoss(reduction="mean")(u_hat[model.x_start:model.x_stop,-1,1], u[model.x_start:model.x_stop,-1,1])

    print(f"loss_1: {loss_1}, loss_2: {loss_2}, loss_3: {loss_3}")
    u_hat = u_hat.detach().cpu()
    u = u.detach().cpu()
    t = t.detach().cpu()
    #plot_tensor(u_hat[...,1])
    # Compute error
    if config.data.name == "data_exp":
        pass
    else:
        mse = nn.MSELoss()(u_hat, u)
        print(f"MSE: {mse}")

    # store diffusion-ad2ss data
    if config.data.type == "diffusion_ad2ss":
        print(config.model.number)
        with open(f"results/{config.model.number}/model.pkl", "wb") as outp:    
            pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
        np.save(f"results/{config.model.number}/u_hat", u_hat)
        np.save(f"results/{config.model.number}/u", u)
        params.save(f"results/{config.model.number}/", filename="params_NN.json")
        config.save(f"results/{config.model.number}/", filename="config_NN.json")

    # Visualize the data
    if config.data.type == "burger" and visualize:
        u_hat = np.transpose(u_hat)
        u = np.transpose(u)
    
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
        # u(t, x) over space
        h = ax[0].imshow(u, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[0].set_xlim(0, t.max())
        ax[0].set_ylim(x.min(), x.max())
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        ax[0].set_title('$u(t,x)$', fontsize = 10)
        
        h = ax[1].imshow(u_hat, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[1].set_xlim(0, t.max())
        ax[1].set_ylim(x.min(), x.max())
        ax[1].legend(loc="upper right")
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x$')
        ax[1].set_title('$u(t,x)$', fontsize = 10)
        
        # u(t, x) over time
        fig, ax = plt.subplots()
        line1, = ax.plot(x, u[:, 0], 'b-', linewidth=2, label='Exact')
        line2, = ax.plot(x, u_hat[:, 0], 'ro', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-1.1, 1.1])
    
        anim = animation.FuncAnimation(fig,
                                       animate_1d,
                                       frames=len(t),
                                       fargs=(line1, line2, u, u_hat),
                                       interval=20)
        plt.tight_layout()
        plt.draw()
        plt.show()

    elif config.data.type == "diffusion_sorption" and visualize:
        print(u_hat.shape)
        u_hat = np.transpose(u_hat[...,0])
        print(u_hat.shape)
        u = np.transpose(u[...,0])
        print(u)
        print(u_hat)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
        # u(t, x) over space
        h = ax[0].imshow(u, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[0].set_xlim(0, t.max())
        ax[0].set_ylim(x.min(), x.max())
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        ax[0].set_title('$u(t,x)$', fontsize = 10)
        
        h = ax[1].imshow(u_hat, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[1].set_xlim(0, t.max())
        ax[1].set_ylim(x.min(), x.max())
        ax[1].legend(loc="upper right")
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x$')
        ax[1].set_title('$u(t,x)$', fontsize = 10)
        
        # u(t, x) over time
        fig, ax = plt.subplots()
        line1, = ax.plot(x, u[:, 0], 'b-', linewidth=2, label='Exact')
        line2, = ax.plot(x, u_hat[:, 0], 'ro', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-1.1, 1.1])
    
        anim = animation.FuncAnimation(fig,
                                       animate_1d,
                                       frames=len(t),
                                       fargs=(line1, line2, u, u_hat),
                                       interval=20)
        plt.tight_layout()
        plt.draw()
        plt.show()
        
        plt.figure()
        plt.plot(x,u_hat[:,-1])
        plt.scatter(x,u[:,-1])
        
    elif config.data.type == "diffusion_reaction" and visualize:
    
        # Plot u over space
        fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    
        im1 = ax[0].imshow(u_hat[-1,:,:,0].squeeze().t().detach(), interpolation='nearest', 
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im1, ax=ax[0])
        im1.set_clim(u[-1,:,:,0].min(), u[-1,:,:,0].max())
        im2 = ax[1].imshow(u[-1,:,:,0].squeeze().t().detach(), interpolation='nearest', 
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im2, ax=ax[1])
        im2.set_clim(u[-1,:,:,0].min(), u[-1,:,:,0].max())
        ax[0].set_xlabel("$x$")
        ax[0].set_ylabel("$y$")
        ax[0].set_title('$u(x,y) predicted$', fontsize = 10)
        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$y$")
        ax[1].set_title('$u(x,y) data$', fontsize = 10)
        plt.show()
        
        # Animate through time
        anim = animation.FuncAnimation(fig,
                                        animate_2d,
                                        frames=len(t),
                                        fargs=(im1, im2, u_hat[...,0], u[...,0]),
                                        interval=20)
        
        plt.tight_layout()
        plt.show()
        
        # Plot v over space
        fig, ax = plt.subplots(2, 1, figsize=(6, 10))
    
        im1 = ax[0].imshow(u_hat[-1,:,:,1].squeeze().t().detach(), interpolation='nearest', 
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im1, ax=ax[0])
        im1.set_clim(u[-1,:,:,1].min(), u[-1,:,:,1].max())
        im2 = ax[1].imshow(u[-1,:,:,1].squeeze().t().detach(), interpolation='nearest', 
                     extent=[x.min(), x.max(),
                             y.min(), y.max()],
                     origin='lower', aspect='auto')
        fig.colorbar(im2, ax=ax[1])
        im2.set_clim(u[-1,:,:,1].min(), u[-1,:,:,1].max())
        ax[0].set_xlabel("$x$")
        ax[0].set_ylabel("$y$")
        ax[0].set_title('$v(x,y) predicted$', fontsize = 10)
        ax[1].set_xlabel("$x$")
        ax[1].set_ylabel("$y$")
        ax[1].set_title('$v(x,y) data$', fontsize = 10)
        
        # Animate through time
        anim = animation.FuncAnimation(fig,
                                        animate_2d,
                                        frames=len(t),
                                        fargs=(im1, im2, u_hat[...,1], u[...,1]),
                                        interval=20)
        
        plt.tight_layout()
        plt.draw()
        plt.show()
        
    elif config.data.type == "allen_cahn" and visualize:
        u_hat = np.transpose(u_hat)
        u = np.transpose(u)
    
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
        # u(t, x) over space
        h = ax[0].imshow(u, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[0].set_xlim(0, t.max())
        ax[0].set_ylim(x.min(), x.max())
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel('$t$')
        ax[0].set_ylabel('$x$')
        ax[0].set_title('$u(t,x)$', fontsize = 10)
        
        h = ax[1].imshow(u_hat, interpolation='nearest', 
                      extent=[t.min(), t.max(),
                              x.min(), x.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
    
        ax[1].set_xlim(0, t.max())
        ax[1].set_ylim(x.min(), x.max())
        ax[1].legend(loc="upper right")
        ax[1].set_xlabel('$t$')
        ax[1].set_ylabel('$x$')
        ax[1].set_title('$u(t,x)$', fontsize = 10)
        
        # u(t, x) over time
        fig, ax = plt.subplots()
        line1, = ax.plot(x, u[:, -1], 'b-', linewidth=2, label='Exact')
        line2, = ax.plot(x, u_hat[:, -1], 'ro', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([-1.1, 1.1])
    
        anim = animation.FuncAnimation(fig,
                                        animate_1d,
                                        frames=len(t),
                                        fargs=(line1, line2, u, u_hat),
                                        interval=20)
        plt.tight_layout()
        plt.draw()
        plt.show()
    
    if config.data.type == "burger_2d" and visualize:
    
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    
        # u(t, x) over space
        h1 = ax[0].imshow(u[0], interpolation='nearest', 
                      extent=[x.min(), x.max(),
                              y.min(), y.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[0])
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h1, cax=cax1)
    
        ax[0].set_xlim(x.min(), x.max())
        ax[0].set_ylim(y.min(), y.max())
        ax[0].set_xlabel(r'$x$')
        ax[0].set_ylabel(r'$y$')
        ax[0].set_title(r'$u(t,x,y)$', fontsize = 10)
        
        h2 = ax[1].imshow(u_hat[0], interpolation='nearest', 
                      extent=[x.min(), x.max(),
                              y.min(), y.max()],
                      origin='upper', aspect='auto')
        divider = make_axes_locatable(ax[1])
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h2, cax=cax2)
    
        ax[1].set_xlim(x.min(), x.max())
        ax[1].set_ylim(y.min(), y.max())
        ax[1].set_xlabel(r'$x$')
        ax[1].set_title(r'$\hat{u}(t,x,y)$', fontsize = 10)

        anim = animation.FuncAnimation(fig,
                                       animate_2d,
                                       frames=len(t),
                                       fargs=(h1, h2, u, u_hat),
                                       interval=10)
        plt.tight_layout()
        plt.draw()
        plt.show()

    return model


def animate_1d(t, axis1, axis2, field, field_hat):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    axis1.set_ydata(field[:, t])
    axis2.set_ydata(field_hat[:, t])


def animate_2d(t, im1, im2, u_hat, u):
    """
    Data animation function animating an image over time.
    :param t: The current time step
    :param axis: The matplotlib image object
    :param field: The data field
    :return: The matplotlib image object updated with the current time step's
        image date
    """
    im1.set_array(u_hat[t,:,:].squeeze().t().detach())
    im2.set_array(u[t,:,:].squeeze().t().detach())



if __name__ == "__main__":
    th.set_num_threads(1)
    
    model = run_testing(print_progress=True, visualize=True)

    print("Done.")
