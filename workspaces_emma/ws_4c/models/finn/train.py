#! env/bin/python3

"""
Main file for training a model with FINN
"""

import os
from pickletools import uint2
import sys
import time
from threading import Thread

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append("..")
import utils.helper_functions as helpers
from utils.configuration import Configuration

from finn import *

def plot_tensor(tensor):
        fig, ax = plt.subplots(1,1)
        h = ax.imshow(tensor.detach().numpy(), interpolation='nearest', 
                extent=[0, 10,
                        0, 55],
                origin='upper', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
        plt.show()

def run_training(print_progress=True, model_number=None):

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
    print(data_path)
    # Set device on GPU if specified in the configuration file, else CPU
    # device = helpers.determine_device()
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
            mode="train",
            learn_coeff=True
        ).to(device=device)
        
    elif config.data.type == "diffusion_sorption":
        # Load samples, together with x, y, and t series

        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))

        # #size: [501, 26]
        sample_c = th.tensor(np.load(os.path.join(data_path, "sample_c.npy")),
                             dtype=th.float).to(device=device)

        # #size: [501, 26]
        sample_ct = th.tensor(np.load(os.path.join(data_path, "sample_ct.npy")),
                             dtype=th.float).to(device=device)

        # #same dx over all x
        dx = x[1]-x[0]

        # #size: [501, 26, 2]
        u = th.stack((sample_c, sample_ct), dim=len(sample_c.shape))
        # #adds noice with mu = 0, std = data.noise
        # #for all rows apart from the firs one
        # # 1 to last in all dimensions
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        # Initialize and set up the model
        model = FINN_DiffSorp(
            u = u,
            D = np.array([0.0005, 0.000145]),
            BC = np.array([[1.0, 1.0], [0.0, 0.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=False
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
            if config.learn.sk_end:
                last_sk = th.zeros(params.X_STEPS)
                last_sk[params.sand.top:params.sand.bot] = params.kin_sorb_end
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
            mode="train",
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
            mode="train",
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
            D = np.array([0.6]),
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
                             dtype=th.float, requires_grad=False).to(device=device)
        
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
    # #numel returns total number of elements in p
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    if print_progress:
        print("Trainable model parameters:", pytorch_total_params)

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if config.training.continue_training:
        if print_progress: 
            print('Restoring model (that is the network\'s weights) from file...')
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                      "checkpoints",
                                      config.model.name,
                                      config.model.name + ".pt")))
        model.train()

    #
    # Set up an optimizer and the criterion (loss)
    optimizer = th.optim.Adam(model.parameters(),
                                lr=config.training.learning_rate)
        
    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    best_train = np.infty
    """
    TRAINING
    """
    a = time.time()

    # Start the training and iterate over all epochs
    for epoch in range(config.training.epochs):
       
        epoch_start_time = time.time()
        
        # Define the closure function that consists of resetting the
        # gradient buffer, loss function calculation, and backpropagation
        # It is necessary for LBFGS optimizer, because it requires multiple
        # function evaluations
       
        def closure():
            # Set the model to train mode
            model.train()

            # Reset the optimizer to clear data from previous iterations
            optimizer.zero_grad()

            # Forward propagate and calculate loss function
            # time x space x (c, sk)
            
            u_hat = model(t=t, u=u)
            #plot_tensor(u_hat[...,1])
            if config.data.type == "diffusion_ad2ss":
                if config.data.name == "data_exp":
                    

                    # In order to prevent sk from being negative get all possibly
                    # negative s_k from calculate solution
                    # elem: fixed row (place), all columns (time)
                    neg_u = nn.ReLU()(-u_hat)
                    
                    ref_u = th.zeros((neg_u.size(dim=0), neg_u.size(dim=1), neg_u.size(dim=2)), device=model.device, requires_grad=True)
                    #print(f"sk: {neg_u[model.x_start:model.x_stop,:,1]}")
                    #print(f"c: {neg_u[:,:,0]}")
                    neg_u_loss = nn.MSELoss(reduction="mean")(ref_u, neg_u)
                    #print(f"neg_loss: {neg_u_loss}")
                    #pen_count = th.tensor(pen_count, dtype=th.float, device=model.device, requires_grad=True)
                    #pen_count_goal = th.tensor(0, dtype=th.float, device=model.device)
                    
                    # calculate loss based on exp btc data
                    mse_btc_points = 0
                    if config.learn.c_out:
                        # calculate loss for experimental btc points (MSE) 
                        #plot_tensor(u[:,loss_indices[1]:,0])
                        #mse_btc_points == nn.MSELoss(reduction="mean")(u_hat[-1,loss_indices[1]:,0], u[-1,loss_indices[1]:,0])
                        #plot_tensor(u[...,0].detach().numpy())        
                        #print(f"f: {model.__dict__['_parameters']['f'].item()}")
                        #print(f"k_d: {model.__dict__['_parameters']['k_d'].item()}")
                        #print(f"beta: {model.__dict__['_parameters']['beta'].item()}")
                        #print(f"alpha: {model.__dict__['_parameters']['alpha_unc'].item()}")
                        for eval_index in loss_indices:
                            if config.learn.loss_calc_c_out == "NL1":
                                if eval_index == 0:
                                    pass
                                else:
                                    mse_btc_points += th.abs(u_hat[-1,eval_index,0]-u[-1,eval_index,0])/u[-1,eval_index,0]
                            else:
                                mse_btc_points += (u_hat[-1,eval_index,0] - u[-1,eval_index,0])**2

                        mse_btc_points = mse_btc_points/(len(loss_indices)-1)
                    # calculate loss base on exp sk data at end
                    mse_sk_end = 0
                    if config.learn.sk_end:
                        mse_sk_end = nn.MSELoss(reduction="mean")(u_hat[model.x_start:model.x_stop,-1,1], u[model.x_start:model.x_stop,-1,1])
                    print(f"sk_end_loss: {mse_sk_end}")
                    model.neg_loss = neg_u_loss
                    mse = mse_btc_points + mse_sk_end + 100000*neg_u_loss
                    print(f"BTC: {mse_btc_points}")
                    print(f"neg_loss: {neg_u_loss}")
                    
                        
                if config.data.name == "data_ext" or config.data.name == "data_train":
                    mse1 = 0
                    mse2 = 0
                    #print(f"f: {1/(1+np.exp(-(model.__dict__['_parameters']['f'].item())))}")
                    #print(f"k_d: {np.abs(model.__dict__['_parameters']['k_d'].item())}")
                    #print(f"beta: {1/(1+np.exp(-(model.__dict__['_parameters']['beta'].item())))}")
                    #print(f"alpha: {np.abs(model.__dict__['_parameters']['alpha'].item())}")
                    if config.learn.c_out:
                        mse1 = nn.MSELoss(reduction="mean")(u_hat[-1,:,0], u[-1,:,0])
                    if config.learn.sk_end:
                        mse2 = nn.MSELoss(reduction="mean")(u_hat[:,-1,1], u[:,-1,1])
                    mse = mse1 + mse2
                    if config.learn.synth:
                        #print(f"f: {1/(1+np.exp(-(model.__dict__['_parameters']['f'].item())))}")
                        print(f"f: {1/(1+np.exp(-(model.__dict__['_parameters']['f'].item())))}")
                        print(f"k_d: {np.abs(model.__dict__['_parameters']['k_d'].item())}")
                        print(f"beta: {1/(1+np.exp(-(model.__dict__['_parameters']['beta'].item())))}")
                        print(f"alpha: {np.abs(model.__dict__['_parameters']['alpha'].item())}")
                        mse = nn.MSELoss(reduction="mean")(u_hat, u)

            else:
                mse = nn.MSELoss(reduction="mean")(u_hat,u)
            print(f"loss: {mse}")
            mse.backward()            
            
            return mse
        
        optimizer.step(closure)
            
        # Extract the MSE value from the closure function
        mse = closure()
        
        epoch_errors_train.append(mse.item())

        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if model.neg_loss == 0:
            print("saving candidate")
            if epoch_errors_train[-1] < best_train:
                train_sign = "(+)"
                best_train = epoch_errors_train[-1]
                # Save the model to file (if desired)
                if config.training.save_model:
                    print("saved")
                    # Start a separate thread to save the model
                    thread = Thread(target=helpers.save_model_to_file(
                        model_src_path=os.path.abspath(""),
                        config=config,
                        epoch=epoch,
                        epoch_errors_train=epoch_errors_train,
                        epoch_errors_valid=epoch_errors_train,
                        net=model))
                    thread.start()

        # Print progress to the console
        if print_progress:
            print(f"Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}/{str(config.training.epochs)} took {str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} seconds. \t\tAverage epoch training error: {train_sign}{str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')}")
    
    b = time.time()
    if print_progress:
        print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')
    
    print(model.__dict__)
if __name__ == "__main__":
    th.set_num_threads(1)
    run_training(print_progress=True)

    print("Done.")
