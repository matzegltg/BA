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
    
    params_N1 = Configuration(f"results/412/init_params.json")
    params_N2 = Configuration("params_N2.json")
    df = pd.read_excel("../../220613_ColumnExperiments_Data_N1.xlsx", "N1", skiprows=140, nrows=16, usecols="B:J")

    # select experimental PFOS row
    exp_conc = df.iloc[[0]].to_numpy(dtype=np.float32).squeeze()

    # ng/L -> mug/L -> mug/cm^3
    exp_conc = exp_conc/1000000

    # c(t=0, x) = 0 \forall x in \Omega_x 
    exp_conc = np.insert(exp_conc, 0, 0)

    # select experimental measure points
    exp_t = df.iloc[[15]].to_numpy(dtype=np.float32).squeeze()

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
    t = np.linspace(exp_mean_t[0], params_N2.T_MAX, num=params_N2.T_STEPS, dtype=np.float32)
    
    
    # get indices of time where experimental data is available
    loss_indices = []
    for meas_point in exp_mean_t:
        for i in range(len(t)):
            if np.abs(t[i]-meas_point) <= 0.01:
                loss_indices.append(i)
                break

    # "upscale" to sizes required by FINN
    sample_c = th.zeros((params_N1.X_STEPS, params_N2.T_STEPS), dtype=th.float32)
    sample_sk = th.zeros((params_N1.X_STEPS, params_N2.T_STEPS), dtype=th.float32)
    
    # add initial conditions
    init_conc = th.zeros(params_N1.X_STEPS)
    init_conc[params_N1.sand.top:params_N1.sand.bot] = params_N2.init_conc
    init_sk = th.zeros(params_N1.X_STEPS)
    init_sk[params_N1.sand.top:params_N1.sand.bot] = params_N2.kin_sorb
    sample_c[:,0] = init_conc
    sample_sk[:,0] = init_sk

    # add measured btc points
    
    for i, index in enumerate(loss_indices):
        sample_c[-1,index] = exp_conc[i]
    
    # add measured kin. sorbed points
    last_sk = th.zeros(params_N1.X_STEPS)
    last_sk[params_N1.sand.top:params_N1.sand.bot] = params_N2.sorb_end
    sample_sk[:,-1]= last_sk
    

    u = th.stack((sample_c, sample_sk), dim=len(sample_c.shape))
    t = th.tensor(t, dtype=th.float)
    dx = params_N1.X_LENGTH/(params_N1.X_STEPS -1)
    
    # Initialize and set up the model
    with open(f"results/412/model.pkl", "rb") as inp:
        model = pickle.load(inp)

    # do N2 specific changes for before using forward path of model, other params are the same like in N1
    model.D = th.tensor(params_N2.v_e*params_N2.alpha_l+params_N2.D_e, dtype=th.float64)
    model.D_sand = th.tensor(params_N2.v_e_sand*params_N2.alpha_l_sand, dtype=th.float64)
    
    model.v_e = th.tensor(params_N2.v_e, dtype=th.float64)
    model.v_e_sand = th.tensor(params_N2.v_e_sand, dtype=th.float64)
    
    model.rho_s = th.tensor(params_N2.rho_s, dtype=th.float64)
    model.n_e = th.tensor(params_N2.n_e, dtype=th.float64)
    
    # Count number of trainable parameters
    pytorch_total_params_N1 = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable model parameters: {pytorch_total_params_N1}\n")
    
    model.eval()
    
    print(model.__dict__)
    
    # no gradients for forward pass needed
    with th.no_grad():
        u_hat = model(t=t, u=u)
    loss_1 = 0
    loss_2 = 0
    loss_3 = 0
    neg_u = nn.ReLU()(-u_hat)
                    
    ref_u = th.zeros((neg_u.size(dim=0), neg_u.size(dim=1), neg_u.size(dim=2)), device=model.device)
    loss_3 = nn.MSELoss(reduction="mean")(ref_u, neg_u)
    
    for eval_index in loss_indices:
        if eval_index == 0:
          pass
        else:
          loss_1 += (th.abs(u_hat[-1,eval_index,0]-u[-1,eval_index,0]))/u[-1,eval_index,0]
    loss_1 = loss_1/(len(loss_indices)-1)

    loss_2 = nn.MSELoss(reduction="mean")(u_hat[model.x_start:model.x_stop,-1,1], u[model.x_start:model.x_stop,-1,1])

    print(f"loss_1: {loss_1}, loss_2: {loss_2}, loss_3: {loss_3}")

    u_hat = u_hat.detach().cpu()
    u = u.detach().cpu()
    t = t.detach().cpu()

    # store diffusion-ad2ss data
    np.save(f"results/412/u_N2", u)
    np.save(f"results/412/u_hat_N2", u_hat)
    np.save(f"results/412/t_N2", t)
    

if __name__ == "__main__":
    th.set_num_threads(1)
    
    run_testing(print_progress=True, visualize=True)

    print("Done.")
