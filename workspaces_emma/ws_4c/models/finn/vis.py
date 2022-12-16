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
sys.path.append("..")
from utils.configuration import Configuration
from finn import *
import pandas as pd


def __add_fig(fig, ax, row:float, column:float, title:str, value:np.ndarray, 
    x:np.ndarray, t:np.ndarray, exp=None, exp_mean_t_N2=None, exp_conc_N2=None):
    """add subplot to fig

    Args:
        fig (_type_): _description_
        ax (_type_): _description_
        row (float): _description_
        column (float): _description_
        title (str): _description_
        value (np.ndarray): _description_
        x (np.ndarray): _description_
        t (np.ndarray): _description_
    """
    font_size = 22
    h = ax[row, column].imshow(value, interpolation='nearest', 
                    extent=[t.min(), t.max(),
                            x.min(), x.max()],
                    origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[row,column])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=font_size)

    ax[row, column].set_xlim(0, t.max())
    ax[row, column].set_ylim(x.min(), x.max())
    ax[row, column].set_xlabel('$t [d]$', fontsize=font_size)
    ax[row, column].set_ylabel('$x [cm]$', fontsize=font_size)
    ax[row, column].set_title(title, fontsize = font_size)
    for label in (ax[row, column].get_xticklabels() + ax[row, column].get_yticklabels()): label.set_fontsize(font_size)
    if exp == True:
        inits = np.zeros(len(x))
        ax[row, column].scatter(inits,x,c="r", s=100)
        print(row)
        if row == 0:
            if exp_mean_t_N2:
                ax[row,column].scatter(exp_mean_t_N2, np.zeros(len(exp_mean_t_N2)),c="r",s=100)
            else:
                df = pd.read_excel("../../220613_ColumnExperiments_Data_N1.xlsx", "N1", skiprows=9, nrows=40, usecols="B:U")
                exp_t = df.iloc[[35]].to_numpy(dtype=np.float32).squeeze()
                exp_t = np.insert(exp_t, 0, 0)
            
                # select PFOS row
                exp_conc = df.iloc[[20]].to_numpy(dtype=np.float32).squeeze()
        
                # ng/L -> mug/L -> mug/cm^3
                exp_conc = exp_conc/1000000
        
                # t=0 initial concentration
                exp_conc = np.insert(exp_conc, 0, 0)
        
                # average concentrations -> shift to middle value of times
                exp_mean_t = []
                for i in range(0,len(exp_t)):
                    if i == 0:
                        exp_mean_t.append(exp_t[i])
                    else:
                        exp_mean_t.append((exp_t[i] + exp_t[i-1])/2)
                print("now!")
                z = np.zeros(len(exp_mean_t))
                ax[row, column].scatter(exp_mean_t,z,c="r",s=100)

def init_model(number:float, config_NN:Configuration):
    """Loads selected model

    Args:
        number (float): model number

    Returns:
        u_NN: NN calculated solution
        u_init_NN: c and sk as initialized in params.json before starting the NN
    """
    with open(f"results/{number}/model.pkl", "rb") as inp:
        model = pickle.load(inp)

    u_NN = np.load(f"results/{number}/u_hat.npy")
    if config_NN.data.name == "data_ext":
        u = np.load(f"results/{number}/u_FD.npy")
    elif config_NN.data.name == "data_exp":
        u = np.load(f"results/{number}/u.npy")
    t = np.load(f"results/{number}/t_series.npy")
    x = np.load(f"results/{number}/x_series.npy")
    return model, u, u_NN, t, x

def vis_FD_NN(model, u_FD:np.ndarray, u_NN:np.ndarray,
    t:np.ndarray, x:np.ndarray, config_NN, exp_mean_t_N2=None,
    exp_conc_N2=None):

    fig, ax = plt.subplots(2, 1)
    ax = np.expand_dims(ax, axis=1)
    print(ax.shape)
    if config_NN.data.name == "data_exp":
        title_c = "Exp. c"
        title_sk = "Exp.  $s_k$"
    else:
        title_c = r"FD: $c(t,x) \left[\frac{\mu g}{cm^3}\right]$"
        title_sk = r"FD: $s_k(t,x) \left[\frac{\mu g}{g}\right]$"
    
    #__add_fig(fig=fig, ax=ax, row=0, column=0, title=title_c, 
    #    value=u_FD[...,0], x=x, t=t, exp=True)
    #__add_fig(fig=fig, ax=ax, row=1, column=0, title=title_sk, 
    #    value=u_FD[...,1], x=x, t=t, exp=True)
    __add_fig(fig=fig, ax=ax, row=0, column=0, title=r"FINN: $c(t,x) \left[\frac{\mu g}{cm^3}\right]$", 
        value=u_NN[...,0], x=x, t=t, exp=True, exp_mean_t_N2=exp_mean_t_N2, exp_conc_N2=exp_conc_N2)
    __add_fig(fig=fig, ax=ax, row=1, column=0, title=r"FINN: $s_k(t,x) \left[\frac{\mu g}{g}\right]$", 
        value=u_NN[...,1], x=x, t=t, exp=True)
    fig.set_size_inches(20,10)
    plt.tight_layout()
    plt.savefig(f"results/{config_NN.model.number}/sol_ov", dpi=500)

def vis_diff(model, u_FD:np.ndarray, u_NN:np.ndarray, t:np.ndarray, x:np.ndarray, config_NN:Configuration):
    """calculates difference of u_NN and u_FD solution

    Args:
        model (_type_): _description_
        u_FD (np.ndarray): _description_
        u_NN (np.ndarray): _description_
        t (np.ndarray): _description_
        x (np.ndarray): _description_
    """
    if config_NN.data.name == "data_ext":
        diff_c = u_FD[...,0] - u_NN[...,0]
        diff_sk = u_FD[...,1] - u_NN[...,1]

        fig, ax = plt.subplots(1,2)
        ax = np.expand_dims(ax, axis=0)
        __add_fig(fig=fig, ax=ax, row=0, column=0, title=r"$c_{FD} - c_{FINN} \left[\frac{\mu g}{cm^3}\right]$",
            value=diff_c, x=x, t=t)
        __add_fig(fig=fig, ax=ax, row=0, column=1, title=r"$s_{k, FD} - s_{k, FINN} \left[\frac{\mu g}{g}\right]$",
            value=diff_sk, x=x, t=t)
        plt.tight_layout()
        plt.savefig(f"results/{config_NN.model.number}/diff")
    else:
        pass
def vis_comp_btc(u, u_NN, u_N2, u_NN_N2, t, t_N2, x, exp_mean_t_N2, exp_conc_N2):
    df = pd.read_excel("../../220613_ColumnExperiments_Data_N1.xlsx", 
    "N1", skiprows=9, nrows=40, usecols="B:U")
    exp_t = df.iloc[[35]].to_numpy(dtype=np.float32).squeeze()
    exp_t = np.insert(exp_t, 0, 0)
    
    # select PFOS row
    exp_conc = df.iloc[[20]].to_numpy(dtype=np.float32).squeeze()

    # ng/L -> mug/L -> mug/cm^3
    exp_conc = exp_conc/1000000

    # t=0 initial concentration
    exp_conc = np.insert(exp_conc, 0, 0)

    # average concentrations -> shift to middle value of times
    exp_mean_t = []
    for i in range(0,len(exp_t)):
        if i == 0:
            exp_mean_t.append(exp_t[i])
        else:
            exp_mean_t.append((exp_t[i] + exp_t[i-1])/2)
    fig, ax = plt.subplots()
    font_size = 22

    ax.set_xlabel("t [d]", fontsize=font_size)
    ax.set_ylabel("$c \left[\\frac{\mu g}{cm^3}\\right]$", fontsize=font_size)
    ax.set_title("Conc. of PFOS at outflow", fontsize=font_size)
    ax.set_yscale("log")
    ax.scatter(exp_mean_t, exp_conc, color="b", label="Exp. BTC N1_1")
    #ax.scatter(exp_mean_t_N2, exp_conc_N2, color="r", label="Exp. BTC N1_3")
    ax.plot(t, u_NN[-1,:,0], color="b", label="FINN BTC N1_1")
    ax.plot(t_N2, u_NN_N2[-1,:,0], color="r", label="FINN BTC N1_3")
    ax.set_ylim([10**(-5),0.04])
    
    ax.legend(fontsize=font_size)
    fig.set_size_inches(20,10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(font_size)
    plt.tight_layout()
    plt.savefig(f"results/412/btc", dpi=500)

def __load_exp():
    df_N2 = pd.read_excel("../../220613_ColumnExperiments_Data_N1.xlsx", "N1", skiprows=140, nrows=16, usecols="B:K")

    # select experimental PFOS row
    exp_conc_N2 = df_N2.iloc[[0]].to_numpy(dtype=np.float32).squeeze()

    # ng/L -> mug/L -> mug/cm^3
    exp_conc_N2 = exp_conc_N2/1000000

    # c(t=0, x) = 0 \forall x in \Omega_x 
    exp_conc_N2 = np.insert(exp_conc_N2, 0, 0)

    # select experimental measure points
    exp_t_N2 = df_N2.iloc[[15]].to_numpy(dtype=np.float32).squeeze()

    # insert t=0
    exp_t_N2 = np.insert(exp_t_N2, 0, 0)
    # average concentrations -> shift to middle value of times
    exp_mean_t_N2 = []
    for i in range(0,len(exp_t_N2)):
        if i == 0:
            exp_mean_t_N2.append(exp_t_N2[i])
        else:
            exp_mean_t_N2.append((exp_t_N2[i] + exp_t_N2[i-1])/2)
            
    return exp_mean_t_N2, exp_conc_N2
    
def vis_btc(model, u, u_hat, t, x, config_NN:Configuration, exp_mean_t_N2=None, exp_conc_N2=None):
    fig, ax = plt.subplots(1,2)
    font_size = 22
    if config_NN.data.name == "data_ext":

        # plot BTC
        ax[0].set_xlabel("t [d]", fontsize=font_size)
        ax[0].set_ylabel("$c \left[\\frac{\mu g}{cm^3}\\right]$", fontsize=font_size)
        ax[0].set_title("Conc. of PFOS at outflow", fontsize=font_size)
        
        ax[0].plot(t, u[-1,:,0], color="b", label="FD")
        ax[0].plot(t, u_hat[-1,:,0], color="y", label="FINN")

    elif config_NN.data.name == "data_exp":
        if exp_mean_t_N2:
            exp_mean_t = exp_mean_t_N2
            exp_conc = exp_conc_N2
        else:
          df = pd.read_excel("../../220613_ColumnExperiments_Data_N1.xlsx", 
              "N1", skiprows=9, nrows=40, usecols="B:U")
          exp_t = df.iloc[[35]].to_numpy(dtype=np.float32).squeeze()
          exp_t = np.insert(exp_t, 0, 0)
          
          # select PFOS row
          exp_conc = df.iloc[[20]].to_numpy(dtype=np.float32).squeeze()
  
          # ng/L -> mug/L -> mug/cm^3
          exp_conc = exp_conc/1000000
      
          # t=0 initial concentration
          exp_conc = np.insert(exp_conc, 0, 0)
      
          # average concentrations -> shift to middle value of times
          exp_mean_t = []
          for i in range(0,len(exp_t)):
              if i == 0:
                  exp_mean_t.append(exp_t[i])
              else:
                  exp_mean_t.append((exp_t[i] + exp_t[i-1])/2)
        
        ax[0].set_xlabel("t [d]", fontsize=font_size)
        ax[0].set_ylabel("$c \left[\\frac{\mu g}{cm^3}\\right]$", fontsize=font_size)
        ax[0].set_title("Conc. of PFOS at outflow", fontsize=font_size)
        ax[0].set_yscale("log")
        ax[0].set_ylim([10**(-5),0.04])
        ax[0].scatter(exp_mean_t, exp_conc, color="b", label="Exp. BTC")
        ax[0].plot(t, u_hat[-1,:,0], color="y", label="FINN BTC")
    
    # plot sk end
    ax[1].set_xlabel("$\left[\\frac{\mu g}{g}\\right]$", fontsize=font_size)
    ax[1].set_ylabel("$x [cm]$", fontsize=font_size)
    t_end = np.round(t[-1],2)
    ax[1].set_title(f"Sorbed conc. of PFOS at t = {t_end}d", fontsize=font_size)
    if config_NN.data.name == "data_ext":
        ax[1].plot(np.flip(u[:,-1,1]), x, color="b", label="FD")
    elif config_NN.data.name == "data_exp":
        ax[1].plot(np.flip(u[:,-1,1]), x,color="b", label="Total sorbed exp. concentration")
    
    ax[1].scatter(np.flip(u_hat[:,-1,1]), x, color="y", label="FINN prediction of $s_k$")
    ax[0].legend(fontsize=font_size)
    plt.locator_params(axis="x", nbins=6)
    ax[1].legend(fontsize=font_size)
    plt.locator_params(axis="x", nbins=6)
    for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()): label.set_fontsize(font_size)
    for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()): label.set_fontsize(font_size)
    fig.set_size_inches(20,10)
    plt.tight_layout()
    plt.savefig(f"results/{config_NN.model.number}/btc_sk_end", dpi=300)
  
def vis_sorption_isotherms(model, u, u_hat, t, x, config_NN:Configuration):
    font_size = 22
    fig, ax = plt.subplots()
    dt = t[1]-t[0]
    # plot sk over cw
    modulo_timesteps = np.ceil(len(t)/10)
    for timestep in range(800,len(t)):
        if timestep%modulo_timesteps == 0:
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(u_hat[model.x_start:model.x_stop,timestep,0], u_hat[model.x_start:model.x_stop,timestep,1], label=f"FINN: {np.round(timestep*dt, 2)}d", color=color)
            if config_NN.data.name == "data_ext":
                ax.plot(u[model.x_start:model.x_stop,timestep,0], u[model.x_start:model.x_stop,timestep,1], "--", label=f"FD: {np.round(timestep*dt, 2)}d",  color=color, alpha=0.9)

    ax.set_xlabel("$c \left[\\frac{\mu g}{cm^3}\\right]$", fontsize = font_size)
    ax.set_ylabel("$s_k \left[\\frac{\mu g}{g}\\right]$", fontsize = font_size)
    ax.set_title("$s_k$ vs. $c$ at different times", fontsize=font_size)
    ax.legend(fontsize=font_size)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(font_size)
    fig.set_size_inches(20,10)
    plt.tight_layout()
    plt.savefig(f"results/{config_NN.model.number}/sorpt_iso", dpi=500)

def init_model_N2():

    u_NN_N2 = np.load(f"results/412/u_hat_N2.npy")
    u_N2 = np.load(f"results/412/u_N2.npy")
    t = np.load(f"results/412/t_N2.npy")
    x = np.load(f"results/412/x_series.npy")
    return u_N2, u_NN_N2, t, x
    
def vis_data(number):

    # load NN params
    params_NN = Configuration(f"results/{number}/params_NN.json")
    params_FD = Configuration(f"results/{number}/init_params.json")
    config_NN = Configuration(f"results/{number}/config_NN.json")

    # load NN data
    model, u, u_NN, t, x = init_model(number, config_NN)
    u_N2, u_NN_N2, t_N2, x = init_model_N2()
    exp_mean_t_N2, exp_conc_N2 = __load_exp()
    #u_FD = np.load("results/412/u_FD.npy")
    # visualize
    #print(model.__dict__)
    #vis_FD_NN(model, u, u_NN, t, x, config_NN)
    vis_FD_NN(model, u_N2, u_NN_N2, t_N2, x, config_NN, exp_mean_t_N2, exp_conc_N2)
    #vis_diff(model, u, u_NN, t, x, config_NN)
    #plot_tensor(u_NN[...,0])
    #vis_btc(model, u, u_NN, t, x, config_NN)
    #vis_btc(model, u_N2, u_NN_N2, t_N2, x, config_NN, exp_mean_t_N2, exp_conc_N2)
    #vis_comp_btc(u, u_NN, u_N2, u_NN_N2, t, t_N2, x, exp_mean_t_N2, exp_conc_N2)
    #vis_sorption_isotherms(model, u, u_NN, t, x, config_NN)
    #vis_sorption_isotherms(model, u_N2, u_NN_N2, t_N2, x, config_NN)
if __name__ == "__main__":
    vis_data(number=412)
