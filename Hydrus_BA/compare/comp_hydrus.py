# @File          :   comp_hydrus.py
# @Last modified :   2022/11/15 14:29:59
# @Author        :   Matthias Gueltig

import pandas as pd
import phydrus as ph
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_hydrus(model_number):
    """converts hydrus dict to np file which is needed for FINN"""
    hyd_dict = ph.read_nod_inf(path="../sol_fd_sand/Nod_Inf.out")

    t_series = []
    sample_c = []
    sample_sk = []
    
    for time, df in hyd_dict.items():
        t_series.append(time)
        # get samples c and sk at given time t  
        sample_c_t = df["Conc(1..NS)"].to_numpy()
        sample_c_t = np.flipud(sample_c_t)
        sample_sk_t  =df["Sorb(1...NS)"].to_numpy() 
        sample_sk_t = np.flipud(sample_sk_t)
        sample_c.append(sample_c_t)
        sample_sk.append(sample_sk_t)
    
    # create new np arrays with all time data
    t_series = np.array(t_series)
    sample_c = np.array(sample_c)
    sample_sk = np.array(sample_sk)

    # bring arrays in form like FD solution
    sample_c = sample_c.swapaxes(0,1)
    sample_sk = sample_sk.swapaxes(0,1)

    # delete first row because no ghost cell in FD solution
    sample_c = np.flip(sample_c, 0)[1:,:]
    sample_sk = np.flip(sample_sk, 0)[1:,:]

    # save
    np.save(f"{model_number}/t_series_hyd.npy", t_series)
    np.save(f"{model_number}/sample_c_hyd.npy", sample_c)
    np.save(f"{model_number}/sample_sk_hyd.npy", sample_sk)

def get_FD(model_number):
    # Load results of FD solver in FINN folder structure
    t_series_pde = np.load(f"../../../../../Documents/GitHub/finn_new - Kopie/models/finn/results/{model_number}/t_series.npy")
    u_FD = np.load(f"../../../../../Documents/GitHub/finn_new - Kopie/models/finn/results/{model_number}/u_FD.npy")

    # select corresponding entries
    sample_c_pde = u_FD[...,0]
    sample_sk_pde = u_FD[...,1]

    # save
    np.save(f"{model_number}/t_series_pde.npy", t_series_pde)
    np.save(f"{model_number}/sample_c_pde.npy", sample_c_pde)
    np.save(f"{model_number}/sample_sk_pde.npy", sample_sk_pde)

def comp_hyd_pde_last_cell(model_number):
    font_size = 22
    t_pde = np.load(f"{model_number}/t_series_pde.npy")
    t_hyd = np.load(f"{model_number}/t_series_hyd.npy")
    print(t_pde)
    print(t_hyd)
    print(t_pde[1])
    print(t_hyd[1])
    btc_hyd = np.load(f"{model_number}/sample_c_hyd.npy")[-1,:]
    btc_pde = np.load(f"{model_number}/sample_c_pde.npy")[-1,:]
    print(btc_hyd.shape)
    print(btc_pde.shape)
    mse = 0
    for index in range(0,len(btc_hyd)-1):
        print(btc_hyd[index])
        print(btc_pde[index*1000])
        print(index)
        mse+= (btc_hyd[index]-btc_pde[index*1000])**2
    
    mse = 1/len(btc_hyd)*mse
    print(mse)
    t_hyd = np.load(f"{model_number}/t_series_hyd.npy")
    t_pde = np.load(f"{model_number}/t_series_pde.npy")

    sk_hyd = np.load(f"{model_number}/sample_sk_hyd.npy")[-22,:]
    sk_pde = np.load(f"{model_number}/sample_sk_pde.npy")[-11,:]

    mse = 0
    for index in range(0,len(sk_hyd)-1):
        mse+= (sk_hyd[index]-sk_pde[index*1000])**2
    
    mse = 1/len(sk_hyd)*mse
    print(mse)
    #df = pd.read_excel("../../../../6. Semester/BA/Collaborations/PFAS/Daten/220613_ColumnExperiments_Data_N1.xlsx", "N1", skiprows=9, nrows=40, usecols="B:U")
    #exp_t = df.iloc[[35]].to_numpy(dtype=np.float32).squeeze()
    #exp_t = np.insert(exp_t, 0, 0)
    #print(df)
    # select PFOS row
    #exp_conc = df.iloc[[20]].to_numpy(dtype=np.float32).squeeze()

    # ng/L -> mug/L -> mug/cm^3
    #exp_conc = exp_conc/1000000
    # t=0 initial concentration
    #exp_conc = np.insert(exp_conc, 0, 0)
    # average concentrations -> shift to middle value of times
    #exp_mean_t = []
    #for i in range(0,len(exp_t)):
    #    if i == 0:
    #        exp_mean_t.append(exp_t[i])
    #    else:
    #        exp_mean_t.append((exp_t[i] + exp_t[i-1])/2)#

    fig, ax = plt.subplots(1,2)
    
    __plot2D(fig=fig, ax=ax, column=0, title = '',
        value1=btc_pde, value2=btc_hyd, t1=t_pde, t2=t_hyd, font_size=font_size,
        x_label=r'$t [d]$', y_label=r'conc PFOS $\left[\frac{\mu g}{cm^3}\right]$')
    __plot2D(fig=fig, ax=ax, column=1, title='',
        value1=sk_pde, value2=sk_hyd, t1=t_pde, t2=t_hyd, font_size=font_size, 
        x_label=r'$t[d]$', y_label=r'$s_k \left[\frac{\mu g}{g}\right]$')
    for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()): label.set_fontsize(font_size)
    for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()): label.set_fontsize(font_size)
    ax[1].set_yscale("linear")
    plt.show()

def __add_fig(fig, ax, row:float, column:float, title:str, value:np.ndarray):
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
                    extent=[0, 200,
                            0, 40],
                    origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[row,column])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=font_size)

    ax[row, column].set_xlim(0, 200)
    ax[row, column].set_ylim(0, 40)
    ax[row, column].set_xlabel('$t [d]$', fontsize=font_size)
    ax[row, column].set_ylabel('$x [cm]$', fontsize=font_size)
    ax[row, column].set_title(title, fontsize = font_size)
    for label in (ax[row, column].get_xticklabels() + ax[row, column].get_yticklabels()): label.set_fontsize(font_size)

def comp_hyd_diff(model_number):
    sample_c_pde = np.load(f"{model_number}/sample_c_pde.npy")
    sample_c_hyd = np.load(f"{model_number}/sample_c_hyd.npy")
    sample_sk_hyd = np.load(f"{model_number}/sample_sk_hyd.npy")
    sample_sk_pde = np.load(f"{model_number}/sample_sk_pde.npy")
    print(sample_c_pde.shape)
    print(sample_c_hyd.shape)
    
    fig, ax = plt.subplots(2, 2)

    __add_fig(fig=fig, ax=ax, row=0, column=0, title=r"FD: $c(t,x) \left[\frac{\mu g}{cm^3}\right]$", 
        value=sample_c_pde)
    __add_fig(fig=fig, ax=ax, row=1, column=0, title=r"FD: $s_k(t,x) \left[\frac{\mu g}{g}\right]$", 
        value=sample_sk_pde)
    __add_fig(fig=fig, ax=ax, row=0, column=1, title=r"Hydrus: $c(t,x) \left[\frac{\mu g}{cm^3}\right]$", 
        value=sample_c_hyd)
    __add_fig(fig=fig, ax=ax, row=1, column=1, title=r"Hydrus: $s_k(t,x) \left[\frac{\mu g}{g}\right]$", 
        value=sample_sk_hyd)
    plt.tight_layout()
    plt.show()

def __plot2D(fig, ax, column, title, value1, value2, t1, t2, font_size, x_label, y_label):
    ax[column].plot(t1, value1, label="FD")
    ax[column].plot(t2, value2, label="Hydrus")
    ax[column].set_xlabel(x_label, fontsize=font_size)
    ax[column].set_ylabel(y_label, fontsize=font_size)
    ax[column].set_title(title, fontsize=font_size)
    ax[column].grid()
    ax[column].legend(fontsize=font_size)
    ax[column].set_yscale("log")
    
    

if __name__ == "__main__":
    # please add folder with name "model number" to save model
    model_number = 101
    
    #get_hydrus(model_number=model_number)
    #get_FD(model_number=model_number)
    #comp_hyd_diff(model_number=model_number)
    comp_hyd_pde_last_cell(model_number=model_number)