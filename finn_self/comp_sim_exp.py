# @File          :   comp_sim_exp.py
# @Last modified :   2022/07/15 19:15:11
# @Author        :   Matthias Gueltig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_generation_2ss import T_STEPS
from data_generation_2ss import T

def vis_breakthrough(t_sim:np.ndarray=None, cw_sim:np.ndarray=None, t_exp:np.ndarray=None, cw_exp:np.ndarray=None):
    fig, ax = plt.subplots()
    
    ax.plot(t_sim, cw_sim*1000, label="simulation")
    
    ax.scatter(t_exp, cw_exp, label="experiment")
         
    ax.set(xlabel='time (d)', ylabel='conc PFOS (ng/L)',
        title='Breakthrough curve')
    ax.grid()
    ax.legend()
    ax.set_yscale("log")
    fig.savefig("vis_break.png")
    plt.show()

def get_sim():
    
    sim_df = pd.read_csv('sim_c.csv').iloc[0]
    print(sim_df.to_string())
    sim_data = sim_df.to_numpy()[1:]
    t_arr = np.linspace(0, len(sim_data)-1, len(sim_data))
    return t_arr, sim_data

def get_exp():
    xls = pd.ExcelFile("C:\\Users\\matth\\OneDrive - bwedu\\6. Semester\\BA\\Collaborations\\PFAS\\Daten\\220613_ColumnExperiments_Data_N1.xlsx")
    df_exp = xls.parse('N1', skiprows=9, nrows=50, usecols="A:U")
    
    # get PFOS values in ng/L
    df_PFOS = df_exp.loc[df_exp["Unnamed: 0"] == "PFOS"]
    PFOS_arr = df_PFOS.to_numpy()[0][1:]
    
    # get volume of water
    df_Vw = df_exp.loc[df_exp["Unnamed: 0"] == "Vw [L]"]
    Vw_arr = np.transpose(df_Vw.to_numpy()[0][1:])

    # get times
    df_times = df_exp.loc[df_exp["Unnamed: 0"] == "t [d]"]
    t_arr = df_times.to_numpy()[0][1:]

    return t_arr, PFOS_arr

def convert_sim_steps(t_sim:np.ndarray):
    """converts from simulated timesteps to days"""
    scaling = T_STEPS/T
    return t_sim/scaling

if __name__ == "__main__":

    t_sim, cw_sim = get_sim()
    print(t_sim)
    t_sim = convert_sim_steps(t_sim)
    print(t_sim)
    t_exp, cw_exp = exp_data = get_exp() 
    vis_breakthrough(t_sim=t_sim, cw_sim=cw_sim, t_exp=t_exp, cw_exp=cw_exp)