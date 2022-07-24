# @File          :   comp_sim_exp.py
# @Last modified :   2022/07/15 19:15:11
# @Author        :   Matthias Gueltig

from data_generation import generate_sample
from simulator import Simulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def vis_breakthrough(t_sim:np.ndarray, mass_sim:np.ndarray, t_exp:np.ndarray, mass_exp:np.ndarray):
    fig, ax = plt.subplots()
    ax.scatter(t_sim, mass_sim, label="simulation")
    ax.scatter(t_exp, mass_exp, label="experiment")

    ax.set(xlabel='time (d)', ylabel='mass PFOS ($\mu$g)',
        title='Breakthrough curve')
    ax.grid()
    ax.legend()
    fig.savefig("vis_break.png")
    plt.show()

def get_sim():
    sim_df = pd.read_csv('sim_c.csv').iloc[0]
    sim_data = sim_df.to_numpy()[1:]
    t_arr = np.linspace(0, len(sim_data)-1, len(sim_data))
    return t_arr, sim_data

def get_exp():
    xls = pd.ExcelFile("C:\\Users\\matth\\OneDrive - bwedu\\6. Semester\\BA\\Collaborations\\PFAS\\Daten\\220613_ColumnExperiments_Data_N1.xlsx")
    df_exp = xls.parse('N1', skiprows=9, nrows=50, usecols="A:U")
    
    df_PFOS = df_exp.loc[df_exp["Unnamed: 0"] == "PFOS"]
    df_Vw = df_exp.loc[df_exp["Unnamed: 0"] == "Vw [L]"]
    PFOS_arr = df_PFOS.to_numpy()[0][1:]
    Vw_arr = np.transpose(df_Vw.to_numpy()[0][1:])
    mass_PFOS = PFOS_arr * Vw_arr * 1/1000

    df_times = df_exp.loc[df_exp["Unnamed: 0"] == "t [d]"]
    t_arr = df_times.to_numpy()[0][1:]

    return t_arr, mass_PFOS

if __name__ == "__main__":
    t_sim, mass_sim = get_sim()
    t_exp, mass_exp = exp_data = get_exp() 
    vis_breakthrough(t_sim, mass_sim, t_exp, mass_exp)