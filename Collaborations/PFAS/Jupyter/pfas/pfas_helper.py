#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A module with helper functions for PFAS
"""

import datetime
import numpy as np
import datetime as dt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib import cm
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy import optimize


__author__ = "Claus Haslauer (mail@planetwater.org)"
__version__ = "$Revision: 1.2 $"
__date__ = datetime.date(2020, 4, 15)
__copyright__ = "Copyright (c) 2017 Claus Haslauer"
__license__ = "Python"


def main():
    plt.ioff()

    print("Done! Yay!")


def calculate_timeseries_SC(data_out, data_in, chosen_experiment, soilmass_reference_N1, relative_mass, material, data_fluoratoms, fluor):
    selection = data_out[data_out['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_input = data_in[data_in['Name'] == chosen_experiment].copy()

    starting_date = selection_input['Date']
    mass = selection_input['MassOfSoil (dry) [kg]:']
    length = selection_input['LengthOfSoilMaterial [m]']
    dia = selection_input['Dia [m]']
    recharge = selection_input['Recharge [mm/yr]']
    porosity = selection_input['Porosity']

    bulk_density = mass / (np.pi * np.square(dia) * length / 4) / 1000  # Kg/L
    pore_volume = np.pi * np.square(dia) * length / 4 * porosity * 1000  # l
    
    if fluor == True:
        selection["∑PFAS"] = 0
        PFBA_col = selection.columns.get_loc("PFBA")
        EtFOSAA_col = selection.columns.get_loc("EtFOSAA")
        for col in range(PFBA_col,EtFOSAA_col):
            selection[selection.columns[col]] = selection[selection.columns[col]]*data_fluoratoms[selection.columns[col]][0]
            selection["∑PFAS"] = selection["∑PFAS"] + selection[selection.columns[col]]

    selection["∑PFAS_standard"] = selection['PFBA'] + selection['PFPeA'] + selection['PFHxA'] + selection['PFHpA'] + selection['PFOA'] + selection['PFNA'] + selection['PFDA'] + selection['PFUnDA'] + selection['PFDoDA'] + selection['PFBS'] + selection['PFPeS'] + selection['PFHxS'] + selection['PFHpS'] + selection['PFOS'] + selection['PFNS'] + selection['PFDS']

    compounds = selection.columns[7:37]
    for i in range (len(compounds)):
        mass_compound = 'm_' + compounds[i]
        selection[mass_compound] = selection[compounds[i]] * selection['Vw [L]'] # ng
    for i in range (len(compounds)):
        mass_compound = 'm_' + compounds[i]
        cum_mass_compound = 'cum_m_' + compounds[i]
        selection[cum_mass_compound] = selection[mass_compound].cumsum() / 1000 # ug
    for i in range (len(compounds)):
        cum_mass_compound = 'cum_m_' + compounds[i]
        cum_mass_soilmass_compound = 'cum_m_sm_' + compounds[i]
        selection[cum_mass_soilmass_compound] = selection[cum_mass_compound].apply(lambda x : x / mass  )
    
    for i in range (len(compounds)):
        c_c0_compound = 'c_c0_' + compounds[i]
        selection[c_c0_compound] = selection[compounds[i]] / max(selection[compounds[i]]) 
    
    if soilmass_reference_N1 == True:
        for i in range (len(material)):
            cum_m_sm_PFBA_col = selection.columns.get_loc("cum_m_sm_PFBA")
            cum_m_sm_PFAS_standard_col = selection.columns.get_loc("cum_m_sm_∑PFAS_standard")
            if material == ['R1'] or material == ['R2']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/97.5
            elif material == ['R3']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/93
            elif material == ['R4']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/6.4
    
    if relative_mass == True:
        cum_m_sm_PFBA_col = selection.columns.get_loc("cum_m_sm_PFBA")
        cum_m_sm_PFAS_standard_col = selection.columns.get_loc("cum_m_sm_∑PFAS")
        for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col+1):
            selection[selection.columns[col]] = selection[selection.columns[col]] / max(selection[selection.columns[col]])

    selection['QS'] = selection['PFBA']/10000 + selection['PFHxA']/6000 + selection['PFOA']/100 + selection['PFNA']/60 + selection['PFBS']/6000 + selection['PFHxS']/100 + selection['PFOS']/100
    
    selection['c/c0'] = selection['∑PFAS']/ (max(selection['∑PFAS']))

    selection['Sampling_Time [d]'] = selection['Date'].apply(lambda x: x - starting_date)
    selection['Sampling_Time [d]'] = selection['Sampling_Time [d]'] / pd.offsets.Day(1)  # Days 
    selection['Sampling_Time_mean [d]'] = (selection['Sampling_Time [d]'] + selection['Sampling_Time [d]'].shift(1)) / 2
    selection['Sampling_Time_mean [d]'].iloc[0] = selection['Sampling_Time [d]'].iloc[0] / 2   # Days

    selection['CumVw_(L)'] = selection['Vw [L]'].cumsum()                                             # Cumulative volume [L]
    selection['WS_sample'] = selection['CumVw_(L)'].apply(lambda x: x / mass)                         # W/S Ratio
    selection['WS_mean'] = ( selection['WS_sample'] + selection['WS_sample'].shift(1) ) / 2           # Mean W/S Ratio
    selection['WS_mean'].iloc[0] = selection['WS_sample'].iloc[0] / 2 

    selection['pore_vol'] = selection['CumVw_(L)']                                              # Pore volume
    
    if selection_input['Type'].values == "SC":
        selection['field_time (Y)'] = selection['WS_mean'].apply(
            lambda x: x * 365 * length * bulk_density / (recharge / 1000) )  # Days
    elif selection_input['Type'].values == "LY":
        selection['field_time (Y)'] = selection['CumVw_(L)'].apply(
            lambda x: ( x / (np.pi * np.square(dia) / 4) ) / recharge * 365  )  # Days

    selection['field_time_mean (Y)'] = (selection['field_time (Y)'] + selection['field_time (Y)'].shift(1)) / 2
    selection['field_time_mean (Y)'].iloc[0] = selection['field_time (Y)'].iloc[0] / 2
    selection['field_time (Y)'] = selection['field_time (Y)'].values.astype("timedelta64[D]")
    selection['field_date'] = selection['field_time (Y)'].apply(lambda x: starting_date + x)
    selection['field_time (Y)'] = selection['field_time (Y)'] / np.timedelta64(1, 'Y')
    selection['field_time_mean (Y)'] = selection['field_time_mean (Y)'].values.astype("timedelta64[D]")
    selection['field_time_mean (Y)'] = selection['field_time_mean (Y)'] / np.timedelta64(1, 'Y')

    return selection

def calculate_timeseries_IS(data_out, data_in, chosen_experiment, soilmass_reference_N1, relative_mass, material, data_fluoratoms, fluor):
    selection = data_out[data_out['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_input = data_in[data_in['Name'] == chosen_experiment].copy()

    starting_date = selection_input['Date']
    mass = selection_input['MassOfSoil (dry) [kg]:']
    length = selection_input['LengthOfSoilMaterial [m]']
    dia = selection_input['Dia [m]']
    recharge = selection_input['Recharge [mm/yr]']
    porosity = selection_input['Porosity']

    bulk_density = mass / (np.pi * np.square(dia) * length / 4) / 1000  # Kg/L
    pore_volume = np.pi * np.square(dia) * length / 4 * porosity * 1000  # l
    
    if fluor == True:
        selection["∑PFAS"] = 0
        PFBA_col = selection.columns.get_loc("PFBA")
        EtFOSAA_col = selection.columns.get_loc("EtFOSAA")
        for col in range(PFBA_col,EtFOSAA_col):
            selection[selection.columns[col]] = selection[selection.columns[col]]*data_fluoratoms[selection.columns[col]][0]
            selection["∑PFAS"] = selection["∑PFAS"] + selection[selection.columns[col]]
    
    selection['∑PFAS_standard'] = selection['PFBA'] + selection['PFPeA'] + selection['PFHxA'] + selection['PFHpA'] + selection['PFOA'] + selection['PFNA'] + selection['PFDA'] + selection['PFUnDA'] + selection['PFDoDA'] + selection['PFBS'] + selection['PFPeS'] + selection['PFHxS'] + selection['PFHpS'] + selection['PFOS'] + selection['PFNS'] + selection['PFDS']

    compounds = selection.columns[7:36]
    for i in range (len(compounds)):
        mass_compound_soilmass = 'm_sm_' + compounds[i]
        cum_mass_compound = 'cum_m_' + compounds[i]
        selection[mass_compound_soilmass] = selection[compounds[i]] /0.1/1000 # µg/kg
        selection[cum_mass_compound] = selection[compounds[i]].cumsum() / 1000 # ug
    for i in range (len(compounds)):
        cum_mass_compound = 'cum_m_' + compounds[i]
        cum_mass_soilmass_compound = 'cum_m_sm_' + compounds[i]
        selection[cum_mass_soilmass_compound] = selection[cum_mass_compound].apply(lambda x : x / mass  )

    if soilmass_reference_N1 == True:
        cum_m_sm_PFBA_col = selection.columns.get_loc("cum_m_sm_PFBA")
        cum_m_sm_PFAS_standard_col = selection.columns.get_loc("cum_m_sm_∑PFAS_standard")
        for i in range (len(material)):
            if material == ['R1'] or material == ['R2']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/97.5
            elif material == ['R3']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/93
            elif material == ['R4']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/6.4
    
    if relative_mass == True:
        cum_m_sm_PFBA_col = selection.columns.get_loc("cum_m_sm_PFBA")
        cum_m_sm_PFAS_standard_col = selection.columns.get_loc("cum_m_sm_∑PFAS")
        for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col+1):
            selection[selection.columns[col]] = selection[selection.columns[col]] / max(selection[selection.columns[col]])

    selection['QS'] = selection['PFBA']/10000 + selection['PFHxA']/6000 + selection['PFOA']/100 + selection['PFNA']/60 + selection['PFBS']/6000 + selection['PFHxS']/100 + selection['PFOS']/100

    selection['Sampling_Time [d]'] = selection['Date'].apply(lambda x: x - starting_date)
    selection['Sampling_Time [d]'] = selection['Sampling_Time [d]'] / pd.offsets.Day(1)  # Days 

    return selection

def calculate_timeseries_IS_w(data_out, data_in, chosen_experiment, soilmass_reference_N1, material, data_fluoratoms, fluor):
    selection = data_out[data_out['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_input = data_in[data_in['Name'] == chosen_experiment].copy()

    starting_date = selection_input['Date']
    mass = selection_input['MassOfSoil (dry) [kg]:']
    length = selection_input['LengthOfSoilMaterial [m]']
    dia = selection_input['Dia [m]']
    recharge = selection_input['Recharge [mm/yr]']
    porosity = selection_input['Porosity']

    bulk_density = mass / (np.pi * np.square(dia) * length / 4) / 1000  # Kg/L
    pore_volume = np.pi * np.square(dia) * length / 4 * porosity * 1000  # l
    
    if fluor == True:
        selection["∑PFAS"] = 0
        PFBA_col = selection.columns.get_loc("PFBA")
        EtFOSAA_col = selection.columns.get_loc("EtFOSAA")
        for col in range(PFBA_col,EtFOSAA_col):
            selection[selection.columns[col]] = selection[selection.columns[col]]*data_fluoratoms[selection.columns[col]][0]
            selection["∑PFAS"] = selection["∑PFAS"] + selection[selection.columns[col]]
    
    selection['∑PFAS_standard'] = selection['PFBA'] + selection['PFPeA'] + selection['PFHxA'] + selection['PFHpA'] + selection['PFOA'] + selection['PFNA'] + selection['PFDA'] + selection['PFUnDA'] + selection['PFDoDA'] + selection['PFBS'] + selection['PFPeS'] + selection['PFHxS'] + selection['PFHpS'] + selection['PFOS'] + selection['PFNS'] + selection['PFDS']

    compounds = selection.columns[7:36]
    for i in range (len(compounds)):
        mass_compound = 'm_' + compounds[i]
        mass_compound_soilmass = 'm_sm_' + compounds[i]
        selection[mass_compound] = selection[compounds[i]] * selection['Vw [L]'] # ng
        selection[mass_compound_soilmass] = selection[compounds[i]] * selection['Vw [L]'] /0.1/1000 # µg/kg
    for i in range (len(compounds)):
        mass_compound = 'm_' + compounds[i]
        cum_mass_compound = 'cum_m_' + compounds[i]
        selection[cum_mass_compound] = selection[mass_compound].cumsum() / 1000 # ug
    for i in range (len(compounds)):
        cum_mass_compound = 'cum_m_' + compounds[i]
        cum_mass_soilmass_compound = 'cum_m_sm_' + compounds[i]
        selection[cum_mass_soilmass_compound] = selection[cum_mass_compound].apply(lambda x : x / mass  )
        
    if soilmass_reference_N1 == True:
        cum_m_sm_PFBA_col = selection.columns.get_loc("cum_m_sm_PFBA")
        cum_m_sm_PFAS_standard_col = selection.columns.get_loc("cum_m_sm_∑PFAS_standard")
        for i in range (len(material)):
            if material == ['R1'] or material == ['R2']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/97.5
            elif material == ['R3']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/93
            elif material == ['R4']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/6.4

    selection['QS'] = selection['PFBA']/10000 + selection['PFHxA']/6000 + selection['PFOA']/100 + selection['PFNA']/60 + selection['PFBS']/6000 + selection['PFHxS']/100 + selection['PFOS']/100

    selection['Sampling_Time [d]'] = selection['Date'].apply(lambda x: x - starting_date)
    selection['Sampling_Time [d]'] = selection['Sampling_Time [d]'] / pd.offsets.Day(1)  # Days 

    return selection

def calculate_timeseries_IS_fit_linear(data_1, data_2, data_in, chosen_experiment, material):
    selection = data_1[data_1['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_2 = data_2[data_2['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_input = data_in[data_in['Name'] == chosen_experiment].copy()

    starting_date = selection_input['Date']
    mass = selection_input['MassOfSoil (dry) [kg]:']
    length = selection_input['LengthOfSoilMaterial [m]']
    dia = selection_input['Dia [m]']
    recharge = selection_input['Recharge [mm/yr]']
    porosity = selection_input['Porosity']

    bulk_density = mass / (np.pi * np.square(dia) * length / 4) / 1000  # Kg/L
    pore_volume = np.pi * np.square(dia) * length / 4 * porosity * 1000  # l
    
    #selection["∑PFAS_standard"] = selection['PFBA'] + selection['PFPeA'] + selection['PFHxA'] + selection['PFHpA'] + selection['PFOA'] + selection['PFNA'] + selection['PFDA'] + selection['PFUnDA'] + selection['PFDoDA'] + selection['PFBS'] + selection['PFPeS'] + selection['PFHxS'] + selection['PFHpS'] + selection['PFOS'] + selection['PFNS'] + selection['PFDS']
    
    #selection["∑PFAS_vorTOP"] = selection["∑PFAS"] - selection["F-Zugewinn"]

    PFBA_col_1 = selection.columns.get_loc("cum_m_sm_PFBA")
    EtFOSAA_col_1 = selection.columns.get_loc("cum_m_sm_EtFOSAA")
    #PFBA_col_afterTOP = selection_2.columns.get_loc("PFBA")
    #diSAmPAP_col_afterTOP = selection_2.columns.get_loc("EtFOSAA")
    
    for col in range(PFBA_col_1,EtFOSAA_col_1+1):
        compound = selection.columns[col]
        data_PFAS = np.concatenate((selection[compound][4:], selection_2[compound][4:]))
        data_t = np.concatenate((selection['Sampling_Time [d]'][4:], selection_2['Sampling_Time [d]'][4:]))
        linear_model = np.polyfit(data_t,data_PFAS,1)
        linear_model_fn = np.poly1d(linear_model)
        selection['slope_'+compound] = linear_model_fn[1]
        selection['intercept_'+compound] = linear_model_fn[0]

    return selection

def calculate_timeseries_IS_fit_langmuir(data_1, m0, data_in, chosen_experiment, material):
    selection = data_1[data_1['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_input = data_in[data_in['Name'] == chosen_experiment].copy()

    starting_date = selection_input['Date']
    mass = selection_input['MassOfSoil (dry) [kg]:']
    length = selection_input['LengthOfSoilMaterial [m]']
    dia = selection_input['Dia [m]']
    recharge = selection_input['Recharge [mm/yr]']
    porosity = selection_input['Porosity']

    bulk_density = mass / (np.pi * np.square(dia) * length / 4) / 1000  # Kg/L
    pore_volume = np.pi * np.square(dia) * length / 4 * porosity * 1000  # l
    
    #selection["∑PFAS_standard"] = selection['PFBA'] + selection['PFPeA'] + selection['PFHxA'] + selection['PFHpA'] + selection['PFOA'] + selection['PFNA'] + selection['PFDA'] + selection['PFUnDA'] + selection['PFDoDA'] + selection['PFBS'] + selection['PFPeS'] + selection['PFHxS'] + selection['PFHpS'] + selection['PFOS'] + selection['PFNS'] + selection['PFDS']
    
    #selection["∑PFAS_vorTOP"] = selection["∑PFAS"] - selection["F-Zugewinn"]

    PFBA_col_1 = selection.columns.get_loc("cum_m_sm_PFBA")
    EtFOSAA_col_1 = selection.columns.get_loc("cum_m_sm_EtFOSAA")
    #PFBA_col_afterTOP = selection_2.columns.get_loc("PFBA")
    #diSAmPAP_col_afterTOP = selection_2.columns.get_loc("EtFOSAA")
    
    #def func(t, K, A):
    #    return K*t/(A+K*t)
    def func2(t, k):
        return (1-np.exp(-k*t))
    from sklearn.metrics import r2_score
    
    for col in range(PFBA_col_1,EtFOSAA_col_1+2):
        compound = selection.columns[col]
        print(compound)
        #if m0[compound[9:]].values > 0:
        if max(selection[compound].values) > 0:
            #data_PFAS = selection[compound].values/(m0[compound[9:]].values)
            data_PFAS = selection[compound].values/ max(selection[compound].values)
        else:
            data_PFAS = selection[compound].values
        t = selection['Sampling_Time [d]'].values
        #popt, pcov = curve_fit(func, t, data_PFAS, bounds=(0, [1.]))
        #y_fit = popt[0]*t/(popt[1]+popt[0]*t)
        #r2 = r2_score(data_PFAS, y_fit)
        #print(r2)
        #selection['K_'+compound] = popt[0]
        #selection['A_'+compound] = popt[1]
        #selection['R2_'+compound] = r2
        popt, pcov = curve_fit(func2, t, data_PFAS)
        y_fit = 1-np.exp(-popt[0]*t)
        r2 = r2_score(data_PFAS, y_fit)
        print(r2)
        selection['k_'+compound] = popt[0]
        selection['R2_'+compound] = r2

    return selection

def calculate_timeseries_SC_fit_linear(data_1, data_2, data_in, chosen_experiment, material):
    selection = data_1[data_1['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_2 = data_2[data_2['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_input = data_in[data_in['Name'] == chosen_experiment].copy()

    starting_date = selection_input['Date']
    mass = selection_input['MassOfSoil (dry) [kg]:']
    length = selection_input['LengthOfSoilMaterial [m]']
    dia = selection_input['Dia [m]']
    recharge = selection_input['Recharge [mm/yr]']
    porosity = selection_input['Porosity']

    bulk_density = mass / (np.pi * np.square(dia) * length / 4) / 1000  # Kg/L
    pore_volume = np.pi * np.square(dia) * length / 4 * porosity * 1000  # l
    
    #selection["∑PFAS_standard"] = selection['PFBA'] + selection['PFPeA'] + selection['PFHxA'] + selection['PFHpA'] + selection['PFOA'] + selection['PFNA'] + selection['PFDA'] + selection['PFUnDA'] + selection['PFDoDA'] + selection['PFBS'] + selection['PFPeS'] + selection['PFHxS'] + selection['PFHpS'] + selection['PFOS'] + selection['PFNS'] + selection['PFDS']
    
    #selection["∑PFAS_vorTOP"] = selection["∑PFAS"] - selection["F-Zugewinn"]

    PFBA_col_1 = selection.columns.get_loc("cum_m_sm_PFBA")
    EtFOSAA_col_1 = selection.columns.get_loc("cum_m_sm_EtFOSAA")
    #PFBA_col_afterTOP = selection_2.columns.get_loc("PFBA")
    #diSAmPAP_col_afterTOP = selection_2.columns.get_loc("EtFOSAA")
    
    for col in range(PFBA_col_1,EtFOSAA_col_1+1):
        compound = selection.columns[col]
        data_PFAS = np.concatenate((selection[compound][7:], selection_2[compound][7:]))
        data_t = np.concatenate((selection['WS_sample'][7:], selection_2['WS_sample'][7:]))
        linear_model = np.polyfit(data_t,data_PFAS,1)
        linear_model_fn = np.poly1d(linear_model)
        selection['slope_'+compound] = linear_model_fn[1]
        selection['intercept_'+compound] = linear_model_fn[0]

    return selection

def calculate_timeseries_SC_TOP(data_out, data_in, chosen_experiment, material):
    selection = data_out[data_out['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_input = data_in[data_in['Name'] == chosen_experiment].copy()

    starting_date = selection_input['Date']
    mass = selection_input['MassOfSoil (dry) [kg]:']
    length = selection_input['LengthOfSoilMaterial [m]']
    dia = selection_input['Dia [m]']
    recharge = selection_input['Recharge [mm/yr]']
    porosity = selection_input['Porosity']

    bulk_density = mass / (np.pi * np.square(dia) * length / 4) / 1000  # Kg/L
    pore_volume = np.pi * np.square(dia) * length / 4 * porosity * 1000  # l
    
    selection["∑PFAS_standard"] = selection['PFBA'] + selection['PFPeA'] + selection['PFHxA'] + selection['PFHpA'] + selection['PFOA'] + selection['PFNA'] + selection['PFDA'] + selection['PFUnDA'] + selection['PFDoDA'] + selection['PFBS'] + selection['PFPeS'] + selection['PFHxS'] + selection['PFHpS'] + selection['PFOS'] + selection['PFNS'] + selection['PFDS']
    
    selection["∑PFAS_vorTOP"] = selection["∑PFAS"] - selection["F-Zugewinn"]

    selection['Sampling_Time [d]'] = selection['Date'].apply(lambda x: x - starting_date)
    selection['Sampling_Time [d]'] = selection['Sampling_Time [d]'] / pd.offsets.Day(1)  # Days 
    
    selection['WS_mean'] = selection['WS']
    selection['CumVw_(L)'] = selection['WS_mean'] * mass
    print(chosen_experiment)
    print(sum(selection['F(TOP)'] * selection['Vw [L]'] / sum(selection['Vw [L]'])))


    if selection_input['Type'].values == "SC":
        selection['field_time (Y)'] = selection['WS_mean'].apply(
            lambda x: x * 365 * length * bulk_density / (recharge / 1000) )  # Days
    elif selection_input['Type'].values == "LY":
        selection['field_time (Y)'] = selection['CumVw_(L)'].apply(
            lambda x: ( x / (np.pi * np.square(dia) / 4) ) / recharge * 365  )  # Days

    selection['field_time_mean (Y)'] = (selection['field_time (Y)'] + selection['field_time (Y)'].shift(1)) / 2
    selection['field_time_mean (Y)'].iloc[0] = selection['field_time (Y)'].iloc[0] / 2
    selection['field_time (Y)'] = selection['field_time (Y)'].values.astype("timedelta64[D]")
    selection['field_date'] = selection['field_time (Y)'].apply(lambda x: starting_date + x)
    selection['field_time (Y)'] = selection['field_time (Y)'] / np.timedelta64(1, 'Y')
    selection['field_time_mean (Y)'] = selection['field_time_mean (Y)'].values.astype("timedelta64[D]")
    selection['field_time_mean (Y)'] = selection['field_time_mean (Y)'] / np.timedelta64(1, 'Y')

    return selection

def calculate_timeseries_TOP_molar(data_beforeTOP, data_afterTOP, data_in, data_fluoratoms, chosen_experiment, material):
    selection = data_beforeTOP[data_beforeTOP['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_afterTOP = data_afterTOP[data_afterTOP['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_input = data_in[data_in['Name'] == chosen_experiment].copy()

    starting_date = selection_input['Date']
    mass = selection_input['MassOfSoil (dry) [kg]:']
    length = selection_input['LengthOfSoilMaterial [m]']
    dia = selection_input['Dia [m]']
    recharge = selection_input['Recharge [mm/yr]']
    porosity = selection_input['Porosity']

    bulk_density = mass / (np.pi * np.square(dia) * length / 4) / 1000  # Kg/L
    pore_volume = np.pi * np.square(dia) * length / 4 * porosity * 1000  # l
    
    #selection["∑PFAS_standard"] = selection['PFBA'] + selection['PFPeA'] + selection['PFHxA'] + selection['PFHpA'] + selection['PFOA'] + selection['PFNA'] + selection['PFDA'] + selection['PFUnDA'] + selection['PFDoDA'] + selection['PFBS'] + selection['PFPeS'] + selection['PFHxS'] + selection['PFHpS'] + selection['PFOS'] + selection['PFNS'] + selection['PFDS']
    
    #selection["∑PFAS_vorTOP"] = selection["∑PFAS"] - selection["F-Zugewinn"]
    
    selection['Sampling_Time [d]'] = selection['Date'].apply(lambda x: x - starting_date)
    selection['Sampling_Time [d]'] = selection['Sampling_Time [d]'] / pd.offsets.Day(1)  # Days 
    selection['Sampling_Time_mean [d]'] = (selection['Sampling_Time [d]'] + selection['Sampling_Time [d]'].shift(1)) / 2
    selection['Sampling_Time_mean [d]'].iloc[0] = selection['Sampling_Time [d]'].iloc[0] / 2   # Days

    selection['CumVw_(L)'] = selection['Vw [L]'].cumsum()                                             # Cumulative volume [L]
    selection['WS_sample'] = selection['CumVw_(L)'].apply(lambda x: x / mass)                         # W/S Ratio
    selection['WS_mean'] = ( selection['WS_sample'] + selection['WS_sample'].shift(1) ) / 2           # Mean W/S Ratio
    selection['WS_mean'].iloc[0] = selection['WS_sample'].iloc[0] / 2 

    PFBA_col_beforeTOP = selection.columns.get_loc("PFBA")
    EtFOSAA_col_beforeTOP = selection.columns.get_loc("EtFOSAA")
    #PFBA_col_afterTOP = selection_afterTOP.columns.get_loc("PFBA")
    #diSAmPAP_col_afterTOP = selection_afterTOP.columns.get_loc("EtFOSAA")
    
    for col in range(PFBA_col_beforeTOP,EtFOSAA_col_beforeTOP+1):
        compound = selection.columns[col]
        fluormass = data_fluoratoms[selection.columns[col]][0]
        molarmass = data_fluoratoms[selection.columns[col]][1]
        selection['mol_'+compound] = selection[compound] / molarmass   #nmol/L
        selection_afterTOP['mol_'+compound] = selection_afterTOP[compound] /fluormass / molarmass  #nmol/L
    
    for col in range(PFBA_col_beforeTOP,EtFOSAA_col_beforeTOP+1):
        compound = selection.columns[col]
        selection_afterTOP['mol_abs_'+compound] = selection_afterTOP['mol_'+compound] * selection_afterTOP['Vw [L]']
        selection['mol_abs_'+compound] = selection['mol_'+compound] * selection['Vw [L]']
        
    
    sum_mol_beforeTOP = selection['PFBA']*0
    sum_mol_afterTOP = selection_afterTOP['PFBA']*0
    mol_PFBA_col_beforeTOP = selection.columns.get_loc("mol_PFBA")
    mol_EtFOSAA_col_beforeTOP = selection.columns.get_loc("mol_EtFOSAA")
    for col in range(mol_PFBA_col_beforeTOP, mol_EtFOSAA_col_beforeTOP+1):
        compound = selection.columns[col]
        sum_mol_beforeTOP = sum_mol_beforeTOP +  selection[compound]
    
    mol_PFBA_col_afterTOP = selection_afterTOP.columns.get_loc("mol_PFBA")
    mol_EtFOSAA_col_afterTOP = selection_afterTOP.columns.get_loc("mol_EtFOSAA")
    for col2 in range(mol_PFBA_col_afterTOP, mol_EtFOSAA_col_afterTOP+1):
        compound = selection_afterTOP.columns[col2]
        sum_mol_afterTOP = sum_mol_afterTOP +  selection_afterTOP[compound]
    
    selection['mol_beforeTOP'] = sum_mol_beforeTOP
    selection_afterTOP['mol_afterTOP'] = sum_mol_afterTOP
    
    diff_mol_TOP = selection['PFBA']*0
    rel_diff_mol_TOP = selection['PFBA']*0
    diff_mol_TOP_excl = selection_afterTOP['PFBA']*0
    rel_diff_mol_TOP_excl = selection_afterTOP['PFBA']*0
    
    j = 0
    for i in range(len(diff_mol_TOP)):
        if j < len(sum_mol_afterTOP):
            if selection.iloc[i]['Number'] == selection_afterTOP.iloc[j]['Number']:
                diff_mol_TOP.iloc[i] = selection_afterTOP.iloc[j]['mol_afterTOP'] - selection.iloc[i]['mol_beforeTOP']
                rel_diff_mol_TOP.iloc[i] = diff_mol_TOP.iloc[i] / selection_afterTOP.iloc[j]['mol_afterTOP']
                diff_mol_TOP_excl.iloc[j] = diff_mol_TOP.iloc[i]
                rel_diff_mol_TOP_excl.iloc[j] = rel_diff_mol_TOP.iloc[i]
                j = j+1
    
    #j = 0
    #delete = 0
    #for i in range(len(diff_mol_TOP)):
    #    if i - delete < len(selection['Number']):
    #        print(selection.iloc[i-delete]['Number'])
    #        if j < len(sum_mol_afterTOP):
    #            if selection.iloc[i-delete]['Number'] != selection_afterTOP.iloc[j]['Number']:
    #                selection = selection[selection.Number != selection.iloc[i-delete]['Number']]
    #                delete = delete+1
    #            else:
    #                j = j+1
    #        elif j == len(sum_mol_afterTOP):
    #            selection = selection[selection.Number != selection.iloc[i-delete]['Number']]
    
    #print(selection['mol_abs_PFBA'])
    #print(selection_afterTOP['mol_abs_PFBA'])
    #for col in range(PFBA_col_beforeTOP,EtFOSAA_col_beforeTOP+1):
    #    compound = selection.columns[col]
    #    if sum(selection_afterTOP['mol_abs_'+compound]) - sum(selection['mol_abs_'+compound]) > 0:
    #        selection_afterTOP['mol_abs_diff_'+compound] = sum(selection_afterTOP['mol_abs_'+compound]) - sum(selection['mol_abs_'+compound])
    #    else:
    #        selection_afterTOP['mol_abs_diff_'+compound] = 0
    
    selection['diff_mol_TOP'] = diff_mol_TOP
    selection_afterTOP['diff_mol_TOP'] = diff_mol_TOP_excl
    selection['re_diff_mol_TOP'] = rel_diff_mol_TOP
    selection_afterTOP['rel_diff_mol_TOP'] = rel_diff_mol_TOP_excl
    
    #print(np.mean(selection_afterTOP['diff_mol_TOP']))
    #print(sum(selection_afterTOP['diff_mol_TOP'] * selection_afterTOP['Vw [L]']))
    #print(np.mean(selection_afterTOP['rel_diff_mol_TOP']))
    
    if selection_input['Type'].values == "SC":
        selection['field_time (Y)'] = selection['WS_mean'].apply(
            lambda x: x * 365 * length * bulk_density / (recharge / 1000) )  # Days
    elif selection_input['Type'].values == "LY":
        selection['field_time (Y)'] = selection['CumVw_(L)'].apply(
            lambda x: ( x / (np.pi * np.square(dia) / 4) ) / recharge * 365  )  # Days

    selection['field_time_mean (Y)'] = (selection['field_time (Y)'] + selection['field_time (Y)'].shift(1)) / 2
    selection['field_time_mean (Y)'].iloc[0] = selection['field_time (Y)'].iloc[0] / 2
    selection['field_time (Y)'] = selection['field_time (Y)'].values.astype("timedelta64[D]")
    selection['field_date'] = selection['field_time (Y)'].apply(lambda x: starting_date + x)
    selection['field_time (Y)'] = selection['field_time (Y)'] / np.timedelta64(1, 'Y')
    selection['field_time_mean (Y)'] = selection['field_time_mean (Y)'].values.astype("timedelta64[D]")
    selection['field_time_mean (Y)'] = selection['field_time_mean (Y)'] / np.timedelta64(1, 'Y')

    return selection_afterTOP

def calculate_timeseries_SC_LY(data_out, data_in, chosen_experiment, soilmass_reference_N1, relative_mass, material, data_fluoratoms, fluor):
    selection = data_out[data_out['Number'].str.startswith(chosen_experiment)].copy().sort_values(by=['Date'])
    selection_input = data_in[data_in['Name'] == chosen_experiment].copy()

    starting_date = selection_input['Date']
    mass = selection_input['MassOfSoil (dry) [kg]:']
    length = selection_input['LengthOfSoilMaterial [m]']
    dia = selection_input['Dia [m]']
    recharge = selection_input['Recharge [mm/yr]']
    porosity = selection_input['Porosity']

    bulk_density = mass / (np.pi * np.square(dia) * length / 4) / 1000  # Kg/L
    pore_volume = np.pi * np.square(dia) * length / 4 * porosity * 1000  # l
    
    if fluor == True:
        selection["∑PFAS"] = 0
        PFBA_col = selection.columns.get_loc("PFBA")
        EtFOSAA_col = selection.columns.get_loc("EtFOSAA")
        for col in range(PFBA_col,EtFOSAA_col):
            selection[selection.columns[col]] = selection[selection.columns[col]]*data_fluoratoms[selection.columns[col]][0]
            selection["∑PFAS"] = selection["∑PFAS"] + selection[selection.columns[col]]

    selection["∑PFAS_standard"] = selection['PFBA'] + selection['PFPeA'] + selection['PFHxA'] + selection['PFHpA'] + selection['PFOA'] + selection['PFNA'] + selection['PFDA'] + selection['PFUnDA'] + selection['PFDoDA'] + selection['PFBS'] + selection['PFPeS'] + selection['PFHxS'] + selection['PFHpS'] + selection['PFOS'] + selection['PFNS'] + selection['PFDS']

    compounds = selection.columns[7:37]
    for i in range (len(compounds)):
        mass_compound = 'm_' + compounds[i]
        selection[mass_compound] = selection[compounds[i]] * selection['Vw [L]'] # ng
    for i in range (len(compounds)):
        mass_compound = 'm_' + compounds[i]
        cum_mass_compound = 'cum_m_' + compounds[i]
        selection[cum_mass_compound] = selection[mass_compound].cumsum() / 1000 # ug
    for i in range (len(compounds)):
        cum_mass_compound = 'cum_m_' + compounds[i]
        cum_mass_soilmass_compound = 'cum_m_sm_' + compounds[i]
        selection[cum_mass_soilmass_compound] = selection[cum_mass_compound].apply(lambda x : x / mass  )
    
    for i in range (len(compounds)):
        c_c0_compound = 'c_c0_' + compounds[i]
        selection[c_c0_compound] = selection[compounds[i]] / max(selection[compounds[i]]) 
    
    if soilmass_reference_N1 == True:
        for i in range (len(material)):
            cum_m_sm_PFBA_col = selection.columns.get_loc("cum_m_sm_PFBA")
            cum_m_sm_PFAS_standard_col = selection.columns.get_loc("cum_m_sm_∑PFAS_standard")
            if material == ['R1'] or material == ['R2']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/97.5
            elif material == ['R3']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/93
            elif material == ['R4']:
                for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col):
                    selection[selection.columns[col]] = selection[selection.columns[col]]*100/6.4

    if relative_mass == True:
        cum_m_sm_PFBA_col = selection.columns.get_loc("cum_m_sm_PFBA")
        cum_m_sm_PFAS_standard_col = selection.columns.get_loc("cum_m_sm_∑PFAS")
        for col in range(cum_m_sm_PFBA_col,cum_m_sm_PFAS_standard_col+1):
            selection[selection.columns[col]] = selection[selection.columns[col]] / max(selection[selection.columns[col]])
    
    selection['QS'] = selection['PFBA']/10000 + selection['PFHxA']/6000 + selection['PFOA']/100 + selection['PFNA']/60 + selection['PFBS']/6000 + selection['PFHxS']/100 + selection['PFOS']/100

    selection['Sampling_Time [d]'] = selection['Date'].apply(lambda x: x - starting_date)
    selection['Sampling_Time [d]'] = selection['Sampling_Time [d]'] / pd.offsets.Day(1)  # Days 
    selection['Sampling_Time_mean [d]'] = (selection['Sampling_Time [d]'] + selection['Sampling_Time [d]'].shift(1)) / 2
    selection['Sampling_Time_mean [d]'].iloc[0] = selection['Sampling_Time [d]'].iloc[0] / 2   # Days
    
    
    selection['CumVw_(L)'] = selection['Vw [L]'].cumsum()                                             # Cumulative volume [L]
    if selection_input['Type'].values == 'IS':
        selection['WS_sample'] = selection['Sampling_Time [d]'] * 2/3                      # W/S with fitting coefficient
        selection['WS_mean'] = selection['Sampling_Time_mean [d]'] * 2/3
    else:
        selection['WS_sample'] = selection['CumVw_(L)'].apply(lambda x: x / mass)                         # W/S Ratio
        selection['WS_mean'] = ( selection['WS_sample'] + selection['WS_sample'].shift(1) ) / 2           # Mean W/S Ratio
        selection['WS_mean'].iloc[0] = selection['WS_sample'].iloc[0] / 2 

    selection['pore_vol'] = selection['CumVw_(L)']                                              # Pore volume

    return selection

def calculate_timeseries_pH(data_out, data_in, chosen_experiment):
    selection = data_out[data_out['Number'].str.startswith(chosen_experiment)].copy()

    #selection['cond/cond0'] = selection['cond']/ (max(selection['cond']))

    return selection

def hydrus_model(h_time, h_cumf, sample_time, vol, A):                 # Hydrus fitted model calculations     
    
    model =  pd.DataFrame({'time [d]':h_time, 'cum flux':abs(h_cumf)})        # Dataframe time and cum-bottom-flux from hydrus
    PFAS = model[model['time [d]'].isin(sample_time)]                         # select cum flux for respective experimental sampling time
       
    M_per_time = ( PFAS.loc[:,'cum flux'] - PFAS.loc[:,'cum flux'].shift(1) ) * A  # mass of solute per sampling time (dt)    
    M_per_time.iloc[0] = PFAS['cum flux'].iloc[0] * A
    
            
    conc = M_per_time.values / vol.values                                          # concentration (mass per volume) ug / L
    
     
    PFAS.insert(2,'M_per_time', M_per_time)                                        # insert mass in selected dataframe 
    PFAS.insert(3,'volume', vol)
    PFAS.insert(4,'conc', conc)                                                    # insert concentration in selected dataframe 
    
    return PFAS

def visualization_lines(x_value, x2_value, y_value, y2_value, second_xaxis, second_yaxis, Log, micro, save, select, ex, ex_type, ex_material, ex_material_nr, N1_soil, file_string):
    
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'legend.fontsize': 30})
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["legend.frameon"]
    #plt.rcParams.update({'legend.handlelength': 2})
    
    ex_nr = len(ex)
    # line properties for different experiments
    linecolor = ('b','darkorange','g','r','dimgrey')
    markers = ('o','^','v','X')
    styles = ('-','--',':','-.')
    
    #color specifications: substances get corresponding color (if single substances are selected)
    subst_color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','cyan','tab:cyan', #ca
'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink',                                           #sa
'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','cyan']             #prec
    subst_markers = ('o','o','o','o','o','o','o','o','o','o','o',                                                            #ca
'x','x','x','x','x','x','x',                                                                                                 #sa
'^','^','^','^','^','^','^','^','^','^')                                                                                     #prec
    substances_all = list(select[0].columns)[7:35]  #list(select[0].columns)[7:34]
    
    substances_sel = []
    if y_value[0].startswith('cum_m_sm_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][9:])
    elif y_value[0].startswith('cum_m_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][6:])
    elif y_value[0].startswith('m_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][2:])
    elif list(y_value) ==  ['all']:
        substances_sel = substances_all
    elif list(y_value) ==  ['standard substances']:
        substances_sel = substances_all[0:18]
    elif list(y_value) ==  ['special selection']:
        substances_sel = [substances_all[0],substances_all[4],substances_all[6],substances_all[15],substances_all[27]]
    else:
        substances_sel = y_value
    print(substances_sel)
    
    subst_color_sel = []
    subst_markers_sel = []
    j=0
    for i in range(len(substances_all)):
        if j < len(substances_sel):
            if substances_sel[j] == substances_all[i]:
                subst_color_sel.append(subst_color[i])
                subst_markers_sel.append(subst_markers[i])
                j = j+1

    print(subst_color_sel)
    
    if ex_nr > 1:
        labels = ['t [d]','t [d]','V [L]','WF [L/kg]','WF [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','t [a]','t [a]','field_date']
        headers = ['Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']
    else:
        labels = ['c [ng/L]','t [Tage]','t [Tage]','V [L]','WF [L/kg]','WF [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','t [a]','t [a]','field_date']
        headers = ['∑PFAS','Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']

    if micro == True:
        for i in range(ex_nr):   
            for j in range(len(y_value)):
                select[i][y_value[j]] = select[i][y_value[j]]/1000

    def label(xaxis):
        
        labelseries =  pd.Series(headers, index=labels) 
        label = list(labelseries[labelseries==xaxis].index)[0]
        
        return label

    if ex_nr > 1 and (len(y_value) > 1 or list(y_value) ==  ['all'] ):
        return (print('only single PFAS for multiple experiment'))
    
    fig, ax = plt.subplots(figsize=(20,10))   
    
    for i in range(ex_nr):
        
        if ex_nr > 1:
            y_value = list(y_value)
            print(y_value)
            
            lcol = linecolor[4]
            markertype = markers[0]
            lstyle = styles[0]
            if ex_material[i] == ['N1']:
                lcol = linecolor[0]
            elif ex_material[i] == ['R1']:
                lcol = linecolor[1]
            elif ex_material[i] == ['R2']:
                lcol = linecolor[2]
            elif ex_material[i] == ['R3']:
                lcol = linecolor[3]
            if ex_material_nr[i] == [2]:
                markertype = markers[1]
                lstyle = styles[1]
            elif ex_material_nr[i] == [3]:
                markertype = markers[2]
                lstyle = styles[2]
            elif ex_material_nr[i] == [4]:
                markertype = markers[3]
                lstyle = styles[3]

            select[i].plot( ax=ax, x = x_value, y = y_value, grid = True, figsize=(15,10),                                           
                            style = lstyle,
                            marker = markertype,
                            lw=2,
                            color = lcol
                            #ylim=(0.0,3000),
                            #color = 'k' if len(y_value) == 1 else None
                            )
        else:
        
            if list(y_value) ==  ['all'] or list(y_value) ==  ['standard substances']:                        #for different colouring depending on PFAS substance classes
                y_value_ca = list(select[i].columns)[7:17]
                y_value_sa = list(select[i].columns)[18:24]
                if list(y_value) ==  ['all']:
                    y_value_prec = list(select[i].columns)[25:34]
                #plotting each substance class
                select[i].plot( ax=ax, x = x_value, y = y_value_ca, grid = True, figsize=(15,10),                                           
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )

                select[i].plot( ax=ax, x = x_value, y = y_value_sa, grid = True, figsize=(15,10),                                           
                                style='-',
                                marker='x',
                                markersize='10',
                                lw=2,
                                #ylim=(0.0,3000),
                                )       
                ax.legend(y_value_ca + y_value_sa, handlelength=4)
                if list(y_value) ==  ['all']:
                    select[i].plot( ax=ax, x = x_value, y = y_value_prec, grid = True, figsize=(15,10),
                                style='--',
                                marker='^',
                                #markersize='10',
                                lw=2,
                                #ylim=(0.0,3000),
                                )
                    ax.legend(y_value_ca + y_value_sa + y_value_prec, handlelength=4)
            elif list(y_value) ==  ['special selection']:                        #PFBA, PFOA, PFDA, PFOS, ∑PFAS
                y_value = ['PFBA','PFOA','PFDA','PFOS','∑PFAS']
                select[i].plot( ax=ax, x = x_value, y = ['PFBA','PFOA','PFDA','PFOS'], grid = True, figsize=(15,10),
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )
                select[i].plot( ax=ax, x = x_value, y = ['∑PFAS'], grid = True, figsize=(15,10),
                                color = 'black',
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )

                ax.legend(substances_sel, handlelength=4)
            
            elif list(y_value) ==  ['cum_m_sm_special selection']:                        #PFBA, PFOA, PFDA, PFOS, ∑PFAS
                y_value = ['cum_m_sm_PFBA','cum_m_sm_PFOA','cum_m_sm_PFDA','cum_m_sm_PFOS','cum_m_sm_∑PFAS']
                select[i].plot( ax=ax, x = x_value, y = ['cum_m_sm_PFBA','cum_m_sm_PFOA','cum_m_sm_PFDA','cum_m_sm_PFOS'], grid = True, figsize=(15,10),
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )
                select[i].plot( ax=ax, x = x_value, y = 'cum_m_sm_∑PFAS', grid = True, figsize=(15,10),
                                color = 'black',
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )

                substances_sel = ['PFBA','PFOA','PFDA','PFOS','∑PFAS']
                ax.legend(substances_sel, handlelength=4)
            else:
                y_value = list(y_value)
                
                select[i].plot( ax=ax, x = x_value, y = y_value, grid = False, figsize=(15,10),
                            style = '-',
                            marker = 'o',   #subst_markers_sel[0],
                            lw=2,
                            #color = subst_color_sel
                            #ylim=(0.0,3000),
                            #color = 'k' if len(y_value) == 1 else None
                            )
                #substances_sel = ['PFOA','PFNA','PFDA','PFUnDA','PFDoDA','PFOS']
                ax.legend(substances_sel, handlelength=4)


        
        

    if x_value == "Date":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        ax.set_xlabel('Sample Date')
        
    if second_xaxis == True and x2_value != x_value and ex_type != 'IS' and ex_nr == 1:
        
        if ( x_value == headers[0] and x2_value == headers[4]) or ( x_value == headers[1] and x2_value == headers[3]) or ( x2_value == headers[0] and x_value == headers[4]) or ( x2_value == headers[1] and x_value == headers[3]):
            print('error: Sampling Time')
        
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
        
        if x2_value == 'field_date': 
            ax2.set_xticks(select[0][x2_value])
            ax2.xaxis.set_major_locator(mdates.YearLocator(5)) if ex_type == 'SC' else None        
            ax2.xaxis.set_major_locator(mdates.DayLocator(5)) if ex_type == 'LY' else None
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        else:         
            ax2.set_xticks(select[0][x_value].iloc[1::3])
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::3], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
        ax2.set_xlabel(label(x2_value))        
    elif second_xaxis == True and ex_nr > 1:
        if x2_value == 'field_time_mean (Y)' or x2_value == 'field_time (Y)':
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
            ax2.set_xticks(select[0][x_value].iloc[1::4])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::4], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
            ax2.set_xlabel(label(x2_value))        
        if x2_value == 'Sampling_Time_mean [d]' or x2_value == 'Sampling_Time [d]':
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
            ax2.set_xticks(select[0][x_value].iloc[1::4])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::4], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
            ax2.set_xlabel(label(x2_value))
    
    if second_yaxis == True and  y2_value != y_value[0] and ex_type != 'IS' and ex_nr == 1:             
        ay = ax.twinx()
        select[i].plot( ax=ay, x = x2_value, y = y2_value, style='--', marker='x', color = 'k') 
        ay.legend(loc='best')
        ay.set_yscale('log') if Log == True  else None 
        ay.set_ylim(ymin=0) if Log != True  else None
        ay.set_ylabel(label(y2_value))
    elif second_yaxis == True:
        print('Error: Second Y Axis')  
    
    if ex_nr > 1: #and bar != True
        ax.legend(ex[:],loc='best')
    elif ( (list(y_value) == ['m_∑PFAS']) or + (list(y_value) == ['cum_m_∑PFAS']) or + (list(y_value) == ['cum_m_sm_∑PFAS']) ):
        ax.legend(['∑PFAS'])
        
    #ax.get_legend().set_bbox_to_anchor((-0.07,1))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='out', top=False)
    ax.xaxis.set_tick_params(which='minor', size=5, width=1.5, direction='out', top=False)
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='out', right=False)
    ax.yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='out', right=False)
    #ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    #ax.minorticks_on()
    ax.set_yscale('log') if Log == True  else None    
    if x_value != "Date":
        ax.set_xlabel(label(x_value)) 
        ax.set_xlim(xmin=0)
    if second_yaxis == True:
        ax.legend([label(y_value[0])],loc='best', bbox_to_anchor=(1, 0.48))
        ay.legend([label(y2_value)],loc='best', bbox_to_anchor=(1, 0.52))
    if micro == True:
        ax.set_ylabel('c [\u03BCg/L]')
    elif y_value[0].startswith('c_c0'):
        ax.set_ylabel('c/c$_{0}$ [-]')
    else:
        ax.set_ylabel('c [ng/L]')
    #ax.set_ylabel('m [\u03BCg]') if ( (list(y_value) == ['cum_m_∑PFAS'] ) or + (list(y_value) == ['m_∑PFAS']) ) else None 
    if y_value[0].startswith('m_'):
        ax.set_ylabel('m [ng]')
    elif y_value[0].startswith('cum_m_sm_'):
        ax.set_ylabel('m [\u03BCg/kg]')
    elif y_value[0].startswith('cum_m_'):
        ax.set_ylabel('m [\u03BCg]') 
    ax.set_ylabel('QS [-]') if  (list(y_value) == ['Quotientensumme'])  else None
    if Log != True:
        ax.set_ylim(ymin=0)
    if list(y_value) == ['QS']:
        ax.set_ylim(ymax=5)
        ax.set_ylabel('QS [-]')
        ax.hlines(1, 0, 1000, colors='r', linestyles='dotted')
    if y_value[0].startswith('cum_m_sm_'):
        if ex_nr > 1:
            subst_color_sel = ['black']
        #for i in range(len(substances_sel)):
            #ax.hlines(N1_soil[substances_sel[i]], 0, 1000, colors=subst_color_sel[i], linestyles='dashed')
   
    #ax.set_ylabel('$ \\beta $ [ng/L]') 
    ax = plt.gca()    
    plt.gcf().autofmt_xdate() if x_value == 'Date' else None
    #plt.legend(ncol=1)
    plt.show()
    
    if save == True:
        fig = ax.get_figure()
        #fig.savefig("output/{}{}{}{}.png".format(file_string, ex, y_value, x_value), fontsize = 15)
        fig.savefig("output/{}.png".format(file_string), fontsize = 15)


    return ax


def visualization_bars(x_value, x2_value, y_value, y2_value, second_xaxis, second_yaxis, Log, stack, bar_PFAS, bar_exp, save, select, ex, ex_type, ex_material, ex_material_nr, file_string):
    
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'legend.fontsize': 30})
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["legend.frameon"]
    #plt.rcParams.update({'legend.handlelength': 2})
    
    ex_nr = len(ex)
    linecolor = ('b','darkorange','g','r','dimgrey')
    markers = ('o','^','v','X')
    styles = ('-','--',':','-.')
    
    if ex_nr > 1:
        labels = ['t [d]','t [d]','V [L]','WS [L/kg]','WS [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']
    else:
        labels = ['c [ng/L]','t [Tage]','t [Tage]','V [L]','WS [L/kg]','WS [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['∑PFAS','Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']
    
    
    bar = False
    if bar_PFAS == True or bar_exp == True:
        bar = True


    def label(xaxis):
        
        labelseries =  pd.Series(headers, index=labels) 
        label = list(labelseries[labelseries==xaxis].index)[0]
        
        return label

    if ex_nr > 1 and (len(y_value) > 1 or list(y_value) ==  ['all'] ) and bar != True:
        return (print('only single PFAS for multiple experiment'))
    
    fig, ax = plt.subplots(figsize=(20,10))   
    
    
    if stack == True:
        for i in range(ex_nr):      

            if list(y_value) ==  ['all'] or list(y_value) ==  ['standard substances']:                        #for different colouring depending on PFAS substance classes
                y_value_ca = list(select[i].columns)[7:17]
                y_value_sa = list(select[i].columns)[18:24]
                if list(y_value) ==  ['all']:
                    y_value_prec = list(select[i].columns)[25:34]
                
            else:
                y_value = list(y_value)
                
                lcol = linecolor[4]
                markertype = markers[0]
                lstyle = styles[0]
                if ex_material[i] == ['N1']:
                    lcol = linecolor[0]
                elif ex_material[i] == ['R1']:
                    lcol = linecolor[1]
                elif ex_material[i] == ['R2']:
                    lcol = linecolor[2]
                elif ex_material[i] == ['R3']:
                    lcol = linecolor[3]
                if ex_material_nr[i] == [2]:
                    markertype = markers[1]
                    lstyle = styles[1]
                elif ex_material_nr[i] == [3]:
                    markertype = markers[2]
                    lstyle = styles[2]
                elif ex_material_nr[i] == [4]:
                    markertype = markers[3]
                    lstyle = styles[3]

            if ex_nr != 1 and stack == True:                                                                   #stacked bars
                return(print('error: Stacked Bar'))
            else:
                #select[0].plot(ax=ax, x = x_value, y = y_value, kind = 'bar', stacked = True, figsize=(15,10))              

                x = select[0].set_index(x_value)                                                               # make x axis values indexed
                bottom = np.zeros(len(select[0][x_value]))
                width = x.index.values.max() / 50

                if list(y_value) ==  ['all'] or list(y_value) ==  ['standard substances']:
                    for i in range(len(y_value_ca)):                                                           #carbonic acids
                        plt.bar(x.index, x[y_value_ca[i]], width=width, bottom=bottom) 
                        bottom += select[0][y_value_ca[i]].values
                        print(y_value_ca)
                    for i in range(len(y_value_sa)):                                                           #sulfonic acids
                        plt.bar(x.index, x[y_value_sa[i]], width=width, bottom=bottom, hatch='//') 
                        bottom += select[0][y_value_sa[i]].values
                        print(y_value_sa)
                    if list(y_value) ==  ['all']:
                        for i in range(len(y_value_prec)):                                                     #precursors
                            plt.bar(x.index, x[y_value_prec[i]], width=width, bottom=bottom, hatch='\\\\') 
                            bottom += select[0][y_value_prec[i]].values                                           
                        plt.grid()
                        ax.legend(y_value_ca + y_value_sa + y_value_prec, handlelength=4, loc ='best')
                    else:
                        plt.grid()
                        ax.legend(y_value_ca + y_value_sa, handlelength=4, loc ='best', ncol=2)
                else:
                    y = []
                    for i in range(len(y_value)):
                        #plt.bar(x.index, x[y_value[i]], width=width, bottom=bottom) 
                        #bottom += select[0][y_value[i]].values
                        y.append(x[y_value[i]])
                    plt.stackplot(x.index, y)     #stacked area plot
                    plt.grid()    
                    ax.legend(y_value, loc ='best')
        
    elif bar == True:
        substance = list(y_value)
        cut = 0
        if 'cum_m_sm_' in str(substance):                                                                    #for labelling
            cut = 9
        elif 'cum_m_' in str(substance):
            cut = 6
        elif 'm_' in str(substance):
            cut = 2
        for i in range(0,len(substance)):
            substance[i] = substance[i][cut:]
        
        
        if bar_PFAS == True:                                                                                          #bar plots
            df = pd.DataFrame(0.00, index=substance, columns=ex)
            for i in range(ex_nr):
                for j in range(len(y_value)):
                    result = max(select[i][y_value[j]].values)
                    df[ex[i]][substance[j]] = result
            ax = df.plot.bar(rot=45, subplots=False, ylim=(0,df.max().max()+0.1*df.max().max()), figsize=(20,10))
            print(df)
        elif bar_exp == True:
            df = pd.DataFrame(0.00, index=ex, columns=substance)
            for i in range(ex_nr):
                for j in range(len(y_value)):
                    result = max(select[i][y_value[j]].values)
                    df[substance[j]][ex[i]] = result
            ax = df.plot.bar(rot=0, subplots=False, ylim=(0,df.max().max()+0.1*df.max().max()), figsize=(20,10))
            print(df)
    else:
        return(print('error: choose bar graphs. If line graphs wanted use other function.'))


    if x_value == "Date":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        ax.set_xlabel('Sample Date')
        
    if second_xaxis == True and x2_value != x_value and ex_type != 'IS' and ex_nr == 1:
        
        if ( x_value == headers[0] and x2_value == headers[4]) or ( x_value == headers[1] and x2_value == headers[3]) or ( x2_value == headers[0] and x_value == headers[4]) or ( x2_value == headers[1] and x_value == headers[3]):
            print('error: Sampling Time')
        
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
        
        if x2_value == 'field_date': 
            ax2.set_xticks(select[0][x2_value])
            ax2.xaxis.set_major_locator(mdates.YearLocator(5)) if ex_type == 'SC' else None        
            ax2.xaxis.set_major_locator(mdates.DayLocator(5)) if ex_type == 'LY' else None
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        else:         
            ax2.set_xticks(select[0][x_value].iloc[1::3])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])            
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::3], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
        ax2.set_xlabel(label(x2_value))        
        
    elif second_xaxis == True and ex_nr > 1:
        if x2_value == 'field_time_mean (Y)' or x2_value == 'field_time (Y)':
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
            ax2.set_xticks(select[0][x_value].iloc[1::3])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::3], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
            ax2.set_xlabel(label(x2_value))        
    
    if second_yaxis == True and  y2_value != y_value[0] and ex_type != 'IS' and ex_nr == 1:             
        ay = ax.twinx()
        select[i].plot( ax=ay, x = x2_value, y = y2_value, style='--', marker='x', color = 'k') 
        ay.legend(loc='best')
        ay.set_yscale('log') if Log == True  else None 
        ay.set_ylim(ymin=0) if Log != True  else None
        ay.set_ylabel(label(y2_value))
    elif second_yaxis == True:
        print('Error: Second Y Axis')  
    
    if ( (list(y_value) == ['m_∑PFAS']) or + (list(y_value) == ['cum_m_∑PFAS']) or + (list(y_value) == ['cum_m_sm_∑PFAS']) ):
        ax.legend(['∑PFAS'])
        
    #ax.get_legend().set_bbox_to_anchor((-0.07,1))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='out', top=False)
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='out', right=False)
    ax.yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='out', right=False)
    #ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax.minorticks_on() if bar != True  else None    
    ax.set_yscale('log') if Log == True  else None    
    if x_value != "Date":
        if bar != True:
            ax.set_xlabel(label(x_value)) 
            ax.set_xlim(xmin=0)
    if second_yaxis == True:
        ax.legend([label(y_value[0])],loc='best', bbox_to_anchor=(1, 0.48))
        ay.legend([label(y2_value)],loc='best', bbox_to_anchor=(1, 0.52))
    ax.set_ylabel('c [ng/L]') if bar != True else None     
    #ax.set_ylabel('m [\u03BCg]') if ( (list(y_value) == ['cum_m_∑PFAS'] ) or + (list(y_value) == ['m_∑PFAS']) ) else None 
    if y_value[0].startswith('m_'):
        ax.set_ylabel('m [ng]')
    elif y_value[0].startswith('cum_m_sm_'):
        ax.set_ylabel('m [\u03BCg/kg]')
    elif y_value[0].startswith('cum_m_'):
        ax.set_ylabel('m [\u03BCg]') 
    ax.set_ylabel('QS [-]') if  (list(y_value) == ['Quotientensumme'])  else None
    if Log != True:
        if bar != True:
            ax.set_ylim(ymin=0)
    if list(y_value) == ['QS']:
        ax.set_ylim(ymax=10)
        ax.set_ylabel('QS [-]')
        ax.hlines(1, 0, 1000, colors='r', linestyles='dotted')
   
    ax = plt.gca()    
    plt.gcf().autofmt_xdate() if x_value == 'Date' else None
    #plt.legend(ncol=1)
    plt.show()
    
    if save == True:
        fig = ax.get_figure()
        #fig.savefig("output/{}{}{}{}.png".format(file_string, ex, y_value, x_value), fontsize = 15)
        fig.savefig("output/{}.png".format(file_string), fontsize = 15)


    return ax

def visualization_TOP_bars(x_value, x2_value, y_value, y2_value, second_xaxis, second_yaxis, Log, aof, stack, molar, save, select, ex, ex_type, ex_material, ex_material_nr, file_string):
    
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'legend.fontsize': 30})
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["legend.frameon"]
    #plt.rcParams.update({'legend.handlelength': 2})
    
    ex_nr = len(ex)
    linecolor = ('b','darkorange','g','r','dimgrey')
    markers = ('o','^','v','X')
    styles = ('-','--',':','-.')
    
    if ex_nr > 1:
        labels = ['t [d]','t [d]','V [L]','WF [L/kg]','WF [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']
    else:
        labels = ['c [ng/L]','t [Tage]','t [Tage]','V [L]','WF [L/kg]','WF [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['∑PFAS','Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']
    


    def label(xaxis):
        
        labelseries =  pd.Series(headers, index=labels) 
        label = list(labelseries[labelseries==xaxis].index)[0]
        
        return label

    
    fig, ax = plt.subplots(figsize=(20,13))   
    
    
    if stack == False:
        if list(y_value) ==  ['F(TOP)']:
            for i in range(ex_nr): 
                y_value = list(y_value)

                lcol = linecolor[4]
                markertype = markers[0]
                lstyle = styles[0]
                if ex_material[i] == ['N1']:
                    lcol = linecolor[0]
                elif ex_material[i] == ['R1']:
                    lcol = linecolor[1]
                elif ex_material[i] == ['R2']:
                    lcol = linecolor[2]
                elif ex_material[i] == ['R3']:
                    lcol = linecolor[3]
                if ex_material_nr[i] == [2]:
                    markertype = markers[1]
                    lstyle = styles[1]
                elif ex_material_nr[i] == [3]:
                    markertype = markers[2]
                    lstyle = styles[2]
                elif ex_material_nr[i] == [4]:
                    markertype = markers[3]
                    lstyle = styles[3]
                
                x = select[i].set_index(x_value)
                plt.plot(x.index, x[y_value[0]], marker = markertype, ms = 10, color = lcol, linewidth = 0)
            plt.legend(ex, loc ='best')
        elif aof == True:
            if ex_nr == 1:
                print(y_value)
                print(x_value)
                x = select[0].set_index(x_value)
                print(x.index)
                width = 0.05
                plt.bar(x.index, x[y_value[1]], width=width, color='gray', edgecolor='black', ecolor='black', label = '∑PFAS') 
                plt.plot(x.index, x[y_value[0]], marker = 'o', ms = 15, color = 'r', linewidth = 0, label = 'AOF')
                plt.legend(loc ='best')
        else:
            for i in range(ex_nr):

                y_value = list(y_value)

                lcol = linecolor[4]
                markertype = markers[0]
                lstyle = styles[0]
                if ex_material[i] == ['N1']:
                    lcol = linecolor[0]
                elif ex_material[i] == ['R1']:
                    lcol = linecolor[1]
                elif ex_material[i] == ['R2']:
                    lcol = linecolor[2]
                elif ex_material[i] == ['R3']:
                    lcol = linecolor[3]
                if ex_material_nr[i] == [2]:
                    markertype = markers[1]
                    lstyle = styles[1]
                elif ex_material_nr[i] == [3]:
                    markertype = markers[2]
                    lstyle = styles[2]
                elif ex_material_nr[i] == [4]:
                    markertype = markers[3]
                    lstyle = styles[3]

                #select[i].plot(ax=ax, x = x_value, y = y_value, kind = 'bar', stacked = True, figsize=(15,10))              

                x = select[i].set_index(x_value)                                                               # make x axis values indexed
                bottom = np.zeros(len(select[i][x_value]))
                width = 0.3

                for j in range(len(y_value)):
                    plt.bar(x.index, x[y_value[j]], width=width, bottom=bottom) 
                    bottom += select[i][y_value[j]].values
                plt.grid() 
                plt.legend(ex, loc ='best')
            
        
    else:
        if molar == True: 
            legend = ["vor TOP","Zugewinn durch TOP"]
            for i in range(ex_nr): 
                x = select[i].set_index(x_value)
                bottom = np.zeros(len(select[i][x_value]))
                width = 0.03
                plt.bar(x.index, x[y_value[0]], width=width, bottom=bottom, color=['black'])
                bottom += select[i][y_value[0]].values
                plt.bar(x.index, x[y_value[1]], width=width, bottom=bottom, color=['darkorange'])
                bottom += select[i][y_value[1]].values
                ax.legend(legend, loc ='best')
                print(y_value)
        else:
            legend = ["∑PFAS vor TOP","F-Zugewinn"]   #[y_value[1], y_value[0]]    #bottom: F before TOP, top: F gain in TOP
            for i in range(ex_nr):      
                x = select[i].set_index(x_value)
                bottom = np.zeros(len(select[i][x_value]))
                width = 0.3
                plt.bar(x.index, x[y_value[1]], width=width, bottom=bottom, color=['black'])
                bottom += select[i][y_value[1]].values
                plt.bar(x.index, x[y_value[0]], width=width, bottom=bottom, color=['darkorange'])
                bottom += select[i][y_value[0]].values
                ax.legend(legend, loc ='best')
                print(y_value)
                #plt.grid()


    if x_value == "Date":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        ax.set_xlabel('Sample Date')
        
    if second_xaxis == True and x2_value != x_value and ex_type != 'IS' and ex_nr == 1:
        
        if ( x_value == headers[0] and x2_value == headers[4]) or ( x_value == headers[1] and x2_value == headers[3]) or ( x2_value == headers[0] and x_value == headers[4]) or ( x2_value == headers[1] and x_value == headers[3]):
            print('error: Sampling Time')
        
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
        
        if x2_value == 'field_date': 
            ax2.set_xticks(select[0][x2_value])
            ax2.xaxis.set_major_locator(mdates.YearLocator(5)) if ex_type == 'SC' else None        
            ax2.xaxis.set_major_locator(mdates.DayLocator(5)) if ex_type == 'LY' else None
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        else:         
            ax2.set_xticks(select[0][x_value].iloc[1::3])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])            
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::3], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
        ax2.set_xlabel(label(x2_value))        
        
    elif second_xaxis == True and ex_nr > 1:
        if x2_value == 'field_time_mean (Y)' or x2_value == 'field_time (Y)':
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
            ax2.set_xticks(select[0][x_value].iloc[1::3])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::3], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
            ax2.set_xlabel(label(x2_value))        
    
    if second_yaxis == True and  y2_value != y_value[0] and ex_type != 'IS' and ex_nr == 1:             
        ay = ax.twinx()
        select[i].plot( ax=ay, x = x2_value, y = y2_value, style='--', marker='x', color = 'k') 
        ay.legend(loc='best')
        ay.set_yscale('log') if Log == True  else None 
        ay.set_ylim(ymin=0) if Log != True  else None
        ay.set_ylabel(label(y2_value))
    elif second_yaxis == True:
        print('Error: Second Y Axis')  
    
    if ( (list(y_value) == ['m_∑PFAS']) or + (list(y_value) == ['cum_m_∑PFAS']) or + (list(y_value) == ['cum_m_sm_∑PFAS']) ):
        ax.legend(['∑PFAS'])
        
    #ax.get_legend().set_bbox_to_anchor((-0.07,1))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='out', top=False)
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='out', right=False)
    ax.yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='out', right=False)
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    #ax.minorticks_on()
    ax.set_yscale('log') if Log == True  else None    
    if x_value != "Date":
        ax.set_xlabel(label(x_value)) 
        ax.set_xlim(xmin=0)
    if second_yaxis == True:
        ax.legend([label(y_value[0])],loc='best', bbox_to_anchor=(1, 0.48))
        ay.legend([label(y2_value)],loc='best', bbox_to_anchor=(1, 0.52))
    if molar == True:
        ax.set_ylabel('[nmol/L]')
    elif list(y_value) ==  ['F(TOP)']:
        ax.set_ylabel('F$_{TOP}$ [-]')
    else:
        ax.set_ylabel('c [ng F/L]')  
    #ax.set_ylabel('m [\u03BCg]') if ( (list(y_value) == ['cum_m_∑PFAS'] ) or + (list(y_value) == ['m_∑PFAS']) ) else None 
    if y_value[0].startswith('m_'):
        ax.set_ylabel('m [ng]')
    elif y_value[0].startswith('cum_m_sm_'):
        ax.set_ylabel('m [\u03BCg/kg]')
    elif y_value[0].startswith('cum_m_'):
        ax.set_ylabel('m [\u03BCg]') 
    ax.set_ylabel('QS [-]') if  (list(y_value) == ['Quotientensumme'])  else None
    if Log != True:
        ax.set_ylim(ymin=0)
    if list(y_value) == ['QS']:
        ax.set_ylim(ymax=10)
        ax.set_ylabel('QS [-]')
        ax.hlines(1, 0, 1000, colors='r', linestyles='dotted')
   

    ax.set_ylim(ymin=0)
    #ax.set_ylim(ymax=14000)
    ax = plt.gca()    
    plt.gcf().autofmt_xdate() if x_value == 'Date' else None
    #plt.legend(ncol=1)
    plt.show()
    
    if save == True:
        fig = ax.get_figure()
        #fig.savefig("output/{}{}{}{}.png".format(file_string, ex, y_value, x_value), fontsize = 15)
        fig.savefig("output/{}.png".format(file_string), fontsize = 15)


    return ax

def visualization_bars_IS(x_value, x2_value, y_value, y2_value, second_xaxis, second_yaxis, Log, stack, bar_PFAS, bar_exp, save, select, select_w, ex, ex_type, ex_material, ex_material_nr, file_string):
    
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'legend.fontsize': 30})
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["legend.frameon"]
    #plt.rcParams.update({'legend.handlelength': 2})
    
    ex_nr = len(ex)
    linecolor = ('b','darkorange','g','r','dimgrey')
    markers = ('o','^','v','X')
    styles = ('-','--',':','-.')
    
    if ex_nr > 1:
        labels = ['t [d]','t [d]','V [L]','WS [L/kg]','WS [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']
    else:
        labels = ['c [ng/L]','t [d]','t [d]','V [L]','WS [L/kg]','WS [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['∑PFAS','Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']
    
    
    bar = False
    if bar_PFAS == True or bar_exp == True:
        bar = True


    def label(xaxis):
        
        labelseries =  pd.Series(headers, index=labels) 
        label = list(labelseries[labelseries==xaxis].index)[0]
        
        return label

    if ex_nr > 1 and (len(y_value) > 1 or list(y_value) ==  ['all'] ) and bar != True:
        return (print('only single PFAS for multiple experiment'))
    
    fig, ax = plt.subplots(figsize=(20,10))   
    
    
    if stack == True:
        for i in range(ex_nr):      

            if list(y_value) ==  ['all'] or list(y_value) ==  ['standard substances']:                        #for different colouring depending on PFAS substance classes
                y_value_ca = list(select[i].columns)[7:17]
                y_value_sa = list(select[i].columns)[18:24]
                if list(y_value) ==  ['all']:
                    y_value_prec = list(select[i].columns)[25:34]
                
            else:
                y_value = list(y_value)
                
                lcol = linecolor[4]
                markertype = markers[0]
                lstyle = styles[0]
                if ex_material[i] == ['N1']:
                    lcol = linecolor[0]
                elif ex_material[i] == ['R1']:
                    lcol = linecolor[1]
                elif ex_material[i] == ['R2']:
                    lcol = linecolor[2]
                elif ex_material[i] == ['R3']:
                    lcol = linecolor[3]
                if ex_material_nr[i] == [2]:
                    markertype = markers[1]
                    lstyle = styles[1]
                elif ex_material_nr[i] == [3]:
                    markertype = markers[2]
                    lstyle = styles[2]
                elif ex_material_nr[i] == [4]:
                    markertype = markers[3]
                    lstyle = styles[3]

            if ex_nr != 1 and stack == True:                                                                   #stacked bars
                return(print('error: Stacked Bar'))
            else:
                #select[0].plot(ax=ax, x = x_value, y = y_value, kind = 'bar', stacked = True, figsize=(15,10))              

                x = select[0].set_index(x_value)                                                               # make x axis values indexed
                x_w = select_w[0].set_index(x_value)
                bottom = np.zeros(len(select[0][x_value]))
                width = x_w.index.values.max() / 50

                if list(y_value) ==  ['all'] or list(y_value) ==  ['standard substances']:
                    for i in range(len(y_value_ca)):                                                           #carbonic acids
                        plt.bar(x.index, x[y_value_ca[i]], width=width, bottom=bottom) 
                        bottom += select[0][y_value_ca[i]].values
                        print(y_value_ca)
                    for i in range(len(y_value_sa)):                                                           #sulfonic acids
                        plt.bar(x.index, x[y_value_sa[i]], width=width, bottom=bottom, hatch='//') 
                        bottom += select[0][y_value_sa[i]].values
                        print(y_value_sa)
                    if list(y_value) ==  ['all']:
                        for i in range(len(y_value_prec)):                                                     #precursors
                            plt.bar(x.index, x[y_value_prec[i]], width=width, bottom=bottom, hatch='\\\\') 
                            bottom += select[0][y_value_prec[i]].values                                           
                        plt.grid()
                        ax.legend(y_value_ca + y_value_sa + y_value_prec, handlelength=4, loc ='best')
                    else:
                        plt.grid()
                        ax.legend(y_value_ca + y_value_sa, handlelength=4, loc ='best', ncol=2)
                else:
                    for i in range(len(y_value)):
                        plt.bar(x.index[0:3], x[y_value[i]][0:7], color = 'black') 
                        bottom += select[0][y_value[i]].values
                        bottom = bottom[0:3]
                        plt.bar(x_w.index, x_w[y_value[i]], bottom=bottom, color = 'deepskyblue') 
                    #plt.grid()    
                    ax.legend(('N-1','water'), loc ='best')
        
    elif bar == True:
        substance = list(y_value)
        cut = 0
        if 'cum_m_sm_' in str(substance):                                                                    #for labelling
            cut = 9
        elif 'cum_m_' in str(substance):
            cut = 6
        elif 'm_' in str(substance):
            cut = 2
        for i in range(0,len(substance)):
            substance[i] = substance[i][cut:]
        
        
        if bar_PFAS == True:                                                                                          #bar plots
            df = pd.DataFrame(0.00, index=substance, columns=ex)
            for i in range(ex_nr):
                for j in range(len(y_value)):
                    result = max(select[i][y_value[j]].values)
                    df[ex[i]][substance[j]] = result
            ax = df.plot.bar(rot=45, subplots=False, ylim=(0,df.max().max()+0.1*df.max().max()), figsize=(20,10))
            print(df)
        elif bar_exp == True:
            df = pd.DataFrame(0.00, index=ex, columns=substance)
            for i in range(ex_nr):
                for j in range(len(y_value)):
                    result = max(select[i][y_value[j]].values)
                    df[substance[j]][ex[i]] = result
            ax = df.plot.bar(rot=0, subplots=False, ylim=(0,df.max().max()+0.1*df.max().max()), figsize=(20,10))
            print(df)
    else:
        return(print('error: choose bar graphs. If line graphs wanted use other function.'))


    if x_value == "Date":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        ax.set_xlabel('Sample Date')
        
    if second_xaxis == True and x2_value != x_value and ex_type != 'IS' and ex_nr == 1:
        
        if ( x_value == headers[0] and x2_value == headers[4]) or ( x_value == headers[1] and x2_value == headers[3]) or ( x2_value == headers[0] and x_value == headers[4]) or ( x2_value == headers[1] and x_value == headers[3]):
            print('error: Sampling Time')
        
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
        
        if x2_value == 'field_date': 
            ax2.set_xticks(select[0][x2_value])
            ax2.xaxis.set_major_locator(mdates.YearLocator(5)) if ex_type == 'SC' else None        
            ax2.xaxis.set_major_locator(mdates.DayLocator(5)) if ex_type == 'LY' else None
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        else:         
            ax2.set_xticks(select[0][x_value].iloc[1::3])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])            
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::3], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
        ax2.set_xlabel(label(x2_value))        
        
    elif second_xaxis == True and ex_nr > 1:
        if x2_value == 'field_time_mean (Y)' or x2_value == 'field_time (Y)':
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
            ax2.set_xticks(select[0][x_value].iloc[1::3])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::3], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
            ax2.set_xlabel(label(x2_value))        
    
    if second_yaxis == True and  y2_value != y_value[0] and ex_type != 'IS' and ex_nr == 1:             
        ay = ax.twinx()
        select[i].plot( ax=ay, x = x2_value, y = y2_value, style='--', marker='x', color = 'k') 
        ay.legend(loc='best')
        ay.set_yscale('log') if Log == True  else None 
        ay.set_ylim(ymin=0) if Log != True  else None
        ay.set_ylabel(label(y2_value))
    elif second_yaxis == True:
        print('Error: Second Y Axis')  
        
    #ax.get_legend().set_bbox_to_anchor((-0.07,1))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='out', top=False)
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='out', right=False)
    ax.yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='out', right=False)
    #ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax.minorticks_on() if bar != True  else None    
    ax.set_yscale('log') if Log == True  else None    
    if x_value != "Date":
        if bar != True:
            ax.set_xlabel(label(x_value)) 
            ax.set_xlim(xmin=0)
    if second_yaxis == True:
        ax.legend([label(y_value[0])],loc='best', bbox_to_anchor=(1, 0.48))
        ay.legend([label(y2_value)],loc='best', bbox_to_anchor=(1, 0.52))
    ax.set_ylabel('c [ng/L]') if bar != True else None     
    #ax.set_ylabel('m [\u03BCg]') if ( (list(y_value) == ['cum_m_∑PFAS'] ) or + (list(y_value) == ['m_∑PFAS']) ) else None 
    if y_value[0].startswith('m_sm_'):
        ax.set_ylabel('m [\u03BCg/kg]')
    elif y_value[0].startswith('m_'):
        ax.set_ylabel('m [ng]')
    elif y_value[0].startswith('cum_m_sm_'):
        ax.set_ylabel('m [\u03BCg/kg]')
    elif y_value[0].startswith('cum_m_'):
        ax.set_ylabel('m [\u03BCg]') 
    ax.set_ylabel('QS [-]') if  (list(y_value) == ['Quotientensumme'])  else None
    if Log != True:
        if bar != True:
            ax.set_ylim(ymin=0)
    if list(y_value) == ['QS']:
        ax.set_ylim(ymax=10)
        ax.set_ylabel('QS [-]')
        ax.hlines(1, 0, 1000, colors='r', linestyles='dotted')
   
    ax = plt.gca()    
    plt.gcf().autofmt_xdate() if x_value == 'Date' else None
    #plt.legend(ncol=1)
    plt.show()
    
    if save == True:
        fig = ax.get_figure()
        #fig.savefig("output/{}{}{}{}.png".format(file_string, ex, y_value, x_value), fontsize = 15)
        fig.savefig("output/{}.png".format(file_string), fontsize = 15)


    return ax

def visualization_comparison(x_value, x2_value, y_value, y2_value, second_xaxis, second_yaxis, Log, save, select_SC, select_IS, ex, ex_type, ex_material, ex_material_nr, N1_soil, file_string):
    
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'legend.fontsize': 30})
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["legend.frameon"]
    #plt.rcParams.update({'legend.handlelength': 2})
    
    ex_nr = len(ex)
    SC_nr = 0
    IS_nr = 0
    for i in range(len(ex)):
        print(ex_type)
        if ex_type[i] == ['SC']:
            SC_nr = SC_nr+1
        else:
            IS_nr = IS_nr+1
    print(SC_nr)
    print(IS_nr)
    # line properties for different experiments
    linecolor = ('b','darkorange','g','r','dimgrey')
    markers = ('o','^','v','X')
    styles = ('-','--',':','-.')
    
    #color specifications: substances get corresponding color (if single substances are selected)
    subst_color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','cyan','tab:cyan', #ca
'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink',                                           #sa
'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','cyan']             #prec
    subst_markers = ('o','o','o','o','o','o','o','o','o','o','o',                                                            #ca
'x','x','x','x','x','x','x',                                                                                                 #sa
'^','^','^','^','^','^','^','^','^','^')                                                                                     #prec
    substances_all = list(select_SC[0].columns)[7:34]
    
    substances_sel = []
    if y_value[0].startswith('cum_m_sm_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][9:])
    elif y_value[0].startswith('cum_m_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][6:])
    elif y_value[0].startswith('m_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][2:])
    elif list(y_value) ==  ['all']:
        substances_sel = substances_all
    elif list(y_value) ==  ['standard substances']:
        substances_sel = substances_all[0:18]
    else:
        substances_sel = y_value
    
    subst_color_sel = []
    subst_markers_sel = []
    j=0
    for i in range(len(substances_all)):
        if j < len(substances_sel):
            if substances_sel[j] == substances_all[i]:
                subst_color_sel.append(subst_color[i])
                subst_markers_sel.append(subst_markers[i])
                j = j+1
    
    if ex_nr > 1:
        labels = ['t [d]','t [d]','V [L]','WF [L/kg]','WF [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']
    else:
        labels = ['c [ng/L]','t [Tage]','t [Tage]','V [L]','WF [L/kg]','WF [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['∑PFAS','Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']


    def label(xaxis):
        
        labelseries =  pd.Series(headers, index=labels) 
        label = list(labelseries[labelseries==xaxis].index)[0]
        
        return label

    if ex_nr > 1 and (len(y_value) > 1 or list(y_value) ==  ['all'] ):
        return (print('only single PFAS for multiple experiment'))
    
    fig, ax = plt.subplots(figsize=(20,10))   
    
    for i in range(ex_nr):
        
        if ex_nr > 1:
            y_value = list(y_value)
            print(y_value)
            
            lcol = linecolor[4]
            markertype = markers[0]
            lstyle = styles[0]
            if ex_material[i] == ['N1']:
                lcol = linecolor[0]
            elif ex_material[i] == ['R1']:
                lcol = linecolor[1]
            elif ex_material[i] == ['R2']:
                lcol = linecolor[2]
            elif ex_material[i] == ['R3']:
                lcol = linecolor[3]
            if ex_material_nr[i] == [2]:
                markertype = markers[1]
                lstyle = styles[1]
            elif ex_material_nr[i] == [3]:
                markertype = markers[2]
                lstyle = styles[2]
            elif ex_material_nr[i] == [4]:
                markertype = markers[3]
                lstyle = styles[3]
            
            if ex_type[i] == ['SC']:
                select_SC[i].plot( ax=ax, x = x_value, y = y_value, grid = True, figsize=(15,10),                                           
                                style = lstyle,
                                marker = markertype,
                                lw=2,
                                color = lcol
                                #ylim=(0.0,3000),
                                #color = 'k' if len(y_value) == 1 else None
                                )
            else:
                select_IS[i-SC_nr].plot( ax=ax, x = x_value, y = y_value, grid = True, figsize=(15,10),                                           
                                style = lstyle,
                                marker = markertype,
                                lw=2,
                                color = lcol
                                #ylim=(0.0,3000),
                                #color = 'k' if len(y_value) == 1 else None
                                )
        else:
        
            if list(y_value) ==  ['all'] or list(y_value) ==  ['standard substances']:                        #for different colouring depending on PFAS substance classes
                y_value_ca = list(select[i].columns)[7:17]
                y_value_sa = list(select[i].columns)[18:24]
                if list(y_value) ==  ['all']:
                    y_value_prec = list(select[i].columns)[25:34]
                #plotting each substance class
                select[i].plot( ax=ax, x = x_value, y = y_value_ca, grid = True, figsize=(15,10),                                           
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )

                select[i].plot( ax=ax, x = x_value, y = y_value_sa, grid = True, figsize=(15,10),                                           
                                style='-',
                                marker='x',
                                markersize='10',
                                lw=2,
                                #ylim=(0.0,3000),
                                )       
                if list(y_value) ==  ['all']:
                    select[i].plot( ax=ax, x = x_value, y = y_value_prec, grid = True, figsize=(15,10),
                                style='--',
                                marker='^',
                                #markersize='10',
                                lw=2,
                                #ylim=(0.0,3000),
                                )
                
            else:
                y_value = list(y_value)
                
                select[i].plot( ax=ax, x = x_value, y = y_value, grid = True, figsize=(15,10),                                           
                            style = '-',
                            marker = 'o',   #subst_markers_sel[0],
                            lw=2,
                            color = subst_color_sel
                            #ylim=(0.0,3000),
                            #color = 'k' if len(y_value) == 1 else None
                            )
            
            ax.legend(substances_sel, handlelength=4)


        
        

    if x_value == "Date":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        ax.set_xlabel('Sample Date')
        
    if second_xaxis == True and x2_value != x_value and ex_type != 'IS' and ex_nr == 1:
        
        if ( x_value == headers[0] and x2_value == headers[4]) or ( x_value == headers[1] and x2_value == headers[3]) or ( x2_value == headers[0] and x_value == headers[4]) or ( x2_value == headers[1] and x_value == headers[3]):
            print('error: Sampling Time')
        
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
        
        if x2_value == 'field_date': 
            ax2.set_xticks(select[0][x2_value])
            ax2.xaxis.set_major_locator(mdates.YearLocator(5)) if ex_type == 'SC' else None        
            ax2.xaxis.set_major_locator(mdates.DayLocator(5)) if ex_type == 'LY' else None
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        else:         
            ax2.set_xticks(select[0][x_value].iloc[1::3])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])            
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::3], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
        ax2.set_xlabel(label(x2_value))        
        
    elif second_xaxis == True and ex_nr > 1:
        if x2_value == 'field_time_mean (Y)' or x2_value == 'field_time (Y)':
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
            ax2.set_xticks(select[0][x_value].iloc[1::4])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::4], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
            ax2.set_xlabel(label(x2_value))        
    
    if second_yaxis == True and  y2_value != y_value[0] and ex_type != 'IS' and ex_nr == 1:             
        ay = ax.twinx()
        select[i].plot( ax=ay, x = x2_value, y = y2_value, style='--', marker='x', color = 'k') 
        ay.legend(loc='best')
        ay.set_yscale('log') if Log == True  else None 
        ay.set_ylim(ymin=0) if Log != True  else None
        ay.set_ylabel(label(y2_value))
    elif second_yaxis == True:
        print('Error: Second Y Axis')  
    
    if ex_nr > 1: #and bar != True
        ax.legend(ex[:])
    elif ( (list(y_value) == ['m_∑PFAS']) or + (list(y_value) == ['cum_m_∑PFAS']) or + (list(y_value) == ['cum_m_sm_∑PFAS']) ):
        ax.legend(['∑PFAS'])
        
    #ax.get_legend().set_bbox_to_anchor((-0.07,1))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='out', top=False)
    ax.xaxis.set_tick_params(which='minor', size=5, width=1.5, direction='out', top=False)
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='out', right=False)
    ax.yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='out', right=False)
    #ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    #ax.minorticks_on()
    ax.set_yscale('log') if Log == True  else None    
    if x_value != "Date":
        ax.set_xlabel(label(x_value)) 
        ax.set_xlim(xmin=0)
    if second_yaxis == True:
        ax.legend([label(y_value[0])],loc='best', bbox_to_anchor=(1, 0.48))
        ay.legend([label(y2_value)],loc='best', bbox_to_anchor=(1, 0.52))
    ax.set_ylabel('c [ng/L]')
    #ax.set_ylabel('m [\u03BCg]') if ( (list(y_value) == ['cum_m_∑PFAS'] ) or + (list(y_value) == ['m_∑PFAS']) ) else None 
    if y_value[0].startswith('m_'):
        ax.set_ylabel('m [ng]')
    elif y_value[0].startswith('cum_m_sm_'):
        ax.set_ylabel('m [\u03BCg/kg]')
    elif y_value[0].startswith('cum_m_'):
        ax.set_ylabel('m [\u03BCg]') 
    ax.set_ylabel('QS [-]') if  (list(y_value) == ['Quotientensumme'])  else None
    if Log != True:
        ax.set_ylim(ymin=0)
    if list(y_value) == ['QS']:
        ax.set_ylim(ymax=10)
        ax.set_ylabel('QS [-]')
        ax.hlines(1, 0, 1000, colors='r', linestyles='dotted')
    if y_value[0].startswith('cum_m_sm_'):
        if ex_nr > 1:
            subst_color_sel = ['black']
        for i in range(len(substances_sel)):
            ax.hlines(N1_soil[substances_sel[i]], 0, 1000, colors=subst_color_sel[i], linestyles='dashed')
   
    ax = plt.gca()    
    plt.gcf().autofmt_xdate() if x_value == 'Date' else None
    #plt.legend(ncol=1)
    plt.show()
    
    if save == True:
        fig = ax.get_figure()
        #fig.savefig("output/{}{}{}{}.png".format(file_string, ex, y_value, x_value), fontsize = 15)
        fig.savefig("output/{}.png".format(file_string), fontsize = 15)


    return ax

def visualization_comparison_all(x_value, x2_value, y_value, y2_value, second_xaxis, second_yaxis, Log, save, select, ex, ex_type, ex_material, ex_material_nr, N1_soil, file_string):
    
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'legend.fontsize': 30})
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["legend.frameon"]
    #plt.rcParams.update({'legend.handlelength': 2})
    
    ex_nr = len(ex)
    # line properties for different experiments
    linecolor = ('black','black','black','black','black')  #('b','darkorange','g','r','dimgrey') #
    markers = ('o','^','v','X')
    styles = ('-',':','--')      # IS:    '-'    columns: ':'      lysimeters: '--'
    
    lcol = []                         #
    markertype = []
    lstyle = []
    
    for i in range(ex_nr):
        if ex_material[i] == ['N1']:
            lcol.append(linecolor[0])
        elif ex_material[i] == ['R1']:
            lcol.append(linecolor[1])
        elif ex_material[i] == ['R2']:
            lcol.append(linecolor[2])
        elif ex_material[i] == ['R3']:
            lcol.append(linecolor[3])
        else:
            lcol.append(linecolor[4])
        
        if ex_material_nr[i] == [1]:
            markertype.append(markers[0])
        elif ex_material_nr[i] == [2]:
            markertype.append(markers[1])
        elif ex_material_nr[i] == [3]:
            markertype.append(markers[2])
        elif ex_material_nr[i] == [4]:
            markertype.append(markers[3])
        
        if ex_type[i] == ['IS']:
            lstyle.append(styles[0])
            #lcol.append(linecolor[3])
        elif ex_type[i] == ['SC']:
            lstyle.append(styles[1])
            #lcol.append(linecolor[0])
        else:
            lstyle.append(styles[2])
            #lcol.append(linecolor[2])
    
    #color specifications: substances get corresponding color (if single substances are selected)
    subst_color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','cyan','tab:cyan', #ca
'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink',                                           #sa
'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','cyan']             #prec
    subst_markers = ('o','o','o','o','o','o','o','o','o','o','o',                                                            #ca
'x','x','x','x','x','x','x',                                                                                                 #sa
'^','^','^','^','^','^','^','^','^','^')                                                                                     #prec
    substances_all = list(select[0].columns)[7:35]  #list(select[0].columns)[7:34]
    
    substances_sel = []
    if y_value[0].startswith('cum_m_sm_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][9:])
    elif y_value[0].startswith('cum_m_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][6:])
    elif y_value[0].startswith('m_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][2:])
    elif list(y_value) ==  ['all']:
        substances_sel = substances_all
    elif list(y_value) ==  ['standard substances']:
        substances_sel = substances_all[0:18]
    elif list(y_value) ==  ['special selection']:
        substances_sel = [substances_all[0],substances_all[4],substances_all[6],substances_all[15],substances_all[27]]
    else:
        substances_sel = y_value
    
    subst_color_sel = []
    subst_markers_sel = []
    j=0
    for i in range(len(substances_all)):
        if j < len(substances_sel):
            if substances_sel[j] == substances_all[i]:
                subst_color_sel.append(subst_color[i])
                subst_markers_sel.append(subst_markers[i])
                j = j+1

    
    if ex_nr > 1:
        labels = ['t [d]','t [d]','V [L]','WS [L/kg]','WS [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']
    else:
        labels = ['c [ng/L]','t [Tage]','t [Tage]','V [L]','WS [L/kg]','WS [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['∑PFAS','Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']


    def label(xaxis):
        
        labelseries =  pd.Series(headers, index=labels) 
        label = list(labelseries[labelseries==xaxis].index)[0]
        
        return label

    #if ex_nr > 1 and (len(y_value) > 1 or list(y_value) ==  ['all'] ):
    #    return (print('only single PFAS for multiple experiment'))
    
    fig, ax = plt.subplots(figsize=(20,10))   
    
    for i in range(ex_nr):
        
        if ex_nr > 1:
            if len(substances_sel) == 1:
                y_value = list(y_value)

                select[i].plot( ax=ax, x = x_value, y = y_value, grid = True, figsize=(15,10),                                           
                                style = lstyle[i],
                                marker = markertype[i],
                                lw=2,
                                color = lcol[i]
                                #ylim=(0.0,3000),
                                #color = 'k' if len(y_value) == 1 else None
                                )
            else:                               #several experiments and several substances
                y_value = list(y_value)
                for j in range(len(y_value)):
                    y_value_j = y_value[j]
                    select[i].plot( ax=ax, x = x_value, y = y_value_j, grid = True, figsize=(15,10),                                           
                                style = lstyle[i],
                                marker = markertype[i],
                                lw=2,
                                color = linecolor[j]
                                #ylim=(0.0,3000),
                                #color = 'k' if len(y_value) == 1 else None
                                )
                ax.legend(substances_sel[:])
        else:
        
            if list(y_value) ==  ['all'] or list(y_value) ==  ['standard substances']:                        #for different colouring depending on PFAS substance classes
                y_value_ca = list(select[i].columns)[7:17]
                y_value_sa = list(select[i].columns)[18:24]
                if list(y_value) ==  ['all']:
                    y_value_prec = list(select[i].columns)[25:34]
                #plotting each substance class
                select[i].plot( ax=ax, x = x_value, y = y_value_ca, grid = True, figsize=(15,10),                                           
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )

                select[i].plot( ax=ax, x = x_value, y = y_value_sa, grid = True, figsize=(15,10),                                           
                                style='-',
                                marker='x',
                                markersize='10',
                                lw=2,
                                #ylim=(0.0,3000),
                                )       
                ax.legend(y_value_ca + y_value_sa, handlelength=4)
                if list(y_value) ==  ['all']:
                    select[i].plot( ax=ax, x = x_value, y = y_value_prec, grid = True, figsize=(15,10),
                                style='--',
                                marker='^',
                                #markersize='10',
                                lw=2,
                                #ylim=(0.0,3000),
                                )
                    ax.legend(y_value_ca + y_value_sa + y_value_prec, handlelength=4)
            elif list(y_value) ==  ['special selection']:                        #PFBA, PFOA, PFDA, PFOS, ∑PFAS
                y_value = ['PFBA','PFOA','PFDA','PFOS','∑PFAS']
                select[i].plot( ax=ax, x = x_value, y = ['PFBA','PFOA','PFDA','PFOS'], grid = True, figsize=(15,10),
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )
                select[i].plot( ax=ax, x = x_value, y = ['∑PFAS'], grid = True, figsize=(15,10),
                                color = 'black',
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )

                ax.legend(substances_sel, handlelength=4)
            
            elif list(y_value) ==  ['cum_m_sm_special selection']:                        #PFBA, PFOA, PFDA, PFOS, ∑PFAS
                y_value = ['cum_m_sm_PFBA','cum_m_sm_PFOA','cum_m_sm_PFDA','cum_m_sm_PFOS','cum_m_sm_∑PFAS']
                select[i].plot( ax=ax, x = x_value, y = ['cum_m_sm_PFBA','cum_m_sm_PFOA','cum_m_sm_PFDA','cum_m_sm_PFOS'], grid = True, figsize=(15,10),
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )
                select[i].plot( ax=ax, x = x_value, y = 'cum_m_sm_∑PFAS', grid = True, figsize=(15,10),
                                color = 'black',
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )

                substances_sel = ['PFBA','PFOA','PFDA','PFOS','∑PFAS']
                ax.legend(substances_sel, handlelength=4)
            else:
                y_value = list(y_value)
                
                select[i].plot( ax=ax, x = x_value, y = y_value, grid = True, figsize=(15,10),
                            style = '-',
                            marker = 'o',   #subst_markers_sel[0],
                            lw=2,
                            color = subst_color_sel
                            #ylim=(0.0,3000),
                            #color = 'k' if len(y_value) == 1 else None
                            )
                ax.legend(substances_sel, handlelength=4)


        
        

    if x_value == "Date":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        ax.set_xlabel('Sample Date')
        
    if second_xaxis == True and x2_value != x_value and ex_type != 'IS' and ex_nr == 1:
        
        if ( x_value == headers[0] and x2_value == headers[4]) or ( x_value == headers[1] and x2_value == headers[3]) or ( x2_value == headers[0] and x_value == headers[4]) or ( x2_value == headers[1] and x_value == headers[3]):
            print('error: Sampling Time')
        
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
        
        if x2_value == 'field_date': 
            ax2.set_xticks(select[0][x2_value])
            ax2.xaxis.set_major_locator(mdates.YearLocator(5)) if ex_type == 'SC' else None        
            ax2.xaxis.set_major_locator(mdates.DayLocator(5)) if ex_type == 'LY' else None
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        else:         
            ax2.set_xticks(select[0][x_value].iloc[1::3])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])            
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::3], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
        ax2.set_xlabel(label(x2_value))        
        
    elif second_xaxis == True and ex_nr > 1:
        if x2_value == 'field_time_mean (Y)' or x2_value == 'field_time (Y)':
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
            ax2.set_xticks(select[0][x_value].iloc[1::4])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::4], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
            ax2.set_xlabel(label(x2_value))        
    
    if second_yaxis == True and  y2_value != y_value[0] and ex_type != 'IS' and ex_nr == 1:             
        ay = ax.twinx()
        select[i].plot( ax=ay, x = x2_value, y = y2_value, style='--', marker='x', color = 'k') 
        ay.legend(loc='best')
        ay.set_yscale('log') if Log == True  else None 
        ay.set_ylim(ymin=0) if Log != True  else None
        ay.set_ylabel(label(y2_value))
    elif second_yaxis == True:
        print('Error: Second Y Axis')  
    
    if len(substances_sel) == 1:
        if ex_nr > 1: #and bar != True
            ax.legend(ex[:])
        elif ( (list(y_value) == ['m_∑PFAS']) or + (list(y_value) == ['cum_m_∑PFAS']) or + (list(y_value) == ['cum_m_sm_∑PFAS']) ):
            ax.legend(['∑PFAS'])
        
    #ax.get_legend().set_bbox_to_anchor((-0.07,1))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='out', top=False)
    ax.xaxis.set_tick_params(which='minor', size=5, width=1.5, direction='out', top=False)
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='out', right=False)
    ax.yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='out', right=False)
    #ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    #ax.minorticks_on()
    ax.set_yscale('log') if Log == True  else None    
    if x_value != "Date":
        ax.set_xlabel(label(x_value)) 
        ax.set_xlim(xmin=0)
    if second_yaxis == True:
        ax.legend([label(y_value[0])],loc='best', bbox_to_anchor=(1, 0.48))
        ay.legend([label(y2_value)],loc='best', bbox_to_anchor=(1, 0.52))
    ax.set_ylabel('c [ng/L]')
    #ax.set_ylabel('m [\u03BCg]') if ( (list(y_value) == ['cum_m_∑PFAS'] ) or + (list(y_value) == ['m_∑PFAS']) ) else None 
    if y_value[0].startswith('m_'):
        ax.set_ylabel('m [ng]')
    elif y_value[0].startswith('cum_m_sm_'):
        ax.set_ylabel('m [\u03BCg/kg]')
    elif y_value[0].startswith('cum_m_'):
        ax.set_ylabel('m [\u03BCg]') 
    ax.set_ylabel('QS [-]') if  (list(y_value) == ['Quotientensumme'])  else None
    if Log != True:
        ax.set_ylim(ymin=0)
    if list(y_value) == ['QS']:
        ax.set_ylim(ymax=10)
        ax.set_ylabel('QS [-]')
        ax.hlines(1, 0, 1000, colors='r', linestyles='dotted')
    if y_value[0].startswith('cum_m_sm_'):
        if ex_nr > 1:
            subst_color_sel = ['black']
        #for i in range(len(substances_sel)):
            #ax.hlines(N1_soil[substances_sel[i]], 0, 1000, colors=subst_color_sel[i], linestyles='dashed')
   
    #ax.set_xlim(xmax=10)
    #ax.set_ylim(ymax=150000)
    ax = plt.gca()    
    plt.gcf().autofmt_xdate() if x_value == 'Date' else None
    #plt.legend(ncol=1)
    plt.show()
    
    if save == True:
        fig = ax.get_figure()
        #fig.savefig("output/{}{}{}{}.png".format(file_string, ex, y_value, x_value), fontsize = 15)
        fig.savefig("output/{}.png".format(file_string), fontsize = 15)


    return ax

def visualization_comparison_SC_LY(x_value, x2_value, y_value, y2_value, second_xaxis, second_yaxis, Log, micro, save, select, ex, ex_type, ex_material, ex_material_nr, N1_soil, file_string):
    
    plt.rcParams.update({'font.size': 25})
    plt.rcParams.update({'legend.fontsize': 30})
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams["legend.frameon"]
    #plt.rcParams.update({'legend.handlelength': 2})
    
    ex_nr = len(ex)
    # line properties for different experiments
    linecolor = ('b','darkorange','g','r','dimgrey')
    markers = ('o','^','v','X')
    styles = (':','-')      #columns: ':'      lysimeters: '-'
    
    lcol = []                         #
    markertype = []
    lstyle = []
    
    for i in range(ex_nr):
        if ex_material[i] == ['N1']:
            lcol.append(linecolor[0])
        elif ex_material[i] == ['R1']:
            lcol.append(linecolor[1])
        elif ex_material[i] == ['R2']:
            lcol.append(linecolor[2])
        elif ex_material[i] == ['R3']:
            lcol.append(linecolor[3])
        else:
            lcol.append(linecolor[4])
        
        if ex_material_nr[i] == [1]:
            markertype.append(markers[0])
        elif ex_material_nr[i] == [2]:
            markertype.append(markers[1])
        elif ex_material_nr[i] == [3]:
            markertype.append(markers[2])
        elif ex_material_nr[i] == [4]:
            markertype.append(markers[3])
        
        if ex_type[i] == ['SC']:
            lstyle.append(styles[0])
            #lcol.append('black')
        else:
            lstyle.append(styles[1])
            #lcol.append('black')
    
    #color specifications: substances get corresponding color (if single substances are selected)
    subst_color = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','cyan','tab:cyan', #ca
'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink',                                           #sa
'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','cyan']             #prec
    subst_markers = ('o','o','o','o','o','o','o','o','o','o','o',                                                            #ca
'x','x','x','x','x','x','x',                                                                                                 #sa
'^','^','^','^','^','^','^','^','^','^')                                                                                     #prec
    substances_all = list(select[0].columns)[7:35]  #list(select[0].columns)[7:34]
    
    substances_sel = []
    if y_value[0].startswith('cum_m_sm_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][9:])
    elif y_value[0].startswith('cum_m_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][6:])
    elif y_value[0].startswith('m_'):
        for i in range(len(y_value)):
            substances_sel.append(y_value[i][2:])
    elif list(y_value) ==  ['all']:
        substances_sel = substances_all
    elif list(y_value) ==  ['standard substances']:
        substances_sel = substances_all[0:18]
    elif list(y_value) ==  ['special selection']:
        substances_sel = [substances_all[0],substances_all[4],substances_all[6],substances_all[15],substances_all[27]]
    else:
        substances_sel = y_value
    
    subst_color_sel = []
    subst_markers_sel = []
    j=0
    for i in range(len(substances_all)):
        if j < len(substances_sel):
            if substances_sel[j] == substances_all[i]:
                subst_color_sel.append(subst_color[i])
                subst_markers_sel.append(subst_markers[i])
                j = j+1

    
    if ex_nr > 1:
        labels = ['t [d]','t [d]','V [L]','WF [L/kg]','WF [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']
    else:
        labels = ['c [ng/L]','t [Tage]','t [Tage]','V [L]','WF [L/kg]','WF [L/kg]','pv [-]','m [ng]','m [µg]','m [µg/kg]','Zeit [Jahre]','Zeit [Jahre]','field_date']
        headers = ['∑PFAS','Sampling_Time [d]', 'Sampling_Time_mean [d]', 'CumVw_(L)', 'WS_sample', 'WS_mean', 'pore_vol', 'm_∑PFAS', 'cum_m_∑PFAS', 'cum_m_sm_∑PFAS', 'field_time (Y)', 'field_time_mean (Y)', 'field_date']

    if micro == True:
        for i in range(ex_nr):   
            for j in range(len(y_value)):
                select[i][y_value[j]] = select[i][y_value[j]]/1000

    def label(xaxis):
        
        labelseries =  pd.Series(headers, index=labels) 
        label = list(labelseries[labelseries==xaxis].index)[0]
        
        return label

    #if ex_nr > 1 and (len(y_value) > 1 or list(y_value) ==  ['all'] ):
    #    return (print('only single PFAS for multiple experiment'))
    
    fig, ax = plt.subplots(figsize=(20,10))   
    
    for i in range(ex_nr):
        
        if ex_nr > 1:
            if len(substances_sel) == 1:
                y_value = list(y_value)

                select[i].plot( ax=ax, x = x_value, y = y_value, grid = True, figsize=(15,10),                                           
                                style = lstyle[i],
                                marker = markertype[i],
                                lw=2,
                                color = lcol[i]
                                #ylim=(0.0,3000),
                                #color = 'k' if len(y_value) == 1 else None
                                )
            else:                               #several experiments and several substances
                y_value = list(y_value)
                for j in range(len(y_value)):
                    y_value_j = y_value[j]
                    select[i].plot( ax=ax, x = x_value, y = y_value_j, grid = True, figsize=(15,10),                                           
                                style = lstyle[i],
                                marker = markertype[i],
                                lw=2,
                                color = linecolor[j]
                                #ylim=(0.0,3000),
                                #color = 'k' if len(y_value) == 1 else None
                                )
                ax.legend(substances_sel[:])
        else:
        
            if list(y_value) ==  ['all'] or list(y_value) ==  ['standard substances']:                        #for different colouring depending on PFAS substance classes
                y_value_ca = list(select[i].columns)[7:17]
                y_value_sa = list(select[i].columns)[18:24]
                if list(y_value) ==  ['all']:
                    y_value_prec = list(select[i].columns)[25:34]
                #plotting each substance class
                select[i].plot( ax=ax, x = x_value, y = y_value_ca, grid = True, figsize=(15,10),                                           
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )

                select[i].plot( ax=ax, x = x_value, y = y_value_sa, grid = True, figsize=(15,10),                                           
                                style='-',
                                marker='x',
                                markersize='10',
                                lw=2,
                                #ylim=(0.0,3000),
                                )       
                ax.legend(y_value_ca + y_value_sa, handlelength=4)
                if list(y_value) ==  ['all']:
                    select[i].plot( ax=ax, x = x_value, y = y_value_prec, grid = True, figsize=(15,10),
                                style='--',
                                marker='^',
                                #markersize='10',
                                lw=2,
                                #ylim=(0.0,3000),
                                )
                    ax.legend(y_value_ca + y_value_sa + y_value_prec, handlelength=4)
            elif list(y_value) ==  ['special selection']:                        #PFBA, PFOA, PFDA, PFOS, ∑PFAS
                y_value = ['PFBA','PFOA','PFDA','PFOS','∑PFAS']
                select[i].plot( ax=ax, x = x_value, y = ['PFBA','PFOA','PFDA','PFOS'], grid = True, figsize=(15,10),
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )
                select[i].plot( ax=ax, x = x_value, y = ['∑PFAS'], grid = True, figsize=(15,10),
                                color = 'black',
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )

                ax.legend(substances_sel, handlelength=4)
            
            elif list(y_value) ==  ['cum_m_sm_special selection']:                        #PFBA, PFOA, PFDA, PFOS, ∑PFAS
                y_value = ['cum_m_sm_PFBA','cum_m_sm_PFOA','cum_m_sm_PFDA','cum_m_sm_PFOS','cum_m_sm_∑PFAS']
                select[i].plot( ax=ax, x = x_value, y = ['cum_m_sm_PFBA','cum_m_sm_PFOA','cum_m_sm_PFDA','cum_m_sm_PFOS'], grid = True, figsize=(15,10),
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )
                select[i].plot( ax=ax, x = x_value, y = 'cum_m_sm_∑PFAS', grid = True, figsize=(15,10),
                                color = 'black',
                                style='-',
                                marker='o',
                                lw=2,
                                #ylim=(0.0,3000),
                                )

                substances_sel = ['PFBA','PFOA','PFDA','PFOS','∑PFAS']
                ax.legend(substances_sel, handlelength=4)
            else:
                y_value = list(y_value)
                
                select[i].plot( ax=ax, x = x_value, y = y_value, grid = True, figsize=(15,10),
                            style = '-',
                            marker = 'o',   #subst_markers_sel[0],
                            lw=2,
                            color = subst_color_sel
                            #ylim=(0.0,3000),
                            #color = 'k' if len(y_value) == 1 else None
                            )
                ax.legend(substances_sel, handlelength=4)


        
        

    if x_value == "Date":
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
        ax.set_xlabel('Sample Date')
        
    if second_xaxis == True and x2_value != x_value and ex_type != 'IS' and ex_nr == 1:
        
        if ( x_value == headers[0] and x2_value == headers[4]) or ( x_value == headers[1] and x2_value == headers[3]) or ( x2_value == headers[0] and x_value == headers[4]) or ( x2_value == headers[1] and x_value == headers[3]):
            print('error: Sampling Time')
        
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
        
        if x2_value == 'field_date': 
            ax2.set_xticks(select[0][x2_value])
            ax2.xaxis.set_major_locator(mdates.YearLocator(5)) if ex_type == 'SC' else None        
            ax2.xaxis.set_major_locator(mdates.DayLocator(5)) if ex_type == 'LY' else None
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        else:         
            ax2.set_xticks(select[0][x_value].iloc[1::3])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])            
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::3], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.set_xlim(xmin=0)
        ax2.set_xlabel(label(x2_value))        
        
    elif second_xaxis == True and ex_nr > 1:
        if x2_value == 'field_time_mean (Y)' or x2_value == 'field_time (Y)':
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()) #if x_value == "Date" else None
            ax2.set_xticks(select[0][x_value].iloc[1::4])
            #ax2.set_xticks(select[0][x_value].iloc[1::2]) if x_value != "Date" else None #ax2.set_xticks(select[0][x2_value])
            ax2.set_xticklabels(round(select[0][x2_value].iloc[1::4], 0).astype(int), ha = "center") if x_value != "Data" else None 
            ax2.xaxis.set_tick_params(which='major', size=5, width=1.5, direction='out', top='on')
            ax2.set_xlim(xmin=0)
            ax2.set_xlabel(label(x2_value))        
    
    if second_yaxis == True and  y2_value != y_value[0] and ex_type != 'IS' and ex_nr == 1:             
        ay = ax.twinx()
        select[i].plot( ax=ay, x = x2_value, y = y2_value, style='--', marker='x', color = 'k') 
        ay.legend(loc='best')
        ay.set_yscale('log') if Log == True  else None 
        ay.set_ylim(ymin=0) if Log != True  else None
        ay.set_ylabel(label(y2_value))
    elif second_yaxis == True:
        print('Error: Second Y Axis')  
    
    if len(substances_sel) == 1:
        if ex_nr > 1: #and bar != True
            ax.legend(ex[:])
        elif ( (list(y_value) == ['m_∑PFAS']) or + (list(y_value) == ['cum_m_∑PFAS']) or + (list(y_value) == ['cum_m_sm_∑PFAS']) ):
            ax.legend(['∑PFAS'])
        
    #ax.get_legend().set_bbox_to_anchor((-0.07,1))
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='out', top=False)
    ax.xaxis.set_tick_params(which='minor', size=5, width=1.5, direction='out', top=False)
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='out', right=False)
    ax.yaxis.set_tick_params(which='minor', size=5, width=1.5, direction='out', right=False)
    #ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
    #ax.minorticks_on()
    ax.set_yscale('log') if Log == True  else None    
    if x_value != "Date":
        ax.set_xlabel(label(x_value)) 
        ax.set_xlim(xmin=0)
    if second_yaxis == True:
        ax.legend([label(y_value[0])],loc='best', bbox_to_anchor=(1, 0.48))
        ay.legend([label(y2_value)],loc='best', bbox_to_anchor=(1, 0.52))
    if micro == True:
        ax.set_ylabel('c [\u03BCg/L]')
    elif y_value[0].startswith('c_c0'):
        ax.set_ylabel('c/c$_{0}$ [-]')
    else:
        ax.set_ylabel('c [ng/L]')
    #ax.set_ylabel('m [\u03BCg]') if ( (list(y_value) == ['cum_m_∑PFAS'] ) or + (list(y_value) == ['m_∑PFAS']) ) else None 
    if y_value[0].startswith('m_'):
        ax.set_ylabel('m [ng]')
    elif y_value[0].startswith('cum_m_sm_'):
        ax.set_ylabel('m [\u03BCg/kg]')
    elif y_value[0].startswith('cum_m_'):
        ax.set_ylabel('m [\u03BCg]') 
    ax.set_ylabel('QS [-]') if  (list(y_value) == ['Quotientensumme'])  else None
    if Log != True:
        ax.set_ylim(ymin=0)
    if list(y_value) == ['QS']:
        ax.set_ylim(ymax=6)
        ax.set_ylabel('QS [-]')
        ax.hlines(1, 0, 1000, colors='r', linestyles='dotted')
    if y_value[0].startswith('cum_m_sm_'):
        if ex_nr > 1:
            subst_color_sel = ['black']
        #for i in range(len(substances_sel)):
            #ax.hlines(N1_soil[substances_sel[i]], 0, 1000, colors=subst_color_sel[i], linestyles='dashed')
   
    #ax.set_xlim(xmax=10)
    #ax.set_ylim(ymax=150000)
    ax = plt.gca()    
    plt.gcf().autofmt_xdate() if x_value == 'Date' else None
    #plt.legend(ncol=1)
    ax.get_legend().remove()
    plt.show()
    
    if save == True:
        fig = ax.get_figure()
        #fig.savefig("output/{}{}{}{}.png".format(file_string, ex, y_value, x_value), fontsize = 15)
        fig.savefig("output/{}.png".format(file_string), fontsize = 15)


    return ax

if __name__ == '__main__':
    main()