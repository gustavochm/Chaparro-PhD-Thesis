import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# figure scripts
from figures_phase_equilibria import mie_phase_equilibria_comparison
from figures_other_properties import mie_other_properties_comparison
from figures_load_data import load_data_function
from figures_phase_equilibria_others import mie_phase_equilibria_others_comparison
from figures_Tc_Tt import plot_Tc_Tt_comparison
# markers
marker_triple = 's'
marker_crit = 'o'
marker_exp_vle = '^'
marker_exp_sle = 'v'
marker_exp_sve = 'D'
markersize = 3.
linewidth_triple = 0.4
# Figure sizes
inTocm = 2.54
base_height = 5.  # cm
width_single_column = 8.  # cm
width_two_columns = 14.  # cm
width_three_columns = 17.  # cm
dpi = 400
format = 'pdf'
fontsize_annotation = 8

# figure style
plt.style.use('seaborn-v0_8-colorblind')
plt.style.use('../../thesis.mplstyle')

##############################
# folder to save the figures # 
##############################

folder_to_save = '../figures'
os.makedirs(folder_to_save, exist_ok=True)

###################################
# Control whether to plot figures #
###################################
plot_figures = True

# Reading the data for the figure
root_folder_to_read = "../computed_files"

# To  get the limits and folders correctly

folder_to_read_names = dict()
folder_to_read_names['argon'] = "1_1_argon"
folder_to_read_names['xenon'] = "1_2_xenon"
folder_to_read_names['krypton'] = "1_3_krypton"
folder_to_read_names['methane'] = "1_4_methane"
folder_to_read_names['nitrogen'] = "1_6_nitrogen"
folder_to_read_names['carbon_monoxide'] = "1_7_co"
folder_to_read_names['cf4'] = "1_8_cf4"


expdata_filenames = dict()
expdata_filenames['argon'] = "argon.xlsx"
expdata_filenames['xenon'] = "xenon.xlsx"
expdata_filenames['krypton'] = "krypton.xlsx"
expdata_filenames['methane'] = "methane.xlsx"
expdata_filenames['nitrogen'] = "nitrogen.xlsx"
expdata_filenames['carbon_monoxide'] = "carbon_monoxide.xlsx"
expdata_filenames['cf4'] = "cf4.xlsx"

component_names = dict()
component_names['argon'] = r"Ar"
component_names['xenon'] = r"Xe"
component_names['krypton'] = r"Kr"
component_names['methane'] = r"CH$_{4}$"
component_names['nitrogen'] = r"N$_{2}$"
component_names['carbon_monoxide'] =  r"CO"
component_names['cf4'] = r"CF$_{4}$"

phase_equilibria_limits = dict()
phase_equilibria_limits['argon'] = {'rho_lower': -1, 'rho_upper': 45, 'T_lower': 73, 'T_upper': 200, 'P_lower': 5e-3, 'P_upper': 1e3, 'H_upper': 9, 'markerevery_sve': 3}
phase_equilibria_limits['xenon'] = {'rho_lower': -1, 'rho_upper': 30, 'T_lower': 110, 'T_upper': 340, 'P_lower': 1e-3, 'P_upper': 1e3, 'H_upper': 18, 'markerevery_sve': 1}
phase_equilibria_limits['krypton'] = {'rho_lower': -1, 'rho_upper': 40, 'T_lower': 100, 'T_upper': 260, 'P_lower': 5e-3, 'P_upper': 1e3, 'H_upper': 10, 'markerevery_sve': 3}
phase_equilibria_limits['methane'] = {'rho_lower': -1, 'rho_upper':35, 'T_lower': 80, 'T_upper': 240, 'P_lower': 1e-3, 'P_upper': 1e3, 'H_upper': 12, 'markerevery_sve': 2}
phase_equilibria_limits['nitrogen'] = {'rho_lower': -1, 'rho_upper': 40, 'T_lower': 55, 'T_upper': 205, 'P_lower': 1e-3, 'P_upper': 5e3, 'H_upper': 8, 'markerevery_sve': 2}
phase_equilibria_limits['carbon_monoxide'] = {'rho_lower': -1, 'rho_upper': 40, 'T_lower': 60, 'T_upper': 160, 'P_lower': 1e-3, 'P_upper': 1e3, 'H_upper': 9, 'markerevery_sve': 2}
phase_equilibria_limits['cf4'] = {'rho_lower': -1, 'rho_upper': 25, 'T_lower': 80, 'T_upper': 300, 'P_lower': 1e-5, 'P_upper': 1e3, 'H_upper': 18, 'markerevery_sve': 1}

other_properties_limits = dict()
other_properties_limits['argon'] = {'T_lower': 73, 'T_upper': 160}
other_properties_limits['xenon'] = {'T_lower': 150, 'T_upper': 320}
other_properties_limits['krypton'] = {'T_lower': 110, 'T_upper': 220}
other_properties_limits['methane'] = {'T_lower': 85, 'T_upper': 210}
other_properties_limits['nitrogen'] = {'T_lower': 60, 'T_upper': 135}
other_properties_limits['carbon_monoxide'] = {'T_lower': 65, 'T_upper': 145}
other_properties_limits['cf4'] = {'T_lower': 80, 'T_upper': 280}

# what models to include in the figures
fluid_types = ['vle', 'vle_visc']
solid_types = ['vle_sle_sve', 'vle', 'vle_visc']

component_list = ['argon', 'krypton', 'xenon', 'nitrogen', 'carbon_monoxide', 'methane', 'cf4']
# component_list = ['cf4']
main_chapter = ['argon', 'nitrogen', 'methane', 'cf4']

if plot_figures:
    for component_name in component_list:
        folder_to_read = folder_to_read_names[component_name]
        expdata_filename = expdata_filenames[component_name]

        out = load_data_function(root_folder_to_read, folder_to_read, expdata_filename)
        ExpDataFile, dict_models = out
        # set limits
        rho_lower = phase_equilibria_limits[component_name]['rho_lower']
        rho_upper = phase_equilibria_limits[component_name]['rho_upper']
        T_lower = phase_equilibria_limits[component_name]['T_lower']
        T_upper = phase_equilibria_limits[component_name]['T_upper']
        P_lower = phase_equilibria_limits[component_name]['P_lower']
        P_upper = phase_equilibria_limits[component_name]['P_upper']
        H_upper = phase_equilibria_limits[component_name]['H_upper']
        markerevery_sve = phase_equilibria_limits[component_name]['markerevery_sve']
        T_lower_other = other_properties_limits[component_name]['T_lower']
        T_upper_other = other_properties_limits[component_name]['T_upper']

        # Phase equilibria plot
        width = width_three_columns / inTocm
        height = 3 * base_height / inTocm

        # plotting the figure
        fig = mie_phase_equilibria_others_comparison(ExpDataFile, dict_models, width=width, height=height,
                                                    fluid_types=fluid_types,
                                                    solid_types=solid_types,
                                                    rho_lower=rho_lower, rho_upper=rho_upper,
                                                    T_lower=T_lower, T_upper=T_upper,
                                                    P_lower=P_lower, P_upper=P_upper,
                                                    H_upper=H_upper, 
                                                    T_lower_other=T_lower_other, T_upper_other=T_upper_other,
                                                    marker_crit=marker_crit, marker_triple=marker_triple,
                                                    marker_exp_vle=marker_exp_vle,
                                                    marker_exp_sle=marker_exp_sve,
                                                    marker_exp_sve=marker_exp_sle,
                                                    markersize=markersize, markerevery_sve=markerevery_sve,
                                                    linewidth_triple=linewidth_triple)

        filename = f'{component_name}_phase_equilibria_others.{format}'
        if component_name not in main_chapter:
            filename = f'appendix_{filename}'
        file_to_save = os.path.join(folder_to_save, filename)
        fig.savefig(file_to_save, transparent=False)

        """
        # plotting the figure
        fig = mie_phase_equilibria_comparison(ExpDataFile, dict_models, width=width, height=height,
                                              fluid_types=fluid_types,
                                              solid_types=solid_types,
                                              rho_lower=rho_lower, rho_upper=rho_upper,
                                              T_lower=T_lower, T_upper=T_upper,
                                              P_lower=P_lower, P_upper=P_upper,
                                              H_upper=H_upper,
                                              marker_crit=marker_crit, marker_triple=marker_triple,
                                              marker_exp_vle=marker_exp_vle,
                                              marker_exp_sle=marker_exp_sve,
                                              marker_exp_sve=marker_exp_sle,
                                              markersize=markersize, markerevery_sve=markerevery_sve,
                                              linewidth_triple=linewidth_triple)

        filename = f'{component_name}_phase_equilibria.{format}'
        if component_name not in main_chapter:
            filename = f'appendix_{filename}'
        file_to_save = os.path.join(folder_to_save, filename)
        fig.savefig(file_to_save, transparent=False)

        # Other properties plot
        width = width_three_columns / inTocm
        height = 1 * base_height / inTocm
        # limits
        T_lower = other_properties_limits[component_name]['T_lower']
        T_upper = other_properties_limits[component_name]['T_upper']
        fig = mie_other_properties_comparison(ExpDataFile, dict_models, width=width, height=height,
                                              solid_types=solid_types,
                                              T_lower=T_lower, T_upper=T_upper,
                                              marker_crit=marker_crit, marker_triple=marker_triple,
                                              marker_exp_vle=marker_exp_vle, markersize=markersize, markerevery_sve=markerevery_sve,
                                              linewidth_triple=linewidth_triple)
        filename = f'{component_name}_other_properties.{format}'
        if component_name not in main_chapter:
            filename = f'appendix_{filename}'
        file_to_save = os.path.join(folder_to_save, filename)
        fig.savefig(file_to_save, transparent=False)
        """

if plot_figures:
    expdata_excel_dict = dict()
    data_dict_models = dict()

    for component_name in component_list:
        folder_to_read = folder_to_read_names[component_name]
        expdata_filename = expdata_filenames[component_name]

        out = load_data_function(root_folder_to_read, folder_to_read, expdata_filename)
        ExpDataFile, dict_models = out

        expdata_excel_dict[component_name] = ExpDataFile
        data_dict_models[component_name] = dict_models

    solid_types = ['vle', 'vle_visc', 'vle_sle_sve'] 

    Tc_Tt_dicts_all = dict()

    for component_name in component_list:
        Tc_Tt_dicts = dict()
        ExpDataFile = expdata_excel_dict[component_name]
        dict_models = data_dict_models[component_name]

        # Experimental data
        df_info_exp = ExpDataFile.parse('info')
        Tc_Tt_exp = float(df_info_exp['Critical Temperature {K}'])/float(df_info_exp['Triple Point Temperature {K}'])
        Tc_Tt_dicts['exp'] = np.round(Tc_Tt_exp, 2)

        # Models
        for model_type in solid_types:
            df_info = dict_models[f'{model_type}_solid_file'].parse('info')
            Tc_Tt = float(df_info['Tc_model']/df_info['T_triple_model'])
            Tc_Tt_dicts[model_type] = np.round(Tc_Tt, 2)
        Tc_Tt_dicts_all[component_name] = Tc_Tt_dicts

    data_types = ['exp', 'vle', 'vle_visc', 'vle_sle_sve'] 
    width = width_three_columns / inTocm
    height = base_height / inTocm

    fig = plot_Tc_Tt_comparison(component_list, Tc_Tt_dicts_all, component_names,
                                width=width, height=height, data_types=data_types, 
                                fontsize_annotation=fontsize_annotation, T_upper=3.)

    filename = f'Tc_Tt_comparison.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False)
