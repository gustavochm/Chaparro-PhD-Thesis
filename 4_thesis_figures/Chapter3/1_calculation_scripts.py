import os
import sys
import numpy as np
import pandas as pd
import nest_asyncio
nest_asyncio.apply()

import jax
from jax import numpy as jnp
from jax.config import config
from flax.training import checkpoints



PRECISSION = 'float64'
if PRECISSION == 'float64':
    config.update("jax_enable_x64", True)
    type_np = np.float64
    type_jax = jnp.float64
else:
    config.update("jax_enable_x32", True)
    type_np = np.float32
    type_jax = jnp.float32

sys.path.append("../../")
# loading computing function files

from python_helpers.feanneos import HelmholtzModel
from python_helpers.feanneos import helper_solver_funs, helper_jitted_funs
from python_helpers import helper_get_alpha

from python_helpers.data_figures import data_phase_equilibria_solid_lr
from python_helpers.data_figures import data_isotherms_lr
from python_helpers.data_figures import latex_description_database
from python_helpers.data_figures import parity_plot_computation
from python_helpers.data_figures import parity_error_distribution_latex
from python_helpers.data_figures import data_critical_and_triple_point_feanneos, data_critical_and_triple_point_feanneos_by_parts
from python_helpers.data_figures import data_brown_curves_solid_lr


#############################
# Compute or not data again #
#############################
compute_data = True

######################
# Loading FE-ANN EoS #
######################

ckpt_folder = '../../3_ann_models/feanns_eos'
prefix_params = 'FE-ANN-EoS-params_'
###
Tscale = 'Tinv'
seed = 17
factor = 0.01
EPOCHS = 50000
traind_model_folder = f'models_{Tscale}_factor{factor:.2f}_seed{seed}'
ckpt_folder_model = os.path.join(ckpt_folder, traind_model_folder)
ckpt_Tinv = checkpoints.restore_checkpoint(ckpt_dir=ckpt_folder_model, target=None, prefix=prefix_params, step=EPOCHS)
helmholtz_features = list(ckpt_Tinv['features'].values())
helmholtz_model = HelmholtzModel(features=helmholtz_features)
helmholtz_params = {'params': ckpt_Tinv['params']}

fun_dic = helper_jitted_funs(helmholtz_model, helmholtz_params)

######
training_database_path = '../../2_databases/mieparticle-data-training.csv'
virial_database_path = '../../2_databases/mieparticle-virial-coefficients.csv'

##########################
# Folder to save results #
##########################

folder_to_save = './computed_files'
os.makedirs(folder_to_save, exist_ok=True)

#############################################################
# Isotherms of LJ (lambda_r = 12, lambda_r=6) for flowchart #
#############################################################
if compute_data:
    lambda_r = 12
    filename = f'isotherms_lr{lambda_r}_flowchart.xlsx'
    T_list = [0.9, 1.0, 1.3, 1.6, 2.0, 2.8]
    isotherms_lr = data_isotherms_lr(fun_dic, T_list=T_list, lambda_r=lambda_r)
    file_to_save = os.path.join(folder_to_save, filename)
    writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
    for key, df in isotherms_lr.items():
        df.to_excel(writer, sheet_name=key, index=False)
    writer.close()

#######################################
# Summary table to describe database # 
#######################################
if compute_data:
    file_database = training_database_path
    df_data = pd.read_csv(file_database)
    df = df_data.copy()
    df['alpha'] = helper_get_alpha(df['lr'], df['la'])
    df['Z'] = df['P_ad'] / (df['rho*'] * df['T*'])
    df['gamma'] = df['Cp_npt'] / df['Cv_nvt']

    file_database = virial_database_path
    df_virial = pd.read_csv(file_database)

    tabular_latex = latex_description_database(df, df_virial)
    filename = "database_latex_table.md"
    path_to_save = os.path.join(folder_to_save, filename)
    text_file = open(path_to_save, "w")
    text_file.write(tabular_latex)
    text_file.close()

    print("Number of solid datapoints:", df['is_solid'].sum())

########################################################
# Phase diagram of (lambda_r = 12, 16, 20, lambda_r=6) #
########################################################
if compute_data:
    lr_list = [10, 12, 16, 18, 20, 26]
    triple_point_guesses = dict(lr12=[1e-3, 0.85, 0.95, 0.67], lr16=[1e-3, 0.83, 0.97, 0.67], lr20=[1e-3, 0.81, 0.99, 0.67],
                                lr10=[6e-4, 0.86, 0.95, 0.68], lr18=[7e-3, 0.82, 0.98, 0.67], lr26=[1e-2, 0.8, 1.02, 0.66])
    for lambda_r in lr_list:
        filename = f'phase_equilibria_lr{lambda_r}.xlsx'
        initial_triple_point = triple_point_guesses[f'lr{lambda_r:.0f}']
        print(lambda_r, initial_triple_point)

        phase_equilibria_lr = data_phase_equilibria_solid_lr(fun_dic, lambda_r=lambda_r, initial_triple_point=initial_triple_point)
        file_to_save = os.path.join(folder_to_save, filename)

        writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
        phase_equilibria_lr['info'].to_excel(writer, sheet_name='info', index=False)
        phase_equilibria_lr['vle'].to_excel(writer, sheet_name='vle', index=False)
        phase_equilibria_lr['sle'].to_excel(writer, sheet_name='sle', index=False)
        phase_equilibria_lr['sve'].to_excel(writer, sheet_name='sve', index=False)
        writer.close()

#####################################################
# Critical and triple points from the FE-ANN(s) EoS #
#####################################################
if compute_data:
    lr_min = 7.
    lr_max = 500.
    n_min = 100
    n_max = 1000
    # df = data_critical_and_triple_point_feanneos(fun_dic, lr_min, lr_max, n=n)
    lr0 = 12
    inc0_triple = [1e-3, 0.85, 0.95, 0.67]
    df = data_critical_and_triple_point_feanneos_by_parts(fun_dic, lr_min, lr_max, n_min=n_min, n_max=n_max, lr0=lr0, inc0_triple=inc0_triple)
    filename = 'triple_critical_points.xlsx'
    file_to_save = os.path.join(folder_to_save, filename)
    df.to_excel(file_to_save, index=False)

####################################################
# Isotherms of (lambda_r = 12, 16, 20, lambda_r=6) #
####################################################
if compute_data:
    lr_list = [10, 12, 18, 16, 20, 26]
    T_list = [0.65, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 2.8, 5.3, 6.0, 7.2]

    for lambda_r in lr_list:
        # FE-ANN(s) EoS
        filename = f'isotherms_lr{lambda_r:.0f}.xlsx'
        isotherms_lr = data_isotherms_lr(fun_dic, T_list=T_list, lambda_r=lambda_r)
        file_to_save = os.path.join(folder_to_save, filename)
        writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
        for key, df in isotherms_lr.items():
            df.to_excel(writer, sheet_name=key, index=False)
        writer.close()

##########################################
# Parity plots obtained from  FE-ANN EoS #
##########################################
if compute_data:

    file_database = training_database_path
    df_data = pd.read_csv(file_database)

    file_database = virial_database_path
    df_virial = pd.read_csv(file_database)

    seed_split = 12
    test_size = 0.1

    dict_results_feanneos_Tinv = parity_plot_computation(df_data, df_virial, fun_dic, seed_split=seed_split, test_size=test_size)
    filename = 'parity_data_feanneos_solid.xlsx'
    file_to_save = os.path.join(folder_to_save, filename)
    writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
    for key, df in dict_results_feanneos_Tinv.items():
        df.to_excel(writer, sheet_name=key, index=False)
    writer.close()

    excel_parity = pd.ExcelFile(file_to_save)
    latex_table_parity = parity_error_distribution_latex(excel_parity)
    path_to_save = os.path.join(folder_to_save, "parity_error_distribution_feanneos_solid.md")
    text_file = open(path_to_save, "w")
    text_file.write(latex_table_parity)
    text_file.close()

########################################################
# Brown curves of (lambda_r = 12, 16, 20, lambda_r=6) #
########################################################
if compute_data:
    # lr_list = [12, 16, 20]
    # triple_point_guesses = dict(lr12=[1e-3, 0.85, 0.95, 0.67], lr16=[1e-3, 0.83, 0.97, 0.67], lr20=[1e-3, 0.81, 0.99, 0.67])
    lr_list = [10, 12, 16, 18, 20, 26]
    triple_point_guesses = dict(lr12=[1e-3, 0.85, 0.95, 0.67], lr16=[1e-3, 0.83, 0.97, 0.67], lr20=[1e-3, 0.81, 0.99, 0.67],
                                lr10=[6e-4, 0.86, 0.95, 0.68], lr18=[7e-3, 0.82, 0.98, 0.67], lr26=[1e-2, 0.8, 1.02, 0.66])
    for lambda_r in lr_list:
        filename = f'brown_curves_lr{lambda_r:.0f}.xlsx'
        initial_triple_point = triple_point_guesses[f'lr{lambda_r:.0f}']

        rho0_amagat = 0.2
        if lambda_r == 26:
            rho0_amagat = 0.3

        brown_curves_lr = data_brown_curves_solid_lr(fun_dic, lambda_r, lambda_a=6., T_lower=0.5,
                                                     n_b2=500, T_upper_b2=50.,
                                                     n_zeno=500, rho0_zeno=1e-2, Tr_upper_zeno=0.99999,
                                                     n_boyle=300, rho0_boyle=1e-2, Tr_min_boyle=0.98, Tr_upper_boyle=0.99999,
                                                     n_charles=2000, rho0_charles=0.8, Tr_upper_charles=1.1,
                                                     n_amagat=200, rho0_amagat=rho0_amagat, T_upper_amagat=15.,
                                                     initial_triple_point=initial_triple_point, initial_crit_point=None)

        file_to_save = os.path.join(folder_to_save, filename)
        writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
        for key, df in brown_curves_lr.items():
            df.to_excel(writer, sheet_name=key, index=False)
        writer.close()
