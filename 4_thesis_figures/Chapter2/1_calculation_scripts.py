import os
import sys
import numpy as np
import pandas as pd

from jax import numpy as jnp
from jax.config import config
from flax.training import checkpoints

import nest_asyncio
nest_asyncio.apply()

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
from python_helpers.feanneos import HelmholtzModel
from python_helpers.feanneos import HelmholtzModel_Tlinear
from python_helpers.feanneos import helper_solver_funs, helper_jitted_funs
from python_helpers import helper_get_alpha

# loading computing function files
from python_helpers.data_figures import data_phase_equilibria_fluid_lr
from python_helpers.data_figures import data_isotherms_lr
from python_helpers.data_figures import latex_description_database
from python_helpers.data_figures import parity_plot_computation
from python_helpers.data_figures import data_isotherms_lr_saft
from python_helpers.data_figures import data_critical_point_feanneos, data_critical_point_saft
from python_helpers.data_figures import data_phase_equilibria_fluid_saft_lr
from python_helpers.data_figures import data_brown_curves_fluid_lr

#############################
# Compute or not data again #
#############################
compute_data = True

######################
# Loading FE-ANN EoS #
######################

ckpt_folder = '../../3_ann_models/feann_eos'

prefix_params = 'FE-ANN-EoS-params_'

###
Tscale = 'Tinv'
seed = 1
factor = 0.05
EPOCHS = 20000
traind_model_folder = f'models_{Tscale}_factor{factor:.2f}_seed{seed}'
ckpt_folder_model = os.path.join(ckpt_folder, traind_model_folder)
ckpt_Tinv = checkpoints.restore_checkpoint(ckpt_dir=ckpt_folder_model, target=None, prefix=prefix_params)
helmholtz_features = list(ckpt_Tinv['features'].values())
helmholtz_model = HelmholtzModel(features=helmholtz_features)
helmholtz_params = {'params': ckpt_Tinv['params']}
fun_dic = helper_jitted_funs(helmholtz_model, helmholtz_params)

####
Tscale = 'Tlinear'
seed = 1337
factor = 0.05
EPOCHS = 20000
traind_model_folder = f'models_{Tscale}_factor{factor:.2f}_seed{seed}'
ckpt_folder_model = os.path.join(ckpt_folder, traind_model_folder)
ckpt_Tlinear = checkpoints.restore_checkpoint(ckpt_dir=ckpt_folder_model, target=None, prefix=prefix_params)
helmholtz_Tlinear_features = list(ckpt_Tlinear['features'].values())
helmholtz_model_Tlinear = HelmholtzModel_Tlinear(features=helmholtz_Tlinear_features)
helmholtz_params_Tlinear = {'params': ckpt_Tlinear['params']}
fun_dic_Tlinear = helper_jitted_funs(helmholtz_model_Tlinear, helmholtz_params_Tlinear)

# Database paths
training_database_path = '../../2_databases/mieparticle-data-training.csv'
virial_database_path = '../../2_databases/mieparticle-virial-coefficients.csv'

##########################
# Folder to save results #
##########################

folder_to_save = './computed_files'
os.makedirs(folder_to_save, exist_ok=True)


#######################################################################
# Isotherms of LJ (lambda_r = 12, lambda_r=6) for flowchart (summary) #
#######################################################################
if compute_data:
    lambda_r = 12.
    filename = f'isotherms_lr{lambda_r:.0f}_flowchart.xlsx'
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
    df_data_fluid = df_data[df_data['is_fluid']].copy().reset_index(drop=True)
    df = df_data_fluid
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

#######################################################
# Parity plots obtained from the different FE-ANN EoS #
#######################################################
if compute_data:

    file_database = training_database_path
    df_data = pd.read_csv(file_database)
    df_data_fluid = df_data[df_data['is_fluid']].copy().reset_index(drop=True)

    file_database = virial_database_path
    df_virial = pd.read_csv(file_database)

    seed_split = 12
    test_size = 0.1

    dict_results_feanneos_Tinv = parity_plot_computation(df_data_fluid, df_virial, fun_dic, seed_split=seed_split, test_size=test_size)
    filename = 'parity_data_feanneos.xlsx'
    file_to_save = os.path.join(folder_to_save, filename)
    writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
    for key, df in dict_results_feanneos_Tinv.items():
        df.to_excel(writer, sheet_name=key, index=False)
    writer.close()

    dict_results_Tlinear = parity_plot_computation(df_data_fluid, df_virial, fun_dic_Tlinear, seed_split=seed_split, test_size=test_size)
    filename = 'parity_data_feanneos_Tlinear.xlsx'
    file_to_save = os.path.join(folder_to_save, filename)
    writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
    for key, df in dict_results_Tlinear.items():
        df.to_excel(writer, sheet_name=key, index=False)
    writer.close()

########################################################
# Phase diagram of (lambda_r = 12, 16, 20, lambda_r=6) #
########################################################
if compute_data:
    lr_list = [10, 12, 16, 18, 20, 26]
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'phase_equilibria_lr{lambda_r:.0f}.xlsx'
        phase_equilibria_lr = data_phase_equilibria_fluid_lr(fun_dic, lambda_r=lambda_r)

        file_to_save = os.path.join(folder_to_save, filename)

        writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
        phase_equilibria_lr['info'].to_excel(writer, sheet_name='info', index=False)
        phase_equilibria_lr['vle'].to_excel(writer, sheet_name='vle', index=False)
        phase_equilibria_lr['sle'].to_excel(writer, sheet_name='sle', index=False)
        writer.close()

        # SAFT-VR-Mie EoS
        phase_equilibria_saft_lr = data_phase_equilibria_fluid_saft_lr(lambda_r=lambda_r, Tlower=0.6, n_vle=100, initial_crit_point=None)
        filename = f'phase_equilibria_saft_lr{lambda_r:.0f}.xlsx'
        file_to_save = os.path.join(folder_to_save, filename)
        writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
        phase_equilibria_saft_lr['info'].to_excel(writer, sheet_name='info', index=False)
        phase_equilibria_saft_lr['vle'].to_excel(writer, sheet_name='vle', index=False)
        writer.close()

####################################################
# Isotherms of (lambda_r = 12, 16, 20, lambda_r=6) #
####################################################
if compute_data:
    lr_list = [10, 12, 16, 18, 20, 26]
    T_list = [0.65, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 2.8, 5.3, 6.0, 7.2]

    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'isotherms_lr{lambda_r:.0f}.xlsx'
        isotherms_lr = data_isotherms_lr(fun_dic, T_list=T_list, lambda_r=lambda_r)
        file_to_save = os.path.join(folder_to_save, filename)
        writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
        for key, df in isotherms_lr.items():
            df.to_excel(writer, sheet_name=key, index=False)
        writer.close()

        # SAFT-VR-Mie EoS
        filename = f'isotherms_saft_lr{lambda_r:.0f}.xlsx'
        isotherms_lr = data_isotherms_lr_saft(T_list=T_list, lambda_r=lambda_r)
        file_to_save = os.path.join(folder_to_save, filename)
        writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
        for key, df in isotherms_lr.items():
            df.to_excel(writer, sheet_name=key, index=False)
        writer.close()

############################################################
# Brown curves of (lambda_r = 12, 16, 20 and lamnda_a = 6) #   
############################################################
if compute_data:
    lr_list = [10, 12, 16, 18, 20, 26]
    rho_max_sle_list = [1.4, 1.4, 1.35, 1.33, 1.31, 1.15]
    # lr_list = [12, 16, 20]
    # rho_max_sle_list = [1.4, 1.35, 1.32]
    for lambda_r, rho_max_sle in zip(lr_list, rho_max_sle_list):
        # FE-ANN EoS
        brown_curves_lr = data_brown_curves_fluid_lr(fun_dic, lambda_r=lambda_r, lambda_a=6.,
                                                     T_lower=0.6, n_b2=500, T_upper_b2=50.,
                                                     n_zeno=500, rho0_zeno=1e-2, Tr_upper_zeno=0.99999,
                                                     n_boyle=300, rho0_boyle=1e-2, Tr_min_boyle=0.98, Tr_upper_boyle=0.99999,
                                                     n_charles=2000, rho0_charles=0.8, Tr_upper_charles=1.1,
                                                     n_amagat=200, rho0_amagat=0.2, T_upper_amagat=15.,
                                                     rho_max_sle=rho_max_sle)
        filename = f'brown_curves_lr{lambda_r:.0f}.xlsx'
        file_to_save = os.path.join(folder_to_save, filename)
        writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
        for key, df in brown_curves_lr.items():
            df.to_excel(writer, sheet_name=key, index=False)
        writer.close()

#############################################################
# Critical points of (lambda_r = 12, 16, 20) and lambda_a=6 #
#############################################################
if compute_data:
    lr_min = 7.
    lr_max = 38.
    lambda_a = 6.
    inc0 = [0.3, 2.0]
    n = 100

    # FE-ANN EoS
    df_crit_feann = data_critical_point_feanneos(fun_dic, lr_min, lr_max, n, lambda_a=lambda_a, inc0=inc0)
    filename = 'critical_points.xlsx'
    file_to_save = os.path.join(folder_to_save, filename)
    df_crit_feann.to_excel(file_to_save, index=False)

    # SAFT-VR-Mie EoS
    df_crit_saft = data_critical_point_saft(lr_min, lr_max, n, lambda_a=lambda_a)
    filename = 'critical_points_saft.xlsx'
    file_to_save = os.path.join(folder_to_save, filename)
    df_crit_saft.to_excel(file_to_save, index=False)

