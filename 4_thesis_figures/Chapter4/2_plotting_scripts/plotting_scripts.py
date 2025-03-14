import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from figures_database_description import plot_database_description
from figures_database_comparison_literature import plot_comparison_database_literature
from figures_hpo import plot_hpo
from figures_tp_representations import plot_self_diffusivity_representation
from figures_tp_representations import plot_viscosity_conductivity_representation
from figures_parity_error_distribution import plot_parity_error_distribution
from figures_parity_lit import plot_parity_tp_lit_data
from figures_tp_dilute_ann_model import plot_dilute_limit_ann_model
from figures_isotherms_tp_models import plot_isotherms_tp_anns

import sys
sys.path.append("../../../")
import nest_asyncio
nest_asyncio.apply()
from jax.config import config
config.update("jax_enable_x64", True)
from flax.training import checkpoints
from python_helpers import helper_get_alpha
from python_helpers.feanneos import HelmholtzModel
from python_helpers.feanneos import helper_jitted_funs


# figure style
plt.style.use('seaborn-v0_8-colorblind')
plt.style.use('../../thesis.mplstyle')

# kwargs for the symbols and lines
fontsize_annotation = 8

# Figure sizes
inTocm = 2.54
base_height = 5.  # cm
width_single_column = 8.  # cm
width_two_columns = 14.  # cm
width_three_columns = 17.  # cm
dpi = 400
format = 'pdf'

######################
# Loading FE-ANN EoS #
######################

ckpt_folder = '../../../3_ann_models/feann_eos'

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

###################
# Loading TP data #
###################

# reading the data
dbpath = "../../../2_databases/mieparticle-diff.csv"
df_diff = pd.read_csv(dbpath)
alpha_diff = helper_get_alpha(df_diff['lr'].to_numpy(), df_diff['la'].to_numpy())
rhoad_diff = df_diff['rho*'].to_numpy()
Tad_diff = df_diff['T*'].to_numpy()
Sres_diff = fun_dic['entropy_residual_fun'](alpha_diff, rhoad_diff, Tad_diff)
df_diff['Sr'] = Sres_diff

dbpath = "../../../2_databases/mieparticle-visc.csv"
df_visc = pd.read_csv(dbpath)
alpha_visc = helper_get_alpha(df_visc['lr'].to_numpy(), df_visc['la'].to_numpy())
rhoad_visc = df_visc['rho*'].to_numpy()
Tad_visc = df_visc['T*'].to_numpy()
Sres_visc = fun_dic['entropy_residual_fun'](alpha_visc, rhoad_visc, Tad_visc)
df_visc['Sr'] = Sres_visc

dbpath = "../../../2_databases/mieparticle-tcond.csv"
df_tcond = pd.read_csv(dbpath)
alpha_tcond = helper_get_alpha(df_tcond['lr'].to_numpy(), df_tcond['la'].to_numpy())
rhoad_tcond = df_tcond['rho*'].to_numpy()
Tad_tcond = df_tcond['T*'].to_numpy()
Sres_tcond = fun_dic['entropy_residual_fun'](alpha_tcond, rhoad_tcond, Tad_tcond)
df_tcond['Sr'] = Sres_tcond
dict_md_data = {'self_diffusivity': df_diff, 'shear_viscosity': df_visc, 'thermal_conductivity': df_tcond}

# literature data
dbpath = "../../../2_databases/MieParticle-TransportProperties-literature/mieparticle-literature-diffusivity.csv"
df_diff_lit = pd.read_csv(dbpath)

dbpath = "../../../2_databases/MieParticle-TransportProperties-literature/mieparticle-literature-viscosity.csv"
df_visc_lit = pd.read_csv(dbpath)

dbpath = "../../../2_databases/MieParticle-TransportProperties-literature/mieparticle-literature-thermal-conductivity.csv"
df_tcond_lit = pd.read_csv(dbpath)

dict_md_lit_data = {'self_diffusivity': df_diff_lit, 'shear_viscosity': df_visc_lit, 'thermal_conductivity': df_tcond_lit}

##########################
# Folder to save results #
##########################
folder_to_save = '../figures'
os.makedirs(folder_to_save, exist_ok=True)

###################################
# Control whether to plot figures #
###################################
plot_figures = True

####################################
# Figures for database description #
####################################
if plot_figures:
    width = width_three_columns / inTocm
    height = 1.75 * base_height / inTocm
    fig = plot_database_description(df_diff, df_visc, df_tcond, width=width, height=height)
    filename = f'tp_mie_fluid_database_description.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

###########################################################
# Figures for database comparison agaisnt literature data #
###########################################################
if plot_figures:
    width = width_three_columns / inTocm
    height = 1.3 * base_height / inTocm
    authors_lit_markers = {'Michels 1985': 'o',
                           'Heyes 1988': 'v',
                           'Heyes 1990': '^',
                           'Rowley 1997': '<',
                           'Vasquez 2004': '>',
                           'Galliero 2005': 'p',
                           'Nasrabad 2006': 'P',
                           'Bugel 2008': '*',
                           'Galliero 2009': 'h',
                           'Baidakov 2011': 'H',
                           'Baidakov 2014': 'X',
                           'Lautenschlaeger 2019': 'D',
                           'Slepavicius 2023': 'd'}
    data_list = [df_diff, df_visc, df_tcond]
    data_lit_list = [df_diff_lit, df_visc_lit, df_tcond_lit]
    fig = plot_comparison_database_literature(data_list, data_lit_list, width=width, height=height, authors_lit_markers=authors_lit_markers) 
    filename = f'tp_lj_fluid_database_comparison_literature.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

##############################################
# Figures HPO for fluid transport properties #
##############################################
if plot_figures:
    # Self diffusivity
    for transport, vrange in zip(['rhodiff', 'visc', 'tcond'], [[1e-6, 1e-3], [1e-3, 1e-1], [1e-3, 1e-1]]):
        hpo_file_path = f"../computed_files/hpo_{transport}.xlsx"
        hpo_file = pd.ExcelFile(hpo_file_path)
        df_hpo = hpo_file.parse("HPO_average", index_col=0)
        df_importance = hpo_file.parse("HPO_importance", index_col=0)
        vmin = vrange[0]
        vmax = vrange[1]
        width = width_three_columns / inTocm
        height = base_height / inTocm
        fig = plot_hpo(df_hpo, df_importance, width=width, height=height, vmin=vmin, vmax=vmax)
        filename = f'tp_hpo_{transport}.{format}'
        file_to_save = os.path.join(folder_to_save, filename)
        fig.savefig(file_to_save, transparent=False, dpi=dpi)

####################################################
# Representation of different transport properties #
####################################################
if plot_figures:
    lambda_r = 12
    # Self diffusivity
    width = width_two_columns / inTocm
    height = 2. * base_height / inTocm
    fig = plot_self_diffusivity_representation(df_diff, width=width, height=height, lambda_r=lambda_r)
    filename = f'tp_lj_self_diffusivity_representation.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

    # Shear viscosity and thermal conductivity
    width = width_three_columns / inTocm
    height = 2. * base_height / inTocm
    fig = plot_viscosity_conductivity_representation(df_visc, df_tcond, width=width, height=height, lambda_r=lambda_r)
    filename = f'tp_lj_viscosity_thermal_conductivity_representation.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

##############################################
# Figures parity plot and error distribution #
##############################################
if plot_figures:
    width = width_three_columns / inTocm
    height = 2. * base_height / inTocm

    path_to_read = "../computed_files/parity_data_self_diffusivity.xlsx"
    parity_diff_excel = pd.ExcelFile(path_to_read)

    path_to_read = "../computed_files/parity_data_shear_viscosity.xlsx"
    parity_visc_excel = pd.ExcelFile(path_to_read)

    path_to_read = "../computed_files/parity_data_thermal_conductivity.xlsx"
    parity_tcond_excel = pd.ExcelFile(path_to_read)

    for model_type in ['ann', 'ann_res']:
        fig = plot_parity_error_distribution(parity_diff_excel, parity_visc_excel, parity_tcond_excel,
                                             width, height, model_type=model_type)
        filename = f'tp_parity_error_distribution_{model_type}.{format}'
        file_to_save = os.path.join(folder_to_save, filename)
        fig.savefig(file_to_save, transparent=False, dpi=dpi)

##############################################
# Figures parity plot and error distribution #
##############################################
if plot_figures:
    width = width_three_columns / inTocm
    height = 1.3 * base_height / inTocm

    path_to_read = "../computed_files/parity_data_lit_self_diffusivity.xlsx"
    parity_lit_diff_excel = pd.read_excel(path_to_read)

    path_to_read = "../computed_files/parity_data_lit_shear_viscosity.xlsx"
    parity_lit_visc_excel = pd.read_excel(path_to_read)

    path_to_read = "../computed_files/parity_data_lit_thermal_conductivity.xlsx"
    parity_lit_tcond_excel = pd.read_excel(path_to_read)

    authors_lit_markers = {'Michels 1985': 'o',
                           'Heyes 1988': 'v',
                           'Heyes 1990': '^',
                           'Rowley 1997': '<',
                           'Vasquez 2004': '>',
                           'Galliero 2005': 'p',
                           'Nasrabad 2006': 'P',
                           'Bugel 2008': '*',
                           'Galliero 2009': 'h',
                           'Baidakov 2011': 'H',
                           'Baidakov 2014': 'X',
                           'Lautenschlaeger 2019': 'D',
                           'Slepavicius 2023': 'd'}

    dict_parity_lit = {'self_diffusivity': parity_lit_diff_excel, 'shear_viscosity': parity_lit_visc_excel, 'thermal_conductivity': parity_lit_tcond_excel}
    fig = plot_parity_tp_lit_data(dict_parity_lit, width=width, height=height, authors_lit_markers=authors_lit_markers)
    filename = f'tp_parity_lit_ann.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

#########################################
# Figure TP dilute limit witn ANN model #
#########################################
if plot_figures:

    path_to_read = "../computed_files/dilute_tp_lrs.xlsx"
    excel_dilute = pd.ExcelFile(path_to_read)
    lr_list = [12, 16, 20]
    color_list = ['C0', 'C2', 'C1']
    width = width_three_columns / inTocm
    height = 1. * base_height / inTocm

    fig = plot_dilute_limit_ann_model(excel_dilute, width=width, height=height, lr_list=lr_list, color_list=color_list)
    filename = f'tp_dilute_ann_model.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

###########################################################
# Figure TP isotherms from models and compared to MD data #
###########################################################
if plot_figures:
    lr_list = [12, 16, 20]
    dict_isotherms_lrs = dict()
    for lr in lr_list:
        dbpath = f"../computed_files/isotherms_tp_lr{lr:.0f}.xlsx"
        excel_lr = pd.ExcelFile(dbpath)
        dict_isotherms_lrs[f'lr={lr:.0f}'] = excel_lr

    T_list = [0.9, 1., 1.3, 2.8, 6.0]
    # color_list = ['C0', 'C1', 'C2', 'C3', 'C5']
    # marker_list = ['s', 'D', '^', 'o', 'P']
    marker_list = ['s', 'o', 'v', 'd', '^']
    color_list = ['C0', 'C2', 'black', 'C3', 'C1']
    width = width_three_columns / inTocm
    height = 3. * base_height / inTocm

    fig = plot_isotherms_tp_anns(dict_md_data, dict_isotherms_lrs,
                                 width=width, height=height,
                                 lr_list=lr_list, T_list=T_list,
                                 color_list=color_list, marker_list=marker_list)
    filename = f'tp_isotherms_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

###################
# Figure appendix #
###################
if plot_figures:
    lr_list = [10, 18, 26]
    dict_isotherms_lrs = dict()
    for lr in lr_list:
        dbpath = f"../computed_files/isotherms_tp_lr{lr:.0f}.xlsx"
        excel_lr = pd.ExcelFile(dbpath)
        dict_isotherms_lrs[f'lr={lr:.0f}'] = excel_lr

    T_list = [0.9, 1., 1.3, 2.8, 6.0]
    # color_list = ['C0', 'C1', 'C2', 'C3', 'C5']
    # marker_list = ['s', 'D', '^', 'o', 'P']
    marker_list = ['s', 'o', 'v', 'd', '^']
    color_list = ['C0', 'C2', 'black', 'C3', 'C1']

    width = width_three_columns / inTocm
    height = 3. * base_height / inTocm

    fig = plot_isotherms_tp_anns(dict_md_data, dict_isotherms_lrs,
                                 width=width, height=height,
                                 lr_list=lr_list, T_list=T_list,
                                 color_list=color_list, marker_list=marker_list)
    filename = f'appendix_tp_isotherms_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)
