import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# figure scripts
from figures_flowchart import plot_phase_space_flowchart, plot_phase_diagram_flowchart, plot_isotherms_flowchart
from figures_database_description import plot_database_description
from figures_parity_plot import plot_parity_plot
from figures_error_distribution import plot_error_distribution
from figures_pressure_isotherms import plot_pressure_isotherms
from figures_cv_isotherms import plot_cv_isotherms
from figures_secondorder_isotherms import plot_second_isotherms, plot_second_isotherms_all
from figures_phase_equilibria_dev import plot_phase_equilibria_dev_parity
from figures_critical_triple_point import plot_critical_triple_points
from figures_brown_curves import plot_brown_curves_virials3
from figures_freeze_method import plot_isobars_freezing
from figures_hpo import plot_hpo

# figure style
plt.style.use('seaborn-v0_8-colorblind')
plt.style.use('../../thesis.mplstyle')

# kwargs for the symbols and lines
fontsize_annotation = 8
# colors
color_vle = 'k'
color_virial = 'k'
color_zeno = 'C0'
color_boyle = 'C1'
color_charles = 'C2'
color_amagat = 'C3'
color_solid_fill = 'darkgrey'
color_solid = 'grey'

# markers
marker_triple = 's'
marker_crit = 'o'
marker_zeno = 's'
marker_boyle = '^'
marker_charles = 'v'
marker_amagat = 'o'
markersize = 4.

# linestyles
ls_feann = '-'
ls_saft = '--'

# Figure sizes
inTocm = 2.54
base_height = 5.  # cm
width_single_column = 8.  # cm
width_two_columns = 14.  # cm
width_three_columns = 17.  # cm
dpi = 400
format = 'pdf'

######################
# Database paths
training_database_path = '../../../2_databases/mieparticle-data-training.csv'
virial_database_path = '../../../2_databases/mieparticle-virial-coefficients.csv'
brown_database_path = '../../../2_databases/mieparticle-brown.csv'
vle_database_path  = '../../../2_databases/mieparticle-vle.csv'
hvap_database_path = '../../../2_databases/mieparticle-hvap.csv'
sle_database_path  = '../../../2_databases/mieparticle-sle.csv'
hmelting_database_path = '../../../2_databases/mieparticle-hmelting.csv'

# folder to save the figures
folder_to_save = '../figures'
os.makedirs(folder_to_save, exist_ok=True)

###################################
# Control whether to plot figures #
###################################
plot_figures = True

#########################
# Figures for flowchart #
#########################
if plot_figures:
    lambda_r = 12
    folder_to_read = '../computed_files'
    filename = f'phase_equilibria_lr{lambda_r:.0f}.xlsx'
    file_to_read = os.path.join(folder_to_read, filename)

    excel_phase_equilibria = pd.ExcelFile(file_to_read)

    filename = f'isotherms_lr{lambda_r:.0f}_flowchart.xlsx'
    file_to_read = os.path.join(folder_to_read, filename)

    excel_isotherms = pd.ExcelFile(file_to_read)

    os.makedirs(os.path.join(folder_to_save, 'flowchart'), exist_ok=True)

    # phase_space
    height = 2.7 / inTocm
    width = 2.8 / inTocm
    fig = plot_phase_space_flowchart(excel_phase_equilibria, height=height, width=width)
    filename = f'flowchart_phase_space_lr{lambda_r}.{format}'
    file_to_save = os.path.join(folder_to_save, 'flowchart', filename)
    fig.savefig(file_to_save, transparent=True)

    # phase_diagram
    height = 2.7 / inTocm
    width = 5.3 / inTocm
    fig = plot_phase_diagram_flowchart(excel_phase_equilibria, height=height, width=width,
                                       marker_triple=marker_triple, marker_crit=marker_crit)
    filename = f'flowchart_phase_diagram_lr{lambda_r}.{format}'
    file_to_save = os.path.join(folder_to_save, 'flowchart', filename)
    fig.savefig(file_to_save, transparent=True)

    # isotherms
    height = 2.7 / inTocm
    width = 5.3 / inTocm
    fig = plot_isotherms_flowchart(excel_isotherms, height=height, width=width)
    filename = f'flowchart_isotherms_lr{lambda_r}.{format}'
    file_to_save = os.path.join(folder_to_save, 'flowchart', filename)
    fig.savefig(file_to_save, transparent=True)


####################################
# Figures for database description #
####################################
if plot_figures:
    file_database = training_database_path
    df_data = pd.read_csv(file_database)
    width = width_three_columns / inTocm
    height = 1.25 * base_height / inTocm
    fig = plot_database_description(df_data, width=width, height=height)
    filename = f'mie_database_solid_description.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

###########################
# Figures for parity plot #
###########################
if plot_figures:
    excel_parity_path = "../computed_files/parity_data_feanneos_solid.xlsx"
    excel_parity = pd.ExcelFile(excel_parity_path)

    width = width_three_columns / inTocm
    height = 3. * base_height / inTocm
    fig = plot_parity_plot(excel_parity, width=width, height=height)
    filename = f'solid_feanneos_parity_plot.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

################################################
# Figures error distribution for FE-ANN(s) EoS #
################################################
if plot_figures:
    # for T-inv model
    excel_parity_path = "../computed_files/parity_data_feanneos_solid.xlsx"
    excel_parity = pd.ExcelFile(excel_parity_path)

    width = width_three_columns / inTocm
    height = 1 * base_height / inTocm
    fig = plot_error_distribution(excel_parity, width=width, height=height)
    filename = f'solid_feanneos_error_distribution.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

#######################################################
# Figures for pressure isotherms for lr=12, 16 and 20 #
#######################################################
if plot_figures:
    file_database = training_database_path
    df_data = pd.read_csv(file_database)

    # reading the data
    folder_to_read = "../computed_files"
    lr_list = [12, 16, 20]
    Tad_list = [0.65, 0.70, 0.90, 1.1, 1.3, 2.8, 5.3, 7.2]
    excel_dict_isotherms = dict()
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'isotherms_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict_isotherms[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

    width = width_three_columns / inTocm
    height = 1.25 * base_height / inTocm

    fig = plot_pressure_isotherms(df_data, excel_dict_isotherms, lr_list=lr_list, Tad_list=Tad_list, width=width, height=height)
    filename = f'solid_feanneos_pressure_isotherms_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

######################################################################
# Figures for isochoric heat capacity isotherms for lr=12, 16 and 20 #
######################################################################
if plot_figures:
    file_database = training_database_path
    df_data = pd.read_csv(file_database)

    # reading the data
    folder_to_read = "../computed_files"
    lr_list = [12, 16, 20]
    Tad_list = [0.90, 1.1, 1.3, 2.8, 5.3, 7.2]
    excel_dict_isotherms = dict()
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'isotherms_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict_isotherms[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

    width = width_three_columns / inTocm
    height = 1. * base_height / inTocm

    fig = plot_cv_isotherms(df_data, excel_dict_isotherms, lr_list=lr_list, Tad_list=Tad_list, width=width, height=height)
    filename = f'solid_feanneos_cv_isotherms_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

##################################################################
# Figures second order properties isotherms for lr=12, 16 and 20 #
##################################################################
if plot_figures:
    file_database = training_database_path
    df_data = pd.read_csv(file_database)

    # reading the data
    folder_to_read = "../computed_files"
    lr_list = [12, 16, 20]
    Tad_list = [0.90, 1.1, 1.3, 2.8, 5.3, 7.2]
    excel_dict_isotherms = dict()
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'isotherms_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict_isotherms[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

    width = width_three_columns / inTocm
    height = 3. * base_height / inTocm

    fig = plot_second_isotherms(df_data, excel_dict_isotherms, lr_list=lr_list, Tad_list=Tad_list, width=width, height=height, rasterized=True)
    filename = f'solid_feanneos_second_isotherms_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)


################################################################
# Figures phase equilibria and deviations for lr=12, 16 and 20 #
################################################################
if plot_figures:
    # reading the data
    folder_to_read = "../computed_files"
    lr_list = [12, 16, 20]
    ls_list = ['-', '--', '-.']
    color_list = ['C0', 'C2', 'C1']
    marker_list = ['s', 'v', '^']
    excel_dict_feanneos = dict()
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'phase_equilibria_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict_feanneos[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

    file_database = vle_database_path
    df_vle_md = pd.read_csv(file_database)

    file_database = hvap_database_path
    df_hvap_md = pd.read_csv(file_database)
    df_hvap_md['Uvap*'] = df_hvap_md['Hvap*'] - df_hvap_md['P*_sim'] * (1./df_hvap_md['rhov*'] - 1./df_hvap_md['rhol*'])

    file_database = sle_database_path
    df_sle_md = pd.read_csv(file_database)

    file_database = hmelting_database_path
    df_melting_md = pd.read_csv(file_database)

    dict_md = dict(vle=df_vle_md, hvap=df_hvap_md, sle=df_sle_md, melting=df_melting_md)

    width = width_three_columns / inTocm
    height = 2. * base_height / inTocm
    fig = plot_phase_equilibria_dev_parity(dict_md, excel_dict_feanneos,
                                           width=width, height=height, lr_list=lr_list,
                                           marker_crit=marker_crit, marker_triple=marker_triple,
                                           color_list=color_list, marker_list=marker_list, ls_list=ls_list,
                                           fontsize_annotation=fontsize_annotation, markevery=2)
    filename = f'solid_feanneos_phase_equilibria_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

#####################################################
# Brown characteristics curves for lr=12, 16 and 20 #
#####################################################
if plot_figures:
    file_database = brown_database_path
    df_brown = pd.read_csv(file_database)

    folder_to_read = "../computed_files"
    lr_list = [12, 16, 20]
    excel_dict = dict()
    for lambda_r in lr_list:
        filename = f'brown_curves_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

    width = width_three_columns / inTocm
    height = 2. * base_height / inTocm

    fig = plot_brown_curves_virials3(excel_dict, df_brown, lr_list, width=width, height=height,
                                     color_vle=color_vle, color_virial=color_virial,
                                     color_zeno=color_zeno, color_boyle=color_boyle, color_charles=color_charles, color_amagat=color_amagat,
                                     color_solid_fill=color_solid_fill,
                                     marker_crit=marker_crit, marker_triple=marker_triple,
                                     marker_zeno=marker_zeno, marker_boyle=marker_boyle, marker_charles=marker_charles, marker_amagat=marker_amagat,
                                     include_md_data=True)

    filename = f'solid_feanneos_brown_curves_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

###############################################################
# Figures critical and triple points obtained from FE-ANN EoS #
###############################################################
if plot_figures:
    # reading the data
    folder_to_read = "../computed_files"

    # FE-ANN data
    filename = 'triple_critical_points.xlsx'
    file_to_read = os.path.join(folder_to_read, filename)
    df_feann = pd.read_excel(file_to_read)

    filename = '../../../2_databases/crit_triple_ms_literature.csv'
    df_literature = pd.read_csv(filename)
    author_list = np.unique(df_literature['AuthorID'])

    width = width_three_columns / inTocm
    height = base_height / inTocm
    marker_list = ['s', 'o', 'v', '^', 'd']
    lr_list = [8, 10, 16, 34]
    fig = plot_critical_triple_points(width, height, df_feann, df_literature, author_list, marker_list, lr_list=lr_list)
    filename = f'solid_feanneos_critical_triple_points.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)


########################
# Figures for appendix #
########################
if plot_figures:
    lr_list = [10, 18, 26]

    ######################
    # Pressure isotherms #
    ######################

    file_database = training_database_path
    df_data = pd.read_csv(file_database)

    # reading the data
    folder_to_read = "../computed_files"
    Tad_list = [0.65, 0.70, 0.90, 1.1, 1.3, 2.8, 5.3, 7.2]
    excel_dict_isotherms = dict()
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'isotherms_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict_isotherms[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

    width = width_three_columns / inTocm
    height = 1.25 * base_height / inTocm

    fig = plot_pressure_isotherms(df_data, excel_dict_isotherms, lr_list=lr_list, Tad_list=Tad_list, width=width, height=height)
    filename = f'appendix_solid_feanneos_pressure_isotherms_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

    ####################
    # Phase Equilibria #
    ####################

    # reading the data
    folder_to_read = "../computed_files"

    ls_list = ['-', '--', '-.']
    color_list = ['C0', 'C2', 'C1']
    marker_list = ['s', 'v', '^']
    excel_dict_feanneos = dict()
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'phase_equilibria_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict_feanneos[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

    file_database = vle_database_path
    df_vle_md = pd.read_csv(file_database)

    file_database = hvap_database_path
    df_hvap_md = pd.read_csv(file_database)
    df_hvap_md['Uvap*'] = df_hvap_md['Hvap*'] - df_hvap_md['P*_sim'] * (1./df_hvap_md['rhov*'] - 1./df_hvap_md['rhol*'])

    file_database = sle_database_path
    df_sle_md = pd.read_csv(file_database)

    file_database = hmelting_database_path
    df_melting_md = pd.read_csv(file_database)

    dict_md = dict(vle=df_vle_md, hvap=df_hvap_md, sle=df_sle_md, melting=df_melting_md)

    width = width_three_columns / inTocm
    height = 2. * base_height / inTocm
    fig = plot_phase_equilibria_dev_parity(dict_md, excel_dict_feanneos,
                                           width=width, height=height, lr_list=lr_list,
                                           marker_crit=marker_crit, marker_triple=marker_triple,
                                           color_list=color_list, marker_list=marker_list, ls_list=ls_list,
                                           fontsize_annotation=fontsize_annotation, markevery=2)
    filename = f'appendix_solid_feanneos_phase_equilibria_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

    ################################
    # Brown characteristics curves #
    ################################

    file_database = brown_database_path
    df_brown = pd.read_csv(file_database)

    folder_to_read = "../computed_files"
    excel_dict = dict()
    for lambda_r in lr_list:
        filename = f'brown_curves_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

    width = width_three_columns / inTocm
    height = 2. * base_height / inTocm

    fig = plot_brown_curves_virials3(excel_dict, df_brown, lr_list, width=width, height=height,
                                     color_vle=color_vle, color_virial=color_virial,
                                     color_zeno=color_zeno, color_boyle=color_boyle, color_charles=color_charles, color_amagat=color_amagat,
                                     color_solid_fill=color_solid_fill,
                                     marker_crit=marker_crit, marker_triple=marker_triple,
                                     marker_zeno=marker_zeno, marker_boyle=marker_boyle, marker_charles=marker_charles, marker_amagat=marker_amagat,
                                     include_md_data=True)

    filename = f'appendix_solid_feanneos_brown_curves_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

    ###########################
    # Second-order properties #
    ###########################
    file_database = training_database_path
    df_data = pd.read_csv(file_database)

    # reading the data
    folder_to_read = "../computed_files"
    Tad_list = [0.90, 1.1, 1.3, 2.8, 5.3, 7.2]
    excel_dict_isotherms = dict()
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'isotherms_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict_isotherms[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

    width = width_three_columns / inTocm
    height = 3. * base_height / inTocm

    fig = plot_second_isotherms_all(df_data, excel_dict_isotherms, lr_list=lr_list, Tad_list=Tad_list, width=width, height=height, rasterized=True)
    filename = f'appendix_solid_feanneos_second_isotherms_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

############################
# Figures for HPO analysis #
############################
# if plot_figures:
if plot_figures:
    hpo_file_path = "../computed_files/hpo_feanns_eos.xlsx"
    hpo_file = pd.ExcelFile(hpo_file_path)
    df_hpo = hpo_file.parse("HPO_average", index_col=0)
    df_importance = hpo_file.parse("HPO_importance", index_col=0)

    width = width_three_columns / inTocm
    height = base_height / inTocm

    fig = plot_hpo(df_hpo, df_importance, width=width, height=height)
    filename = f'hpo_solid_feanneos.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)
