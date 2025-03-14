import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# figure scripts
from figures_flowchart  import plot_phase_space_flowchart, plot_phase_diagram_flowchart, plot_isotherms_flowchart
from figures_database_description import plot_database_description
from figures_sanity_check import plot_sanity_check
from figures_hpo import plot_hpo
from figures_parity_plot import plot_parity_plot
from figures_error_distribution import plot_error_distribution
from figures_pressure_isotherms import plot_pressure_isotherms
from figures_second_order_isotherms import plot_second_properties_isotherms, plot_lrs_second_isotherms
from figures_brown_curves import plot_brown_curves_second_virial, plot_brown_curves_virials3
from figures_phase_equilibria_dev import plot_phase_equilibria_dev, plot_phase_equilibria_dev_parity
from figures_critical_point import plot_critical_points

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
folder_to_save = '../figures'
os.makedirs(folder_to_save, exist_ok=True)

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

# Database paths
training_database_path = '../../../2_databases/mieparticle-data-training.csv'
virial_database_path = '../../../2_databases/mieparticle-virial-coefficients.csv'
brown_database_path = '../../../2_databases/mieparticle-brown.csv'
vle_database_path  = '../../../2_databases/mieparticle-vle.csv'
hvap_database_path = '../../../2_databases/mieparticle-hvap.csv'

###################################
# Control whether to plot figures #
###################################
plot_figures = True

#########################
# Figures for flowchart #
#########################
if plot_figures:
    lambda_r = 12.
    folder_to_read = '../computed_files'
    filename = f'phase_equilibria_lr{lambda_r:.0f}.xlsx'
    file_to_read = os.path.join(folder_to_read, filename)

    excel_phase_equilibria = pd.ExcelFile(file_to_read)

    filename = f'isotherms_lr{lambda_r:.0f}_flowchart.xlsx'
    file_to_read = os.path.join(folder_to_read, filename)

    excel_isotherms = pd.ExcelFile(file_to_read)

    flowchart_folder = os.path.join(folder_to_save, 'flowchart')
    os.makedirs(flowchart_folder, exist_ok=True)

    # phase_space
    height = 2.7 / inTocm
    width = 2.8 / inTocm
    fig = plot_phase_space_flowchart(excel_phase_equilibria, height=height, width=width)
    filename = f'flowchart_phase_space_lr{lambda_r:.0f}.{format}'
    file_to_save = os.path.join(flowchart_folder, filename)
    fig.savefig(file_to_save, transparent=True, dpi=dpi)

    # phase_diagram
    height = 2.7 / inTocm
    width = 5.3 / inTocm
    fig = plot_phase_diagram_flowchart(excel_phase_equilibria, height=height, width=width, marker_crit=marker_crit)
    filename = f'flowchart_phase_diagram_lr{lambda_r:.0f}.{format}'
    file_to_save = os.path.join(flowchart_folder, filename)
    fig.savefig(file_to_save, transparent=True, dpi=dpi)

    # isotherms
    height = 2.7 / inTocm
    width = 5.3 / inTocm
    fig = plot_isotherms_flowchart(excel_isotherms, height=height, width=width)
    filename = f'flowchart_isotherms_lr{lambda_r:.0f}.{format}'
    file_to_save = os.path.join(flowchart_folder, filename)
    fig.savefig(file_to_save, transparent=True, dpi=dpi)


####################################
# Figures for database description #
####################################
if plot_figures:
    file_database = training_database_path
    df_data = pd.read_csv(file_database)
    df_data_fluid = df_data[df_data['is_fluid']].copy().reset_index(drop=True)

    width = width_three_columns / inTocm
    height = 1.25 * base_height / inTocm
    fig = plot_database_description(df_data_fluid, width=width, height=height)
    filename = f'mie_database_fluid_description.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)


############################
# Figures for HPO analysis #
############################
if plot_figures:
    hpo_file_path = "../computed_files/hpo_feann_eos.xlsx"
    hpo_file = pd.ExcelFile(hpo_file_path)
    df_hpo = hpo_file.parse("HPO_average", index_col=0)
    df_importance = hpo_file.parse("HPO_importance", index_col=0)

    width = width_three_columns / inTocm
    height = base_height / inTocm

    fig = plot_hpo(df_hpo, df_importance, width=width, height=height)
    filename = f'hpo_fluid_feanneos.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

###########################
# Figures for parity plot #
###########################
if plot_figures:
    excel_parity_path = "../computed_files/parity_data_feanneos.xlsx"
    excel_parity = pd.ExcelFile(excel_parity_path)

    width = width_three_columns / inTocm
    height = 3. * base_height / inTocm
    fig = plot_parity_plot(excel_parity, width=width, height=height)
    filename = f'fluid_feanneos_parity_plot.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

    # plot for T-linear model
    excel_parity_path = "../computed_files/parity_data_feanneos_Tlinear.xlsx"
    excel_parity = pd.ExcelFile(excel_parity_path)

    fig = plot_parity_plot(excel_parity, width=width, height=height)
    filename = f'fluid_feanneos_parity_plot_Tlinear.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

#######################################################
# Figures error distribution between T and 1/T models #
#######################################################
if plot_figures:
    # for T-inv model
    excel_parity_path = "../computed_files/parity_data_feanneos.xlsx"
    excel_parity = pd.ExcelFile(excel_parity_path)

    # for T-linear model
    excel_parity_path_Tlinear = "../computed_files/parity_data_feanneos_Tlinear.xlsx"
    excel_parity_Tlinear = pd.ExcelFile(excel_parity_path_Tlinear)

    width = width_three_columns / inTocm
    height = 2 * base_height / inTocm
    fig = plot_error_distribution(excel_parity, excel_parity_path_Tlinear, width=width, height=height)
    filename = f'fluid_feanneos_error_distribution.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

#######################################################
# Figures for pressure isotherms for lr=12, 16 and 20 #
#######################################################
if plot_figures:
    file_database = training_database_path
    df_data = pd.read_csv(file_database)
    df_data_fluid = df_data[df_data['is_fluid']].copy().reset_index(drop=True)

    width = width_three_columns / inTocm
    height = 1.2 * base_height / inTocm
    lr_list = [12, 16, 20]
    Tad_list = [0.8, 1., 1.3, 2.8, 6.0]
    symbol_list = ['s', 'o', 'v', 'd', '^']
    color_list = ['C0', 'C2', 'black', 'C3', 'C1']
    fig = plot_pressure_isotherms(df_data_fluid, lr_list=lr_list, Tad_list=Tad_list, width=width, height=height, 
                                  markersize=markersize, symbol_list=symbol_list, color_list=color_list,
                                  color_solid_fill=color_solid_fill)
    filename = f'fluid_feanneos_pressure_isotherms_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

##########################################################
# Second order properties isotherms for lr=12, 16 and 20 #
##########################################################
if False:
    file_database = training_database_path
    df_data = pd.read_csv(file_database)
    df_data_fluid = df_data[df_data['is_fluid']].copy().reset_index(drop=True)

    lr_list = [12, 16, 20]
    Tad_list = [0.8, 1., 1.3, 2.8, 6.0]
    symbol_list = ['s', 'o', 'v', 'd', '^']
    color_list = ['C0', 'C2', 'black', 'C3', 'C1']
    folder_to_read = "../computed_files"
    for lambda_r in lr_list:
        # Dataframe with the PVT data
        df_data_lr = df_data_fluid.query(f'lr=={lambda_r}').reset_index(drop=True)

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        filename = f'isotherms_lr{lambda_r:.0f}.xlsx'
        filename_saft = f'isotherms_saft_lr{lambda_r:.0f}.xlsx'

        file_to_read = os.path.join(folder_to_read, filename)
        file_to_read_saft = os.path.join(folder_to_read, filename_saft)

        excel_feann = pd.ExcelFile(file_to_read)
        excel_saft = pd.ExcelFile(file_to_read_saft)

        width = width_three_columns / inTocm
        height = 2. * base_height / inTocm
        fig = plot_second_properties_isotherms(df_data_lr, excel_feann, excel_saft, markersize=markersize, Tad_list=Tad_list, width=width, height=height, 
                                               symbol_list=symbol_list, color_list=color_list)
        filename = f'fluid_feanneos_second_properties_isotherms_lr{lambda_r:.0f}.{format}'
        file_to_save = os.path.join(folder_to_save, filename)
        fig.savefig(file_to_save, transparent=False, dpi=dpi)

if plot_figures:
     # reading the data
    file_database = training_database_path
    df_data = pd.read_csv(file_database)
    df_data_fluid = df_data[df_data['is_fluid']].copy().reset_index(drop=True)

    folder_to_read = "../computed_files"
    lr_list = [12, 16, 20]
    Tad_list = [0.8, 1., 1.3, 2.8, 6.0]
    symbol_list = ['s', 'o', 'v', 'd', '^']
    color_list = ['C0', 'C2', 'black', 'C3', 'C1']
    excel_dict_isotherms = dict()
    excel_dict_isotherms_saft = dict()
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'isotherms_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict_isotherms[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

        # SAFT-VR Mie EoS
        filename_saft = f'isotherms_saft_lr{lambda_r:.0f}.xlsx'
        path_to_read_saft = os.path.join(folder_to_read, filename_saft)
        excel_dict_isotherms_saft[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read_saft)

    width = width_three_columns / inTocm
    height = 3.7 * base_height / inTocm
    fig = plot_lrs_second_isotherms(df_data_fluid, excel_dict_isotherms, excel_dict_isotherms_saft,
                                    width=width, height=height,
                                    lr_list=lr_list, Tad_list=Tad_list, symbol_list=symbol_list, color_list=color_list)
    filename = f'fluid_feanneos_second_isotherms_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

###############################################################
# Figures for phase equilibria for lr=12, 16 and 20  and la=6 #
###############################################################
if plot_figures:
    file_database = vle_database_path
    df_vle_md = pd.read_csv(file_database)

    file_database = hvap_database_path
    df_hvap_md = pd.read_csv(file_database)

    # reading the data
    folder_to_read = "../computed_files"
    lr_list = [12, 16, 20]
    excel_dict = dict()
    excel_dict_saft = dict()
    excel_dict_pohl = dict()
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'phase_equilibria_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

        # SAFT-VR Mie
        filename = f'phase_equilibria_saft_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict_saft[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

    width = width_three_columns / inTocm
    height = 2.3 * base_height / inTocm
    color_list = ['C0', 'C2', 'C1']
    marker_list = ['s', 'v', '^']
    fig = plot_phase_equilibria_dev_parity(df_vle_md, df_hvap_md, excel_dict, excel_dict_saft=excel_dict_saft, 
                                           width=width, height=height, lr_list=lr_list,
                                           markersize=markersize, color_list=color_list, marker_list=marker_list, ls_feann=ls_feann, ls_saft=ls_saft,
                                           fontsize_annotation=fontsize_annotation)

    filename = f'fluid_feanneos_phase_equilibria_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

####################################
# Figure of critical points trends #
####################################
if plot_figures:

    # reading the data
    folder_to_read = "../computed_files"

    # FE-ANN data
    filename = 'critical_points.xlsx'
    file_to_read = os.path.join(folder_to_read, filename)
    df_crit_feann = pd.read_excel(file_to_read)
    # SAFT data
    filename = 'critical_points_saft.xlsx'
    file_to_read = os.path.join(folder_to_read, filename)
    df_crit_saft = pd.read_excel(file_to_read)

    filename = '../../../2_databases/crit_triple_ms_literature.csv'
    df_literature = pd.read_csv(filename)
    author_list = np.unique(df_literature['AuthorID'])

    width = width_two_columns / inTocm
    height = base_height / inTocm
    marker_list = ['s', 'o', 'v', '^', 'd']
    lr_list = [8, 10, 12, 16, 24]

    print('Critical point literature: ')
    print(author_list)
    print(marker_list)

    fig = plot_critical_points(width, height, df_crit_feann, df_crit_saft, df_literature, author_list, marker_list,
                               lr_list=lr_list, ls_feann=ls_feann, ls_saft=ls_saft,
                               markersize=markersize, fontsize_annotation=fontsize_annotation-1)
    filename = f'fluid_feanneos_critical_points.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

###########################################################
# Figures for Brown curves for lr=12, 16 and 20  and la=6 #
###########################################################
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
    """
    width = width_three_columns / inTocm
    height = 1.5 * base_height / inTocm
    fig = plot_brown_curves_second_virial(excel_dict, df_brown, lr_list, width=width, height=height, markersize=markersize,
                                          T_lower=0.6, T_upper=10., 
                                          P_lower=1e-3, P_upper=1e2, 
                                          B_lower=-10., B_upper=3,
                                          color_vle=color_vle, color_virial=color_virial,
                                          color_zeno=color_zeno, color_boyle=color_boyle, color_charles=color_charles, color_amagat=color_amagat,
                                          color_solid_fill=color_solid_fill,
                                          marker_crit=marker_crit, marker_zeno=marker_zeno, marker_boyle=marker_boyle, marker_charles=marker_charles, marker_amagat=marker_amagat,
                                          include_md_data=True)
    """
    width = width_three_columns / inTocm
    height = 2. * base_height / inTocm
    fig = plot_brown_curves_virials3(excel_dict, df_brown, lr_list, width=width, height=height, markersize=markersize,
                                          T_lower=0.6, T_upper=10., 
                                          P_lower=1e-3, P_upper=1e2, 
                                          B2_lower=-10., B2_upper=3,
                                          B3_lower=-10., B3_upper=5,
                                          color_vle=color_vle, color_virial=color_virial,
                                          color_zeno=color_zeno, color_boyle=color_boyle, color_charles=color_charles, color_amagat=color_amagat,
                                          color_solid_fill=color_solid_fill,
                                          marker_crit=marker_crit, marker_zeno=marker_zeno, marker_boyle=marker_boyle, marker_charles=marker_charles, marker_amagat=marker_amagat,
                                          include_md_data=True)
    filename = f'fluid_feanneos_brown_curves_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

################################################
# Figures for appendix lr= 10, 18, 26 and la=6 #
################################################

if plot_figures:
    lr_list = [10, 18, 26]
    ##########################
    # PVT pressure isotherms #
    ##########################

    file_database = training_database_path
    df_data = pd.read_csv(file_database)
    df_data_fluid = df_data[df_data['is_fluid']].copy().reset_index(drop=True)

    width = width_three_columns / inTocm
    height = 1.2 * base_height / inTocm

    Tad_list = [0.8, 1., 1.3, 2.8, 6.0]
    symbol_list = ['s', 'o', 'v', 'd', '^']
    color_list = ['C0', 'C2', 'black', 'C3', 'C1']
    fig = plot_pressure_isotherms(df_data_fluid, lr_list=lr_list, Tad_list=Tad_list, width=width, height=height, 
                                  markersize=markersize, symbol_list=symbol_list, color_list=color_list,
                                  color_solid_fill=color_solid_fill)
    filename = f'appendix_fluid_feanneos_pressure_isotherms_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

    ####################
    # Phase Equilibria #
    ####################

    # Phase equilibria
    file_database = vle_database_path
    df_vle_md = pd.read_csv(file_database)

    file_database = hvap_database_path
    df_hvap_md = pd.read_csv(file_database)

    # reading the data
    folder_to_read = "../computed_files"
    excel_dict = dict()
    excel_dict_saft = dict()
    excel_dict_pohl = dict()
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'phase_equilibria_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

        # SAFT-VR Mie
        filename = f'phase_equilibria_saft_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict_saft[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

    width = width_three_columns / inTocm
    height = 2.3 * base_height / inTocm
    color_list = ['C0', 'C2', 'C1']
    marker_list = ['s', 'v', '^']
    fig = plot_phase_equilibria_dev_parity(df_vle_md, df_hvap_md, excel_dict, excel_dict_saft=excel_dict_saft,
                                           width=width, height=height, lr_list=lr_list,
                                           markersize=markersize, color_list=color_list, marker_list=marker_list, ls_feann=ls_feann, ls_saft=ls_saft,
                                           fontsize_annotation=fontsize_annotation, 
                                           T_upper=1.6, T_ticks=[0.6, 0.8, 1.0, 1.2, 1.4, 1.6], Tinv_lower=0.65,
                                           P_upper=1.5e-1, H_upper=9., H_ticks=[0, 3, 6, 9])

    filename = f'appendix_fluid_feanneos_phase_equilibria_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

    ###############################
    # Brown characteristic curves #
    ###############################

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
    fig = plot_brown_curves_virials3(excel_dict, df_brown, lr_list, width=width, height=height, markersize=markersize,
                                          T_lower=0.6, T_upper=10., 
                                          P_lower=1e-3, P_upper=1e2, 
                                          B2_lower=-10., B2_upper=3,
                                          B3_lower=-10., B3_upper=5,
                                          color_vle=color_vle, color_virial=color_virial,
                                          color_zeno=color_zeno, color_boyle=color_boyle, color_charles=color_charles, color_amagat=color_amagat,
                                          color_solid_fill=color_solid_fill,
                                          marker_crit=marker_crit, marker_zeno=marker_zeno, marker_boyle=marker_boyle, marker_charles=marker_charles, marker_amagat=marker_amagat,
                                          include_md_data=True)
    filename = f'appendix_fluid_feanneos_brown_curves_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)

    ###########################
    # Second-order properties #
    ###########################
    file_database = training_database_path
    df_data = pd.read_csv(file_database)
    df_data_fluid = df_data[df_data['is_fluid']].copy().reset_index(drop=True)

    folder_to_read = "../computed_files"
    Tad_list = [0.8, 1., 1.3, 2.8, 6.0]
    symbol_list = ['s', 'o', 'v', 'd', '^']
    color_list = ['C0', 'C2', 'black', 'C3', 'C1']
    excel_dict_isotherms = dict()
    excel_dict_isotherms_saft = dict()
    for lambda_r in lr_list:
        # FE-ANN EoS
        filename = f'isotherms_lr{lambda_r:.0f}.xlsx'
        path_to_read = os.path.join(folder_to_read, filename)
        excel_dict_isotherms[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read)

        # SAFT-VR Mie EoS
        filename_saft = f'isotherms_saft_lr{lambda_r:.0f}.xlsx'
        path_to_read_saft = os.path.join(folder_to_read, filename_saft)
        excel_dict_isotherms_saft[f'lr={lambda_r:.0f}'] = pd.ExcelFile(path_to_read_saft)

    width = width_three_columns / inTocm
    height = 3.7 * base_height / inTocm
    fig = plot_lrs_second_isotherms(df_data_fluid, excel_dict_isotherms, excel_dict_isotherms_saft,
                                    width=width, height=height,
                                    lr_list=lr_list, Tad_list=Tad_list, symbol_list=symbol_list, color_list=color_list)
    filename = f'appendix_fluid_feanneos_second_isotherms_lr{lr_list[0]:.0f}_lr{lr_list[1]:.0f}_lr{lr_list[2]:.0f}.{format}'
    file_to_save = os.path.join(folder_to_save, filename)
    fig.savefig(file_to_save, transparent=False, dpi=dpi)
