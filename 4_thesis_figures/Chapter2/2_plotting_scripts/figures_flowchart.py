import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()

def plot_phase_space_flowchart(excel_phase_equilibria, width=1., height=1., fontsize=8):

    df_vle = pd.read_excel(excel_phase_equilibria, sheet_name='vle')
    df_sle = pd.read_excel(excel_phase_equilibria, sheet_name='sle')

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(df_vle['rhov_vle_model'], df_vle['T_vle_model'], color='k')
    ax.plot(df_vle['rhol_vle_model'], df_vle['T_vle_model'], color='k')
    ax.plot(df_sle['rhol_sle_model'], df_sle['T_sle_model'], color='k')

    ax.set_xlim([0., 1.2])
    ax.set_ylim([0.65, 4])
    ax.set_xlabel(r'$\rho$', fontsize=fontsize)
    ax.set_ylabel(r'$T$', fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_phase_diagram_flowchart(excel_phase_equilibria, width=2., height=1.,
                                 T_lower=0.6, T_upper=2., rho_lower=-4e-2, rho_upper=1.2, P_lower=5e-4, P_upper=2e1,
                                 linewidth=0.7,  markersize=1.5, fontsize=8, marker_crit='o'):

    kwargs_vle = {'color': 'k', 'linestyle': '-', 'linewidth': linewidth}
    kwargs_crit = {'color': 'k', 'linestyle': '', 'linewidth': linewidth,
                   'marker': marker_crit, 'markersize':markersize, 'markerfacecolor': 'k'}

    df_info = pd.read_excel(excel_phase_equilibria, sheet_name='info')
    df_vle = pd.read_excel(excel_phase_equilibria, sheet_name='vle')

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax = fig.add_subplot(121)
    ax.set_xlim([rho_lower, rho_upper])
    ax.set_ylim([T_lower, T_upper])
    ax.set_xlabel(r'$\rho$', fontsize=fontsize)
    ax.set_ylabel(r'$T$', fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.plot(df_vle['rhov_vle_model'], df_vle['T_vle_model'], **kwargs_vle)
    ax.plot(df_vle['rhol_vle_model'], df_vle['T_vle_model'], **kwargs_vle)
    ax.plot(df_info['rhocad_model'].values[0], df_info['Tcad_model'].values[0], **kwargs_crit)

    ax2 = fig.add_subplot(122)

    ax2.set_xlim([T_lower, T_upper])
    ax2.set_ylim([P_lower, P_upper])
    ax2.tick_params(direction='in', which='both')
    ax2.set_xlabel(r'$T$', fontsize=fontsize)
    ax2.set_ylabel(r'$P$', fontsize=fontsize)
    ax2.set_yscale('log')
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.plot(df_vle['T_vle_model'], df_vle['P_vle_model'], **kwargs_vle)
    ax2.plot(df_info['Tcad_model'], df_info['Pcad_model'], **kwargs_crit)
    return fig


def plot_isotherms_flowchart(excel_isotherms, height=1., width=2., 
                             rho_lower=0.0, rho_upper=1.2, P_lower=-1, P_upper=1.8e1, cv_lower=1.4, cv_upper=7.,
                             linewidth=0.7, fontsize=8):

    plot_kwars = {'linestyle': '-', 'linewidth': linewidth, 'color': 'k'}

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for sheet_name in excel_isotherms.sheet_names:
        # Tad = float(sheet_name.replace("T=", ""))
        df_isotherm = pd.read_excel(excel_isotherms, sheet_name=sheet_name)
        ax1.plot(df_isotherm['density'], df_isotherm['pressure'], **plot_kwars)
        ax2.plot(df_isotherm['density'], df_isotherm['isochoric_heat_capacity'],  **plot_kwars)

    ax1.tick_params(direction='in', which='both')
    ax1.set_xlabel(r'$\rho$', fontsize=fontsize)
    ax1.set_ylabel(r'$P$', fontsize=fontsize)
    ax1.set_xlim([rho_lower, rho_upper])
    ax1.set_ylim([P_lower, P_upper])
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.tick_params(direction='in', which='both')
    ax2.set_xlabel(r'$\rho$', fontsize=fontsize)
    ax2.set_ylabel(r'$C_V$', fontsize=fontsize)
    ax2.set_xlim([rho_lower, rho_upper])
    ax2.set_ylim([cv_lower, cv_upper])
    ax2.set_xticks([])
    ax2.set_yticks([])
    return fig
