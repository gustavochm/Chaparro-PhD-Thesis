import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def mie_other_properties_comparison(ExpDataFile, dict_models, width=3, height=2.,
                                    solid_types=['vle_sle_sve', 'vle', 'vle_visc'],
                                    T_lower=73, T_upper=200,
                                    marker_crit='o', marker_triple='s',
                                    marker_exp_vle='^', markersize=3, markerevery_sve=3,
                                    linewidth_triple=0.4):
    ######
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, sharex=ax1)
    ax3 = fig.add_subplot(133, sharex=ax1)

    axs = [ax1, ax2, ax3]
    titles = ['(a)', '(b)', '(c)']
    for ax, title in zip(axs, titles):
        ax.grid(True)
        ax.tick_params(direction='in', which='both')
        ax.set_title(title)
        ax.set_xlabel(r'$T$ / K')
        ax.set_xlim([T_lower, T_upper])

    ax1.set_ylabel(r'$w$ / m s$^{-1}$')
    ax2.set_ylabel(r'$\eta$ / Pa s')
    ax3.set_ylabel(r'$\kappa$ / W m$^{-1}$ K$^{-1}$')

    kwargs_types = dict()
    kwargs_types['vle'] = {'color': 'C0', 'linestyle': '--'}
    kwargs_types['vle_visc'] = {'color': 'C2', 'linestyle': ':'}
    kwargs_types['vle_sle_sve'] = {'color': 'C1', 'linestyle': '-'}
    kwargs_types['vle_sle_sve_visc'] = {'color': 'C4', 'linestyle': '.-'}
    kwargs_exp = {'marker': marker_exp_vle, 'color': 'black', 'linestyle': 'None', 'markerfacecolor': 'white', 'markersize': markersize}

    # VLE/SLE/SVE plots
    for model_type in solid_types:
        kwargs = kwargs_types[model_type]
        # VLE
        df_vle = dict_models[f'{model_type}_solid_file'].parse('vle')
        ax1.plot(df_vle['T_vle_model'], df_vle['speed_of_sound_liq_vle_model'], **kwargs)
        ax2.plot(df_vle['T_vle_model'], df_vle['viscl_vle_model'], **kwargs)
        ax3.plot(df_vle['T_vle_model'], df_vle['tcondl_vle_model'], **kwargs)

        # fake triple points
        ax1.plot(df_vle['T_vle_model'][0], df_vle['speed_of_sound_liq_vle_model'][0], 
                marker=marker_triple, linestyle='', markersize=markersize, color=kwargs['color'])
        ax2.plot(df_vle['T_vle_model'][0], df_vle['viscl_vle_model'][0], 
                marker=marker_triple, linestyle='', markersize=markersize, color=kwargs['color'])
        ax3.plot(df_vle['T_vle_model'][0], df_vle['tcondl_vle_model'][0], 
                marker=marker_triple, linestyle='', markersize=markersize, color=kwargs['color'])
        # fake critical points
        ax1.plot(df_vle['T_vle_model'].values[-1], df_vle['speed_of_sound_liq_vle_model'].values[-1], 
                marker=marker_crit, linestyle='', markersize=markersize, color=kwargs['color'])
        ax2.plot(df_vle['T_vle_model'].values[-1], df_vle['viscl_vle_model'].values[-1], 
                marker=marker_crit, linestyle='', markersize=markersize, color=kwargs['color'])
        ax3.plot(df_vle['T_vle_model'].values[-1], df_vle['tcondl_vle_model'].values[-1], 
                marker=marker_crit, linestyle='', markersize=markersize, color=kwargs['color'])

    #####################
    # Experimental data #
    #####################

    # component info
    df_component_info = pd.read_excel(ExpDataFile, sheet_name='info')
    Mw = float(df_component_info['Mw'])  # g/mol
    Tc = float(df_component_info.iloc[0, 1])  # K
    rhoc = float(df_component_info.iloc[0, 2]) * 1000 / Mw  # mol/m3
    Pc = float(df_component_info.iloc[0, 3]) * 1000  # Pa
    Ttriple = float(df_component_info.iloc[0, 4])  # K

    # component VLE
    df_component_vle = pd.read_excel(ExpDataFile, sheet_name='vle')
    Tvle = np.asarray(df_component_vle.iloc[:, 0], dtype=np.float64)  # K
    Pvle = np.asarray(df_component_vle.iloc[:, 1], dtype=np.float64) * 1000.  # Pa
    speed_of_sound_vle = np.asarray(df_component_vle.iloc[:, 5], dtype=np.float64)  # m/s
    thermal_conductivity_vle = np.asarray(df_component_vle.iloc[:, 6], dtype=np.float64)  # W/m K
    viscosity_vle = np.asarray(df_component_vle.iloc[:, 7], dtype=np.float64) # Pa s

    ax1.plot(Tvle, speed_of_sound_vle, **kwargs_exp)
    ax2.plot(Tvle, viscosity_vle, **kwargs_exp)
    ax3.plot(Tvle, thermal_conductivity_vle, **kwargs_exp)
    # triple temperature
    ax1.axvline(x=Ttriple, color='black', linestyle='-', linewidth=linewidth_triple, zorder=0.5)
    ax2.axvline(x=Ttriple, color='black', linestyle='-', linewidth=linewidth_triple, zorder=0.5)
    ax3.axvline(x=Ttriple, color='black', linestyle='-', linewidth=linewidth_triple, zorder=0.5)

    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    return fig
