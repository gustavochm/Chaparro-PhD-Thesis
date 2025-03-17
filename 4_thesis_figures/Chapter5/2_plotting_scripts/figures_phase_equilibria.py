import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def mie_phase_equilibria_comparison(ExpDataFile, dict_models, width=3, height=2.,
                                    fluid_types=['vle', 'vle_visc'],
                                    solid_types=['vle_sle_sve', 'vle', 'vle_visc'],
                                    rho_lower=-1, rho_upper=45,
                                    T_lower=73, T_upper=200,
                                    P_lower=5e-3, P_upper=1e3,
                                    H_upper=9,
                                    marker_crit='o', marker_triple='s',
                                    marker_exp_vle='^', 
                                    marker_exp_sle='D',
                                    marker_exp_sve='v',
                                    markersize=3, markerevery_sve=3,
                                    linewidth_triple=0.4):
    ######
    rho_factor = 1000
    P_factor = 1e6
    H_factor = 1e3
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234, sharex=ax1, sharey=ax1)
    ax5 = fig.add_subplot(235, sharex=ax2, sharey=ax2)
    ax6 = fig.add_subplot(236, sharex=ax3, sharey=ax3)

    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    axs_fluid = [ax1, ax2, ax3]
    axs_solid = [ax4, ax5, ax6]
    titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for ax, title in zip(axs, titles):
        ax.grid(True)
        ax.tick_params(direction='in', which='both')
        ax.set_title(title)

    for ax in [ax1, ax4]:
        ax.set_xlabel(r'$\rho$ / kmol m$^{-3}$')
        ax.set_ylabel(r'$T$ / K')

    for ax in [ax2, ax5]:
        ax.set_yscale('log')
        ax.set_xlabel(r'$1/T$ / K$^{-1}$')
        ax.set_ylabel(r'$P$ / MPa')

    for ax in [ax3, ax6]:
        ax.set_xlabel(r'$T$ / K')
        ax.set_ylabel(r'$\Delta H$ / kJ mol$^{-1}$')

    kwargs_types = dict()
    kwargs_types['vle'] = {'color': 'C0', 'linestyle': '--'}
    kwargs_types['vle_visc'] = {'color': 'C2', 'linestyle': ':'}
    kwargs_types['vle_sle_sve'] = {'color': 'C1', 'linestyle': '-'}
    kwargs_types['vle_sle_sve_visc'] = {'color': 'C4', 'linestyle': '.-'}
    kwargs_exp_vle = {'marker': marker_exp_vle,'color': 'black', 'linestyle': 'None', 'markerfacecolor': 'white', 'markersize': markersize}
    kwargs_exp_sle = {'marker': marker_exp_sle,'color': 'black', 'linestyle': 'None', 'markerfacecolor': 'white', 'markersize': markersize}
    kwargs_exp_sve = {'marker': marker_exp_sve,'color': 'black', 'linestyle': 'None', 'markerfacecolor': 'white', 'markersize': markersize}

    # VLE only
    for model_type in fluid_types:
        kwargs = kwargs_types[model_type]
        # VLE
        df_vle = dict_models[f'{model_type}_fluid_file'].parse('vle')
        ax1.plot(df_vle['rhov_vle_model']/rho_factor, df_vle['T_vle_model'], **kwargs)
        ax1.plot(df_vle['rhol_vle_model']/rho_factor, df_vle['T_vle_model'], **kwargs)
        ax2.plot(1. / df_vle['T_vle_model'], df_vle['P_vle_model']/P_factor, **kwargs)
        ax3.plot(df_vle['T_vle_model'], df_vle['Hvap_vle_model']/H_factor, **kwargs)
        # Info
        df_info = dict_models[f'{model_type}_fluid_file'].parse('info')

        ax1.plot(df_info['rhoc_model']/rho_factor, df_info['Tc_model'], 
                marker=marker_crit, color=kwargs['color'], markersize=markersize)

        ax2.plot(1/df_info['Tc_model'], df_info['Pc_model']/P_factor, 
                marker=marker_crit, color=kwargs['color'], markersize=markersize)

        ax3.plot(df_info['Tc_model'], [0], 
                marker=marker_crit, color=kwargs['color'], markersize=markersize, clip_on=False, zorder=5.)

    # VLE/SLE/SVE plots
    for model_type in solid_types:
        kwargs = kwargs_types[model_type]
        # VLE
        df_vle = dict_models[f'{model_type}_solid_file'].parse('vle')
        ax4.plot(df_vle['rhov_vle_model']/rho_factor, df_vle['T_vle_model'], **kwargs)
        ax4.plot(df_vle['rhol_vle_model']/rho_factor, df_vle['T_vle_model'], **kwargs)
        ax5.plot(1. / df_vle['T_vle_model'], df_vle['P_vle_model']/P_factor, **kwargs)
        ax6.plot(df_vle['T_vle_model'], df_vle['Hvap_vle_model']/H_factor, **kwargs)
        # SLE
        df_sle = dict_models[f'{model_type}_solid_file'].parse('sle')
        ax4.plot(df_sle['rhos_sle_model']/rho_factor, df_sle['T_sle_model'], **kwargs)
        ax4.plot(df_sle['rhol_sle_model']/rho_factor, df_sle['T_sle_model'], **kwargs)
        ax5.plot(1. / df_sle['T_sle_model'], df_sle['P_sle_model']/P_factor, **kwargs)
        ax6.plot(df_sle['T_sle_model'], df_sle['Hmelting_sle_model']/H_factor, **kwargs)
        # SVE
        df_sve = dict_models[f'{model_type}_solid_file'].parse('sve')
        ax4.plot(df_sve['rhov_sve_model']/rho_factor, df_sve['T_sve_model'], **kwargs)
        ax4.plot(df_sve['rhos_sve_model']/rho_factor, df_sve['T_sve_model'], **kwargs)
        ax5.plot(1. / df_sve['T_sve_model'], df_sve['P_sve_model']/P_factor, **kwargs)
        ax6.plot(df_sve['T_sve_model'], df_sve['Hsub_sve_model']/H_factor, **kwargs)
        # Info
        df_info = dict_models[f'{model_type}_solid_file'].parse('info')
        # Critical point
        ax4.plot(df_info['rhoc_model']/rho_factor, df_info['Tc_model'], 
                marker=marker_crit, color=kwargs['color'], markersize=markersize)

        ax5.plot(1/df_info['Tc_model'], df_info['Pc_model']/P_factor, 
                marker=marker_crit, color=kwargs['color'], markersize=markersize)

        ax6.plot(df_info['Tc_model'], [0], 
                marker=marker_crit, color=kwargs['color'], markersize=markersize, clip_on=False, zorder=5.)
        # Triple point
        ax4.plot(df_info[['rhov_triple_model', 'rhol_triple_model', 'rhos_triple_model']].values[0]/rho_factor,
                df_info[3* ['T_triple_model']].values[0], marker=marker_triple, markersize=markersize,
                **kwargs)
        ax5.plot(1/df_info['T_triple_model'], df_info['P_triple_model']/P_factor, 
                marker=marker_triple, markersize=markersize, **kwargs)
        dHvap_triple = df_info['dUvap_triple_model'] + df_info['P_triple_model'] * (1/df_info['rhov_triple_model'] - 1/df_info['rhol_triple_model'])
        dHmel_triple = df_info['dUmel_triple_model'] + df_info['P_triple_model'] * (1/df_info['rhol_triple_model'] - 1/df_info['rhos_triple_model'])
        dHsub_triple = df_info['dUsub_triple_model'] + df_info['P_triple_model'] * (1/df_info['rhov_triple_model'] - 1/df_info['rhos_triple_model'])
        ax6.plot(df_info[3*['T_triple_model']].values[0], np.hstack([dHsub_triple, dHvap_triple, dHmel_triple])/H_factor, 
                marker=marker_triple, markersize=markersize, color=kwargs['color'], linestyle='None')

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
    rhov_vle = np.asarray(df_component_vle.iloc[:, 2], dtype=np.float64) * 1000. / Mw  # mol/m3
    rhol_vle = np.asarray(df_component_vle.iloc[:, 3], dtype=np.float64) * 1000. / Mw  # mol/m3
    Hvap_vle = np.asarray(df_component_vle.iloc[:, 4], dtype=np.float64) * 1000.  # J/mol
    speed_of_sound_vle = np.asarray(df_component_vle.iloc[:, 5], dtype=np.float64)  # m/s
    thermal_conductivity_vle = np.asarray(df_component_vle.iloc[:, 6], dtype=np.float64)  # W/m K
    viscosity_vle = np.asarray(df_component_vle.iloc[:, 7], dtype=np.float64) # Pa s

    # component SLE
    df_component_sle = pd.read_excel(ExpDataFile, sheet_name='sle')
    Tsle = np.asarray(df_component_sle.iloc[:, 0], dtype=np.float64)  # K
    Psle = np.asarray(df_component_sle.iloc[:, 1], dtype=np.float64)*1000.  # Pa

    # component SVE
    df_component_sve = pd.read_excel(ExpDataFile, sheet_name='sve')
    Tsve = np.asarray(df_component_sve.iloc[:, 0], dtype=np.float64)  # K
    Psve = np.asarray(df_component_sve.iloc[:, 1], dtype=np.float64)*1000.  # Pa
    Hsub_sve = np.asarray(df_component_sve.iloc[:, 2], dtype=np.float64) * 1000.  # J/mol

    # plotting experimental data

    # VLE only plots
    ax1.plot(rhov_vle/rho_factor, Tvle, **kwargs_exp_vle)
    ax1.plot(rhol_vle/rho_factor, Tvle, **kwargs_exp_vle)
    ax2.plot(1/Tvle, Pvle/P_factor, **kwargs_exp_vle)
    ax3.plot(Tvle, Hvap_vle/H_factor, **kwargs_exp_vle)

    # VLE/SLE/SVE plots
    # VLE
    ax4.plot(rhov_vle/rho_factor, Tvle, **kwargs_exp_vle)
    ax4.plot(rhol_vle/rho_factor, Tvle, **kwargs_exp_vle)
    ax5.plot(1/Tvle, Pvle/P_factor, **kwargs_exp_vle)
    ax6.plot(Tvle, Hvap_vle/H_factor, **kwargs_exp_vle)
    # SLE
    ax5.plot(1/Tsle, Psle/P_factor, **kwargs_exp_sle)
    # SVE
    ax5.plot(1/Tsve, Psve/P_factor, **kwargs_exp_sve, markevery=markerevery_sve)
    ax6.plot(Tsve, Hsub_sve/H_factor, **kwargs_exp_sve, markevery=markerevery_sve)

    # Experimental triple point
    ax1.plot([rho_lower, rho_upper], [Ttriple, Ttriple], color='black', linestyle='-', linewidth=linewidth_triple, zorder=0.5)
    ax2.plot([1/Ttriple, 1/Ttriple], [P_lower, P_upper], color='black', linestyle='-', linewidth=linewidth_triple, zorder=0.5)
    ax3.plot([Ttriple, Ttriple], [0, H_upper], color='black', linestyle='-', linewidth=linewidth_triple, zorder=0.5)

    ax4.plot([rho_lower, rho_upper], [Ttriple, Ttriple], color='black', linestyle='-', linewidth=linewidth_triple, zorder=0.5)
    ax5.plot([1/Ttriple, 1/Ttriple], [P_lower, P_upper], color='black', linestyle='-', linewidth=linewidth_triple, zorder=0.5)
    ax6.plot([Ttriple, Ttriple], [0, H_upper], color='black', linestyle='-', linewidth=linewidth_triple, zorder=0.5)

    # set limits
    ax1.set_xlim([rho_lower, rho_upper])
    ax1.set_ylim([T_lower, T_upper])

    ax2.set_xlim([1/T_upper, 1/T_lower])
    ax2.set_ylim([P_lower, P_upper])

    ax3.set_ylim([0, H_upper])
    ax3.set_xlim([T_lower, T_upper])

    return fig
