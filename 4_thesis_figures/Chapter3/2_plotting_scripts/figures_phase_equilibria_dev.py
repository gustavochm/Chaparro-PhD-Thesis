import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.interpolate import interp1d


plt.ioff()


def plot_phase_equilibria_dev_parity(dict_md, excel_dict_feanneos,
                                     width=3., height=2., lr_list=[12, 16, 20],
                                     rho_lower=-3e-2, rho_upper=1.2,
                                     T_lower=0.5, T_upper=2.5,
                                     Tinv_lower=0.5, Tinv_upper=1.8,
                                     P_lower=5e-4, P_upper=5e1,
                                     U_lower=0., U_upper=8.,
                                     zorder=3., markersize=3.5,
                                     marker_crit='o', marker_triple='s',
                                     color_list=['C0', 'C2', 'C1'], marker_list=['s', 'v', '^'], ls_list=['-', '--', '-.'],
                                     fontsize_annotation=8, markevery=2):

    zorder_last = zorder + 1.
    df_vle_md = dict_md['vle']
    df_hvap_md = dict_md['hvap']
    df_sle_md = dict_md['sle']
    df_melting_md = dict_md['melting']

    # figure
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 3, height_ratios=(1.5, 1))
    ax1 = fig.add_subplot(grid_spec[0, 0])
    ax2 = fig.add_subplot(grid_spec[0, 1])
    ax3 = fig.add_subplot(grid_spec[0, 2])
    ax4 = fig.add_subplot(grid_spec[1, 0])
    ax5 = fig.add_subplot(grid_spec[1, 1])
    ax6 = fig.add_subplot(grid_spec[1, 2])

    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    label_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for ax, label in zip(axs, label_list):
        ax.tick_params(direction='in', which='both')
        ax.grid(True)
        ax.set_title(label)

    # limits
    ax1.set_xlim([rho_lower, rho_upper])
    ax1.set_ylim([T_lower, T_upper])

    ax2.set_xlim([Tinv_lower, Tinv_upper])
    ax2.set_ylim([P_lower, P_upper])

    ax3.set_xlim([T_lower, T_upper])
    ax3.set_ylim([U_lower, U_upper])

    ax4.set_xlim([rho_lower, rho_upper])
    ax4.set_ylim([rho_lower, rho_upper])
    ax4.plot([rho_lower, rho_upper], [rho_lower, rho_upper], color='black')

    ax5.set_xlim([P_lower, P_upper])
    ax5.set_ylim([P_lower, P_upper])
    ax5.plot([P_lower, P_upper], [P_lower, P_upper], color='black')

    ax6.set_xlim([U_lower, U_upper])
    ax6.set_ylim([U_lower, U_upper])
    ax6.plot([U_lower, U_upper], [U_lower, U_upper], color='black')

    # scales
    ax2.set_yscale('log')

    # labels
    ax1.set_xlabel(r'$\rho^*$')
    ax1.set_ylabel(r'$T^*$')

    ax2.set_xlabel(r'$1/T^*$')
    ax2.set_ylabel(r'$P^{*}$')

    ax3.set_xlabel(r'$T^*$')
    ax3.set_ylabel(r'$\Delta U^{*}$')

    ax4.set_xlabel(r'$\rho^*$ (MD)')
    ax4.set_ylabel(r'$\rho^*$ (Predicted)')

    ax5.set_xlabel(r'$P^{*}$ (MD)')
    ax5.set_ylabel(r'$P^{*}$ (Predicted)')

    ax6.set_xlabel(r'$\Delta U^{*}$ (MD)')
    ax6.set_ylabel(r'$\Delta U^{*}$ (Predicted)')

    # scales and ticks
    ax5.set_xscale('log')
    ax5.set_yscale('log')

    T_ticks = [0.5, 1.0, 1.5, 2.0, 2.5]
    rho_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    P_ticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
    U_ticks = [0, 2, 4, 6, 8]

    ax1.set_xticks(rho_ticks)
    ax1.set_yticks(T_ticks)

    ax3.set_xticks(T_ticks)
    ax3.set_yticks(U_ticks)

    ax4.set_xticks(rho_ticks)
    ax4.set_yticks(rho_ticks)

    ax5.set_xticks(P_ticks)
    ax5.set_yticks(P_ticks)

    ax6.set_xticks(U_ticks)
    ax6.set_yticks(U_ticks)

    # list to store values and compute deviations
    list_rhol_vle_md = []
    list_rhov_vle_md = []
    list_P_vle_md = []
    list_Uvap_vle_md = []
    list_rhol_sle_md = []
    list_rhos_sle_md = []
    list_P_sle_md = []
    list_Umelting_sle_md = []
    #
    list_rhol_vle_feann = []
    list_rhov_vle_feann = []
    list_P_vle_feann = []
    list_Uvap_vle_feann = []
    list_rhol_sle_feann = []
    list_rhos_sle_feann = []
    list_P_sle_feann = []
    list_Umelting_sle_feann = []

    for lambda_r, color, marker, ls in zip(lr_list, color_list, marker_list, ls_list):
        ##############
        # FE-ANN EoS #
        ##############
        excel_file = excel_dict_feanneos[f'lr={lambda_r:.0f}']
        # reading the data from the excel file
        df_info_model = pd.read_excel(excel_file, sheet_name='info')
        df_vle_model = pd.read_excel(excel_file, sheet_name='vle')
        df_sle_model = pd.read_excel(excel_file, sheet_name='sle')
        df_sle_model.dropna(how='all', inplace=True)
        df_sve_model = pd.read_excel(excel_file, sheet_name='sve')
        T_triple_model = df_info_model['T_triple'].values[0]

        # VLE
        ax1.plot(df_vle_model['rhov_vle_model'], df_vle_model['T_vle_model'], color=color, linestyle=ls)
        ax1.plot(df_vle_model['rhol_vle_model'], df_vle_model['T_vle_model'], color=color, linestyle=ls)
        ax2.plot(1./df_vle_model['T_vle_model'], df_vle_model['P_vle_model'], color=color, linestyle=ls)
        ax3.plot(df_vle_model['T_vle_model'], df_vle_model['Uvap_vle_model'], color=color, linestyle=ls)
        # SLE
        ax1.plot(df_sle_model['rhol_sle_model'], df_sle_model['T_sle_model'], color=color, linestyle=ls)
        ax1.plot(df_sle_model['rhos_sle_model'], df_sle_model['T_sle_model'], color=color, linestyle=ls)
        ax2.plot(1./df_sle_model['T_sle_model'], df_sle_model['P_sle_model'], color=color, linestyle=ls)
        ax3.plot(df_sle_model['T_sle_model'], df_sle_model['Umelting_sle_model'], color=color, linestyle=ls)
        # SVE
        ax1.plot(df_sve_model['rhov_sve_model'], df_sve_model['T_sve_model'], color=color, linestyle=ls)
        ax1.plot(df_sve_model['rhos_sve_model'], df_sve_model['T_sve_model'], color=color, linestyle=ls)
        ax2.plot(1./df_sve_model['T_sve_model'], df_sve_model['P_sve_model'], color=color, linestyle=ls)
        ax3.plot(df_sve_model['T_sve_model'], df_sve_model['Usub_sve_model'], color=color, linestyle=ls)

        # Critical Point
        ax1.plot(df_info_model['rhocad_model'].values[0], df_info_model['Tcad_model'].values[0], marker_crit, markersize=markersize, color=color, linestyle='', zorder=zorder_last)
        ax2.plot(1./df_info_model['Tcad_model'].values[0], df_info_model['Pcad_model'].values[0], marker_crit, markersize=markersize, color=color, linestyle='', zorder=zorder_last)
        ax3.plot(df_info_model['Tcad_model'].values[0], 0, marker_crit, markersize=markersize, color=color, linestyle='', zorder=zorder_last, clip_on=False)
        # Triple Point
        ax1.plot(df_info_model[['rhovad_triple', 'rholad_triple', 'rhosad_triple']].values[0],
                 df_info_model[['T_triple', 'T_triple', 'T_triple']].values[0], 
                 color=color, linestyle=ls, marker=marker_triple, markersize=markersize, zorder=zorder_last)
        ax2.plot(1./df_info_model['T_triple'].values[0], df_info_model['P_triple'].values[0],
                 color=color, linestyle='', marker=marker_triple, markersize=markersize, zorder=zorder_last)
        ax3.plot(df_info_model[['T_triple', 'T_triple', 'T_triple']].values[0], 
                 df_info_model[['dUsub_triple', 'dUvap_triple', 'dUmel_triple']].values[0],
                 color=color, linestyle='', marker=marker_triple, markersize=markersize, zorder=zorder_last)     

        ###########
        # MD data #
        ###########
        kwargs_md = dict(color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        # MD data
        df_vle_md_lr = df_vle_md[(df_vle_md['lr'] == lambda_r) & (df_vle_md['Tr'] < 1.) & (df_vle_md['T*'] > T_triple_model) & (df_vle_md['T*'] < T_upper)].reset_index(drop=True)
        df_hvap_md_lr = df_hvap_md[(df_hvap_md['lr'] == lambda_r) & (df_hvap_md['T*'] > T_triple_model)].reset_index(drop=True)
        df_sle_lr = df_sle_md[(df_sle_md['lr'] == lambda_r) & (df_sle_md['T*'] < T_upper)].reset_index(drop=True)
        df_melting_lr = df_melting_md[(df_melting_md['lr'] == lambda_r) & (df_melting_md['T*'] < T_upper)].reset_index(drop=True)

        # VLE
        ax1.plot(df_vle_md_lr['rhol*'], df_vle_md_lr['T*'], markevery=markevery, **kwargs_md)
        ax1.plot(df_vle_md_lr['rhov*'], df_vle_md_lr['T*'], markevery=markevery, **kwargs_md)
        ax2.plot(1./df_vle_md_lr['T*'], df_vle_md_lr['P*'], markevery=markevery, **kwargs_md)
        ax3.plot(df_hvap_md_lr['T*'], df_hvap_md_lr['Uvap*'], markevery=markevery, **kwargs_md)
        # SLE
        ax1.plot(df_sle_lr['rhos*'], df_sle_lr['T*'], markevery=markevery, **kwargs_md)
        ax1.plot(df_sle_lr['rhol*'], df_sle_lr['T*'], markevery=markevery, **kwargs_md)
        ax2.plot(1./df_sle_lr['T*'], df_sle_lr['P*'], markevery=markevery, **kwargs_md)
        ax3.plot(df_melting_lr['T*'], df_melting_lr['dUmelting*'], **kwargs_md)

        ################
        # Parity plots #
        ################
        # VLE
        T_vle_md = df_vle_md_lr['T*'].to_numpy()
        rhov_vle_intp = interp1d(df_vle_model['T_vle_model'], df_vle_model['rhov_vle_model'], kind='cubic', fill_value="extrapolate")(T_vle_md)
        rhol_vle_intp = interp1d(df_vle_model['T_vle_model'], df_vle_model['rhol_vle_model'], kind='cubic', fill_value="extrapolate")(T_vle_md)
        P_vle_intp = interp1d(df_vle_model['T_vle_model'], df_vle_model['P_vle_model'], kind='cubic', fill_value="extrapolate")(T_vle_md)
        # Uvap
        T_Uvap_md = df_hvap_md_lr['T*'].to_numpy()
        Uvap_intp = interp1d(df_vle_model['T_vle_model'], df_vle_model['Uvap_vle_model'], kind='cubic', fill_value="extrapolate")(T_Uvap_md)
        # SLE
        T_sle_md = df_sle_lr['T*'].to_numpy()
        rhos_sle_intp = interp1d(df_sle_model['T_sle_model'], df_sle_model['rhos_sle_model'], kind='cubic', fill_value="extrapolate")(T_sle_md)
        rhol_sle_intp = interp1d(df_sle_model['T_sle_model'], df_sle_model['rhol_sle_model'], kind='cubic', fill_value="extrapolate")(T_sle_md)
        P_sle_intp = interp1d(df_sle_model['T_sle_model'], df_sle_model['P_sle_model'], kind='cubic', fill_value="extrapolate")(T_sle_md)
        # Umelting
        T_melting_md = df_melting_lr['T*'].to_numpy()
        dUmelting_intp = interp1d(df_sle_model['T_sle_model'], df_sle_model['Umelting_sle_model'], kind='cubic', fill_value="extrapolate")(T_melting_md)
        # VLE
        ax4.plot(df_vle_md_lr['rhov*'], rhov_vle_intp, markevery=markevery, **kwargs_md)
        ax4.plot(df_vle_md_lr['rhol*'], rhol_vle_intp, markevery=markevery, **kwargs_md)
        ax5.plot(df_vle_md_lr['P*'], P_vle_intp, **kwargs_md)
        ax6.plot(df_hvap_md_lr['Uvap*'], Uvap_intp, **kwargs_md)
        # SLE
        ax4.plot(df_sle_lr['rhos*'], rhos_sle_intp, markevery=markevery, **kwargs_md)
        ax4.plot(df_sle_lr['rhol*'], rhol_sle_intp, markevery=markevery, **kwargs_md)
        ax5.plot(df_sle_lr['P*'], P_sle_intp, **kwargs_md)
        ax6.plot(df_melting_lr['dUmelting*'], dUmelting_intp, **kwargs_md)

        # appending values
        list_rhol_vle_md.append(df_vle_md_lr['rhol*'])
        list_rhov_vle_md.append(df_vle_md_lr['rhov*'])
        list_P_vle_md.append(df_vle_md_lr['P*'])
        list_Uvap_vle_md.append(df_hvap_md_lr['Uvap*'])
        list_rhol_sle_md.append(df_sle_lr['rhol*'])
        list_rhos_sle_md.append(df_sle_lr['rhos*'])
        list_P_sle_md.append(df_sle_lr['P*'])
        list_Umelting_sle_md.append(df_melting_lr['dUmelting*'])
        #
        list_rhol_vle_feann.append(rhol_vle_intp)
        list_rhov_vle_feann.append(rhov_vle_intp)
        list_P_vle_feann.append(P_vle_intp)
        list_Uvap_vle_feann.append(Uvap_intp)
        list_rhol_sle_feann.append(rhol_sle_intp)
        list_rhos_sle_feann.append(rhos_sle_intp)
        list_P_sle_feann.append(P_sle_intp)
        list_Umelting_sle_feann.append(dUmelting_intp)

    # Getting mean average deviations
    list_rhol_vle_md = np.hstack(list_rhol_vle_md)
    list_rhov_vle_md = np.hstack(list_rhov_vle_md)
    list_P_vle_md = np.hstack(list_P_vle_md)
    list_Uvap_vle_md = np.hstack(list_Uvap_vle_md)
    list_rhol_sle_md = np.hstack(list_rhol_sle_md)
    list_rhos_sle_md = np.hstack(list_rhos_sle_md)
    list_P_sle_md = np.hstack(list_P_sle_md)
    list_Umelting_sle_md = np.hstack(list_Umelting_sle_md)
    #
    list_rhol_vle_feann = np.hstack(list_rhol_vle_feann)
    list_rhov_vle_feann = np.hstack(list_rhov_vle_feann)
    list_P_vle_feann = np.hstack(list_P_vle_feann)
    list_Uvap_vle_feann = np.hstack(list_Uvap_vle_feann)
    list_rhol_sle_feann = np.hstack(list_rhol_sle_feann)
    list_rhos_sle_feann = np.hstack(list_rhos_sle_feann)
    list_P_sle_feann = np.hstack(list_P_sle_feann)
    list_Umelting_sle_feann = np.hstack(list_Umelting_sle_feann)

    # AAD FE-ANN(s) EoS
    AAD_rhol_vle_feann = 100 * np.mean(np.abs(list_rhol_vle_md - list_rhol_vle_feann) / list_rhol_vle_md)
    AAD_rhov_vle_feann = 100 * np.mean(np.abs(list_rhov_vle_md - list_rhov_vle_feann) / list_rhov_vle_md)
    AAD_P_vle_feann = 100 * np.mean(np.abs(list_P_vle_md - list_P_vle_feann) / list_P_vle_md)
    AAD_Uvap_vle_feann = 100 * np.mean(np.abs(list_Uvap_vle_md - list_Uvap_vle_feann) / list_Uvap_vle_md)
    AAD_rhol_sle_feann = 100 * np.mean(np.abs(list_rhol_sle_md - list_rhol_sle_feann) / list_rhol_sle_md)
    AAD_rhos_sle_feann = 100 * np.mean(np.abs(list_rhos_sle_md - list_rhos_sle_feann) / list_rhos_sle_md)
    AAD_P_sle_feann = 100 * np.mean(np.abs(list_P_sle_md - list_P_sle_feann) / list_P_sle_md)
    AAD_Umelting_sle_feann = 100 * np.mean(np.abs(list_Umelting_sle_md - list_Umelting_sle_feann) / list_Umelting_sle_md)

    # coordinates for annotations
    x_vle = 0.95
    y_vle = 0.05
    x_sle = 0.05
    y_sle = 0.92

    # rho-rho parity
    ax4.text(x_vle, y_vle, f'VLE\nMAPE(l): {AAD_rhol_vle_feann:.2f}\%\nMAPE(v): {AAD_rhov_vle_feann:.2f}\%', transform=ax4.transAxes, fontsize=fontsize_annotation, ha='right', va='bottom')
    ax4.text(x_sle, y_sle, f'SLE\nMAPE(l): {AAD_rhol_sle_feann:.2f}\%\nMAPE(s): {AAD_rhos_sle_feann:.2f}\%', transform=ax4.transAxes, fontsize=fontsize_annotation, ha='left', va='top')

    # P-P parity
    ax5.text(x_vle, y_vle, f'VLE\nMAPE: {AAD_P_vle_feann:.2f}\%', transform=ax5.transAxes, fontsize=fontsize_annotation, ha='right', va='bottom')
    ax5.text(x_sle, y_sle, f'SLE\nMAPE: {AAD_P_sle_feann:.2f}\%', transform=ax5.transAxes, fontsize=fontsize_annotation, ha='left', va='top')

    # U-U parity
    ax6.text(x_vle, y_vle, f'VLE\nMAPE: {AAD_Uvap_vle_feann:.2f}\%', transform=ax6.transAxes, fontsize=fontsize_annotation, ha='right', va='bottom')
    ax6.text(x_sle, y_sle, f'SLE\nMAPE: {AAD_Umelting_sle_feann:.2f}\%', transform=ax6.transAxes, fontsize=fontsize_annotation, ha='left', va='top')
    return fig
