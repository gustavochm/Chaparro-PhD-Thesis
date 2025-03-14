import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d



def plot_phase_equilibria_dev(df_vle_md, df_hvap_md,
                              excel_dict, excel_dict_saft=None, excel_dict_pohl=None,
                              width=3., height=2., lr_list=[12, 16, 20],
                              rho_lower=-5e-2, rho_upper=1.0, T_lower=0.6, T_upper=1.5, Tinv_lower=0.7, Tinv_upper=1.75,
                              P_lower=5e-4, P_upper=1.2e-1, H_lower=0., H_upper=8., 
                              zorder=3.,
                              markersize=3.5, color_list=['C0', 'C2', 'C1'], marker_list=['s', 'v', '^'], ls_feann='-', ls_saft='--', ls_pohl='-.',
                              marker_feann='^', marker_saft='o', marker_pohl='s', include_saft=True, include_pohl=False, alpha_zoom=0.8):


    # figure
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

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
    ax3.set_ylim([H_lower, H_upper])

    ax4.set_xlim([rho_lower, rho_upper])
    ax4.set_ylim([rho_lower, rho_upper])
    ax4.plot([rho_lower, rho_upper], [rho_lower, rho_upper], color='black')

    ax5.set_xlim([P_lower-1e-2, P_upper])
    ax5.set_ylim([P_lower-1e-2, P_upper])
    ax5.plot([P_lower-1e-2, P_upper], [P_lower-1e-2, P_upper], color='black')

    ax6.set_xlim([H_lower, H_upper])
    ax6.set_ylim([H_lower, H_upper])
    ax6.plot([H_lower, H_upper], [H_lower, H_upper], color='black')

    # scales
    ax2.set_yscale('log')

    # labels
    ax1.set_xlabel(r'$\rho^*$')
    ax1.set_ylabel(r'$T^*$')

    ax2.set_xlabel(r'$1/T^*$')
    ax2.set_ylabel(r'$P^{*, \mathrm{vap}}$')

    ax3.set_xlabel(r'$T^*$')
    ax3.set_ylabel(r'$\Delta H^{*, \mathrm{vap}}$')

    ax4.set_xlabel(r'$\rho^*$ (MD)')
    ax4.set_ylabel(r'$\rho^*$ (Predicted)')

    ax5.set_xlabel(r'$P^{*, \mathrm{vap}}$ (MD)')
    ax5.set_ylabel(r'$P^{*, \mathrm{vap}}$ (Predicted)')

    ax6.set_xlabel(r'$\Delta H^{*, \mathrm{vap}}$ (MD)')
    ax6.set_ylabel(r'$\Delta H^{*, \mathrm{vap}}$ (Predicted)')

    # setting ticks
    T_ticks = [0.6, 0.8, 1.0, 1.2, 1.4]
    rho_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    P_ticks = [0, 0.04, 0.08, 0.12]
    H_ticks = [0, 2, 4, 6, 8]

    ax1.set_xticks(rho_ticks)
    ax1.set_yticks(T_ticks)

    ax3.set_xticks(T_ticks)
    ax3.set_yticks(H_ticks)

    ax4.set_xticks(rho_ticks)
    ax4.set_yticks(rho_ticks)

    ax5.set_xticks(P_ticks)
    ax5.set_yticks(P_ticks)

    ax6.set_xticks(H_ticks)
    ax6.set_yticks(H_ticks)

    # Inset axis
    axins = ax4.inset_axes([0.05, 0.55, 0.4, 0.4])
    # sub region of the original image

    x1lim, x2lim = rho_lower, 0.25
    axins.set_xlim(x1lim, x2lim)
    axins.set_ylim(x1lim, x2lim)
    axins.tick_params(direction='in')
    axins.grid(True)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    ax4.indicate_inset_zoom(axins, alpha=alpha_zoom)
    axins.plot([x1lim, x2lim], [x1lim, x2lim], color='black')

    for lambda_r, color, marker in zip(lr_list, color_list, marker_list):

        # MD data
        df_vle_md_lr = df_vle_md[(df_vle_md['lr'] == lambda_r) & (df_vle_md['Tr'] < 1.)].reset_index(drop=True)
        df_hvap_md_lr = df_hvap_md[(df_hvap_md['lr'] == lambda_r)].reset_index(drop=True)

        ax1.plot(df_vle_md_lr['rhol*'], df_vle_md_lr['T*'], color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax1.plot(df_vle_md_lr['rhov*'], df_vle_md_lr['T*'], color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax2.plot(1./df_vle_md_lr['T*'], df_vle_md_lr['P*'], color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax3.plot(df_hvap_md_lr['T*'], df_hvap_md_lr['Hvap*'], color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)

        ##############
        # FE-ANN EoS #
        ##############
        excel_file = excel_dict[f'lr={lambda_r:.0f}']

        # reading the data from the excel file
        df_info_model = pd.read_excel(excel_file, sheet_name='info')
        df_vle_model = pd.read_excel(excel_file, sheet_name='vle')

        # VLE
        rho_vle_envelope = np.hstack([df_vle_model['rhov_vle_model'], df_info_model['rhocad_model'], df_vle_model['rhol_vle_model'][::-1]])
        T_vle_envelope = np.hstack([df_vle_model['T_vle_model'], df_info_model['Tcad_model'], df_vle_model['T_vle_model'][::-1]])
        T_vle = np.hstack([df_vle_model['T_vle_model'], df_info_model['Tcad_model']])
        P_vle = np.hstack([df_vle_model['P_vle_model'], df_info_model['Pcad_model']])
        rhov_vle = np.hstack([df_vle_model['rhov_vle_model'], df_info_model['rhocad_model']])
        rhol_vle = np.hstack([df_vle_model['rhol_vle_model'], df_info_model['rhocad_model']])
        Hvap_vle = np.hstack([df_vle_model['Hvap_vle_model'], 0.])

        ax1.plot(rho_vle_envelope, T_vle_envelope, color=color, linestyle=ls_feann)
        ax2.plot(1./T_vle, P_vle, color=color, linestyle=ls_feann)
        ax3.plot(T_vle, Hvap_vle, color=color, linestyle=ls_feann)

        # Parity plots
        T_vle_md = df_vle_md_lr['T*'].to_numpy()
        rhov_intp = interp1d(T_vle, rhov_vle, kind='cubic')(T_vle_md)
        rhol_intp = interp1d(T_vle, rhol_vle, kind='cubic')(T_vle_md)
        Psat_intp = interp1d(T_vle, P_vle, kind='cubic')(T_vle_md)

        T_hvap_md = df_hvap_md_lr['T*'].to_numpy()
        Hvap_intp = interp1d(T_vle, Hvap_vle, kind='cubic')(T_hvap_md)

        ax4.plot(df_vle_md_lr['rhol*'], rhol_intp, color=color, linestyle='', marker=marker_feann, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax4.plot(df_vle_md_lr['rhov*'], rhov_intp, color=color, linestyle='', marker=marker_feann, markersize=markersize, markerfacecolor='white', zorder=zorder)
        axins.plot(df_vle_md_lr['rhov*'], rhov_intp, color=color, linestyle='', marker=marker_feann, markersize=markersize, markerfacecolor='white', zorder=zorder)

        ax5.plot(df_vle_md_lr['P*'], Psat_intp, color=color, linestyle='', marker=marker_feann, markersize=markersize, markerfacecolor='white', zorder=zorder)

        ax6.plot(df_hvap_md_lr['Hvap*'], Hvap_intp, color=color, linestyle='', marker=marker_feann, markersize=markersize, markerfacecolor='white', zorder=zorder)

        ###################
        # SAFT-VR-Mie EoS #
        ###################
        if include_saft:
            excel_file = excel_dict_saft[f'lr={lambda_r:.0f}']

            # reading the data from the excel file
            df_info_model = pd.read_excel(excel_file, sheet_name='info')
            df_vle_model = pd.read_excel(excel_file, sheet_name='vle')
            df_vle_model.sort_values('T_vle_model', inplace=True)

            # VLE
            rho_vle_envelope = np.hstack([df_vle_model['rhov_vle_model'], df_info_model['rhocad_model'], df_vle_model['rhol_vle_model'][::-1]])
            T_vle_envelope = np.hstack([df_vle_model['T_vle_model'], df_info_model['Tcad_model'], df_vle_model['T_vle_model'][::-1]])
            T_vle = np.hstack([df_vle_model['T_vle_model'], df_info_model['Tcad_model']])
            P_vle = np.hstack([df_vle_model['P_vle_model'], df_info_model['Pcad_model']])
            rhov_vle = np.hstack([df_vle_model['rhov_vle_model'], df_info_model['rhocad_model']])
            rhol_vle = np.hstack([df_vle_model['rhol_vle_model'], df_info_model['rhocad_model']])
            Hvap_vle = np.hstack([df_vle_model['Hvap_vle_model'], 0.])

            ax1.plot(rho_vle_envelope, T_vle_envelope, color=color, linestyle=ls_saft)
            ax2.plot(1./T_vle, P_vle, color=color, linestyle=ls_saft)
            ax3.plot(T_vle, Hvap_vle, color=color, linestyle=ls_saft)

            # Parity plots
            T_vle_md = df_vle_md_lr['T*'].to_numpy()
            rhov_intp = interp1d(T_vle, rhov_vle, kind='cubic')(T_vle_md)
            rhol_intp = interp1d(T_vle, rhol_vle, kind='cubic')(T_vle_md)
            Psat_intp = interp1d(T_vle, P_vle, kind='cubic')(T_vle_md)

            T_hvap_md = df_hvap_md_lr['T*'].to_numpy()
            Hvap_intp = interp1d(T_vle, Hvap_vle, kind='cubic')(T_hvap_md)

            ax4.plot(df_vle_md_lr['rhol*'], rhol_intp, color=color, linestyle='', marker=marker_saft, markersize=markersize, zorder=zorder-1)
            ax4.plot(df_vle_md_lr['rhov*'], rhov_intp, color=color, linestyle='', marker=marker_saft, markersize=markersize, zorder=zorder-1)
            axins.plot(df_vle_md_lr['rhov*'], rhov_intp, color=color, linestyle='', marker=marker_saft, markersize=markersize, zorder=zorder-1)

            ax5.plot(df_vle_md_lr['P*'], Psat_intp, color=color, linestyle='', marker=marker_saft, markersize=markersize, zorder=zorder-1)

            ax6.plot(df_hvap_md_lr['Hvap*'], Hvap_intp, color=color, linestyle='', marker=marker_saft, markersize=markersize, zorder=zorder-1)

        ###################
        # Pohl EoS #
        ###################
        if include_pohl:
            excel_file = excel_dict_pohl[f'lr={lambda_r:.0f}']

            # reading the data from the excel file
            df_info_model = pd.read_excel(excel_file, sheet_name='info')
            df_vle_model = pd.read_excel(excel_file, sheet_name='vle')
            df_vle_model.sort_values('T_vle_model', inplace=True)

            # VLE
            rho_vle_envelope = np.hstack([df_vle_model['rhov_vle_model'], df_info_model['rhocad_model'], df_vle_model['rhol_vle_model'][::-1]])
            T_vle_envelope = np.hstack([df_vle_model['T_vle_model'], df_info_model['Tcad_model'], df_vle_model['T_vle_model'][::-1]])
            T_vle = np.hstack([df_vle_model['T_vle_model'], df_info_model['Tcad_model']])
            P_vle = np.hstack([df_vle_model['P_vle_model'], df_info_model['Pcad_model']])
            rhov_vle = np.hstack([df_vle_model['rhov_vle_model'], df_info_model['rhocad_model']])
            rhol_vle = np.hstack([df_vle_model['rhol_vle_model'], df_info_model['rhocad_model']])
            Hvap_vle = np.hstack([df_vle_model['Hvap_vle_model'], 0.])

            ax1.plot(rho_vle_envelope, T_vle_envelope, color=color, linestyle=ls_pohl)
            ax2.plot(1./T_vle, P_vle, color=color, linestyle=ls_pohl)
            ax3.plot(T_vle, Hvap_vle, color=color, linestyle=ls_pohl)

            # Parity plots
            T_vle_md = df_vle_md_lr['T*'].to_numpy()
            rhov_intp = interp1d(T_vle, rhov_vle, kind='cubic')(T_vle_md)
            rhol_intp = interp1d(T_vle, rhol_vle, kind='cubic')(T_vle_md)
            Psat_intp = interp1d(T_vle, P_vle, kind='cubic')(T_vle_md)

            T_hvap_md = df_hvap_md_lr['T*'].to_numpy()
            Hvap_intp = interp1d(T_vle, Hvap_vle, kind='cubic')(T_hvap_md)

            ax4.plot(df_vle_md_lr['rhol*'], rhol_intp, color=color, linestyle='', marker=marker_pohl, markersize=markersize, zorder=zorder-1)
            ax4.plot(df_vle_md_lr['rhov*'], rhov_intp, color=color, linestyle='', marker=marker_pohl, markersize=markersize, zorder=zorder-1)
            axins.plot(df_vle_md_lr['rhov*'], rhov_intp, color=color, linestyle='', marker=marker_pohl, markersize=markersize, zorder=zorder-1)

            ax5.plot(df_vle_md_lr['P*'], Psat_intp, color=color, linestyle='', marker=marker_pohl, markersize=markersize, zorder=zorder-1)

            ax6.plot(df_hvap_md_lr['Hvap*'], Hvap_intp, color=color, linestyle='', marker=marker_pohl, markersize=markersize, zorder=zorder-1)

    return fig



def plot_phase_equilibria_dev_parity(df_vle_md, df_hvap_md, excel_dict, excel_dict_saft,
                                     width=3., height=2., lr_list=[12, 16, 20],
                                     rho_lower=-5e-2, rho_upper=1.0, T_lower=0.6, T_upper=1.5, Tinv_lower=0.7, Tinv_upper=1.75,
                                     P_lower=5e-4, P_upper=1.2e-1, H_lower=0., H_upper=8., 
                                     zorder=3., markersize=3.5, color_list=['C0', 'C2', 'C1'], marker_list=['s', 'v', '^'], ls_feann='-', ls_saft='--',
                                     fontsize_annotation=8, T_ticks=[0.6, 0.8, 1.0, 1.2, 1.4], H_ticks=[0, 2, 4, 6, 8]):

    # figure
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=(2.1, 2))
    nested_subfigs = subfigs[1].subfigures(1, 3)

    ax1 = subfigs[0].add_subplot(1, 3, 1)
    ax2 = subfigs[0].add_subplot(1, 3, 2)
    ax3 = subfigs[0].add_subplot(1, 3, 3)
    ax4 = nested_subfigs[0].add_subplot(2, 1, 1)
    ax5 = nested_subfigs[1].add_subplot(2, 1, 1)
    ax6 = nested_subfigs[2].add_subplot(2, 1, 1)
    ax7 = nested_subfigs[0].add_subplot(2, 1, 2, sharex=ax4, sharey=ax4)
    ax8 = nested_subfigs[1].add_subplot(2, 1, 2, sharex=ax5, sharey=ax5)
    ax9 = nested_subfigs[2].add_subplot(2, 1, 2, sharex=ax6, sharey=ax6)

    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    label_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for ax, label in zip(axs, label_list):
        ax.tick_params(direction='in', which='both')
        ax.grid(True)
        ax.set_title(label)

    for ax in [ax7, ax8, ax9]:
        ax.tick_params(direction='in', which='both')
        ax.grid(True)

    # limits
    ax1.set_xlim([rho_lower, rho_upper])
    ax1.set_ylim([T_lower, T_upper])

    ax2.set_xlim([Tinv_lower, Tinv_upper])
    ax2.set_ylim([P_lower, P_upper])

    ax3.set_xlim([T_lower, T_upper])
    ax3.set_ylim([H_lower, H_upper])

    ax4.set_xlim([rho_lower, rho_upper])
    ax4.set_ylim([rho_lower, rho_upper])
    ax4.plot([rho_lower, rho_upper], [rho_lower, rho_upper], color='black')
    ax7.plot([rho_lower, rho_upper], [rho_lower, rho_upper], color='black')

    ax5.set_xlim([P_lower-1e-2, P_upper])
    ax5.set_ylim([P_lower-1e-2, P_upper])
    ax5.plot([P_lower-1e-2, P_upper], [P_lower-1e-2, P_upper], color='black')
    ax8.plot([P_lower-1e-2, P_upper], [P_lower-1e-2, P_upper], color='black')

    ax6.set_xlim([H_lower, H_upper])
    ax6.set_ylim([H_lower, H_upper])
    ax6.plot([H_lower, H_upper], [H_lower, H_upper], color='black')
    ax9.plot([H_lower, H_upper], [H_lower, H_upper], color='black')

    ax4.tick_params('x', labelbottom=False)
    ax5.tick_params('x', labelbottom=False)
    ax6.tick_params('x', labelbottom=False)

    # scales
    ax2.set_yscale('log')

    # labels
    ax1.set_xlabel(r'$\rho^*$')
    ax1.set_ylabel(r'$T^*$')

    ax2.set_xlabel(r'$1/T^*$')
    ax2.set_ylabel(r'$P^{*, \mathrm{vap}}$')

    ax3.set_xlabel(r'$T^*$')
    ax3.set_ylabel(r'$\Delta H^{*, \mathrm{vap}}$')

    nested_subfigs[0].supxlabel(r'$\rho^*$ (MD)')
    nested_subfigs[0].supylabel(r'$\rho^*$ (Predicted)')

    nested_subfigs[1].supxlabel(r'$P^{*, \mathrm{vap}}$ (MD)')
    nested_subfigs[1].supylabel(r'$P^{*, \mathrm{vap}}$ (Predicted)')

    nested_subfigs[2].supxlabel(r'$\Delta H^{*, \mathrm{vap}}$ (MD)')
    nested_subfigs[2].supylabel(r'$\Delta H^{*, \mathrm{vap}}$ (Predicted)')

    # setting ticks
    rho_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rho_ticks = [0, 0.25, 0.5, 0.75,  1.0]

    P_ticks = [0, 0.04, 0.08, 0.12]
    

    ax1.set_xticks(rho_ticks)
    ax1.set_yticks(T_ticks)

    ax3.set_xticks(T_ticks)
    ax3.set_yticks(H_ticks)

    ax4.set_xticks(rho_ticks)
    ax4.set_yticks(rho_ticks)

    ax5.set_xticks(P_ticks)
    ax5.set_yticks(P_ticks)

    ax6.set_xticks(H_ticks)
    ax6.set_yticks(H_ticks)

    list_rhol_md = []
    list_rhov_md = []
    list_P_md = []
    list_Hvap_md = []

    list_rhol_feann = []
    list_rhov_feann = []
    list_P_feann = []
    list_Hvap_feann = []

    list_rhol_saft = []
    list_rhov_saft = []
    list_P_saft = []
    list_Hvap_saft = []

    for lambda_r, color, marker in zip(lr_list, color_list, marker_list):

        # MD data
        df_vle_md_lr = df_vle_md[(df_vle_md['lr'] == lambda_r) & (df_vle_md['Tr'] < 1.)].reset_index(drop=True)
        df_hvap_md_lr = df_hvap_md[(df_hvap_md['lr'] == lambda_r)].reset_index(drop=True)

        ax1.plot(df_vle_md_lr['rhol*'], df_vle_md_lr['T*'], color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax1.plot(df_vle_md_lr['rhov*'], df_vle_md_lr['T*'], color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax2.plot(1./df_vle_md_lr['T*'], df_vle_md_lr['P*'], color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax3.plot(df_hvap_md_lr['T*'], df_hvap_md_lr['Hvap*'], color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)

        ##############
        # FE-ANN EoS #
        ##############
        excel_file = excel_dict[f'lr={lambda_r:.0f}']

        # reading the data from the excel file
        df_info_model = pd.read_excel(excel_file, sheet_name='info')
        df_vle_model = pd.read_excel(excel_file, sheet_name='vle')

        # VLE
        rho_vle_envelope = np.hstack([df_vle_model['rhov_vle_model'], df_info_model['rhocad_model'], df_vle_model['rhol_vle_model'][::-1]])
        T_vle_envelope = np.hstack([df_vle_model['T_vle_model'], df_info_model['Tcad_model'], df_vle_model['T_vle_model'][::-1]])
        T_vle = np.hstack([df_vle_model['T_vle_model'], df_info_model['Tcad_model']])
        P_vle = np.hstack([df_vle_model['P_vle_model'], df_info_model['Pcad_model']])
        rhov_vle = np.hstack([df_vle_model['rhov_vle_model'], df_info_model['rhocad_model']])
        rhol_vle = np.hstack([df_vle_model['rhol_vle_model'], df_info_model['rhocad_model']])
        Hvap_vle = np.hstack([df_vle_model['Hvap_vle_model'], 0.])

        ax1.plot(rho_vle_envelope, T_vle_envelope, color=color, linestyle=ls_feann)
        ax2.plot(1./T_vle, P_vle, color=color, linestyle=ls_feann)
        ax3.plot(T_vle, Hvap_vle, color=color, linestyle=ls_feann)

        # Parity plots
        T_vle_md = df_vle_md_lr['T*'].to_numpy()
        rhov_intp = interp1d(T_vle, rhov_vle, kind='cubic')(T_vle_md)
        rhol_intp = interp1d(T_vle, rhol_vle, kind='cubic')(T_vle_md)
        Psat_intp = interp1d(T_vle, P_vle, kind='cubic')(T_vle_md)

        T_hvap_md = df_hvap_md_lr['T*'].to_numpy()
        Hvap_intp = interp1d(T_vle, Hvap_vle, kind='cubic')(T_hvap_md)

        ax4.plot(df_vle_md_lr['rhol*'], rhol_intp, color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax4.plot(df_vle_md_lr['rhov*'], rhov_intp, color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax5.plot(df_vle_md_lr['P*'], Psat_intp, color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax6.plot(df_hvap_md_lr['Hvap*'], Hvap_intp, color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)

        list_rhol_md.append(df_vle_md_lr['rhol*'])
        list_rhov_md.append(df_vle_md_lr['rhov*'])
        list_P_md.append(df_vle_md_lr['P*'])
        list_Hvap_md.append(df_hvap_md_lr['Hvap*'])

        list_rhol_feann.append(rhol_intp)
        list_rhov_feann.append(rhov_intp)
        list_P_feann.append(Psat_intp)
        list_Hvap_feann.append(Hvap_intp)

        ###################
        # SAFT-VR-Mie EoS #
        ###################
        excel_file = excel_dict_saft[f'lr={lambda_r:.0f}']

        # reading the data from the excel file
        df_info_model = pd.read_excel(excel_file, sheet_name='info')
        df_vle_model = pd.read_excel(excel_file, sheet_name='vle')
        df_vle_model.sort_values('T_vle_model', inplace=True)

        # VLE
        rho_vle_envelope = np.hstack([df_vle_model['rhov_vle_model'], df_info_model['rhocad_model'], df_vle_model['rhol_vle_model'][::-1]])
        T_vle_envelope = np.hstack([df_vle_model['T_vle_model'], df_info_model['Tcad_model'], df_vle_model['T_vle_model'][::-1]])
        T_vle = np.hstack([df_vle_model['T_vle_model'], df_info_model['Tcad_model']])
        P_vle = np.hstack([df_vle_model['P_vle_model'], df_info_model['Pcad_model']])
        rhov_vle = np.hstack([df_vle_model['rhov_vle_model'], df_info_model['rhocad_model']])
        rhol_vle = np.hstack([df_vle_model['rhol_vle_model'], df_info_model['rhocad_model']])
        Hvap_vle = np.hstack([df_vle_model['Hvap_vle_model'], 0.])

        ax1.plot(rho_vle_envelope, T_vle_envelope, color=color, linestyle=ls_saft)
        ax2.plot(1./T_vle, P_vle, color=color, linestyle=ls_saft)
        ax3.plot(T_vle, Hvap_vle, color=color, linestyle=ls_saft)

        # Parity plots
        T_vle_md = df_vle_md_lr['T*'].to_numpy()
        rhov_intp = interp1d(T_vle, rhov_vle, kind='cubic')(T_vle_md)
        rhol_intp = interp1d(T_vle, rhol_vle, kind='cubic')(T_vle_md)
        Psat_intp = interp1d(T_vle, P_vle, kind='cubic')(T_vle_md)

        T_hvap_md = df_hvap_md_lr['T*'].to_numpy()
        Hvap_intp = interp1d(T_vle, Hvap_vle, kind='cubic')(T_hvap_md)

        ax7.plot(df_vle_md_lr['rhol*'], rhol_intp, color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax7.plot(df_vle_md_lr['rhov*'], rhov_intp, color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax8.plot(df_vle_md_lr['P*'], Psat_intp, color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)
        ax9.plot(df_hvap_md_lr['Hvap*'], Hvap_intp, color=color, linestyle='', marker=marker, markersize=markersize, markerfacecolor='white', zorder=zorder)

        list_rhol_saft.append(rhol_intp)
        list_rhov_saft.append(rhov_intp)
        list_P_saft.append(Psat_intp)
        list_Hvap_saft.append(Hvap_intp)

    x = 0.05
    y = 0.8
    for ax in [ax4, ax5, ax6]:
        ax.text(x=x, y=y, s='FE-ANN EoS', transform=ax.transAxes, fontsize=fontsize_annotation)
    for ax in [ax7, ax8, ax9]:
        ax.text(x=x, y=y, s='SAFT-VR-Mie EoS', transform=ax.transAxes, fontsize=fontsize_annotation)

    # getting AADs
    list_rhol_md = np.hstack(list_rhol_md)
    list_rhov_md = np.hstack(list_rhov_md)
    list_P_md = np.hstack(list_P_md)
    list_Hvap_md = np.hstack(list_Hvap_md)
    #
    list_rhol_feann = np.hstack(list_rhol_feann)
    list_rhov_feann = np.hstack(list_rhov_feann)
    list_P_feann = np.hstack(list_P_feann)
    list_Hvap_feann = np.hstack(list_Hvap_feann)
    #
    list_rhol_saft = np.hstack(list_rhol_saft)
    list_rhov_saft = np.hstack(list_rhov_saft)
    list_P_saft = np.hstack(list_P_saft)
    list_Hvap_saft = np.hstack(list_Hvap_saft)
    # AAD FE-ANN
    AAD_rhol_feann = 100 * np.mean(np.abs(list_rhol_md - list_rhol_feann) / list_rhol_md)
    AAD_rhov_feann = 100 * np.mean(np.abs(list_rhov_md - list_rhov_feann) / list_rhov_md)
    AAD_P_feann = 100 * np.mean(np.abs(list_P_md - list_P_feann) / list_P_md)
    AAD_Hvap_feann = 100 * np.mean(np.abs(list_Hvap_md - list_Hvap_feann) / list_Hvap_md)
    # AAD SAFT
    AAD_rhol_saft = 100 * np.mean(np.abs(list_rhol_md - list_rhol_saft) / list_rhol_md)
    AAD_rhov_saft = 100 * np.mean(np.abs(list_rhov_md - list_rhov_saft) / list_rhov_md)
    AAD_P_saft = 100 * np.mean(np.abs(list_P_md - list_P_saft) / list_P_md)
    AAD_Hvap_saft = 100 * np.mean(np.abs(list_Hvap_md - list_Hvap_saft) / list_Hvap_md)

    x = 0.98
    y2 = 0.30
    y1 = 0.10
    ax4.text(x, y2, f'MAPE(l): {AAD_rhol_feann:.2f}\%', transform=ax4.transAxes, fontsize=fontsize_annotation, ha='right')
    ax4.text(x, y1, f'MAPE(v): {AAD_rhov_feann:.2f}\%', transform=ax4.transAxes, fontsize=fontsize_annotation, ha='right')
    ax7.text(x, y2, f'MAPE(l): {AAD_rhol_saft:.2f}\%', transform=ax7.transAxes, fontsize=fontsize_annotation, ha='right')
    ax7.text(x, y1, f'MAPE(v): {AAD_rhov_saft:.2f}\%', transform=ax7.transAxes, fontsize=fontsize_annotation, ha='right')
    y = 0.10
    ax5.text(x, y, f'MAPE: {AAD_P_feann:.2f}\%', transform=ax5.transAxes, fontsize=fontsize_annotation, ha='right')
    ax8.text(x, y, f'MAPE: {AAD_P_saft:.2f}\%', transform=ax8.transAxes, fontsize=fontsize_annotation, ha='right')
    ax6.text(x, y, f'MAPE: {AAD_Hvap_feann:.2f}\%', transform=ax6.transAxes, fontsize=fontsize_annotation, ha='right')
    ax9.text(x, y, f'MAPE: {AAD_Hvap_saft:.2f}\%', transform=ax9.transAxes, fontsize=fontsize_annotation, ha='right')

    return fig
