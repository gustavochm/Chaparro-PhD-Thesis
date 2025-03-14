import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.lines import Line2D

plt.ioff()


def plot_isotherms_tp_anns(dict_md_data, dict_isotherms_lrs, width=3, height=3, 
                           lr_list=[12, 16, 20], T_list=[0.9, 1., 1.3, 2.8, 6.0],
                           color_list=['C0', 'C1', 'C2', 'C3', 'C5'], marker_list=['s', 'D', '^', 'o', 'P'],
                           markersize=4, markerevery=3, capsize=2, fontsize_annotation=8,
                           rho_lower=0., rho_upper=1.2, 
                           diff_lower=1e-2, diff_upper=1e2, 
                           visc_lower=8e-2, visc_upper=4e1,
                           tcond_lower=4e-1, tcond_upper=4e1):
    ###########
    rho_ticks = np.linspace(0, 1.2, 5)
    df_diff_md = dict_md_data['self_diffusivity']
    df_visc_md = dict_md_data['shear_viscosity']
    df_tcond_md = dict_md_data['thermal_conductivity']
    kwargs_data = dict(markerfacecolor='white', markersize=markersize, capsize=capsize, zorder=3, linestyle='')

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332, sharex=ax1)
    ax3 = fig.add_subplot(333, sharex=ax1)
    ax4 = fig.add_subplot(334, sharex=ax1)
    ax5 = fig.add_subplot(335, sharex=ax1)
    ax6 = fig.add_subplot(336, sharex=ax1)
    ax7 = fig.add_subplot(337, sharex=ax1)
    ax8 = fig.add_subplot(338, sharex=ax1)
    ax9 = fig.add_subplot(339, sharex=ax1)

    axs_diff = [ax1, ax2, ax3]
    axs_visc = [ax4, ax5, ax6]
    axs_tcond = [ax7, ax8, ax9]

    for lr, ax, letter in zip(lr_list, axs_diff, ['i.', 'ii.', 'iii.']):
        ax.set_title(f'{letter} $\lambda_\mathrm{{r}}={lr:.0f}, \lambda_\mathrm{{a}}=6$')

    # setting labels
    ax1.set_ylabel(r'(a) $D^*$')
    ax4.set_ylabel(r'(b) $\eta^*$')
    ax7.set_ylabel(r'(c) $\kappa^*$')

    # deactivating yticks
    for i in range(1, 3):
        axs_diff[i].tick_params('y', labelleft=False)
        axs_visc[i].tick_params('y', labelleft=False)
        axs_tcond[i].tick_params('y', labelleft=False)   
    # deactivating xticks
    for i in range(3):
        axs_diff[i].tick_params('x', labelbottom=False)
        axs_visc[i].tick_params('x', labelbottom=False)
        axs_tcond[i].set_xticks(rho_ticks)
        axs_tcond[i].set_xlim(rho_lower, rho_upper)
        axs_tcond[i].set_xlabel(r'$\rho^*$')

    for lr, ax_diff, ax_visc, ax_tcond in zip(lr_list, axs_diff, axs_visc, axs_tcond):
        excel_lr = dict_isotherms_lrs[f'lr={lr:.0f}']

        df_diff_md_lr = df_diff_md[df_diff_md['lr'] == lr].reset_index(drop=True)
        df_visc_md_lr = df_visc_md[df_visc_md['lr'] == lr].reset_index(drop=True)
        df_tcond_md_lr = df_tcond_md[df_tcond_md['lr'] == lr].reset_index(drop=True)

        # setting grid
        ax_diff.grid(True)
        ax_visc.grid(True)
        ax_tcond.grid(True)
        # seting labels
        ax_diff.tick_params(direction='in', which='both')
        ax_visc.tick_params(direction='in', which='both')
        ax_tcond.tick_params(direction='in', which='both')
        # setting scale
        ax_diff.set_yscale('log')
        ax_visc.set_yscale('log')
        ax_tcond.set_yscale('log')
        # setting limits
        ax_diff.set_ylim(diff_lower, diff_upper)
        ax_visc.set_ylim(visc_lower, visc_upper)
        ax_tcond.set_ylim(tcond_lower, tcond_upper)

        for Tad, color, marker in zip(T_list, color_list, marker_list):
            # Models
            df = pd.read_excel(excel_lr, sheet_name=f'T={Tad:.2f}')

            ax_diff.plot(df['rho*'], df[f'self_diffusivity_ann'], color=color)
            ax_visc.plot(df['rho*'], df[f'shear_viscosity_ann'], color=color)
            ax_tcond.plot(df['rho*'], df[f'thermal_conductivity_ann'], color=color)

            ax_diff.plot(df['rho*'], df[f'self_diffusivity_ann_res'], color=color, linestyle='--')
            ax_visc.plot(df['rho*'], df[f'shear_viscosity_ann_res'], color=color, linestyle='--')
            ax_tcond.plot(df['rho*'], df[f'thermal_conductivity_ann_res'], color=color, linestyle='--')

            # MD data
            df_diff_md_lr_T = df_diff_md_lr[df_diff_md_lr['T*'] == Tad].reset_index(drop=True)[::markerevery]
            df_visc_md_lr_T = df_visc_md_lr[df_visc_md_lr['T*'] == Tad].reset_index(drop=True)[::markerevery]
            df_tcond_md_lr_T = df_tcond_md_lr[df_tcond_md_lr['T*'] == Tad].reset_index(drop=True)[::markerevery]

            ax_diff.errorbar(df_diff_md_lr_T['rho*'], df_diff_md_lr_T['self_diffusivity'],
                             yerr=1.96*df_diff_md_lr_T['self_diffusivity_std'], 
                             color=color, marker=marker, **kwargs_data, label=f'$T^*={Tad:.1f}$')
            ax_visc.errorbar(df_visc_md_lr_T['rho*'], df_visc_md_lr_T['shear_viscosity'],
                             yerr=1.96*df_visc_md_lr_T['shear_viscosity_std'],
                             color=color, marker=marker, **kwargs_data)
            ax_tcond.errorbar(df_tcond_md_lr_T['rho*'], df_tcond_md_lr_T['thermal_conductivity'],
                              yerr=1.96*df_tcond_md_lr_T['thermal_conductivity_std'],
                              color=color, marker=marker, **kwargs_data)

    custom_legend = []
    for Tad, color, marker in zip(T_list, color_list, marker_list):
        custom_legend.append(Line2D([0], [0], marker=marker, color=color, label=f'$T^*={Tad:.1f}$',
                                    linestyle=kwargs_data['linestyle'], markersize=kwargs_data['markersize'],
                                    markerfacecolor=kwargs_data['markerfacecolor']))
    legend = ax4.legend(handles=custom_legend, ncol=1, frameon=False, fontsize=fontsize_annotation, framealpha=0)
    for text, color in zip(legend.get_texts(), color_list):
        text.set_color(color)

    return fig
