import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feanneos import helper_get_alpha


def plot_critical_points(width, height, df_crit_feann, df_crit_saft, df_literature, author_list, marker_list,
                         lr_list=[8, 10, 12, 16, 24], 
                         ls_feann='-', ls_saft='--',
                         color_feann='C0', color_saft='C2', color_literature='grey',
                         alpha_lower=0.55, alpha_upper=1.45, T_lower=0.8, T_upper=2.,  rho_lower=0.25, rho_upper=0.5,
                         markersize=3.5, fontsize_annotation=7, linewidth_alpha=0.8):

    df_literature['alpha'] = helper_get_alpha(df_literature['lambda_r'], df_literature['lambda_a'])

    # making the plot
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharey=ax1)

    for ax in [ax1, ax2]:
        ax.grid(True)
        ax.tick_params(which='both', direction='in')

    ax2.tick_params('y', labelleft=False)

    # labels
    ax1.set_xlabel(r'$T_\mathrm{c}^*$')
    ax2.set_xlabel(r'$\rho_\mathrm{c}^*$')
    ax1.set_ylabel(r'$\alpha_\mathrm{vdw}$')

    # titles
    ax1.set_title('(a)')
    ax2.set_title('(b)')

    annotation_kwargs = dict(ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", lw=0.6),
                             fontsize=fontsize_annotation)

    for lambda_r in lr_list:
        alpha_vdw = helper_get_alpha(lambda_r, 6.)
        ax1.plot([T_lower, T_upper], [alpha_vdw, alpha_vdw], ':', color='k', linewidth=linewidth_alpha)
        ax2.plot([rho_lower, rho_upper], [alpha_vdw, alpha_vdw], ':', color='k', linewidth=linewidth_alpha)
        x = 0.45
        ax2.annotate(f'$\lambda_\mathrm{{r}}={lambda_r}$, $\lambda_\mathrm{{a}}=6$', xy=(x, alpha_vdw), **annotation_kwargs)

    # plot data
    ax1.plot(df_crit_feann['Tcad'], df_crit_feann['alpha'], linestyle=ls_feann, color=color_feann)
    ax1.plot(df_crit_saft['Tcad'], df_crit_saft['alpha'], linestyle=ls_saft, color=color_saft)

    ax2.plot(df_crit_feann['rhocad'], df_crit_feann['alpha'], linestyle=ls_feann, color=color_feann)
    ax2.plot(df_crit_saft['rhocad'], df_crit_saft['alpha'], linestyle=ls_saft, color=color_saft)

    ax1.set_ylim([alpha_lower, alpha_upper])
    ax1.set_xlim([T_lower, T_upper])
    ax2.set_xlim([rho_lower, rho_upper])

    for author, marker in zip(author_list, marker_list):
        df_filtered = df_literature[df_literature['AuthorID'] == author]
        kwargs = dict(marker=marker, linestyle='None', color=color_literature, markersize=markersize, markerfacecolor='white')
        ax1.plot(df_filtered['T_crit'], df_filtered['alpha'], **kwargs)
        ax2.plot(df_filtered['rho_crit'], df_filtered['alpha'], **kwargs)

    return fig
