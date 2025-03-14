import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from feanneos import helper_get_alpha


plt.ioff()


def plot_critical_triple_points(width, height, df_feann, df_literature, author_list, marker_list,
                                lr_list=[8, 10, 12, 16, 24], 
                                color_crit='C0', color_triple='C2', color_literature='grey',
                                alpha_lower=0.3, alpha_upper=1.4,
                                T_lower=0.6, T_upper=2.,  
                                Tratio_lower=0.96, Tratio_upper=2.6,
                                rho_lower=0.2, rho_upper=1.2,
                                markersize=3.5, fontsize_annotation=6, linewidth_alpha=0.8):

    df_literature['alpha'] = helper_get_alpha(df_literature['lambda_r'], df_literature['lambda_a'])
    df_feann_training = df_feann[df_feann['lr'] <= 34]
    df_feann_extra = df_feann[df_feann['lr'] > 34]

    # making the plot
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, sharey=ax1)
    ax3 = fig.add_subplot(133, sharey=ax1)

    for ax in [ax1, ax2, ax3]:
        ax.grid(True)
        ax.tick_params(which='both', direction='in')

    ax2.tick_params('y', labelleft=False)

    # labels
    ax1.set_xlabel(r'$T_\mathrm{c}^*$')
    ax2.set_xlabel(r'$T_\mathrm{c}^*/T_\mathrm{tr}^*$')
    ax3.set_xlabel(r'$\rho_\mathrm{c}^*$')
    ax1.set_ylabel(r'$\alpha_\mathrm{vdw}$')

    # titles
    ax1.set_title('(a)')
    ax2.set_title('(b)')
    ax3.set_title('(c)')

    # settting limits
    ax1.set_ylim([alpha_lower, alpha_upper])
    ax1.set_xlim([T_lower, T_upper])
    ax2.set_xlim([Tratio_lower, Tratio_upper])
    ax3.set_xlim([rho_lower, rho_upper])

    # reference lines
    annotation_kwargs = dict(ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", lw=0.6),
                             fontsize=fontsize_annotation)
    lr_end = np.round(df_feann_extra['lr'].values[-1], 1)
    lr_list.append(lr_end)
    for lambda_r in lr_list:
        alpha_vdw = helper_get_alpha(lambda_r, 6.)
        ax1.plot([T_lower, T_upper], [alpha_vdw, alpha_vdw], ':', color='k', linewidth=linewidth_alpha)
        ax2.plot([Tratio_lower, Tratio_upper], [alpha_vdw, alpha_vdw], ':', color='k', linewidth=linewidth_alpha)
        ax3.plot([rho_lower, rho_upper], [alpha_vdw, alpha_vdw], ':', color='k', linewidth=linewidth_alpha)
        x = 2.2
        ax2.annotate(f'$\lambda_\mathrm{{r}}={lambda_r}$, $\lambda_\mathrm{{a}}=6$', xy=(x, alpha_vdw), **annotation_kwargs)

    # FEANN data
    ax1.plot(df_feann_training['Tad_crit'], df_feann_training['alpha'], color=color_crit)
    ax1.plot( df_feann_training['Tad_triple'], df_feann_training['alpha'], color=color_triple)
    ax1.plot(df_feann_extra['Tad_crit'], df_feann_extra['alpha'], color=color_crit, linestyle=':')
    ax1.plot( df_feann_extra['Tad_triple'], df_feann_extra['alpha'], color=color_triple, linestyle=':')
    # 
    ax2.plot(df_feann_training['Tc/Tt'], df_feann_training['alpha'], color='k')
    ax2.plot(df_feann_extra['Tc/Tt'], df_feann_extra['alpha'], color='k', linestyle=':')
    # 
    ax3.plot(df_feann_training['rhoad_crit'], df_feann_training['alpha'], color=color_crit)
    ax3.plot(df_feann_training['rholad_triple'], df_feann_training['alpha'], color=color_triple)
    ax3.plot(df_feann_training['rhosad_triple'], df_feann_training['alpha'], color=color_triple)
    ax3.plot(df_feann_extra['rhoad_crit'], df_feann_extra['alpha'], color=color_crit, linestyle=':')
    ax3.plot(df_feann_extra['rholad_triple'], df_feann_extra['alpha'], color=color_triple, linestyle=':')
    ax3.plot(df_feann_extra['rhosad_triple'], df_feann_extra['alpha'], color=color_triple, linestyle=':')
    # End Point
    ax1.plot(df_feann_extra['Tad_crit'].values[-1], df_feann_extra['alpha'].values[-1], '*', markerfacecolor='white', color='k', clip_on=False)
    ax2.plot(df_feann_extra['Tc/Tt'].values[-1], df_feann_extra['alpha'].values[-1], '*', markerfacecolor='white', color='k', clip_on=False)
    ax3.plot(df_feann_extra['rhoad_crit'].values[-1], df_feann_extra['alpha'].values[-1], '*', markerfacecolor='white', color='k', clip_on=False)
    ax3.plot(df_feann_extra['rhosad_triple'].values[-1], df_feann_extra['alpha'].values[-1], '*', markerfacecolor='white', color='k', clip_on=False)

    # plot data from literature
    for author, marker in zip(author_list, marker_list):
        df_filtered = df_literature[df_literature['AuthorID'] == author]
        kwargs = dict(marker=marker, linestyle='None', color=color_literature, markersize=markersize, markerfacecolor='white')
        ax1.plot(df_filtered['T_crit'], df_filtered['alpha'], **kwargs)
        ax1.plot(df_filtered['T_triple'], df_filtered['alpha'], **kwargs)
        ax2.plot(df_filtered['T_crit']/df_filtered['T_triple'], df_filtered['alpha'], **kwargs)
        ax3.plot(df_filtered['rho_crit'], df_filtered['alpha'], **kwargs)
        ax3.plot(df_filtered['rho_l_triple'], df_filtered['alpha'], **kwargs)
        ax3.plot(df_filtered['rho_s_triple'], df_filtered['alpha'], **kwargs)

    # annotations
    annotation_kwargs = dict(ha='center', va='center', fontsize=fontsize_annotation+2)
    ax1.annotate(r'$T^{*}_\mathrm{c}$', xy=(1.5, 1.2), **annotation_kwargs, color=color_crit)
    ax1.annotate(r'$T^{*}_\mathrm{tr}$', xy=(0.9, 1.2), **annotation_kwargs, color=color_triple)
    #
    ax1.annotate(r'Extrapolation', xy=(0.9, 0.44), xytext=(1.4, 0.44), **annotation_kwargs,
                 arrowprops=dict(facecolor='black', shrink=0.01, width=0.8, headlength=4., headwidth=4.))
    #
    ax3.annotate(r'$\rho^{*}_\mathrm{c}$', xy=(0.4, 1.2), **annotation_kwargs, color=color_crit)
    ax3.annotate(r'$\rho^{*}_\mathrm{tr, l}$', xy=(0.75, 1.2), **annotation_kwargs, color=color_triple)
    ax3.annotate(r'$\rho^{*}_\mathrm{tr, s}$', xy=(1.1, 1.2), **annotation_kwargs, color=color_triple)

    return fig
