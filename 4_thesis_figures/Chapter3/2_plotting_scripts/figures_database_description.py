import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def plot_database_description(df_data,
                              width=3., height=1., marker='.', color_fluid='C0', color_solid='C2',
                              alpha=0.1, markersize=3.,  bins_fluid=25, bins_solid=15, hmax=2500, linewidth=0.5,
                              edgecolor_fluid='#0A4C86', edgecolor_solid='#983902', alpha_grid=0.2, rasterized=True):

    df_data_fluid = df_data[df_data['is_fluid']].copy().reset_index(drop=True)
    df_data_solid = df_data[df_data['is_solid']].copy().reset_index(drop=True)

    fig = plt.figure(figsize=(width, height), constrained_layout=True)

    leftfig, centerfig, rightfig = fig.subfigures(1, 3)

    # needed for density axis
    xticks = np.linspace(0., 1.2, 5)
    xlower = -2e-2
    xupper = 1.25

    #################
    #################
    gs = leftfig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                              left=0.1, right=0.9, bottom=0.1, top=0.9,
                              wspace=0.02, hspace=0.02)

    ax1 = leftfig.add_subplot(gs[1, 0])

    scatter_f = ax1.scatter(df_data_fluid['rho*'], df_data_fluid['T*'], marker=marker, color=color_fluid, alpha=alpha, clip_on=True, s=markersize)
    scatter_f.set_rasterized(rasterized)

    scatter_s = ax1.scatter(df_data_solid['rho*'], df_data_solid['T*'], marker=marker, color=color_solid, alpha=alpha, clip_on=True, s=markersize)
    scatter_s.set_rasterized(rasterized)

    yticks = np.linspace(0., 10., 6)
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)

    ax1.set_xlim([xlower, xupper])
    ax1.set_ylim([0.5, 10.5])
    ax1.set_xlabel(r'$\rho^*$')
    ax1.set_ylabel(r'$T^*$')
    ax1.tick_params(direction='in')

    # including histograms
    ax1_histx = leftfig.add_subplot(gs[0, 0], sharex=ax1)
    ax1_histy = leftfig.add_subplot(gs[1, 1], sharey=ax1)
    ax1_histx.grid(visible=True, alpha=alpha_grid)
    ax1_histy.grid(visible=True, alpha=alpha_grid)
    ax1_histx.tick_params(axis="x", labelbottom=False)
    ax1_histy.tick_params(axis="y", labelleft=False)
    ax1_histx.tick_params(direction='in')
    ax1_histy.tick_params(direction='in')

    ax1_histx.hist(df_data_fluid['rho*'], bins=bins_fluid, color=color_fluid,
                   linewidth=linewidth, edgecolor=edgecolor_fluid)
    ax1_histy.hist(df_data_fluid['T*'], bins=bins_fluid, orientation='horizontal', 
                   color=color_fluid, linewidth=linewidth, edgecolor=edgecolor_fluid)

    ax1_histx.hist(df_data_solid['rho*'], bins=bins_solid, color=color_solid,
                   linewidth=linewidth, edgecolor=edgecolor_solid)

    ax1_histy.hist(df_data_solid['T*'], bins=bins_solid, orientation='horizontal',
                   color=color_solid, linewidth=linewidth, edgecolor=edgecolor_solid)

    ax1_histx.set_ylim([0, hmax])
    ax1_histy.set_xlim([0, hmax])

    #################
    #################
    gs = centerfig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                                left=0.1, right=0.9, bottom=0.1, top=0.9,
                                wspace=0.02, hspace=0.02)

    ax2 = centerfig.add_subplot(gs[1, 0])

    scatter_f2 = ax2.scatter(df_data_fluid['rho*'], 1./df_data_fluid['T*'], marker=marker, color=color_fluid, alpha=alpha, clip_on=True, s=markersize)
    scatter_f2.set_rasterized(rasterized)

    scatter_s2 = ax2.scatter(df_data_solid['rho*'], 1./df_data_solid['T*'], marker=marker, color=color_solid, alpha=alpha, clip_on=True, s=markersize)
    scatter_s2.set_rasterized(rasterized)

    ax2.set_xlabel(r'$\rho^*$')
    ax2.set_ylabel(r'$1 / T^*$')
    ax2.tick_params(direction='in')

    ax2.set_xticks(xticks)
    ax2.set_xlim([xlower, xupper])

    # including histograms
    ax2_histx = rightfig.add_subplot(gs[0, 0], sharex=ax2)
    ax2_histy = rightfig.add_subplot(gs[1, 1], sharey=ax2)
    ax2_histx.grid(visible=True, alpha=alpha_grid)
    ax2_histy.grid(visible=True, alpha=alpha_grid)
    ax2_histx.tick_params(axis="x", labelbottom=False)
    ax2_histy.tick_params(axis="y", labelleft=False)
    ax2_histx.tick_params(direction='in')
    ax2_histy.tick_params(direction='in')

    ax2_histx.hist(df_data_fluid['rho*'], bins=bins_fluid, color=color_fluid,
                   linewidth=linewidth, edgecolor=edgecolor_fluid)

    ax2_histy.hist(1./df_data_fluid['T*'], bins=bins_fluid, orientation='horizontal',
                   color=color_fluid, linewidth=linewidth, edgecolor=edgecolor_fluid)

    ax2_histx.hist(df_data_solid['rho*'], bins=bins_solid, color=color_solid,
                   linewidth=linewidth, edgecolor=edgecolor_solid)

    ax2_histy.hist(1./df_data_solid['T*'], bins=bins_solid, orientation='horizontal',
                   color=color_solid, linewidth=linewidth, edgecolor=edgecolor_solid)

    ax2_histx.set_ylim([0, hmax])
    ax2_histy.set_xlim([0, hmax])

    #############################
    #############################

    count_lr = []
    lr_unique = np.unique(df_data['lr'])
    for lr in lr_unique:
        count_lr.append(np.sum(df_data['lr'] == lr))  

    count_lr_solid = []
    for lr in lr_unique:
        count_lr_solid.append(np.sum(df_data_solid['lr'] == lr))  

    ax3 = rightfig.add_subplot(111)
    ax3.bar(lr_unique, count_lr, color=color_fluid, linewidth=linewidth, edgecolor=edgecolor_fluid, width=0.5)
    ax3.bar(lr_unique, count_lr_solid, color=color_solid, linewidth=linewidth, edgecolor=edgecolor_solid, width=0.5)

    ax3.set_xticks(np.arange(7, 35)[::3])
    ax3.set_xlabel(r'$\lambda_\mathrm{r}$')
    ax3.set_ylabel('Number of data points')
    ax3.tick_params(direction='in')
    ax3.grid(visible=True, alpha=alpha_grid)

    ax1_histx.set_title('(a)', horizontalalignment='center') #, x=-0.01, y=1.0)
    ax2_histx.set_title('(b)', horizontalalignment='center') #, x=-0.01, y=1.0)
    ax3.set_title('(c)', horizontalalignment='center') #, x=-0.01, y=1.0)

    return fig
