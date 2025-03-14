import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def plot_database_description(df_diff, df_visc, df_tcond, width=3, height=2, 
                              color_list=['C0', 'C1', 'C2'], edgecolor_list=['#169acf', '#1be093', '#f2a983'],
                              rho_lower=-1e-2, rho_upper=1.25, T_lower=0.6, T_upper=10.1,
                              bins=20, hmax=2500, lrmax=750,
                              rasterized=True, marker='.', markersize=3.0, alpha=0.1, linewidth=0.5):

    df_list = [df_diff, df_visc, df_tcond]

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    topfig, bttmfig = fig.subfigures(2, 1, height_ratios=[1, 0.4])

    gs = topfig.add_gridspec(2, 6, width_ratios=(4, 1, 4, 1, 4, 1), height_ratios=(1, 4))
    rho_ticks = np.linspace(0, 1.2, 5)

    i = 0
    title_list = ['(a) Self-diffusivity', '(b) Shear viscosity', '(c) Thermal conductivity']
    for df, color, edgecolor, title in zip(df_list, color_list, edgecolor_list, title_list):

        ax = topfig.add_subplot(gs[1, 2*i])
        ax_histx = topfig.add_subplot(gs[0, 2*i], sharex=ax)
        ax_histy = topfig.add_subplot(gs[1, 2*i+1], sharey=ax)

        ax_hist_lr = bttmfig.add_subplot(1, 3, i+1)

        ax.tick_params(direction='in')
        ax_histx.tick_params(direction='in')
        ax_histy.tick_params(direction='in')
        ax_hist_lr.tick_params(direction='in')

        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        ax.grid(True)
        ax_histx.grid(True)
        ax_histy.grid(True)
        ax_hist_lr.grid(True)

        # limits
        ax.set_xlim(rho_lower, rho_upper)
        ax.set_ylim(T_lower, T_upper)
        ax_histx.set_ylim([0, hmax])
        ax_histy.set_xlim([0, hmax])
        ax_hist_lr.set_ylim([0, lrmax])

        # setting ticks
        ax.set_xticks(rho_ticks)
        ax_hist_lr.set_xticks(np.arange(7, 35)[::3])
        ax_hist_lr.set_yticks([0, 300, 600])

        # Label
        if i == 0:
            ax.set_ylabel(r'$T^*$')
            # ax_hist_lr.set_ylabel('\# of data points')
            ax_hist_lr.set_ylabel('Count')
        ax.set_xlabel(r'$\rho^*$')
        ax_hist_lr.set_xlabel(r'$\lambda_\mathrm{r}$')

        i += 1

        #####################
        # Plotting the data
        ####################

        scatter = ax.scatter(df['rho*'], df['T*'], color=color, marker=marker, alpha=alpha, clip_on=True, s=markersize)
        scatter.set_rasterized(rasterized)

        ax_histx.hist(df['rho*'], bins=bins, color=color, linewidth=linewidth, edgecolor=edgecolor)
        ax_histy.hist(df['T*'], bins=bins, color=color, linewidth=linewidth, orientation='horizontal', edgecolor=edgecolor)

        ax_histx.set_title(title, x=0.7)

        # lambda_r count
        count_lr = []
        lr_unique = np.unique(df['lr'])
        for lr in lr_unique:
            count_lr.append(np.sum(df['lr'] == lr))
        ax_hist_lr.bar(lr_unique, count_lr, color=color, linewidth=linewidth, edgecolor=edgecolor, width=0.5)

    return fig
