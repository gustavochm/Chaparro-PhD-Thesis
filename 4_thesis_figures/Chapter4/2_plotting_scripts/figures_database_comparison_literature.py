import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

plt.ioff()


def plot_comparison_database_literature(data_list, data_lit_list, width=3, height=2,
                                        lambda_r=12, lambda_a=6,
                                        T_list=[1.2, 2.0, 6.0, 10.0], color_list=['C0', 'C2', 'C1', 'C3'],
                                        rho_lower=-2e-2, rho_upper=1.25,
                                        diff_lower=8e-3, diff_upper=3e2,
                                        visc_lower=0, visc_upper=12,
                                        tcond_lower=0, tcond_upper=30,
                                        marker_mine='s',
                                        markersize=3,
                                        capsize=3,
                                        fontsize_annotation=7,
                                        authors_lit_markers={'Michels 1985': 'o', 'Heyes 1988': 'v', 'Heyes 1990': '^', 'Rowley 1997': '<',
                                                             'Vasquez 2004': '>', 'Galliero 2005': 'p', 'Nasrabad 2006': 'P', 'Bugel 2008': '*',
                                                             'Galliero 2009': 'h', 'Baidakov 2011': 'H', 'Baidakov 2014': 'X',
                                                             'Lautenschlaeger 2019': 'D', 'Slepavicius 2023': 'd'}):

    author_list = list(authors_lit_markers.keys())
    kwargs_symbols = { 'markerfacecolor': 'white', 'linestyle': '', 'markersize': markersize, 'clip_on': True}

    rho_ticks = np.linspace(0, 1.2, 5)
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 0.3])

    ax1 = subfigs[0].add_subplot(131)
    ax2 = subfigs[0].add_subplot(132)
    ax3 = subfigs[0].add_subplot(133)

    ax1.set_yscale('log')

    ax1.set_ylim([diff_lower, diff_upper])
    ax2.set_ylim([visc_lower, visc_upper])
    ax3.set_ylim([tcond_lower, tcond_upper])
    ax_list = [ax1, ax2, ax3]
    title_list = ['(a) Self-diffusivity', '(b) Shear viscosity', '(c) Thermal conductivity']
    property_list = ['self_diffusivity', 'shear_viscosity', 'thermal_conductivity']

    ylabel_list = [r'$D^*$ ', r'$\eta^*$', r'$\kappa^*$']
    for ax, df, df_lit, title, ylabel, transport_property in zip(ax_list, data_list, data_lit_list, title_list, ylabel_list, property_list):
        ax.grid(True)
        ax.tick_params(direction='in', which='both')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r'$\rho^*$')
        ax.set_xlim(rho_lower, rho_upper)
        ax.set_xticks(rho_ticks)

        df = df[df['rho*'] <= rho_upper].reset_index(drop=True)
        df_lit = df_lit[df_lit['rho*'] <= rho_upper].reset_index(drop=True)

        df_lr = df[(df['lr'] == lambda_r) & (df['la'] == lambda_a)]
        df_lit_lr = df_lit[(df_lit['lr'] == lambda_r) & (df_lit['la'] == lambda_a)]

        for Tad, color in zip(T_list, color_list):
            df_T = df_lr[df_lr['T*'] == Tad]
            df_lit_T = df_lit_lr[df_lit_lr['T*'] == Tad]

            for author in author_list:
                df_author = df_lit_T[df_lit_T['author_id'] == author]
                ax.plot(df_author['rho*'], df_author[transport_property], color=color, marker=authors_lit_markers[author], **kwargs_symbols)

            ax.errorbar(df_T['rho*'], df_T[transport_property], yerr=df_T[transport_property + '_std'], color=color, marker=marker_mine, capsize=capsize, **kwargs_symbols)

    T_legend = []
    for Tad, color in zip(T_list, color_list):
        T_legend.append(Patch(color=color, label=f'$T^*$ = {Tad}'))
    legend = ax3.legend(handles=T_legend, loc='upper left', ncol=1, frameon=False, fontsize=fontsize_annotation)
    for legend_handle, color in zip(legend.texts, color_list):
        legend_handle.set_color(color)

    custom_legend = []
    for key, value in authors_lit_markers.items():
        custom_legend.append(Line2D([0], [0], marker=value, color='k', label=key, **kwargs_symbols))
    custom_legend.append(Line2D([0], [0], marker=marker_mine, color='k', label='This work', **kwargs_symbols))
    subfigs[1].legend(handles=custom_legend, loc='lower center', ncol=5, frameon=False, fontsize=fontsize_annotation)

    # Adjust subplot layout to make room for the legend
    # fig.tight_layout()
    # fig.subplots_adjust(bottom=0.3)  # Adjust the bottom margin to fit the legend
    return fig
