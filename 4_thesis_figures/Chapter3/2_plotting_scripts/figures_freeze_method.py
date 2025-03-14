import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def plot_isobars_freezing(df_heating, df_cooling, width=1, height=1,
                          capsize = 1., markersize = 2, fontsize = 10, fontsize_annotation = 10,
                          color_heating = 'C2', color_cooling = 'C0', rho_min = 0.98, rho_max = 1.22, T_min = 0.9,
                          T_max = 2.7, linewidth = 0.6):

    fig_heating = plt.figure(figsize=(width, height), constrained_layout=True)
    ax = fig_heating.add_subplot(111)
    ax.errorbar(df_heating['rho*'], df_heating['T*'], 
                yerr=df_heating['T*_std'],
                xerr=df_heating['rho*_std'],
                fmt='o', color=color_heating, capsize=capsize, markersize=markersize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([rho_min, rho_max])
    ax.set_ylim([T_min, T_max])
    ax.set_xlabel(r"$\rho^*$", fontsize=fontsize)
    ax.set_ylabel(r"$T^*$", fontsize=fontsize)

    # annotation
    rho_text = 1.07
    Tupper = 2.32
    ax.plot([rho_min, rho_max], [Tupper, Tupper], ':', color=color_heating, linewidth=linewidth)
    annotation_kwargs = dict(ha='center', va='center', color=color_heating, fontsize=fontsize_annotation)
    ax.annotate(r'$T^{*, \mathrm{upp}}, \rho_\mathrm{l}^{*, \mathrm{upp}}, \rho_\mathrm{s}^{*, \mathrm{upp}} $', 
                xy=(rho_text, Tupper-0.1), xytext=(rho_text, Tupper-1), 
                arrowprops=dict(color=color_heating, shrink=0.01, width=0.8, headlength=4., headwidth=4.),
                **annotation_kwargs)

    fig_cooling = plt.figure(figsize=(width, height), constrained_layout=True)
    ax = fig_cooling.add_subplot(111)
    ax.errorbar(df_cooling['rho*'], df_cooling['T*'],
                yerr=df_cooling['T*_std'],
                xerr=df_cooling['rho*_std'],
                fmt='o', color=color_cooling, capsize=capsize, markersize=markersize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([rho_min, rho_max])
    ax.set_ylim([T_min, T_max])
    ax.set_xlabel(r"$\rho^*$", fontsize=fontsize)
    ax.set_ylabel(r"$T^*$", fontsize=fontsize)

    # annotation
    rho_text = 1.13
    Tlower = 1.3
    ax.plot([rho_min, rho_max], [Tlower, Tlower], ':', color=color_cooling, linewidth=linewidth)
    annotation_kwargs = dict(ha='center', va='center', color=color_cooling, fontsize=fontsize_annotation)
    ax.annotate(r'$T^{*, \mathrm{low}}, \rho_\mathrm{l}^{*, \mathrm{low}}, \rho_\mathrm{s}^{*, \mathrm{low}} $', 
                xy=(rho_text, Tlower+0.1), xytext=(rho_text, Tlower+1), 
                arrowprops=dict(color=color_cooling, shrink=0.01, width=0.8, headlength=4., headwidth=4.),
                **annotation_kwargs)
    return fig_heating, fig_cooling
