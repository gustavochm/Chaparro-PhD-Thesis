import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.colors import LogNorm

plt.ioff()


def plot_hpo(df_hpo, df_importance, width=3., height=1., markersize=8., vmin=1e-4, vmax=1e0):

    dict_style = {'1': {'marker': '^', 'color': 'C4', 's': markersize},
                  '2': {'marker': 's', 'color': 'C1', 's': markersize},
                  '3': {'marker': 'o', 'color': 'C2', 's': markersize},
                  '4': {'marker': 'v', 'color': 'C3', 's': markersize},
                  '5': {'marker': 'D', 'color': 'C0', 's': markersize}}

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    ax3 = fig.add_subplot(131)

    #######
    ax1.grid()
    for layers in [1, 2, 3, 4, 5]:
        df_filtered = df_hpo[df_hpo['params_num_layers'] == layers]
        style = dict_style[str(layers)]
        ax1.scatter(df_filtered['values_0'], df_filtered['params_num_units'], label=f'\# layers = {layers}', **style)

    ax1.set_xscale('log')
    ax1.tick_params(direction='in', which='both')
    ax1.set_xlabel('Objective function')
    ax1.set_ylabel('\# Neurons per layer')
    ax1.legend(ncols=1, frameon=False)

    #######
    ax2.grid()
    of_values = df_hpo['values_0'].to_numpy()
    # norm=LogNorm(vmin=of_values.min(), vmax=of_values.max())
    norm=LogNorm(vmin=vmin, vmax=vmax)
    cmap_plot = ax2.scatter(df_hpo['params_num_layers'], df_hpo['params_num_units'], c=df_hpo['values_0'], cmap='viridis', norm=norm, s=markersize)
    ax2.tick_params(direction='in', which='both')
    ax2.set_xlabel('\# layers')
    cb = fig.colorbar(cmap_plot, ax=ax2)

    ######
    ax3.grid()
    ax3.bar([0, 1, 2], df_importance.values[0], tick_label=['\# Neurons', '\# Layers', 'Seed'], color='C0')
    ax3.tick_params(direction='in', which='both')
    ax3.set_ylabel('Relative importance')
    ax3.set_xlabel('Hyperparameter')

    ax1.set_title('(b)', horizontalalignment='center')
    ax2.set_title('(c)', horizontalalignment='center')
    ax3.set_title('(a)', horizontalalignment='center')
    return fig
