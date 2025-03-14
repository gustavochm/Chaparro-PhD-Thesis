import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def plot_sanity_check(df_rdf_glassy, df_rdf_fluid, df_data_filtered, width=3., height=1.,
                       marker='s', color_delete='C2', color_normal='C0', markersize=6.):

    ## plot
    fig = plt.figure(figsize = (width, height), constrained_layout=True)
    ax1 = fig.add_subplot(121)
    ax1.plot(df_rdf_glassy['c_myRDF[1]'], df_rdf_glassy['c_myRDF[2]'], '-', color=color_delete)
    ax1.annotate('Metastable state', xy=(1.2, 2.8), xytext=(2.2, 3.1), color=color_delete, verticalalignment='center',
                 arrowprops=dict(arrowstyle='->', color=color_delete))

    ax1.plot(df_rdf_fluid['c_myRDF[1]'], df_rdf_fluid['c_myRDF[2]'], '-', color=color_normal)
    ax1.annotate('Liquid state', xy=(1.2, 2.1), xytext=(2.6, 2.5), color=color_normal, verticalalignment='center',
                 arrowprops=dict(arrowstyle='->', color=color_normal))
    ax1.set_xlim([0, 5])
    ax1.set_ylim([0, 3.5])
    # ax1.set_xticks([])
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_xlabel(r'$r/\sigma$')
    ax1.set_ylabel(r'$g(r)$')
    ax1.tick_params(direction='in')

    ax2 = fig.add_subplot(122)
    where_max = df_data_filtered['Cv_nvt'].idxmax()
    where_normal = [x for x in df_data_filtered['Cv_nvt'].index if x != where_max]
    ax2.plot(df_data_filtered.loc[where_normal, 'rho*'], df_data_filtered.loc[where_normal, 'Cv_nvt'], marker, color=color_normal, markersize=markersize)
    ax2.annotate('Normal', xy=(0.9, 2.6), xytext=(0.95, 2.45), color=color_normal, verticalalignment='center',
                 arrowprops=dict(arrowstyle='->', color=color_normal))
    ax2.annotate('', xy=(1.05, 2.7), xytext=(1.02, 2.5), color=color_normal, verticalalignment='center',
                 arrowprops=dict(arrowstyle='->', color=color_normal))

    ax2.plot(df_data_filtered.loc[where_max, 'rho*'], df_data_filtered.loc[where_max, 'Cv_nvt'], marker, color=color_delete, markersize=markersize)
    ax2.annotate('Outlier', xy=(0.92, 3.26), xytext=(0.82, 3.26), color=color_delete, verticalalignment='center',
                 arrowprops=dict(arrowstyle='->', color=color_delete))
    ax2.set_xlabel(r"$\rho^*$")
    ax2.set_ylabel(r"$C_V^*$")
    ax2.set_ylim([2.3, 3.5])
    ax2.set_xlim([0.8, 1.1])
    ax2.tick_params(direction='in')

    ax1.set_title('(a)', horizontalalignment='center')
    ax2.set_title('(b)', horizontalalignment='center')
    return fig
