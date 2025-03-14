import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.lines import Line2D

plt.ioff()


def plot_dilute_limit_ann_model(excel_dilute,  width=3, height=1, 
                                lr_list=[12, 16, 20], color_list=['C0', 'C2', 'C1'],
                                markersize=5, markevery=20, 
                                T_lower=0.6, T_min=10., 
                                fontsize_annotation=8):
    kwargs_ideal = dict(linestyle='', marker='.', markersize=markersize,
                        markevery=markevery, markerfacecolor='white', zorder=3.)

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    title_list = ['(a) Self-diffusivity', '(b) Shear viscosity', '(c) Thermal conductivity']
    label_list = [r'$\rho^* D^*$ ', r'$\eta^*$', r'$\kappa^*$']

    for ax, title, ylabel in zip([ax1, ax2, ax3], title_list, label_list):
        ax.grid(True)
        ax.tick_params(direction='in', which='both')
        ax.set_xlabel(r'$T^*$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(T_lower, T_min)

    y_text_list = [0.34, 0.22, 0.1]
    for lr, color, ytext in zip(lr_list, color_list, y_text_list):
        df_dilute = pd.read_excel(excel_dilute, sheet_name=f'lr={lr:.0f}')
        # self-diffusivity
        ax1.plot(df_dilute['T*'], df_dilute['rho_self_diffusivity_ann_res'], color=color, **kwargs_ideal)
        ax1.plot(df_dilute['T*'], df_dilute['rho_self_diffusivity_ann'], color=color, linestyle='-')

        # shear viscosity
        ax2.plot(df_dilute['T*'], df_dilute['shear_viscosity_ann_res'], color=color, **kwargs_ideal)
        ax2.plot(df_dilute['T*'], df_dilute['shear_viscosity_ann'], color=color, linestyle='-')

        # shear viscosity
        ax3.plot(df_dilute['T*'], df_dilute['thermal_conductivity_ann_res'], color=color, **kwargs_ideal)
        ax3.plot(df_dilute['T*'], df_dilute['thermal_conductivity_ann'], color=color, linestyle='-')

        ax1.text(x=0.95, y=ytext, s=f'$\lambda_\mathrm{{r}}={lr}$', color=color, horizontalalignment='right', transform=ax1.transAxes, fontsize=fontsize_annotation)
    ax1.set_ylim(0.1, 0.9)
    ax2.set_ylim(0., 0.75)
    ax3.set_ylim(0.2, 2.8)
    return fig
