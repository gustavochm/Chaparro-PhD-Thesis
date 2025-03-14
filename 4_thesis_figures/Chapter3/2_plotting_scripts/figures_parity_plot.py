import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def plot_parity_plot(excel_parity, width=3., height=9., color_train='C0', color_test='C2',
                     marker='.', alpha_data=0.4, fontsize_annotation=8, rasterized=True):

    # reading the data from the excel file
    df_model_train = pd.read_excel(excel_parity, sheet_name='model_train')
    df_model_test = pd.read_excel(excel_parity, sheet_name='model_test')
    df_model_virial_train = pd.read_excel(excel_parity, sheet_name='model_virial_train')
    df_model_virial_test = pd.read_excel(excel_parity, sheet_name='model_virial_test')

    df_train = pd.read_excel(excel_parity, sheet_name='PVT_train')
    df_test = pd.read_excel(excel_parity, sheet_name='PVT_test')
    df_virial_train = pd.read_excel(excel_parity, sheet_name='virials_train')
    df_virial_test = pd.read_excel(excel_parity, sheet_name='virials_test')

    # making the figure
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ncols = 3
    nrows = 3
    ax1 = fig.add_subplot(ncols, nrows, 1)
    ax2 = fig.add_subplot(ncols, nrows, 2)
    ax3 = fig.add_subplot(ncols, nrows, 3)
    #
    ax4 = fig.add_subplot(ncols, nrows, 4)
    ax5 = fig.add_subplot(ncols, nrows, 5)
    ax6 = fig.add_subplot(ncols, nrows, 6)
    #
    ax7 = fig.add_subplot(ncols, nrows, 7)
    ax8 = fig.add_subplot(ncols, nrows, 8)
    ax9 = fig.add_subplot(ncols, nrows, 9)

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    title_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)']
    i = 0
    for ax in axs:
        ax.tick_params(direction='in', which='both')
        ax.grid(True)
        ax.set_title(title_list[i])
        i += 1

    property_kwargs = dict()
    property_kwargs['compressibility_factor'] = dict(ax=ax1, lims=[-5, 42], ticks=[0, 10, 20, 30, 40], label=r'$Z$')
    property_kwargs['internal_energy'] = dict(ax=ax2, lims=[-16, 26], ticks=[-10,  0, 10, 20], label=r'$U^*$')
    property_kwargs['B2'] = dict(ax=ax3, lims=[-11, 3], ticks=[-10, -6, -2, 2], label=r'$B_2^*$', plot_index='(c)')
    property_kwargs['isochoric_heat_capacity'] = dict(ax=ax4, lims=[1, 4], ticks=[1, 2, 3, 4], label=r'$C_V^*$')
    property_kwargs['rho_isothermal_compressibility'] = dict(ax=ax5, lims=[-0.5, 4.5], ticks=[0., 1., 2., 3., 4.], label=r'$\rho^*\kappa_T^*$')
    property_kwargs['thermal_pressure_coefficient'] = dict(ax=ax6, lims=[-1., 21], ticks=[0., 5, 10, 15, 20], label=r'$\gamma_V^*$')
    property_kwargs['thermal_expansion_coefficient'] = dict(ax=ax7, lims=[-1., 10], ticks=[0., 3., 6.,  9.], label=r'$\alpha_P^*$')
    property_kwargs['adiabatic_index'] = dict(ax=ax8, lims=[0.5, 10.5], ticks=[2, 4, 6, 8, 10.], label=r'$\gamma$', plot_index='(h)')
    property_kwargs['joule_thomson_coefficient'] = dict(ax=ax9, lims=[-1, 13], ticks=[0., 3, 6., 9., 12.], label=r'$\mu_\mathrm{JT}^*$')
    # property_kwargs['B3'] = dict(ax=ax9, lims=[-3, 3], ticks=[-3, -1, 1, 3], label=r'$B_3^*$')

    for property_name, dict_kwargs in property_kwargs.items():
        ax = dict_kwargs['ax']
        lims = dict_kwargs['lims']
        ticks = dict_kwargs['ticks']
        label = dict_kwargs['label']
        if property_name != 'B2' and property_name != 'B3':
            xlabel = label + r' $(\mathrm{MD})$'
            property_PVT_train = df_train[property_name].to_numpy()
            property_model_train = df_model_train[property_name].to_numpy()
            property_PVT_test = df_test[property_name].to_numpy()
            property_model_test = df_model_test[property_name].to_numpy()
        else: 
            xlabel = label
            property_PVT_train = df_virial_train[property_name].to_numpy()
            property_model_train = df_model_virial_train[property_name].to_numpy()
            property_PVT_test = df_virial_test[property_name].to_numpy()
            property_model_test = df_model_virial_test[property_name].to_numpy()

        scatter1 = ax.scatter(property_PVT_train, property_model_train, marker=marker, color=color_train, alpha=alpha_data)
        scatter2 = ax.scatter(property_PVT_test, property_model_test, marker=marker, color=color_test, alpha=alpha_data)
        scatter1.set_rasterized(rasterized)
        scatter2.set_rasterized(rasterized)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(label + r' $(\mathrm{FE-ANN \ EoS})$')
        ax.plot(lims, lims, color='k')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # annotating the plot
        mse_train = np.nanmean((property_PVT_train - property_model_train)**2)
        mse_test = np.nanmean((property_PVT_test - property_model_test)**2)
        ax.text(0.05, 0.86, f'MSE Train: {mse_train:.1e}', fontsize=fontsize_annotation, transform=ax.transAxes, color=color_train)
        ax.text(0.05, 0.74, f'MSE Test: {mse_test:.1e}', fontsize=fontsize_annotation, transform=ax.transAxes, color=color_test)
    return fig
