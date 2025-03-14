import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def plot_error_distribution(excel_parity,
                            width=5., height=1., rho_lower=-1e-2, rho_upper=1.25, T_lower=0.6, T_upper=10.,
                            alpha_data=0.3, marksersize=4, colormap='viridis', rasterized=True):

    # Reading the data from the excel file
    df_model_train = pd.read_excel(excel_parity, sheet_name='model_train')
    df_model_test = pd.read_excel(excel_parity, sheet_name='model_test')
    df_train = pd.read_excel(excel_parity, sheet_name='PVT_train')
    df_test = pd.read_excel(excel_parity, sheet_name='PVT_test')

    property_name = 'compressibility_factor'
    for property_name in ['compressibility_factor', 'internal_energy', 'isochoric_heat_capacity', 'rho_isothermal_compressibility', 'thermal_pressure_coefficient']:

        df_model_train['mse_' + property_name] = np.abs((df_model_train[property_name] - df_train[property_name])**2)
        df_model_test['mse_' + property_name] = np.abs((df_model_test[property_name] - df_test[property_name])**2)

    # Generating arrays with training and test data
    density_Tinv = np.hstack([df_model_train['density'], df_model_test['density']])
    temperature_Tinv = np.hstack([df_model_train['temperature'], df_model_test['temperature']])
    mse_Z_Tinv = np.hstack([df_model_train['mse_compressibility_factor'], df_model_test['mse_compressibility_factor']])
    mse_U_Tinv = np.hstack([df_model_train['mse_internal_energy'], df_model_test['mse_internal_energy']])
    mse_Cv_Tinv = np.hstack([df_model_train['mse_isochoric_heat_capacity'], df_model_test['mse_isochoric_heat_capacity']])
    mse_kappaT_Tinv = np.hstack([df_model_train['mse_rho_isothermal_compressibility'], df_model_test['mse_rho_isothermal_compressibility']])
    mse_gammaV_Tinv = np.hstack([df_model_train['mse_thermal_pressure_coefficient'], df_model_test['mse_thermal_pressure_coefficient']])


    # Making the figure
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ncols = 5
    nrows = 1
    ax1 = fig.add_subplot(nrows, ncols, 1)
    ax2 = fig.add_subplot(nrows, ncols, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(nrows, ncols, 3, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(nrows, ncols, 4, sharex=ax1, sharey=ax1)
    ax5 = fig.add_subplot(nrows, ncols, 5, sharex=ax1, sharey=ax1)

    axs = [ax1, ax2, ax3, ax4, ax5]
    for ax in axs:
        ax.tick_params(direction='in', which='both')
        ax.grid(True)

    # setting up axis limits and ticks
    ax2.tick_params('y', labelleft=False)
    ax3.tick_params('y', labelleft=False)
    ax4.tick_params('y', labelleft=False)
    ax5.tick_params('y', labelleft=False)

    ax1.set_xlim([rho_lower, rho_upper])
    ax1.set_ylim([T_lower, T_upper])
    xticks = np.linspace(0., 1.2, 4)
    ax1.set_xticks(xticks)

    # 
    norm = mpl.colors.LogNorm(vmin=1e-4, vmax=1e1)

    #######
    order = np.argsort(mse_Z_Tinv)
    scatter2 = ax1.scatter(density_Tinv[order], temperature_Tinv[order], c=mse_Z_Tinv[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter2.set_rasterized(rasterized)

    order = np.argsort(mse_U_Tinv)
    scatter2 = ax2.scatter(density_Tinv[order], temperature_Tinv[order], c=mse_U_Tinv[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter2.set_rasterized(rasterized)

    order = np.argsort(mse_Cv_Tinv)
    scatter8 = ax3.scatter(density_Tinv[order], temperature_Tinv[order], c=mse_Cv_Tinv[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter8.set_rasterized(rasterized)

    order = np.argsort(mse_kappaT_Tinv)
    scatter9 = ax4.scatter(density_Tinv[order], temperature_Tinv[order], c=mse_kappaT_Tinv[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter9.set_rasterized(rasterized)

    order = np.argsort(mse_gammaV_Tinv)
    scatter10 = ax5.scatter(density_Tinv[order], temperature_Tinv[order], c=mse_gammaV_Tinv[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter10.set_rasterized(rasterized)
    # color bar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax = axs, pad=0.02, aspect=20, location='right')
    cbar.ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])

    # labels
    # fig.supylabel(r'$T^*$')
    ax1.set_ylabel(r'$T^*$')
    fig.supxlabel(r'$\rho^*$')

    ax1.set_title(r'(a) $Z$')
    ax2.set_title(r'(b) $U^*$')
    ax3.set_title(r'(c) $C_V^*$')
    ax4.set_title(r'(d) $\rho^* \kappa_T^*$')
    ax5.set_title(r'(e) $\gamma_V^*$')
    return fig
