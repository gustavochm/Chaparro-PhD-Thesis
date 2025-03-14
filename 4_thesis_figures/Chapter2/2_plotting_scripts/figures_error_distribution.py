import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def plot_error_distribution(excel_parity, excel_parity_Tlinear,
                            width=5., height=2., rho_lower=-1e-2, rho_upper=1.25, T_lower=0.6, T_upper=10.,
                            alpha_data=0.3, alpha_labels=0.8, marksersize=4, colormap='viridis', rasterized=True):

    # Reading the data from the excel file
    df_model_train = pd.read_excel(excel_parity, sheet_name='model_train')
    df_model_test = pd.read_excel(excel_parity, sheet_name='model_test')
    df_train = pd.read_excel(excel_parity, sheet_name='PVT_train')
    df_test = pd.read_excel(excel_parity, sheet_name='PVT_test')

    df_model_train_Tlinear = pd.read_excel(excel_parity_Tlinear, sheet_name='model_train')
    df_model_test_Tlinear = pd.read_excel(excel_parity_Tlinear, sheet_name='model_test')

    property_name = 'compressibility_factor'
    for property_name in ['compressibility_factor', 'internal_energy', 'isochoric_heat_capacity', 'rho_isothermal_compressibility', 'thermal_pressure_coefficient']:

        df_model_train['mse_' + property_name] = np.abs((df_model_train[property_name] - df_train[property_name])**2)
        df_model_test['mse_' + property_name] = np.abs((df_model_test[property_name] - df_test[property_name])**2)

        df_model_train_Tlinear['mse_' + property_name] = np.abs((df_model_train_Tlinear[property_name] - df_train[property_name])**2)
        df_model_test_Tlinear['mse_' + property_name] = np.abs((df_model_test_Tlinear[property_name] - df_test[property_name])**2)

    # Generating arrays with training and test data
    density_Tlinear = np.hstack([df_model_train_Tlinear['density'], df_model_test_Tlinear['density']])
    temperature_Tlinear = np.hstack([df_model_train_Tlinear['temperature'], df_model_test_Tlinear['temperature']])
    mse_Z_Tlinear = np.hstack([df_model_train_Tlinear['mse_compressibility_factor'], df_model_test_Tlinear['mse_compressibility_factor']])
    mse_U_Tlinear = np.hstack([df_model_train_Tlinear['mse_internal_energy'], df_model_test_Tlinear['mse_internal_energy']])
    mse_Cv_Tlinear = np.hstack([df_model_train_Tlinear['mse_isochoric_heat_capacity'], df_model_test_Tlinear['mse_isochoric_heat_capacity']])
    mse_kappaT_Tlinear = np.hstack([df_model_train_Tlinear['mse_rho_isothermal_compressibility'], df_model_test_Tlinear['mse_rho_isothermal_compressibility']])
    mse_gammaV_Tlinear = np.hstack([df_model_train_Tlinear['mse_thermal_pressure_coefficient'], df_model_test_Tlinear['mse_thermal_pressure_coefficient']])

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
    nrows = 2
    ax1 = fig.add_subplot(nrows, ncols, 1)
    ax2 = fig.add_subplot(nrows, ncols, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(nrows, ncols, 3, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(nrows, ncols, 4, sharex=ax1, sharey=ax1)
    ax5 = fig.add_subplot(nrows, ncols, 5, sharex=ax1, sharey=ax1)
    ax6 = fig.add_subplot(nrows, ncols, 6, sharex=ax1, sharey=ax1)
    ax7 = fig.add_subplot(nrows, ncols, 7, sharex=ax1, sharey=ax1)
    ax8 = fig.add_subplot(nrows, ncols, 8, sharex=ax1, sharey=ax1)
    ax9 = fig.add_subplot(nrows, ncols, 9, sharex=ax1, sharey=ax1)
    ax10 = fig.add_subplot(nrows, ncols, 10, sharex=ax1, sharey=ax1)

    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    title_list = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
    bbox_labels = dict(facecolor='white',  edgecolor='white',  boxstyle='round,pad=0.2', alpha=alpha_labels)
    i = 0
    for ax in axs:
        ax.tick_params(direction='in', which='both')
        ax.grid(True)
        ax.text(0.15, 0.87, title_list[i], transform=ax.transAxes, bbox=bbox_labels, horizontalalignment='center', verticalalignment='center')
        i += 1

    # setting up axis limits and ticks
    ax1.tick_params('x', labelbottom=False)
    ax2.tick_params('x', labelbottom=False)
    ax2.tick_params('y', labelleft=False)
    ax3.tick_params('x', labelbottom=False)
    ax3.tick_params('y', labelleft=False)
    ax4.tick_params('x', labelbottom=False)
    ax4.tick_params('y', labelleft=False)
    ax5.tick_params('x', labelbottom=False)
    ax5.tick_params('y', labelleft=False)
    ax7.tick_params('y', labelleft=False)
    ax8.tick_params('y', labelleft=False)
    ax9.tick_params('y', labelleft=False)
    ax10.tick_params('y', labelleft=False)

    ax1.set_xlim([rho_lower, rho_upper])
    ax1.set_ylim([T_lower, T_upper])
    xticks = np.linspace(0., 1.2, 4)
    ax1.set_xticks(xticks)

    # 
    norm = mpl.colors.LogNorm(vmin=1e-4, vmax=1e1)

    order = np.argsort(mse_Z_Tlinear)
    scatter1 = ax1.scatter(density_Tlinear[order], temperature_Tlinear[order], c=mse_Z_Tlinear[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter1.set_rasterized(rasterized)

    order = np.argsort(mse_U_Tlinear)
    scatter2 = ax2.scatter(density_Tlinear[order], temperature_Tlinear[order], c=mse_U_Tlinear[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter2.set_rasterized(rasterized)

    order = np.argsort(mse_Cv_Tlinear)
    scatter3 = ax3.scatter(density_Tlinear[order], temperature_Tlinear[order], c=mse_Cv_Tlinear[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter3.set_rasterized(rasterized)

    order = np.argsort(mse_kappaT_Tlinear)
    scatter4 = ax4.scatter(density_Tlinear[order], temperature_Tlinear[order], c=mse_kappaT_Tlinear[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter4.set_rasterized(rasterized)

    order = np.argsort(mse_gammaV_Tlinear)
    scatter5 = ax5.scatter(density_Tlinear[order], temperature_Tlinear[order], c=mse_gammaV_Tlinear[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter5.set_rasterized(rasterized)

    #######
    order = np.argsort(mse_Z_Tinv)
    scatter2 = ax6.scatter(density_Tinv[order], temperature_Tinv[order], c=mse_Z_Tinv[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter2.set_rasterized(rasterized)

    order = np.argsort(mse_U_Tinv)
    scatter2 = ax7.scatter(density_Tinv[order], temperature_Tinv[order], c=mse_U_Tinv[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter2.set_rasterized(rasterized)

    order = np.argsort(mse_Cv_Tinv)
    scatter8 = ax8.scatter(density_Tinv[order], temperature_Tinv[order], c=mse_Cv_Tinv[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter8.set_rasterized(rasterized)

    order = np.argsort(mse_kappaT_Tinv)
    scatter9 = ax9.scatter(density_Tinv[order], temperature_Tinv[order], c=mse_kappaT_Tinv[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter9.set_rasterized(rasterized)

    order = np.argsort(mse_gammaV_Tinv)
    scatter10 = ax10.scatter(density_Tinv[order], temperature_Tinv[order], c=mse_gammaV_Tinv[order], s=marksersize, alpha=alpha_data, norm=norm, cmap=colormap)
    scatter10.set_rasterized(rasterized)
    # color bar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax = axs, pad=0.02, aspect=30, location='right')

    # labels
    fig.supylabel(r'$T^*$', x=0.05)
    fig.supxlabel(r'$\rho^*$')

    ax1.set_title(r'$Z$')
    ax2.set_title(r'$U^*$')
    ax3.set_title(r'$C_V^*$')
    ax4.set_title(r'$\rho^* \kappa_T^*$')
    ax5.set_title(r'$\gamma_V^*$')

    x_annotation = -0.65
    ax1.text(x_annotation, 0.5, r'$A^\mathrm{*, res}(\alpha_\mathrm{vdw}, \rho^*, T^*)$', transform=ax1.transAxes, verticalalignment='center', rotation=90)
    ax6.text(x_annotation, 0.5, r'$A^\mathrm{*, res}(\alpha_\mathrm{vdw}, \rho^*, 1/T^*)$', transform=ax6.transAxes, verticalalignment='center', rotation=90)
    return fig
