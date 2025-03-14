import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def plot_second_isotherms(df_data, excel_dict_isotherms, lr_list=[12, 16, 20], Tad_list=[0.8, 1., 1.3, 2.8, 6.0],
                          width=3, height=5,
                          rho_lower=-1e-2, rho_upper=1.2, T_lower = 0.6, T_upper = 10.0,
                          alpha_data=0.3, markersize=3., rasterized=False):

    kappaT_lower = -1
    kappaT_upper = 4
    #
    gammaV_lower = -1
    gammaV_upper = 30
    #
    alphaP_lower = 0.
    alphaP_upper = 5.
    #
    gamma_lower = 0.
    gamma_upper = 10.
    #
    muJT_lower = -1
    muJT_upper = 11

    #####
    colormap = mpl.colormaps['viridis']
    norm = mpl.colors.LogNorm(vmin=T_lower, vmax=T_upper)

    kwargs_symbols = {'linestyle': '',  'markerfacecolor': 'white', 'markersize': markersize, 'zorder': 2.}
    rho_ticks = np.linspace(0, 1.2, 5)

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    grid_spec = fig.add_gridspec(5, 3)
    # First
    ax1 = fig.add_subplot(grid_spec[0, 0])
    ax2 = fig.add_subplot(grid_spec[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(grid_spec[0, 2], sharex=ax1, sharey=ax1)
    # Second
    ax4 = fig.add_subplot(grid_spec[1, 0], sharex=ax1)
    ax5 = fig.add_subplot(grid_spec[1, 1], sharex=ax1, sharey=ax4)
    ax6 = fig.add_subplot(grid_spec[1, 2], sharex=ax1, sharey=ax4)
    # Third
    ax7 = fig.add_subplot(grid_spec[2, 0], sharex=ax1)
    ax8 = fig.add_subplot(grid_spec[2, 1], sharex=ax1, sharey=ax7)
    ax9 = fig.add_subplot(grid_spec[2, 2], sharex=ax1, sharey=ax7)
    # Fourth
    ax10 = fig.add_subplot(grid_spec[3, 0], sharex=ax1)
    ax11 = fig.add_subplot(grid_spec[3, 1], sharex=ax1, sharey=ax10)
    ax12 = fig.add_subplot(grid_spec[3, 2], sharex=ax1, sharey=ax10)
    # fifth
    ax13 = fig.add_subplot(grid_spec[4, 0], sharex=ax1)
    ax14 = fig.add_subplot(grid_spec[4, 1], sharex=ax1, sharey=ax13)
    ax15 = fig.add_subplot(grid_spec[4, 2], sharex=ax1, sharey=ax13)

    axs_first = [ax1, ax2, ax3]
    axs_second = [ax4, ax5, ax6]
    axs_third = [ax7, ax8, ax9]
    axs_fourth = [ax10, ax11, ax12]
    axs_fifth = [ax13, ax14, ax15]

    for i in range(1, 3):
        axs_first[i].tick_params('y', labelleft=False)
        axs_second[i].tick_params('y', labelleft=False)
        axs_third[i].tick_params('y', labelleft=False)
        axs_fourth[i].tick_params('y', labelleft=False)
        axs_fifth[i].tick_params('y', labelleft=False)    
    for i in range(3):
        axs_first[i].tick_params('x', labelbottom=False)
        axs_second[i].tick_params('x', labelbottom=False)
        axs_third[i].tick_params('x', labelbottom=False)
        axs_fourth[i].tick_params('x', labelbottom=False)

    ax1.set_xlim([rho_lower, rho_upper])
    ax1.set_xticks(rho_ticks)

    # Isothermal Compressibility
    axs = axs_first
    axs[0].set_ylim([kappaT_lower, kappaT_upper])
    axs[0].set_ylabel(r"(a) $\rho^* \kappa_T^*$")
    titles_list = ['i. ', 'ii. ', 'iii. ']
    for lambda_r, ax, title in zip(lr_list, axs, titles_list):
        # Configuring the plot
        ax.set_title(title + f' $\lambda_\mathrm{{r}}={lambda_r}, \lambda_\mathrm{{a}}=6$')
        ax.grid(True)
        ax.tick_params(direction='in', which='both')

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Plotting the data in the bakcground
        sct = ax.scatter(df_data_lr['rho*'], df_data_lr['rho*']*df_data_lr['IsothermalCompressibility_npt'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color = colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['rho_isothermal_compressibility'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['rho*'] * df_data_lr_T['IsothermalCompressibility_npt'], color=color, marker='o', **kwargs_symbols)

    # Thermal Pressure Coefficient
    axs = axs_second
    axs[0].set_ylim([gammaV_lower, gammaV_upper])
    axs[0].set_ylabel(r"(b) $\gamma_V^*$")
    axs[0].set_yticks([0, 10, 20, 30])
    for lambda_r, ax in zip(lr_list, axs):
        # Configuring the plot
        ax.grid(True)
        ax.tick_params(direction='in', which='both')

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Plotting the data in the bakcground
        sct = ax.scatter(df_data_lr['rho*'], df_data_lr['ThermalPressureCoeff_nvt'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color = colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['thermal_pressure_coefficient'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['ThermalPressureCoeff_nvt'], color=color, marker='o', **kwargs_symbols)

    # Thermal Expansion Coefficient
    axs = axs_third
    axs[0].set_ylim([alphaP_lower, alphaP_upper])
    axs[0].set_ylabel(r"(c) $\alpha_P^*$")
    for lambda_r, ax in zip(lr_list, axs):
        # Configuring the plot
        ax.grid(True)
        ax.tick_params(direction='in', which='both')

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Plotting the data in the bakcground
        sct = ax.scatter(df_data_lr['rho*'], df_data_lr['ThermalExpansionCoeff_npt'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color = colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['thermal_expansion_coefficient'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['ThermalExpansionCoeff_npt'], color=color, marker='o', **kwargs_symbols)

    # Adiabatic index
    axs = axs_fourth
    axs[0].set_ylim([gamma_lower, gamma_upper])
    axs[0].set_ylabel(r"(d) $\gamma$")
    for lambda_r, ax in zip(lr_list, axs):
        # Configuring the plot
        ax.grid(True)
        ax.tick_params(direction='in', which='both')

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Plotting the data in the bakcground
        sct = ax.scatter(df_data_lr['rho*'], df_data_lr['Cp_npt']/df_data_lr['Cv_npt'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color = colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['adiabatic_index'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['Cp_npt']/df_data_lr_T['Cv_npt'], color=color, marker='o', **kwargs_symbols)

    # Joule Thomson Coefficient
    axs = axs_fifth
    axs[0].set_ylim([muJT_lower, muJT_upper])
    axs[0].set_ylabel(r"(e) $\mu_\mathrm{JT}^*$")
    for lambda_r, ax in zip(lr_list, axs):
        # Configuring the plot
        ax.grid(True)
        ax.tick_params(direction='in', which='both')

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Plotting the data in the bakcground
        sct = ax.scatter(df_data_lr['rho*'], df_data_lr['JouleThomson_npt'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color = colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['joule_thomson_coefficient'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['JouleThomson_npt'], color=color, marker='o', **kwargs_symbols)

    for ax in axs_fifth:
        ax.set_xlabel(r'$\rho^*$')

    # color bar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap),
                        ax=axs_fifth, pad=0.1,  location='bottom', orientation='horizontal', aspect=80)
    cbar.ax.set_ylabel(r'$T^*$', rotation='horizontal', loc='bottom', labelpad=15.)
    cbar.set_ticks(ticks=[0.6, 1, 2, 6, 10], labels=[0.6, 1, 2, 6, 10])

    return fig


def plot_second_isotherms_all(df_data, excel_dict_isotherms, lr_list=[12, 16, 20], Tad_list=[0.8, 1., 1.3, 2.8, 6.0],
                              width=3, height=5,
                              rho_lower=-1e-2, rho_upper=1.2, T_lower = 0.6, T_upper = 10.0,
                              alpha_data=0.3, markersize=3., rasterized=False):
    cv_lower = 1.
    cv_upper = 8.
    # 
    kappaT_lower = -1
    kappaT_upper = 4
    #
    gammaV_lower = -1
    gammaV_upper = 30
    #
    alphaP_lower = 0.
    alphaP_upper = 5.
    #
    gamma_lower = 0.
    gamma_upper = 10.
    #
    muJT_lower = -1
    muJT_upper = 11

    #####
    colormap = mpl.colormaps['viridis']
    norm = mpl.colors.LogNorm(vmin=T_lower, vmax=T_upper)

    kwargs_symbols = {'linestyle': '',  'markerfacecolor': 'white', 'markersize': markersize, 'zorder': 2.}
    rho_ticks = np.linspace(0, 1.2, 5)

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    grid_spec = fig.add_gridspec(6, 3)
    # First
    ax1 = fig.add_subplot(grid_spec[0, 0])
    ax2 = fig.add_subplot(grid_spec[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(grid_spec[0, 2], sharex=ax1, sharey=ax1)
    # Second
    ax4 = fig.add_subplot(grid_spec[1, 0], sharex=ax1)
    ax5 = fig.add_subplot(grid_spec[1, 1], sharex=ax1, sharey=ax4)
    ax6 = fig.add_subplot(grid_spec[1, 2], sharex=ax1, sharey=ax4)
    # Third
    ax7 = fig.add_subplot(grid_spec[2, 0], sharex=ax1)
    ax8 = fig.add_subplot(grid_spec[2, 1], sharex=ax1, sharey=ax7)
    ax9 = fig.add_subplot(grid_spec[2, 2], sharex=ax1, sharey=ax7)
    # Fourth
    ax10 = fig.add_subplot(grid_spec[3, 0], sharex=ax1)
    ax11 = fig.add_subplot(grid_spec[3, 1], sharex=ax1, sharey=ax10)
    ax12 = fig.add_subplot(grid_spec[3, 2], sharex=ax1, sharey=ax10)
    # fifth
    ax13 = fig.add_subplot(grid_spec[4, 0], sharex=ax1)
    ax14 = fig.add_subplot(grid_spec[4, 1], sharex=ax1, sharey=ax13)
    ax15 = fig.add_subplot(grid_spec[4, 2], sharex=ax1, sharey=ax13)
    # sixth
    ax16 = fig.add_subplot(grid_spec[5, 0], sharex=ax1)
    ax17 = fig.add_subplot(grid_spec[5, 1], sharex=ax1, sharey=ax16)
    ax18 = fig.add_subplot(grid_spec[5, 2], sharex=ax1, sharey=ax16)

    axs_first = [ax1, ax2, ax3]
    axs_second = [ax4, ax5, ax6]
    axs_third = [ax7, ax8, ax9]
    axs_fourth = [ax10, ax11, ax12]
    axs_fifth = [ax13, ax14, ax15]
    axs_sixth = [ax16, ax17, ax18]


    for i in range(1, 3):
        axs_first[i].tick_params('y', labelleft=False)
        axs_second[i].tick_params('y', labelleft=False)
        axs_third[i].tick_params('y', labelleft=False)
        axs_fourth[i].tick_params('y', labelleft=False)
        axs_fifth[i].tick_params('y', labelleft=False) 
        axs_sixth[i].tick_params('y', labelleft=False)    
    for i in range(3):
        axs_first[i].tick_params('x', labelbottom=False)
        axs_second[i].tick_params('x', labelbottom=False)
        axs_third[i].tick_params('x', labelbottom=False)
        axs_fourth[i].tick_params('x', labelbottom=False)
        axs_fifth[i].tick_params('x', labelbottom=False)

    ax1.set_xlim([rho_lower, rho_upper])
    ax1.set_xticks(rho_ticks)

    # Isochoric heat capacity
    axs = axs_first
    axs[0].set_ylim([cv_lower, cv_upper])
    axs[0].set_ylabel(r"(a) $C_V^*$")
    titles_list = ['i. ', 'ii. ', 'iii. ']
    for lambda_r, ax, title in zip(lr_list, axs, titles_list):
        # Configuring the plot
        ax.set_title(title + f' $\lambda_\mathrm{{r}}={lambda_r}, \lambda_\mathrm{{a}}=6$')
        ax.grid(True)
        ax.tick_params(direction='in', which='both')

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Plotting the data in the bakcground
        sct = ax.scatter(df_data_lr['rho*'], df_data_lr['Cv_nvt'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color = colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['isochoric_heat_capacity'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['Cv_nvt'], color=color, marker='o', **kwargs_symbols)

    # Isothermal Compressibility
    axs = axs_second
    axs[0].set_ylim([kappaT_lower, kappaT_upper])
    axs[0].set_ylabel(r"(b) $\rho^* \kappa_T^*$")
    titles_list = ['i. ', 'ii. ', 'iii. ']
    for lambda_r, ax in zip(lr_list, axs):
        # Configuring the plot
        # ax.set_title(title + f' $\lambda_\mathrm{{r}}={lambda_r}, \lambda_\mathrm{{a}}=6$')
        ax.grid(True)
        ax.tick_params(direction='in', which='both')

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Plotting the data in the bakcground
        sct = ax.scatter(df_data_lr['rho*'], df_data_lr['rho*']*df_data_lr['IsothermalCompressibility_npt'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color = colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['rho_isothermal_compressibility'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['rho*'] * df_data_lr_T['IsothermalCompressibility_npt'], color=color, marker='o', **kwargs_symbols)

    # Thermal Pressure Coefficient
    axs = axs_third
    axs[0].set_ylim([gammaV_lower, gammaV_upper])
    axs[0].set_ylabel(r"(c) $\gamma_V^*$")
    axs[0].set_yticks([0, 10, 20, 30])
    for lambda_r, ax in zip(lr_list, axs):
        # Configuring the plot
        ax.grid(True)
        ax.tick_params(direction='in', which='both')

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Plotting the data in the bakcground
        sct = ax.scatter(df_data_lr['rho*'], df_data_lr['ThermalPressureCoeff_nvt'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color = colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['thermal_pressure_coefficient'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['ThermalPressureCoeff_nvt'], color=color, marker='o', **kwargs_symbols)

    # Thermal Expansion Coefficient
    axs = axs_fourth
    axs[0].set_ylim([alphaP_lower, alphaP_upper])
    axs[0].set_ylabel(r"(d) $\alpha_P^*$")
    for lambda_r, ax in zip(lr_list, axs):
        # Configuring the plot
        ax.grid(True)
        ax.tick_params(direction='in', which='both')

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Plotting the data in the bakcground
        sct = ax.scatter(df_data_lr['rho*'], df_data_lr['ThermalExpansionCoeff_npt'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color = colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['thermal_expansion_coefficient'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['ThermalExpansionCoeff_npt'], color=color, marker='o', **kwargs_symbols)

    # Adiabatic index
    axs = axs_fifth
    axs[0].set_ylim([gamma_lower, gamma_upper])
    axs[0].set_ylabel(r"(e) $\gamma$")
    for lambda_r, ax in zip(lr_list, axs):
        # Configuring the plot
        ax.grid(True)
        ax.tick_params(direction='in', which='both')

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Plotting the data in the bakcground
        sct = ax.scatter(df_data_lr['rho*'], df_data_lr['Cp_npt']/df_data_lr['Cv_npt'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color = colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['adiabatic_index'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['Cp_npt']/df_data_lr_T['Cv_npt'], color=color, marker='o', **kwargs_symbols)

    # Joule Thomson Coefficient
    axs = axs_sixth
    axs[0].set_ylim([muJT_lower, muJT_upper])
    axs[0].set_ylabel(r"(f) $\mu_\mathrm{JT}^*$")
    for lambda_r, ax in zip(lr_list, axs):
        # Configuring the plot
        ax.grid(True)
        ax.tick_params(direction='in', which='both')

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Plotting the data in the bakcground
        sct = ax.scatter(df_data_lr['rho*'], df_data_lr['JouleThomson_npt'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color = colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['joule_thomson_coefficient'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['JouleThomson_npt'], color=color, marker='o', **kwargs_symbols)

    for ax in axs_sixth:
        ax.set_xlabel(r'$\rho^*$')

    # color bar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap),
                        ax=axs_sixth, pad=0.1,  location='bottom', orientation='horizontal', aspect=80)
    cbar.ax.set_ylabel(r'$T^*$', rotation='horizontal', loc='bottom', labelpad=15.)
    cbar.set_ticks(ticks=[0.6, 1, 2, 6, 10], labels=[0.6, 1, 2, 6, 10])

    return fig
