import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_second_properties_isotherms(df_data_lr, excel_feann, excel_saft,
                                     Tad_list=[0.9, 1., 1.3, 2.8, 6.0],  width=3., height=2.,
                                     rho_lower=0, rho_upper=1.2, markersize=3.5,
                                     color_list = ['C0', 'C2', 'black', 'C3', 'C1'], symbol_list=['s', 'o', 'v', 'D', '^'],):

    df_data_lr['rhoIsothermalCompressibility_npt'] = df_data_lr['rho*'] * df_data_lr['IsothermalCompressibility_npt']
    df_data_lr['adiabatic_index'] = df_data_lr['Cp_npt'] / df_data_lr['Cv_npt']

    ###########
    kwargs_symbols = {'linestyle': '',  'markerfacecolor':'white', 'markersize':markersize}
    rho_ticks = np.linspace(0, 1.2, 5)

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232, sharex=ax1)
    ax3 = fig.add_subplot(233, sharex=ax1)
    ax4 = fig.add_subplot(234, sharex=ax1)
    ax5 = fig.add_subplot(235, sharex=ax1)
    ax6 = fig.add_subplot(236, sharex=ax1)

    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for ax, label in zip(axs, labels):
        ax.set_xlim([rho_lower, rho_upper])
        ax.set_xticks(rho_ticks)
        ax.set_title(label)
        ax.grid(True)
        ax.tick_params(direction='in', which='both')
        ax.set_xlabel(r'$\rho^*$')

    property_kwargs = dict()
    property_kwargs['isochoric_heat_capacity'] = dict(ax=ax1, lims=[1.4, 4.], ticks=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0], label=r'$C_V^*$', label_db='Cv_npt')
    property_kwargs['rho_isothermal_compressibility'] = dict(ax=ax2, lims=[-0.2, 4.2], ticks=[0., 1., 2., 3., 4.], label=r'$\rho^*\kappa_T^*$', label_db='rhoIsothermalCompressibility_npt')
    property_kwargs['thermal_pressure_coefficient'] = dict(ax=ax3, lims=[-1., 13], ticks=[0., 3., 6., 9., 12.], label=r'$\gamma_V^*$', label_db='ThermalPressureCoeff_nvt')
    property_kwargs['thermal_expansion_coefficient'] = dict(ax=ax4, lims=[-1., 10], ticks=[0., 3., 6.,  9.], label=r'$\alpha_P^*$', label_db='ThermalExpansionCoeff_npt')
    property_kwargs['adiabatic_index'] = dict(ax=ax5, lims=[0.5, 10.5], ticks=[2, 4, 6, 8, 10.], label=r'$\gamma$', plot_index='(h)', label_db='adiabatic_index')
    property_kwargs['joule_thomson_coefficient'] = dict(ax=ax6, lims=[-1, 12.], ticks=[0., 3, 6., 9., 12.], label=r'$\mu_\mathrm{JT}^*$', label_db='JouleThomson_npt')

    # property_kwargs['joule_thomson_coefficient'] = dict(ax=ax6, lims=[-1, 1], ticks=[-20., 3, 6., 9., 20.], label=r'$\mu_\mathrm{JT}^*$', label_db='JouleThomson_npt')


    for property_name, dict_kwargs in property_kwargs.items():
        ax = dict_kwargs['ax']
        ax.set_ylabel(dict_kwargs['label'])
        ax.set_ylim(dict_kwargs['lims'])
        ax.set_yticks(dict_kwargs['ticks'])
        property_name_db = dict_kwargs['label_db']

        for Tad, symbol, color in zip(Tad_list, symbol_list, color_list):
            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)
            df_saft = pd.read_excel(excel_saft, sheet_name=sheet_name)

            # FE-ANN EoS data
            ax.plot(df_feann['density'], df_feann[property_name], color=color)

            # SAFT EoS data
            ax.plot(df_saft['density'], df_saft[property_name], color=color, linestyle='--')

            # MD data
            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T[property_name_db], color=color, marker=symbol, **kwargs_symbols, zorder=3)
    return fig


def plot_lrs_second_isotherms(df_data, excel_dict_isotherms, excel_dict_isotherms_saft,
                              width=3, height=5,
                              lr_list=[12, 16, 20], Tad_list=[0.8, 1., 1.3, 2.8, 6.0],
                              symbol_list=['s', 'o', 'v', 'd', '^'],
                              color_list=['C0', 'C2', 'black', 'C3', 'C1'],
                              rho_lower=-1e-2, rho_upper=1.2, markersize=3.5):

    df_data['rhoIsothermalCompressibility_npt'] = df_data['rho*'] * df_data['IsothermalCompressibility_npt']
    df_data['adiabatic_index'] = df_data['Cp_npt'] / df_data['Cv_npt']

    property_kwargs = dict()
    property_kwargs['isochoric_heat_capacity'] = dict(lims=[1., 5.], ticks=[1, 2, 3, 4, 5], label=r'$C_V^*$', label_db='Cv_npt')
    property_kwargs['rho_isothermal_compressibility'] = dict(lims=[-0.2, 4.2], ticks=[0., 1., 2., 3., 4.], label=r'$\rho^*\kappa_T^*$', label_db='rhoIsothermalCompressibility_npt')
    property_kwargs['thermal_pressure_coefficient'] = dict(lims=[-1., 13], ticks=[0., 3., 6., 9., 12.], label=r'$\gamma_V^*$', label_db='ThermalPressureCoeff_nvt')
    property_kwargs['thermal_expansion_coefficient'] = dict(lims=[-1., 10], ticks=[0., 3., 6.,  9.], label=r'$\alpha_P^*$', label_db='ThermalExpansionCoeff_npt')
    property_kwargs['adiabatic_index'] = dict(lims=[0.5, 10.5], ticks=[2, 4, 6, 8, 10.], label=r'$\gamma$', plot_index='(h)', label_db='adiabatic_index')
    property_kwargs['joule_thomson_coefficient'] = dict(lims=[-1, 12.], ticks=[0., 3, 6., 9., 12.], label=r'$\mu_\mathrm{JT}^*$', label_db='JouleThomson_npt')

    #####
    kwargs_symbols = {'linestyle': '',  'markerfacecolor':'white', 'markersize': markersize, 'zorder': 2.}
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

    titles_list = ['i. ', 'ii. ', 'iii. ']
    for lambda_r, ax, title in zip(lr_list, axs_first, titles_list):
        ax.set_title(title + f' $\lambda_\mathrm{{r}}={lambda_r}, \lambda_\mathrm{{a}}=6$')

    titles_list = ['(a) ', '(b) ', '(c) ', '(d) ', '(e) ', '(f) ']
    for property_name, axs, title in zip(property_kwargs.keys(), [axs_first, axs_second, axs_third, axs_fourth, axs_fifth, axs_sixth], titles_list):
        axs[0].set_ylim(property_kwargs[property_name]['lims'])
        axs[0].set_yticks(property_kwargs[property_name]['ticks'])
        axs[0].set_ylabel(title + property_kwargs[property_name]['label'])
        property_name_db = property_kwargs[property_name]['label_db']

        for lambda_r, ax, title in zip(lr_list, axs, titles_list):
            # Configuring the plot
            ax.grid(True)
            ax.tick_params(direction='in', which='both')

            # Dataframe with the PVT data
            df_data_lr = df_data.query(f'lr=={lambda_r}')

            # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
            excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])
            excel_saft = pd.ExcelFile(excel_dict_isotherms_saft[f'lr={lambda_r:.0f}'])

            for Tad, symbol, color in zip(Tad_list, symbol_list, color_list):
                sheet_name = f'T={Tad:.2f}'
                df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)
                df_saft = pd.read_excel(excel_saft, sheet_name=sheet_name)

                # FE-ANN EoS data
                ax.plot(df_feann['density'], df_feann[property_name], color=color)

                # SAFT EoS data
                ax.plot(df_saft['density'], df_saft[property_name], color=color, linestyle='--')

                # MD data
                df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
                ax.plot(df_data_lr_T['rho*'], df_data_lr_T[property_name_db], color=color, marker=symbol, **kwargs_symbols)

    fig.supxlabel(r'$\rho^*$')

    return fig
