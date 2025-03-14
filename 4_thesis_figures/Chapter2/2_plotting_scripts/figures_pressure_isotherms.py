import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_pressure_isotherms(df_data, lr_list=[12, 16, 20], Tad_list=[0.8, 1., 1.3, 2.8, 6.0],
                            width=3, height=1,
                            P_lower=-1, P_upper=20, rho_lower=-1e-2, rho_upper=1.2,
                            color_list=['C0', 'C2', 'black', 'C3', 'C1'], symbol_list=['s', 'o', 'v', 'D', '^'],
                            markersize=3.5, alpha_zoom=0.8, alpha_sle=0.5, color_solid_fill='grey'):

    folder_to_read = "../computed_files"
    kwargs_symbols = {'linestyle': '',  'markerfacecolor':'white', 'markersize':markersize}
    rho_ticks = np.linspace(0, 1.2, 5)

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, sharey=ax1, sharex=ax1)
    ax3 = fig.add_subplot(133, sharey=ax1, sharex=ax1)

    ax1.set_ylabel(r'$P^*$')
    ax2.tick_params('y', labelleft=False)
    ax3.tick_params('y', labelleft=False)
    ax1.set_xlim([rho_lower, rho_upper])
    ax1.set_ylim([P_lower, P_upper])
    ax1.set_xticks(rho_ticks)

    axs = [ax1, ax2, ax3]
    titles = ['(a)', '(b)', '(c)']

    for lambda_r, ax, title in zip(lr_list, axs, titles):

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        filename = f'isotherms_lr{lambda_r:.0f}.xlsx'
        filename_saft = f'isotherms_saft_lr{lambda_r:.0f}.xlsx'

        file_to_read = os.path.join(folder_to_read, filename)
        file_to_read_saft = os.path.join(folder_to_read, filename_saft)

        excel_feann = pd.ExcelFile(file_to_read)
        excel_saft = pd.ExcelFile(file_to_read_saft)

        # phase equilibria file
        file_sle = f'phase_equilibria_lr{lambda_r:.0f}.xlsx'
        file_to_read_sle = os.path.join(folder_to_read, file_sle)
        df_sle = pd.read_excel(file_to_read_sle, sheet_name='sle')

        # Configuring the plot
        ax.set_title(title + f' $\lambda_\mathrm{{r}}={lambda_r}, \lambda_\mathrm{{a}}=6$')
        ax.tick_params(direction='in', which='both')
        ax.grid(True)
        ax.set_xlabel(r'$\rho^*$')

        # Inset axis
        axins = ax.inset_axes([0.05, 0.55, 0.4, 0.4])
        # sub region of the original image

        # x1lim, x2lim, y1lim, y2lim = 0.0, 0.8, -0.6, 0.6
        x1lim, x2lim, y1lim, y2lim = 0.0, 0.9, -0.85, 0.7
        axins.set_xlim(x1lim, x2lim)
        axins.set_ylim(y1lim, y2lim)
        axins.tick_params(direction='in')
        axins.grid(True)
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        ax.indicate_inset_zoom(axins, alpha=alpha_zoom)

        # ploting SLE region
        rho_sle = df_sle['rhol_sle_model'].to_numpy()
        P_sle = df_sle['P_sle_model'].to_numpy()
        ax.fill(np.hstack([rho_sle[-1], rho_sle]), np.hstack([P_sle[0], P_sle]), color=color_solid_fill, alpha=alpha_sle)
        axins.fill(np.hstack([rho_sle[-1], rho_sle]), np.hstack([P_sle[0], P_sle]), color=color_solid_fill, alpha=alpha_sle)

        for Tad, symbol, color in zip(Tad_list, symbol_list, color_list):
            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)
            df_saft = pd.read_excel(excel_saft, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['pressure'], color=color)
            ax.plot(df_saft['density'], df_saft['pressure'], color=color, linestyle='--')

            axins.plot(df_feann['density'], df_feann['pressure'], color=color)
            axins.plot(df_saft['density'], df_saft['pressure'], color=color, linestyle='--')

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['P_ad'], color=color, marker=symbol, **kwargs_symbols)
            axins.plot(df_data_lr_T['rho*'], df_data_lr_T['P_ad'], color=color, marker=symbol, **kwargs_symbols)

    return fig
