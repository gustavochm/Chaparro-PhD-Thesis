import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def plot_pressure_isotherms(df_data, excel_dict_isotherms, lr_list=[12, 16, 20], Tad_list=[0.65, 0.70, 0.90, 1.1, 1.3, 2.8, 5.3, 7.2],
                            width=3, height=1,
                            rho_lower=-1e-2, rho_upper=1.2, P_lower=-2, P_upper=20,
                            P_lower_b=-1.2, P_upper_b=0.3, T_lower=0.6, T_upper=10.0,
                            alpha_data=0.3, markersize=3., rasterized=False):

    colormap = mpl.colormaps['viridis']
    norm = mpl.colors.LogNorm(vmin=T_lower, vmax=T_upper)

    kwargs_symbols = {'linestyle': '',  'markerfacecolor':'white', 'markersize':markersize, 'zorder': 2}
    rho_ticks = np.linspace(0, 1.2, 5)

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 3, height_ratios=(2.5, 1))
    ax1 = fig.add_subplot(grid_spec[0, 0])
    ax2 = fig.add_subplot(grid_spec[0, 1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(grid_spec[0, 2], sharex=ax1, sharey=ax1)
    ax1b = fig.add_subplot(grid_spec[1, 0], sharex=ax1)
    ax2b = fig.add_subplot(grid_spec[1, 1], sharex=ax1, sharey=ax1b)
    ax3b = fig.add_subplot(grid_spec[1, 2], sharex=ax1, sharey=ax1b)

    fig.supylabel(r'$P^*$')

    ax2.tick_params('y', labelleft=False)
    ax3.tick_params('y', labelleft=False)
    ax1.tick_params('x', labelbottom=False)
    ax2.tick_params('x', labelbottom=False)
    ax3.tick_params('x', labelbottom=False)
    ax2b.tick_params('y', labelleft=False)
    ax3b.tick_params('y', labelleft=False)

    ax1.set_xlim([rho_lower, rho_upper])
    ax1.set_ylim([P_lower, P_upper])
    ax1b.set_ylim([P_lower_b, P_upper_b])
    ax1.set_xticks(rho_ticks)

    axs = [ax1, ax2, ax3]
    axs_bottom = [ax1b, ax2b, ax3b]

    titles_list = ['(a)', '(b)', '(c)']
    for lambda_r, ax, axb, title in zip(lr_list, axs, axs_bottom, titles_list):

        # Dataframe with the PVT data
        df_data_lr = df_data.query(f'lr=={lambda_r}')

        # reading the isotherms obtained from the FE-ANN and SAFT-VR Mie EoS
        excel_feann = pd.ExcelFile(excel_dict_isotherms[f'lr={lambda_r:.0f}'])

        # Configuring the plot
        ax.set_title(title + f' $\lambda_\mathrm{{r}}={lambda_r}, \lambda_\mathrm{{a}}=6$')
        ax.tick_params(direction='in', which='both')
        ax.grid(True)

        axb.tick_params(direction='in', which='both')
        axb.grid(True)
        axb.set_xlabel(r'$\rho^*$')

        # Plotting the data in the bakcground
        sct1 = ax.scatter(df_data_lr['rho*'], df_data_lr['P_ad'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct2 = axb.scatter(df_data_lr['rho*'], df_data_lr['P_ad'], alpha=alpha_data, s=markersize, c=df_data_lr['T_ad'], cmap=colormap, norm=norm)
        sct1.set_rasterized(rasterized)
        sct2.set_rasterized(rasterized)

        # plotting the isotherms in Tad_list
        for Tad in Tad_list:

            color_norm = norm(Tad)
            color =  colormap(color_norm)

            sheet_name = f'T={Tad:.2f}'
            df_feann = pd.read_excel(excel_feann, sheet_name=sheet_name)

            ax.plot(df_feann['density'], df_feann['pressure'], color=color)
            axb.plot(df_feann['density'], df_feann['pressure'], color=color)

            df_data_lr_T = df_data_lr[df_data_lr['T*'] == Tad]
            ax.plot(df_data_lr_T['rho*'], df_data_lr_T['P_ad'], color=color, marker='o', **kwargs_symbols)
            axb.plot(df_data_lr_T['rho*'], df_data_lr_T['P_ad'], color=color, marker='o', **kwargs_symbols)

    # color bar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap),
                        ax=[ax1, ax2, ax3, ax1b, ax2b, ax3b], pad=0.01,
                        location='right')
    # cbar.set_label(r'$T^*$', loc='center', fontsize=fontsize)
    cbar.ax.set_title(r'$T^*$')
    cbar.set_ticks(ticks=[0.6, 1, 2, 6, 10], labels=[0.6, 1, 2, 6, 10])

    return fig
