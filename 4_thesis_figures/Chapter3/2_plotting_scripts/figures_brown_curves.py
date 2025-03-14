import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy.interpolate import interp1d

plt.ioff()


def plot_brown_curves_second_virial(excel_dict, df_brown, lr_list, width=6., height=4.,
                                    T_lower=0.5, T_upper=10., P_lower=5e-4, P_upper=1e2, B2_lower=-10., B2_upper=3,
                                    zorder=3, alpha_solid_fill=0.5, markersize=3.5, markevery=3, markevery_amagat=1,
                                    color_vle='k', color_virial='k',
                                    color_zeno='C0', color_boyle='C1', color_charles='C2', color_amagat='C3',
                                    color_solid_fill='darkgrey',
                                    marker_crit='o', marker_triple='s',
                                    marker_zeno='s', marker_boyle='^', marker_charles='v', marker_amagat='o',
                                    include_md_data=True):

    # Phase Equilibria
    kwargs_vle = dict(color=color_vle, linestyle='-')
    kwargs_crit = dict(color=color_vle, linestyle='', marker=marker_crit, markersize=markersize)
    kwargs_triple = dict(color=color_vle, linestyle='', marker=marker_triple, markersize=markersize)
    # virial coefficients 
    kwargs_virial = dict(color=color_virial, linestyle='-')
    # characteristic curves
    kwargs_zeno = dict(color=color_zeno, linestyle=':')
    kwargs_boyle = dict(color=color_boyle, linestyle='--')
    kwargs_charles = dict(color=color_charles, linestyle='-.')
    kwargs_amagat = dict(color=color_amagat, linestyle=(0, (3, 1, 1, 1, 1, 1)))
    # characteristic temperatures
    kwargs_Tboyle = dict(color=color_boyle, linestyle='', marker=marker_boyle, 
                        zorder=zorder, clip_on=False, markersize=markersize)
    kwargs_Tcharles = dict(color=color_charles, linestyle='', marker=marker_charles, 
                            zorder=zorder, clip_on=False, markersize=markersize)
    # MD data
    kwargs_zeno_md = dict(color=color_zeno, linestyle='', marker=marker_zeno,
                        markerfacecolor='white', markersize=markersize, markevery=markevery)
    kwargs_boyle_md = dict(color=color_boyle, linestyle='', marker=marker_boyle, 
                        markerfacecolor='white', markersize=markersize, markevery=markevery)
    kwargs_charles_md = dict(color=color_charles, linestyle='', marker=marker_charles,
                            markerfacecolor='white', markersize=markersize, markevery=markevery)
    kwargs_amagat_md = dict(color=color_amagat, linestyle='', marker=marker_amagat,
                            markerfacecolor='white', markersize=markersize, markevery=markevery_amagat)

    # making the figure
    fig = plt.figure(figsize=(width, height), constrained_layout=True)

    # Configuring the axes
    grid_spec = fig.add_gridspec(2, 3, height_ratios=(3, 1))
    ax1 = fig.add_subplot(grid_spec[0, 0])
    ax2 = fig.add_subplot(grid_spec[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(grid_spec[0, 2], sharey=ax1)
    ax1b = fig.add_subplot(grid_spec[1, 0], sharex=ax1)
    ax2b = fig.add_subplot(grid_spec[1, 1], sharex=ax1, sharey=ax1b)
    ax3b = fig.add_subplot(grid_spec[1, 2], sharex=ax1, sharey=ax1b)

    axs = [ax1, ax2, ax3]
    axs_b = [ax1b, ax2b, ax3b]
    T_ticks = [0.6, 1., 2., 6., 10.]
    B2_ticks = [-10, -5, 0]
    for ax, axb in zip(axs, axs_b):
        ax.tick_params(direction='in', which='both')
        ax.tick_params('x', labelbottom=False)
        ax.grid(True)
        ax.set_xlim([T_lower, T_upper])
        ax.set_ylim([P_lower, P_upper])
        ax.set_xscale('log')
        ax.set_yscale('log')

        axb.tick_params(direction='in', which='both')
        axb.grid(True)
        axb.set_xticks(T_ticks)
        axb.set_yticks(B2_ticks)
        axb.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axb.set_ylim([B2_lower, B2_upper])

    ax2.tick_params('y', labelleft=False)
    ax3.tick_params('y', labelleft=False)
    ax2b.tick_params('y', labelleft=False)
    ax3b.tick_params('y', labelleft=False)

    ax1.set_ylabel(r'$P^*$')
    fig.supxlabel(r'$T^*$')
    ax1b.set_ylabel(r'$B_2^*$')

    # reading the data
    plot_labels = ['(a)', '(b)', '(c)']
    for lambda_r, ax, axb, plot_label in zip(lr_list, axs, axs_b, plot_labels):
        ax.set_title(plot_label + f' $\lambda_\mathrm{{r}}={lambda_r:.0f}, \lambda_\mathrm{{a}}=6$')
        excel_file = excel_dict[f'lr={lambda_r:.0f}']

        # reading the data from the excel file
        df_info = pd.read_excel(excel_file, sheet_name='info')
        df_vle = pd.read_excel(excel_file, sheet_name='vle')
        df_sle = pd.read_excel(excel_file, sheet_name='sle')
        df_sve = pd.read_excel(excel_file, sheet_name='sve')
        df_characteristic = pd.read_excel(excel_file, sheet_name='characteristic_temperatures')
        df_zeno = pd.read_excel(excel_file, sheet_name='zeno_curve')
        df_boyle = pd.read_excel(excel_file, sheet_name='boyle_curve')
        df_charles = pd.read_excel(excel_file, sheet_name='charles_curve')
        df_amagat = pd.read_excel(excel_file, sheet_name='amagat_curve')
        df_virial = pd.read_excel(excel_file, sheet_name='virial')

        # plotting the data

        # Zeno Curve
        ax.plot(df_zeno['T_zeno'], df_zeno['pressure_zeno'], **kwargs_zeno)
        # Boyle Curve
        ax.plot(df_boyle['T_boyle'], df_boyle['pressure_boyle'], **kwargs_boyle)
        # Charles Curve
        ax.plot(df_charles['T_charles'], df_charles['pressure_charles'], **kwargs_charles)
        # Amagat Curve
        ax.plot(df_amagat['T_amagat'], df_amagat['pressure_amagat'], **kwargs_amagat)

        # VLE
        ax.plot(df_vle['T_vle_model'], df_vle['P_vle_model'], **kwargs_vle)
        # SLE
        ax.plot(df_sle['T_sle_model'], df_sle['P_sle_model'], **kwargs_vle)
            # SLE
        ax.plot(df_sve['T_sve_model'], df_sve['P_sve_model'], **kwargs_vle)
        
        # Critical Point
        ax.plot(df_info['Tcad_model'], df_info['Pcad_model'], **kwargs_crit)
        # Triple Point
        ax.plot(df_info['T_triple'], df_info['P_triple'], **kwargs_triple)
        # Fillinf solid region
        Ts = np.hstack([df_sve['T_sve_model'][::-1], df_sle['T_sle_model']])
        Ps = np.hstack([df_sve['P_sve_model'][::-1], df_sle['P_sle_model']])
        ax.fill_betweenx(Ps, Ts, T_lower, color=color_solid_fill, zorder=0.1, alpha=alpha_solid_fill)

        # virial coefficient
        axb.plot(df_virial['T_virial'], df_virial['B2'], **kwargs_virial)

        # characteristic temperatures
        ax.plot(df_characteristic['T_boyle'].values[0], P_lower, **kwargs_Tboyle)
        ax.plot(df_characteristic['T_charles'].values[0], P_lower, **kwargs_Tcharles)
        B_boyle = float(interp1d(df_virial['T_virial'], df_virial['B2'])(df_characteristic['T_boyle'].values[0]))
        B_charles = float(interp1d(df_virial['T_virial'], df_virial['B2'])(df_characteristic['T_charles'].values[0]))
        axb.plot(df_characteristic['T_boyle'].values[0], B_boyle, **kwargs_Tboyle)
        axb.plot(df_characteristic['T_charles'].values[0], B_charles, **kwargs_Tcharles)

        if include_md_data:
            # MD data, Data from simon
            df_brown_lr = df_brown[df_brown['T*'] < 1.1 * T_upper].reset_index(drop=True)
            df_brown_lr = df_brown_lr[df_brown_lr['lr'] == lambda_r].reset_index(drop=True)

            df_brown_zeno = df_brown_lr[df_brown_lr['Brown_type'] == 'Zeno']
            df_brown_boyle = df_brown_lr[df_brown_lr['Brown_type'] == 'Boyle']
            df_brown_charles = df_brown_lr[df_brown_lr['Brown_type'] == 'Charles']
            df_brown_amagat = df_brown_lr[df_brown_lr['Brown_type'] == 'Amagat']

            ax.plot(df_brown_zeno['T*'], df_brown_zeno['P_ad'], **kwargs_zeno_md)
            ax.plot(df_brown_boyle['T*'], df_brown_boyle['P_ad'], **kwargs_boyle_md)
            ax.plot(df_brown_charles['T*'], df_brown_charles['P_ad'], **kwargs_charles_md)
            ax.plot(df_brown_amagat['T*'], df_brown_amagat['P_ad'], **kwargs_amagat_md)  

    return fig

def plot_brown_curves_virials3(excel_dict, df_brown, lr_list, width=6., height=4.,
                               T_lower=0.5, T_upper=10.,
                               P_lower=5e-4, P_upper=1e2,
                               B2_lower=-10., B2_upper=3,
                               B3_lower=-10., B3_upper=5,
                               zorder=3, alpha_solid_fill=0.5, markersize=3.5, markevery=3, markevery_amagat=1,
                               color_vle='k', color_virial='k',
                               color_zeno='C0', color_boyle='C1', color_charles='C2', color_amagat='C3',
                               color_solid_fill='darkgrey',
                               marker_crit='o', marker_triple='s',
                               marker_zeno='s', marker_boyle='^', marker_charles='v', marker_amagat='o',
                               include_md_data=True):

    plt.ioff()

    # Phase Equilibria
    kwargs_vle = dict(color=color_vle, linestyle='-')
    kwargs_crit = dict(color=color_vle, linestyle='', marker=marker_crit, markersize=markersize)
    kwargs_triple = dict(color=color_vle, linestyle='', marker=marker_triple, markersize=markersize)
    # virial coefficients 
    kwargs_virial = dict(color=color_virial, linestyle='-')
    # characteristic curves
    kwargs_zeno = dict(color=color_zeno, linestyle=':')
    kwargs_boyle = dict(color=color_boyle, linestyle='--')
    kwargs_charles = dict(color=color_charles, linestyle='-.')
    kwargs_amagat = dict(color=color_amagat, linestyle=(0, (3, 1, 1, 1, 1, 1)))
    # characteristic temperatures
    kwargs_Tboyle = dict(color=color_boyle, linestyle='', marker=marker_boyle, 
                         zorder=zorder, clip_on=False, markersize=markersize)
    kwargs_Tcharles = dict(color=color_charles, linestyle='', marker=marker_charles, 
                           zorder=zorder, clip_on=False, markersize=markersize)
    # MD data
    kwargs_zeno_md = dict(color=color_zeno, linestyle='', marker=marker_zeno,
                          markerfacecolor='white', markersize=markersize, markevery=markevery)
    kwargs_boyle_md = dict(color=color_boyle, linestyle='', marker=marker_boyle, 
                           markerfacecolor='white', markersize=markersize, markevery=markevery)
    kwargs_charles_md = dict(color=color_charles, linestyle='', marker=marker_charles,
                             markerfacecolor='white', markersize=markersize, markevery=markevery)
    kwargs_amagat_md = dict(color=color_amagat, linestyle='', marker=marker_amagat,
                            markerfacecolor='white', markersize=markersize, markevery=markevery_amagat)

    # making the figure
    fig = plt.figure(figsize=(width, height), constrained_layout=True)

    # Configuring the axes
    grid_spec = fig.add_gridspec(3, 3, height_ratios=(3, 1, 1))
    ax1 = fig.add_subplot(grid_spec[0, 0])
    ax2 = fig.add_subplot(grid_spec[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(grid_spec[0, 2], sharey=ax1)
    ax1b = fig.add_subplot(grid_spec[1, 0], sharex=ax1)
    ax2b = fig.add_subplot(grid_spec[1, 1], sharex=ax1, sharey=ax1b)
    ax3b = fig.add_subplot(grid_spec[1, 2], sharex=ax1, sharey=ax1b)
    ax1c = fig.add_subplot(grid_spec[2, 0], sharex=ax1)
    ax2c = fig.add_subplot(grid_spec[2, 1], sharex=ax1, sharey=ax1c)
    ax3c = fig.add_subplot(grid_spec[2, 2], sharex=ax1, sharey=ax1c)

    axs = [ax1, ax2, ax3]
    axs_b = [ax1b, ax2b, ax3b]
    axs_c = [ax1c, ax2c, ax3c]
    T_ticks = [0.6, 1., 2., 6., 10.]
    B2_ticks = [-10, -5, 0]
    B3_ticks = [-10, -5, 0]
    for ax, axb, axc in zip(axs, axs_b, axs_c):
        ax.tick_params(direction='in', which='both')
        ax.tick_params('x', labelbottom=False)
        ax.grid(True)
        ax.set_xlim([T_lower, T_upper])
        ax.set_ylim([P_lower, P_upper])
        ax.set_xscale('log')
        ax.set_yscale('log')

        axb.tick_params(direction='in', which='both')
        axb.grid(True)
        # axb.set_xticks(T_ticks)
        axb.set_yticks(B2_ticks)
        # axb.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axb.set_ylim([B2_lower, B2_upper])
        axb.tick_params('x', labelbottom=False)

        axc.tick_params(direction='in', which='both')
        axc.grid(True)
        axc.set_xticks(T_ticks)
        axc.set_yticks(B3_ticks)
        axc.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axc.set_ylim([B3_lower, B3_upper])

    ax2.tick_params('y', labelleft=False)
    ax3.tick_params('y', labelleft=False)
    ax2b.tick_params('y', labelleft=False)
    ax3b.tick_params('y', labelleft=False)
    ax2c.tick_params('y', labelleft=False)
    ax3c.tick_params('y', labelleft=False)

    ax1.set_ylabel(r'(a) $P^*$')
    fig.supxlabel(r'$T^*$')
    ax1b.set_ylabel(r'(b) $B_2^*$')
    ax1c.set_ylabel(r'(c) $B_3^*$')
    """
    x = -0.2
    y = 0.9
    ax1.set_title('(a)', x=x, y=y)
    ax1b.set_title('(b)', x=x, y=y)
    ax1c.set_title('(c)', x=x, y=y)
    """

    # reading the data
    plot_labels = ['i. ', 'ii. ', 'iii. ']
    for lambda_r, ax, axb, axc,  plot_label in zip(lr_list, axs, axs_b, axs_c, plot_labels):
        ax.set_title(plot_label + f' $\lambda_\mathrm{{r}}={lambda_r:.0f}, \lambda_\mathrm{{a}}=6$')
        excel_file = excel_dict[f'lr={lambda_r:.0f}']

        # reading the data from the excel file
        df_info = pd.read_excel(excel_file, sheet_name='info')
        df_vle = pd.read_excel(excel_file, sheet_name='vle')
        df_sle = pd.read_excel(excel_file, sheet_name='sle')
        df_sve = pd.read_excel(excel_file, sheet_name='sve')
        df_characteristic = pd.read_excel(excel_file, sheet_name='characteristic_temperatures')
        df_zeno = pd.read_excel(excel_file, sheet_name='zeno_curve')
        df_boyle = pd.read_excel(excel_file, sheet_name='boyle_curve')
        df_charles = pd.read_excel(excel_file, sheet_name='charles_curve')
        df_amagat = pd.read_excel(excel_file, sheet_name='amagat_curve')
        df_virial = pd.read_excel(excel_file, sheet_name='virial')

        # plotting the data

        # Zeno Curve
        ax.plot(df_zeno['T_zeno'], df_zeno['pressure_zeno'], **kwargs_zeno)
        # Boyle Curve
        ax.plot(df_boyle['T_boyle'], df_boyle['pressure_boyle'], **kwargs_boyle)
        # Charles Curve
        ax.plot(df_charles['T_charles'], df_charles['pressure_charles'], **kwargs_charles)
        # Amagat Curve
        ax.plot(df_amagat['T_amagat'], df_amagat['pressure_amagat'], **kwargs_amagat)

        # VLE
        ax.plot(df_vle['T_vle_model'], df_vle['P_vle_model'], **kwargs_vle)
        # SLE
        ax.plot(df_sle['T_sle_model'], df_sle['P_sle_model'], **kwargs_vle)
        # SLE
        ax.plot(df_sve['T_sve_model'], df_sve['P_sve_model'], **kwargs_vle)

        # Critical Point
        ax.plot(df_info['Tcad_model'], df_info['Pcad_model'], **kwargs_crit)
        # Triple Point
        ax.plot(df_info['T_triple'], df_info['P_triple'], **kwargs_triple)
        # Fillinf solid region
        Ts = np.hstack([df_sve['T_sve_model'][::-1], df_sle['T_sle_model']])
        Ps = np.hstack([df_sve['P_sve_model'][::-1], df_sle['P_sle_model']])
        ax.fill_betweenx(Ps, Ts, T_lower, color=color_solid_fill, zorder=0.1, alpha=alpha_solid_fill)

        # second virial coefficient
        axb.plot(df_virial['T_virial'], df_virial['B2'], **kwargs_virial)

        # third virial coefficient
        axc.plot(df_virial['T_virial'], df_virial['B3'], **kwargs_virial)

        # characteristic temperatures
        ax.plot(df_characteristic['T_boyle'].values[0], P_lower, **kwargs_Tboyle)
        ax.plot(df_characteristic['T_charles'].values[0], P_lower, **kwargs_Tcharles)
        B2_boyle = float(interp1d(df_virial['T_virial'], df_virial['B2'])(df_characteristic['T_boyle'].values[0]))
        B2_charles = float(interp1d(df_virial['T_virial'], df_virial['B2'])(df_characteristic['T_charles'].values[0]))
        axb.plot(df_characteristic['T_boyle'].values[0], B2_boyle, **kwargs_Tboyle)
        axb.plot(df_characteristic['T_charles'].values[0], B2_charles, **kwargs_Tcharles)
        B3_boyle = float(interp1d(df_virial['T_virial'], df_virial['B3'])(df_characteristic['T_boyle'].values[0]))
        B3_charles = float(interp1d(df_virial['T_virial'], df_virial['B3'])(df_characteristic['T_charles'].values[0]))
        axc.plot(df_characteristic['T_boyle'].values[0], B3_boyle, **kwargs_Tboyle)
        axc.plot(df_characteristic['T_charles'].values[0], B3_charles, **kwargs_Tcharles)

        if include_md_data:
            # MD data, Data from simon
            df_brown_lr = df_brown[df_brown['T*'] < 1.1 * T_upper].reset_index(drop=True)
            df_brown_lr = df_brown_lr[df_brown_lr['lr'] == lambda_r].reset_index(drop=True)

            df_brown_zeno = df_brown_lr[df_brown_lr['Brown_type'] == 'Zeno']
            df_brown_boyle = df_brown_lr[df_brown_lr['Brown_type'] == 'Boyle']
            df_brown_charles = df_brown_lr[df_brown_lr['Brown_type'] == 'Charles']
            df_brown_amagat = df_brown_lr[df_brown_lr['Brown_type'] == 'Amagat']

            ax.plot(df_brown_zeno['T*'], df_brown_zeno['P_ad'], **kwargs_zeno_md)
            ax.plot(df_brown_boyle['T*'], df_brown_boyle['P_ad'], **kwargs_boyle_md)
            ax.plot(df_brown_charles['T*'], df_brown_charles['P_ad'], **kwargs_charles_md)
            ax.plot(df_brown_amagat['T*'], df_brown_amagat['P_ad'], **kwargs_amagat_md)

    return fig
