import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

plt.ioff()


def plot_parity_error_distribution(parity_diff_excel, parity_visc_excel, parity_tcond_excel,
                                   width, height, model_type='ann_res',
                                   marker='.', markersize=5, alpha_data=0.4, color_train='C0', color_test='C2',
                                   rasterized=True, colormap='viridis',
                                   rho_lower=-0.02, rho_upper=1.25,
                                   T_lower=0.6, T_upper=10.0,
                                   diff_lower=-5., diff_upper=210.,
                                   visc_lower=-1, visc_upper=22,
                                   tcond_lower=-1, tcond_upper=42,
                                   vmin=0, vmax=20,
                                   fontsize_annotation=8):

    #########
    transport_list = ['self_diffusivity', 'shear_viscosity', 'thermal_conductivity']
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    # setting limits
    ax1.set_xlim(diff_lower, diff_upper)
    ax1.set_ylim(diff_lower, diff_upper)
    ax2.set_xlim(visc_lower, visc_upper)
    ax2.set_ylim(visc_lower, visc_upper)
    ax3.set_xlim(tcond_lower, tcond_upper)
    ax3.set_ylim(tcond_lower, tcond_upper)
    ax4.set_xlim(rho_lower, rho_upper)
    ax4.set_ylim(T_lower, T_upper)
    ax5.set_xlim(rho_lower, rho_upper)
    ax5.set_ylim(T_lower, T_upper)
    ax6.set_xlim(rho_lower, rho_upper)
    ax6.set_ylim(T_lower, T_upper)

    # diagonal of parity plot
    ax1.plot([diff_lower, diff_upper], [diff_lower, diff_upper], color='k')
    ax2.plot([visc_lower, visc_upper], [visc_lower, visc_upper], color='k')
    ax3.plot([tcond_lower, tcond_upper], [tcond_lower, tcond_upper], color='k')

    title_list = ['(a) Self-diffusivity', '(b) Shear viscosity', '(c) Thermal conductivity',
                   '(d) Self-diffusivity', '(e) Shear viscosity', '(f) Thermal conductivity']
    for ax, title in zip([ax1, ax2, ax3, ax4, ax5, ax6], title_list):
        ax.set_title(title)

    excel_list = [parity_diff_excel, parity_visc_excel, parity_tcond_excel]
    label_list = [r'$D^*$', r'$\eta^*$', r'$\kappa^*$']
    for ax_parity, ax_errors, transport, excelfile, label  in zip([ax1, ax2, ax3], [ax4, ax5, ax6], transport_list, excel_list, label_list):
        ax_parity.grid(True)
        ax_parity.tick_params(direction='in', which='both')
        ax_parity.set_xlabel(f'{label} (MD)')
        ax_parity.set_ylabel(f'{label} (ANN)')

        ax_errors.grid(True)
        ax_errors.tick_params(direction='in', which='both')
        ax_errors.set_xlabel(r'$\rho^*$')
        ax_errors.set_ylabel(r'$T^*$')

        # parity plot
        df_train = pd.read_excel(excelfile, sheet_name='train')
        df_test = pd.read_excel(excelfile, sheet_name='test')

        scatter_train = ax_parity.scatter(df_train[transport], df_train[f'{transport}_{model_type}'], 
                                          marker=marker, color=color_train, s=markersize, alpha=alpha_data)
        scatter_test = ax_parity.scatter(df_test[transport], df_test[f'{transport}_{model_type}'], 
                                         linestyle='', marker=marker, color=color_test, s=markersize, alpha=alpha_data)
        scatter_train.set_rasterized(rasterized)
        scatter_test.set_rasterized(rasterized)
        mape_train = np.mean(100 * np.abs(df_train[f'{transport}_{model_type}']/df_train[transport] - 1))
        mape_test = np.mean(100 * np.abs(df_test[f'{transport}_{model_type}']/df_test[transport] - 1))
        ax_parity.text(0.05, 0.86, f'MAPE Train: {mape_train:.1f} \%', fontsize=fontsize_annotation, transform=ax_parity.transAxes, color=color_train)
        ax_parity.text(0.05, 0.74, f'MAPE Test: {mape_test:.1f} \%', fontsize=fontsize_annotation, transform=ax_parity.transAxes, color=color_test)
        # error_distribution plot
        rho_train = df_train['rhoad']
        T_train = df_train['Tad']
        tp_train = df_train[transport]
        tp_model_train = df_train[f'{transport}_{model_type}']

        rho_test = df_test['rhoad']
        T_test = df_test['Tad']
        tp_test = df_test[transport]
        tp_model_test = df_test[f'{transport}_{model_type}']

        rhoad = np.hstack((rho_train, rho_test))
        Tad = np.hstack((T_train, T_test))
        tp = np.hstack((tp_train, tp_test))
        tp_model = np.hstack((tp_model_train, tp_model_test))

        mape = 100. * np.abs((tp_model - tp) / tp)
        order = np.argsort(mape)
        scatter_mape = ax_errors.scatter(rhoad[order], Tad[order], c=mape[order], norm=norm, cmap=colormap, marker=marker, s=markersize, alpha=alpha_data)
        scatter_mape.set_rasterized(rasterized)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax = [ax4, ax5, ax6], pad=0.02, aspect=20, location='right')
    cbar.set_label(r'MAPE')
    return fig
