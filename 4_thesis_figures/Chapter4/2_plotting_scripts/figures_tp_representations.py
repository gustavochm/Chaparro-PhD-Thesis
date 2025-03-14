import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from feanneos.TransportProperties import viscosity_scaling
from feanneos.TransportProperties import diffusivity_scaling
from feanneos.TransportProperties import thermal_conductivity_scaling

plt.ioff()


def plot_self_diffusivity_representation(df_diff, width=4, height=4, lambda_r=12,
                                         rho_lower=-2e-2, rho_upper=1.25,
                                         T_lower=0.6, T_upper=10.,
                                         diff_lower=-1, diff_upper=2e2,
                                         logdiff_lower=1e-2, logdiff_upper=2e2,
                                         rhodiff_lower=0., rhodiff_upper=1.,
                                         Sres_lower=-2e-2, Sres_upper=4.,
                                         diff_scaled_lower=0., diff_scaled_upper=10.,
                                         rho_lower_inset=0., rho_upper_inset=1.2,
                                         diff_lower_inset=-2, diff_upper_inset=15,
                                         Sres_lower_inset=0., Sres_upper_inset=1.,
                                         diff_scaled_lower_inset=0., diff_scaled_upper_inset=3.,
                                         markersize=4):

    colormap = mpl.colormaps['viridis']
    norm = mpl.colors.LogNorm(vmin=T_lower, vmax=T_upper)
    kwargs = dict(cmap=colormap, norm=norm, marker='s', clip_on=False, s=markersize)
    kwargs_inset = kwargs.copy()
    kwargs_inset['clip_on'] = True
    rho_ticks = np.linspace(0., 1.2, 5)
    df_diff_lr = df_diff[df_diff['lr'] == lambda_r].reset_index(drop=True)

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    axs_list = [ax1, ax2, ax3]
    for ax in axs_list:
        ax.set_xlim(rho_lower, rho_upper)
        ax.tick_params(which='both', direction='in')
        ax.grid(True)
        ax.set_xlabel(r'$\rho^*$')
        ax.set_xticks(rho_ticks)

    ax4.tick_params(which='both', direction='in')
    ax4.grid(True)

    # Setting limits
    ax1.set_ylim([diff_lower, diff_upper])
    ax2.set_ylim([logdiff_lower, logdiff_upper])
    ax3.set_ylim([rhodiff_lower, rhodiff_upper])

    ax4.set_xlim([Sres_lower, Sres_upper])
    ax4.set_ylim([diff_scaled_lower, diff_scaled_upper])


    # Setting labels
    ax1.set_ylabel(r'$D^*$')
    ax2.set_ylabel(r'$D^*$')
    ax3.set_ylabel(r'$\rho^*D^*$')
    ax4.set_ylabel(r'$\tilde{D}^*$')
    ax4.set_xlabel(r'$-S^{*, \mathrm{res}}$')

    # Setting titles
    ax1.set_title(r'(a)')
    ax2.set_title(r'(b)')
    ax3.set_title(r'(c)')
    ax4.set_title(r'(d)')

    # Plots
    # Linear scale
    ax1.scatter(df_diff_lr['rho*'], df_diff_lr['self_diffusivity'], c=df_diff_lr['T*'], **kwargs)

    axins1 = ax1.inset_axes([0.15, 0.35, 0.8, 0.6])
    axins1.set_xlim(rho_lower_inset, rho_upper_inset)
    axins1.set_ylim(diff_lower_inset, diff_upper_inset)
    axins1.tick_params(direction='in')
    axins1.set_xticklabels('')
    axins1.set_yticklabels('')
    axins1.grid(True)
    ax1.indicate_inset_zoom(axins1)
    axins1.scatter(df_diff_lr['rho*'], df_diff_lr['self_diffusivity'], c=df_diff_lr['T*'], **kwargs_inset)

    # Log scale
    ax2.scatter(df_diff_lr['rho*'], df_diff_lr['self_diffusivity'], c=df_diff_lr['T*'], **kwargs)
    ax2.set_yscale('log')

    # rho*D scale
    ax3.scatter(df_diff_lr['rho*'], df_diff_lr['rho*'] * df_diff_lr['self_diffusivity'], c=df_diff_lr['T*'], **kwargs)

    # Entropy scaling
    scaled_diffusivity = diffusivity_scaling(df_diff_lr['rho*'], df_diff_lr['T*'], df_diff_lr['self_diffusivity'])
    ax4.scatter(-df_diff_lr['Sr'], scaled_diffusivity, c=df_diff_lr['T*'], **kwargs)

    axins4 = ax4.inset_axes([0.15, 0.35, 0.8, 0.60])
    axins4.set_xlim(Sres_lower_inset, Sres_upper_inset)
    axins4.set_ylim(diff_scaled_lower_inset, diff_scaled_upper_inset)
    axins4.tick_params(direction='in')
    axins4.set_xticklabels('')
    axins4.set_yticklabels('')
    axins4.grid(True)
    ax4.indicate_inset_zoom(axins4)
    axins4.scatter(-df_diff_lr['Sr'], scaled_diffusivity, c=df_diff_lr['T*'], **kwargs_inset)

    # color bar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax = [ax1, ax2, ax3, ax4], pad=0.01, location='right')
    cbar.ax.set_title(r'$T^*$')
    cbar.set_ticks(ticks=[0.6, 1, 2, 6, 10], labels=[0.6, 1, 2, 6, 10])

    return fig


def plot_viscosity_conductivity_representation(df_visc, df_tcond, width=4, height=4, lambda_r=12,
                                               rho_lower=-2e-2, rho_upper=1.25,
                                               T_lower=0.6, T_upper=10.,
                                               Sres_lower=-2e-2, Sres_upper=4.,
                                               visc_lower=0., visc_upper=12.,
                                               logvisc_lower=8e-2, logvisc_upper=2e1,
                                               visc_lower_scaled=0., visc_upper_scaled=6.,
                                               tcond_lower=0., tcond_upper=26.,
                                               logtcond_lower=5e-1, logtcond_upper=4e1,
                                               tcond_lower_scaled=1., tcond_upper_scaled=12,
                                               markersize=4):

    colormap = mpl.colormaps['viridis']
    norm = mpl.colors.LogNorm(vmin=T_lower, vmax=T_upper)
    kwargs = dict(cmap=colormap, norm=norm, marker='s', clip_on=True, s=markersize)
    rho_ticks = np.linspace(0., 1.2, 5)
    Sres_ticks = [0, 1, 2, 3, 4]
    df_visc_lr = df_visc[df_visc['lr'] == lambda_r].reset_index(drop=True)
    df_tcond_lr = df_tcond[df_tcond['lr'] == lambda_r].reset_index(drop=True)

    ##########
    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax1v = fig.add_subplot(231)
    ax2v = fig.add_subplot(232)
    ax3v = fig.add_subplot(233)
    ax1t = fig.add_subplot(234)
    ax2t = fig.add_subplot(235)
    ax3t = fig.add_subplot(236)

    axs_rho = [ax1v, ax2v, ax1t, ax2t]
    for ax in axs_rho:
        ax.set_xlim(rho_lower, rho_upper)
        ax.set_xticks(rho_ticks)
        ax.tick_params(which='both', direction='in')
        ax.grid(True)
        ax.set_xlabel(r'$\rho^*$')

    axs_ent = [ax3v, ax3t]
    for ax in axs_ent:
        ax.set_xlim(Sres_lower, Sres_upper)
        ax.set_xticks(Sres_ticks)
        ax.tick_params(which='both', direction='in')
        ax.grid(True)
        ax.set_xlabel(r'$-S^{*, \mathrm{res}}$')

    axs_all = [ax1v, ax2v, ax3v, ax1t, ax2t, ax3t]
    titles = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for ax, title in zip(axs_all, titles):
        ax.set_title(title)

    # Setting labels
    ax1v.set_ylabel(r'$\eta^*$')
    ax2v.set_ylabel(r'$\eta^*$')
    ax3v.set_ylabel(r'$\tilde{\eta}^*$')

    ax1t.set_ylabel(r'$\kappa^*$')
    ax2t.set_ylabel(r'$\kappa^*$')
    ax3t.set_ylabel(r'$\tilde{\kappa}^*$')

    # setting limits
    ax1v.set_ylim(visc_lower, visc_upper)
    ax2v.set_ylim(logvisc_lower, logvisc_upper)
    ax3v.set_ylim(visc_lower_scaled, visc_upper_scaled)

    ax1t.set_ylim(tcond_lower, tcond_upper)
    ax2t.set_ylim(logtcond_lower, logtcond_upper)
    ax3t.set_ylim(tcond_lower_scaled, tcond_upper_scaled)

    #####################
    # Plotting the data #
    #####################

    # Viscosity
    # Linear scale
    ax1v.scatter(df_visc_lr['rho*'], df_visc_lr['shear_viscosity'], c=df_visc_lr['T*'], **kwargs)

    # Log scale
    ax2v.set_yscale('log')
    ax2v.scatter(df_visc_lr['rho*'], df_visc_lr['shear_viscosity'], c=df_visc_lr['T*'], **kwargs)

    # Entropy scaling reduced viscosity
    scaled_viscosity = viscosity_scaling(df_visc_lr['rho*'], df_visc_lr['T*'], df_visc_lr['shear_viscosity'])
    ax3v.scatter(-df_visc_lr['Sr'], scaled_viscosity, c=df_visc_lr['T*'], **kwargs)

    # Thermal conductivity
    # Linear scale
    ax1t.scatter(df_tcond_lr['rho*'], df_tcond_lr['thermal_conductivity'], c=df_tcond_lr['T*'], **kwargs)

    # Log scale
    ax2t.set_yscale('log')
    ax2t.scatter(df_tcond_lr['rho*'], df_tcond_lr['thermal_conductivity'], c=df_tcond_lr['T*'], **kwargs)

    # Entropy scaling reduced thermal conductivity
    scaled_tcond = thermal_conductivity_scaling(df_tcond_lr['rho*'], df_tcond_lr['T*'], df_tcond_lr['thermal_conductivity'])
    ax3t.scatter(-df_tcond_lr['Sr'], scaled_tcond, c=df_tcond_lr['T*'], **kwargs)

    # color bar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax = axs_all, pad=0.01, location='right')
    cbar.ax.set_title(r'$T^*$')
    cbar.set_ticks(ticks=[0.6, 1, 2, 6, 10], labels=[0.6, 1, 2, 6, 10])

    return fig
