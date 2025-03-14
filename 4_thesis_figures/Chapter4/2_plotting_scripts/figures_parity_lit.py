import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.lines import Line2D

plt.ioff()


def plot_parity_tp_lit_data(dict_parity_lit, width, height,
                            markersize=10, markersize_label=5, colormap='viridis', fontsize_annotation=8,
                            diff_lower=-2, diff_upper=46,
                            visc_lower=-1, visc_upper=21,
                            tcond_lower=-1, tcond_upper=36,
                            alpha_min=0.2, alha_max=1.0,
                            authors_lit_markers={'Michels 1985': 'o', 'Heyes 1988': 'v', 'Heyes 1990': '^', 'Rowley 1997': '<',
                                                 'Vasquez 2004': '>', 'Galliero 2005': 'p', 'Nasrabad 2006': 'P', 'Bugel 2008': '*',
                                                 'Galliero 2009': 'h', 'Baidakov 2011': 'H', 'Baidakov 2014': 'X',
                                                 'Lautenschlaeger 2019': 'D', 'Slepavicius 2023': 'd'}):

    #########
    norm = mpl.colors.Normalize(vmin=alpha_min, vmax=alha_max)

    author_list = list(authors_lit_markers.keys())
    transport_list = ['self_diffusivity', 'shear_viscosity', 'thermal_conductivity']

    kwargs_symbols = {'s': markersize, 'clip_on': True, 'norm': norm, 'cmap': colormap, 'edgecolor': 'k', 'linewidths': 0.15}

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[1, 0.3])
    ax1 = subfigs[0].add_subplot(131)
    ax2 = subfigs[0].add_subplot(132)
    ax3 = subfigs[0].add_subplot(133)

    # setting limits
    ax1.set_xlim(diff_lower, diff_upper)
    ax1.set_ylim(diff_lower, diff_upper)
    ax2.set_xlim(visc_lower, visc_upper)
    ax2.set_ylim(visc_lower, visc_upper)
    ax3.set_xlim(tcond_lower, tcond_upper)
    ax3.set_ylim(tcond_lower, tcond_upper)

    ax1.plot([diff_lower, diff_upper], [diff_lower, diff_upper], color='k')
    ax2.plot([visc_lower, visc_upper], [visc_lower, visc_upper], color='k')
    ax3.plot([tcond_lower, tcond_upper], [tcond_lower, tcond_upper], color='k')

    ax1.set_xticks([0, 10, 20, 30, 40])
    ax1.set_yticks([0, 10, 20, 30, 40])

    ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.set_yticks([0, 5, 10, 15, 20])

    ax3.set_xticks([0, 10, 20, 30])
    ax3.set_yticks([0, 10, 20, 30])

    title_list = ['(a) Self-diffusivity', '(b) Shear viscosity', '(c) Thermal conductivity']
    label_list = [r'$D^*$ ', r'$\eta^*$', r'$\kappa^*$']

    for ax, title, transport, label in zip([ax1, ax2, ax3], title_list, transport_list, label_list):
        ax.set_title(title)
        ax.grid(True)
        ax.tick_params(direction='in', which='both')
        ax.set_xlabel(f'{label} (MD)')
        ax.set_ylabel(f'{label} (ANN)')

        df_lit = dict_parity_lit[transport]

        for author in author_list:
            df_author = df_lit[df_lit['author_id'] == author]
            ax.scatter(df_author[transport], df_author[f'{transport}_ann'], 
                    c=df_author['alpha'], marker=authors_lit_markers[author], **kwargs_symbols)

        mape = np.mean(100 * np.abs(df_lit[f'{transport}_ann']/df_lit[transport] - 1.))
        ax.text(0.05, 0.9, f'MAPE: {mape:.1f} \%', fontsize=fontsize_annotation, transform=ax.transAxes)

    cbar = subfigs[0].colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap), ax=[ax1, ax2, ax3], pad=0.02, aspect=20, location='right')
    cbar.ax.set_ylabel(r'$\alpha_\mathrm{vdw}$')

    custom_legend = []
    for key, value in authors_lit_markers.items():
        custom_legend.append(Line2D([0], [0], marker=value, color='k', label=key, markersize=markersize_label, linestyle=''))
    subfigs[1].legend(handles=custom_legend, loc='lower center', ncol=5, frameon=False, fontsize=fontsize_annotation)

    return fig
