import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.patches import Patch

plt.ioff()

def plot_Tc_Tt_comparison(component_list, Tc_Tt_dicts_all, component_names,
                          width=5, height=3, data_types=['exp', 'vle', 'vle_visc', 'vle_sle_sve'], fontsize_annotation=8, T_lower=1., T_upper=3.):

    dict_kwargs = dict()
    dict_kwargs['exp'] = dict(color='grey', hatch='//', label=r'NIST TRC')
    dict_kwargs['vle'] = dict(color='C0', hatch='o', label=r'OF$_1$ (VLE)')
    dict_kwargs['vle_visc'] = dict(color='C2', hatch=r'*', label=r'OF$_2$ (VLE + $\eta$)')
    dict_kwargs['vle_sle_sve'] = dict(color='C1', hatch='..', label=r'OF$_3$ (VLE + SLE + SVE)')

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.tick_params(direction='in', axis='y')

    ax.grid(True)
    x = np.arange(len(component_list))  # the label locations
    bar_width = 0.2  # the width of the bars

    for i, component in enumerate(component_list):
        Tc_Tt_dicts = Tc_Tt_dicts_all[component]

        # experimental
        for j, data_type in enumerate(data_types):
            kwargs = dict_kwargs[data_type]
            shift = j - len(data_types)/2 + 0.5
            rects = ax.bar(x[i] + shift * bar_width, Tc_Tt_dicts[data_type], width=bar_width, **kwargs)
            ax.bar_label(rects, padding=5, color=kwargs['color'], fontsize=fontsize_annotation,  rotation=90)

    component_name_list = [component_names[x] for x in component_list]
    ax.set_xticks(x)
    ax.set_xticklabels(component_name_list)
    ax.set_ylabel(r'$T_\mathrm{c}/T_\mathrm{tr}$')
    ax.set_ylim([T_lower, T_upper])

    legend_handles = []
    for data_type in data_types:
        legend_handles.append(Patch(color=dict_kwargs[data_type]['color'], label=dict_kwargs[data_type]['label'], hatch=dict_kwargs[data_type]['hatch']))
    legend = ax.legend(handles=legend_handles, loc='upper left', ncol=4, frameon=False, fontsize=fontsize_annotation)
    for j, data_type in enumerate(data_types):
        legend.texts[j].set_color(dict_kwargs[data_type]['color'])

    return fig
