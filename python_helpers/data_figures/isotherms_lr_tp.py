import pandas as pd
import numpy as np
from ..helpers import helper_get_alpha
import jax.numpy as jnp


def data_tp_isotherms_lr(dict_models,
                         dict_res_models, dict_dilute_functions,
                         T_list,
                         lambda_r=12., rho_min=0., rho_max=1.25, n=200,
                         transport_list=['self_diffusivity', 'shear_viscosity', 'thermal_conductivity']):

    lambda_a = 6.
    alpha = helper_get_alpha(lambda_r, lambda_a)

    rho_plot = jnp.linspace(rho_min, rho_max, n)
    alpha_plot = alpha * jnp.ones(n)
    lr_plot = lambda_r * jnp.ones(n)

    dict_tp_isotherms = dict()
    for Tad in T_list:
        T_plot = Tad * jnp.ones(n)
        dict_data = {'lr': lr_plot, 'alpha': alpha_plot, 'rho*': rho_plot, 'T*': T_plot}

        for transport_type in transport_list:

            ##############
            # Full Model #
            ##############
            tp_model = dict_models[transport_type]
            transport_property_ann = tp_model(alpha_plot, rho_plot, T_plot)
            if transport_type == 'self_diffusivity':
                dict_data[f'rho_{transport_type}_ann'] = transport_property_ann
                transport_property_ann = transport_property_ann / rho_plot

            elif transport_type == 'shear_viscosity' or transport_type == 'thermal_conductivity':
                transport_property_ann = np.exp(transport_property_ann)

            ##################
            # Residual Model #
            ##################
            transport_ideal = np.zeros(n)
            for i in range(n):
                transport_ideal[i] = dict_dilute_functions[transport_type](lr_plot[i], T_plot[i])

            tp_model_res = dict_res_models[transport_type]
            transport_property_res = tp_model_res(lr_plot, rho_plot, T_plot)

            if transport_type == 'self_diffusivity':
                transport_property_res = transport_property_res + transport_ideal
                dict_data[f'rho_{transport_type}_ann_res'] = transport_property_res
                transport_property_res = transport_property_res / rho_plot

            elif transport_type == 'shear_viscosity' or transport_type == 'thermal_conductivity':
                transport_property_res = np.exp(transport_property_res)
                transport_property_res = transport_property_res * transport_ideal

            dict_data[f'{transport_type}_ann'] = transport_property_ann
            dict_data[f'{transport_type}_ann_res'] = transport_property_res

        dict_tp_isotherms[f'T={Tad:.2f}'] = pd.DataFrame(dict_data)

    return dict_tp_isotherms
