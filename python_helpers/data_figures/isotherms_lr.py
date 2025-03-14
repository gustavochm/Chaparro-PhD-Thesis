import numpy as np
import pandas as pd
import nest_asyncio
nest_asyncio.apply()

import jax
from jax import numpy as jnp
from jax.config import config

from ..helpers import helper_get_alpha

PRECISSION = 'float64'
if PRECISSION == 'float64':
    config.update("jax_enable_x64", True)
    type_np = np.float64
    type_jax = jnp.float64
else:
    config.update("jax_enable_x32", True)
    type_np = np.float32
    type_jax = jnp.float32


def data_isotherms_lr(fun_dic, T_list, lambda_r=12., rho_min=0., rho_max=1.25, n=200):

    thermophysical_properties_fun = fun_dic['thermophysical_properties_fun']

    lambda_a = 6.
    alpha = helper_get_alpha(lambda_r, lambda_a)

    rho_plot = jnp.linspace(rho_min, rho_max, n)
    alpha_plot = alpha * jnp.ones(n)

    dict_isotherms = dict()
    for Tad in T_list:
        T_plot = Tad * jnp.ones(n)
        out = thermophysical_properties_fun(alpha_plot, rho_plot, T_plot)
        dict_isotherms[f'T={Tad:.2f}'] = pd.DataFrame(out)

    return dict_isotherms