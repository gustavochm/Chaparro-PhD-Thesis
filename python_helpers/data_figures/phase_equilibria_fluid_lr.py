import numpy as np
import pandas as pd
import nest_asyncio
nest_asyncio.apply()

import jax
from jax import numpy as jnp
from jax.config import config

from ..helpers import helper_get_alpha
from ..feanneos import vle_solver, sve_solver, sle_solver
from ..feanneos import critical_point_solver, triple_point_solver

from .sle_mie_correlation_lr import Tsle_correlation_khrapak

PRECISSION = 'float64'
if PRECISSION == 'float64':
    config.update("jax_enable_x64", True)
    type_np = np.float64
    type_jax = jnp.float64
else:
    config.update("jax_enable_x32", True)
    type_np = np.float32
    type_jax = jnp.float32


def data_phase_equilibria_fluid_lr(fun_dic, lambda_r=12., Tlower=0.6, n_vle=100,
                                   rho_min_sle=0.8, rho_max_sle=1.3, n_sle=100,
                                   initial_crit_point=None):

    enthalpy_residual_fun = fun_dic['enthalpy_residual_fun']
    internal_energy_residual_fun = fun_dic['internal_energy_residual_fun']

    lambda_a = 6.
    alpha = helper_get_alpha(lambda_r, lambda_a)

    # solving critical point
    if initial_crit_point is None:
        if lambda_r < 9:
            initial_crit_point = [0.3, 1.8]
        elif lambda_r > 30:
            initial_crit_point = [0.3, 0.9]
        else:
            initial_crit_point = [0.3, 1.3]

    sol_crit = critical_point_solver(alpha, fun_dic, inc0=initial_crit_point)
    rhocad_model, Tcad_model, Pcad_model = sol_crit
    critical_point = [rhocad_model, Tcad_model, Pcad_model]

    # getting VLE
    T_vle_model = np.linspace(Tlower, 0.99999*Tcad_model, n_vle)

    rhov_vle_model = np.zeros(n_vle)
    rhol_vle_model = np.zeros(n_vle)
    P_vle_model = np.zeros(n_vle)

    i = 0
    sol_vle = vle_solver(alpha, T_vle_model[i], fun_dic=fun_dic, critical=critical_point)
    P_vle_model[i], rhov_vle_model[i], rhol_vle_model[i] = sol_vle

    for i in range(1, n_vle):
        sol_vle = vle_solver(alpha, T_vle_model[i], Pad0=P_vle_model[i-1], fun_dic=fun_dic,
                             critical=critical_point, rho0=[rhov_vle_model[i-1], rhol_vle_model[i-1]],
                             good_initial=True)
        P_vle_model[i], rhov_vle_model[i], rhol_vle_model[i] = sol_vle

    alpha_vle = alpha*np.ones(n_vle)
    enthalpy_vap = enthalpy_residual_fun(alpha_vle, rhov_vle_model, T_vle_model)
    enthalpy_liq = enthalpy_residual_fun(alpha_vle, rhol_vle_model, T_vle_model)
    Hvap_vle_model = np.asarray(enthalpy_vap - enthalpy_liq)

    internal_vap = internal_energy_residual_fun(alpha_vle, rhov_vle_model, T_vle_model)
    internal_liq = internal_energy_residual_fun(alpha_vle, rhol_vle_model, T_vle_model)
    Uvap_vle_model = np.asarray(internal_vap - internal_liq)

    # getting SLE from correlation
    rhol_sle_model = np.linspace(rho_min_sle, rho_max_sle, n_sle)
    T_sle_model = Tsle_correlation_khrapak(rhol_sle_model, lambda_r)
    alpha_sle = alpha*np.ones(n_sle)
    P_sle_model = fun_dic['pressure_fun'](alpha_sle, rhol_sle_model, T_sle_model)

    df_info = pd.DataFrame({'lambda_r': [lambda_r], 'lambda_a': [lambda_a], 'alpha': [alpha],
                            'Tcad_model': [Tcad_model], 'Pcad_model': [Pcad_model], 'rhocad_model': [rhocad_model]})
    df_vle = pd.DataFrame({'T_vle_model': T_vle_model, 'P_vle_model': P_vle_model,
                           'rhov_vle_model': rhov_vle_model, 'rhol_vle_model': rhol_vle_model,
                           'Hvap_vle_model': Hvap_vle_model, 'Uvap_vle_model': Uvap_vle_model})
    df_sle = pd.DataFrame({'T_sle_model': T_sle_model, 'rhol_sle_model': rhol_sle_model, 'P_sle_model': P_sle_model})

    data_df = {'info': df_info, 'vle': df_vle, 'sle': df_sle}
    return data_df
