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

PRECISSION = 'float64'
if PRECISSION == 'float64':
    config.update("jax_enable_x64", True)
    type_np = np.float64
    type_jax = jnp.float64
else:
    config.update("jax_enable_x32", True)
    type_np = np.float32
    type_jax = jnp.float32


def data_phase_equilibria_solid_lr(fun_dic, lambda_r=12., lambda_a=6.,
                                   Tlower=0.4, Tupper=10., n_vle=100, n_sle=1000, n_sve=50,
                                   initial_triple_point=None, initial_crit_point=None):

    enthalpy_residual_fun = fun_dic['enthalpy_residual_fun']
    internal_energy_residual_fun = fun_dic['internal_energy_residual_fun']

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

    # solving triple point
    if initial_triple_point is None:
        initial_triple_point = [1e-3, 0.85, 0.95, 0.67] # initial guesses for lr=12

    sol_triple = triple_point_solver(alpha, fun_dic, inc0=initial_triple_point)
    rhovad_triple = sol_triple[0]
    rholad_triple = sol_triple[1]
    rhosad_triple = sol_triple[2]
    T_triple = sol_triple[3]
    P_triple = sol_triple[4]

    Uv_triple = internal_energy_residual_fun(np.array([alpha]), np.array([rhovad_triple]), np.array([T_triple]))
    Ul_triple = internal_energy_residual_fun(np.array([alpha]), np.array([rholad_triple]), np.array([T_triple]))
    Us_triple = internal_energy_residual_fun(np.array([alpha]), np.array([rhosad_triple]), np.array([T_triple]))

    dUvap_triple = float((Uv_triple - Ul_triple)[0])
    dUmel_triple = float((Ul_triple - Us_triple)[0])
    dUsub_triple = float((Uv_triple - Us_triple)[0])

    # getting VLE
    T_vle_model = np.linspace(1.01 * T_triple, 0.99999*Tcad_model, n_vle)

    rhov_vle_model = np.zeros(n_vle)
    rhol_vle_model = np.zeros(n_vle)
    P_vle_model = np.zeros(n_vle)

    i = 0
    rho0 = [rhovad_triple, rholad_triple]
    sol_vle = vle_solver(alpha, T_vle_model[i], Pad0=P_triple, rho0=rho0, fun_dic=fun_dic, critical=critical_point)
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

    # getting sle
    T_sle_model = np.linspace(1.0 * T_triple, Tupper, n_sle)

    rhos_sle_model = np.zeros(n_sle)
    rhol_sle_model = np.zeros(n_sle)
    P_sle_model = np.zeros(n_sle)

    i = 0
    rho0 = [rholad_triple, rhosad_triple]
    sol_sle = sle_solver(alpha, T_sle_model[i], fun_dic, rho0=rho0)
    rhol_sle_model[i] = sol_sle[1]
    rhos_sle_model[i] = sol_sle[2]
    P_sle_model[i] = sol_sle[0]

    for i in range(1, n_sle):
        rho0 = [rhol_sle_model[i-1], rhos_sle_model[i-1]]
        sol_sle = sle_solver(alpha, T_sle_model[i], fun_dic, rho0=rho0)
        rhol_sle_model[i] = sol_sle[1]
        rhos_sle_model[i] = sol_sle[2]
        P_sle_model[i] = sol_sle[0]
        if np.isclose(rhol_sle_model[i], rhos_sle_model[i]):
            break
        if np.isnan(rhol_sle_model[i]) or np.isnan(rhos_sle_model[i]):
            break

    alpha_sle = alpha * np.ones(n_sle)
    # melting vaporization
    enthalpy_sle_liq = enthalpy_residual_fun(alpha_sle, rhol_sle_model, T_sle_model)
    enthalpy_sle_sol = enthalpy_residual_fun(alpha_sle, rhos_sle_model, T_sle_model)
    Hmelting_sle_model = np.asarray(enthalpy_sle_liq - enthalpy_sle_sol)

    internal_sle_liq = internal_energy_residual_fun(alpha_sle, rhol_sle_model, T_sle_model)
    internal_sle_sol = internal_energy_residual_fun(alpha_sle, rhos_sle_model, T_sle_model)
    Umelting_sle_model = np.asarray(internal_sle_liq - internal_sle_sol)

    T_sle_model = T_sle_model[:i]
    rhol_sle_model = rhol_sle_model[:i]
    rhos_sle_model = rhos_sle_model[:i]
    P_sle_model = P_sle_model[:i]
    Hmelting_sle_model = Hmelting_sle_model[:i]
    Umelting_sle_model = Umelting_sle_model[:i]

    # getting sve 
    T_sve_model = np.linspace(1.0 * T_triple, Tlower, n_sve)

    rhov_sve_model = np.zeros(n_sve)
    rhos_sve_model = np.zeros(n_sve)
    P_sve_model = np.zeros(n_sve)

    i = 0
    rho0 = [rhovad_triple, rhosad_triple]
    sol_sve = sve_solver(alpha, T_sve_model[i], fun_dic, rho0=rho0)
    rhov_sve_model[i] = sol_sve[1]
    rhos_sve_model[i] = sol_sve[2]
    P_sve_model[i] = sol_sve[0]

    for i in range(1, n_sve):
        rho0 = [rhov_sve_model[i-1], rhos_sve_model[i-1]]
        sol_sve = sve_solver(alpha, T_sve_model[i], fun_dic, rho0=rho0)
        rhov_sve_model[i] = sol_sve[1]
        rhos_sve_model[i] = sol_sve[2]
        P_sve_model[i] = sol_sve[0]

    alpha_sve = alpha * np.ones(n_sve)

    # sublimation changes
    enthalpy_sve_vap = enthalpy_residual_fun(alpha_sve, rhov_sve_model, T_sve_model)
    enthalpy_sve_sol = enthalpy_residual_fun(alpha_sve, rhos_sve_model, T_sve_model)
    Hsub_sve_model = np.asarray(enthalpy_sve_vap - enthalpy_sve_sol)

    internal_sve_vap = internal_energy_residual_fun(alpha_sve, rhov_sve_model, T_sve_model)
    internal_sve_sol = internal_energy_residual_fun(alpha_sve, rhos_sve_model, T_sve_model)
    Usub_sve_model = np.asarray(internal_sve_vap - internal_sve_sol)

    df_info = pd.DataFrame({'lambda_r': [lambda_r], 'lambda_a': [lambda_a], 'alpha': [alpha],
                            'Tcad_model': [Tcad_model], 'Pcad_model': [Pcad_model], 'rhocad_model': [rhocad_model],
                            'T_triple': [T_triple], 'P_triple': [P_triple], 'rhovad_triple': [rhovad_triple],
                            'rholad_triple': [rholad_triple], 'rhosad_triple': [rhosad_triple],
                            'dUvap_triple': [dUvap_triple], 'dUmel_triple': [dUmel_triple], 'dUsub_triple': [dUsub_triple]})
    df_vle = pd.DataFrame({'T_vle_model': T_vle_model, 'P_vle_model': P_vle_model, 
                           'rhov_vle_model': rhov_vle_model, 'rhol_vle_model': rhol_vle_model,
                           'Hvap_vle_model': Hvap_vle_model, 'Uvap_vle_model': Uvap_vle_model})
    df_vle.dropna(how='any', inplace=True)
    df_sle = pd.DataFrame({'T_sle_model': T_sle_model, 'P_sle_model': P_sle_model,
                           'rhol_sle_model': rhol_sle_model, 'rhos_sle_model': rhos_sle_model,
                           'Hmelting_sle_model': Hmelting_sle_model, 'Umelting_sle_model': Umelting_sle_model})
    df_sle.dropna(how='any', inplace=True)
    df_sve = pd.DataFrame({'T_sve_model': T_sve_model, 'P_sve_model': P_sve_model, 
                           'rhov_sve_model': rhov_sve_model, 'rhos_sve_model': rhos_sve_model,
                           'Hsub_sve_model': Hsub_sve_model, 'Usub_sve_model': Usub_sve_model})
    df_sve.dropna(how='any', inplace=True)

    data_df = {'info': df_info, 'vle': df_vle, 'sle': df_sle, 'sve': df_sve}
    return data_df
