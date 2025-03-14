import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root
from ..helpers import helper_get_alpha
from ..feanneos.BrownCurves import of_boyle_temp, of_charles_temp, of_amagat_temp
from ..feanneos.BrownCurves import zeno_obj, boyle_obj, charles_obj, amagat_obj
from .phase_equilibria_fluid_lr import data_phase_equilibria_fluid_lr


def data_brown_curves_fluid_lr(fun_dic, lambda_r, lambda_a=6., T_lower=0.6, 
                               n_b2=1000, T_upper_b2=50., 
                               n_zeno=500, rho0_zeno=1e-2, Tr_upper_zeno=0.99999,
                               n_boyle=300, rho0_boyle=1e-2, Tr_min_boyle=0.98, Tr_upper_boyle=0.99999,
                               n_charles=2000, rho0_charles=0.8, Tr_upper_charles=1.1,
                               n_amagat=200, rho0_amagat=0.2, T_upper_amagat=15.,
                               rho_max_sle=1.3):

    pressure_fun = fun_dic['pressure_fun']
    dhelmholtz_drho_fun = fun_dic['dhelmholtz_drho_fun']
    d2helmholtz_drho2_dT_fun = fun_dic['d2helmholtz_drho2_dT_fun']
    d2helmholtz_drho2_fun = fun_dic['d2helmholtz_drho2_fun']
    thermal_expansion_coeff_fun = fun_dic['thermal_expansion_coeff_fun']
    d2helmholtz_fun = fun_dic['d2helmholtz_fun']

    #############
    alpha = helper_get_alpha(lambda_r, lambda_a)

    # getting phase equilibria data
    data_lr = data_phase_equilibria_fluid_lr(fun_dic, lambda_r=lambda_r, rho_max_sle=rho_max_sle)
    df_info = data_lr['info']
    df_vle = data_lr['vle']
    # df_sle = data_lr['sle']
    Tcad_model = df_info['Tcad_model'].values[0]
    #######################
    # Virial coefficients #
    #######################
    alpha_virial = alpha * np.ones(n_b2)
    T_virial = np.linspace(T_lower, T_upper_b2, n_b2)
    B2, dB2_dT, B3 = fun_dic['virial_coefficients_fun'](alpha_virial, T_virial)

    # initial guesses for characteristic temperatures
    T_boyle_guess = T_virial[np.argmin(B2**2)]
    T_charles_guess = T_virial[np.argmin((B2/T_virial - dB2_dT)**2)]
    T_amagat_guess = T_virial[np.argmax(B2)]

    # Solving for characteristic temperatures
    # Boyle temperature
    sol_Tboyle = root(of_boyle_temp, T_boyle_guess, args=(alpha, dhelmholtz_drho_fun))
    Tboyle_model = sol_Tboyle.x[0]

    # Charles temperature
    sol_Tcharles = root(of_charles_temp, T_charles_guess, args=(alpha, d2helmholtz_drho2_dT_fun))
    Tcharles_model = sol_Tcharles.x[0]

    # Amagat temperature
    sol_Tamagat = root(of_amagat_temp, T_amagat_guess, args=(alpha, d2helmholtz_drho2_dT_fun))
    Tamagat_model = sol_Tamagat.x[0]

    ##############
    # Zeno Curve #
    ##############
    T_zeno_model = np.linspace(T_lower, Tr_upper_zeno*Tboyle_model, n_zeno)[::-1]
    rho_zeno_model = np.zeros(n_zeno)
    i = 0

    sol_zeno = root(zeno_obj, rho0_zeno, args=(alpha, T_zeno_model[i], dhelmholtz_drho_fun))
    rho_zeno_model[i] = sol_zeno.x

    for i in range(1, n_zeno):
        rho0 = rho_zeno_model[i-1]
        sol_zeno = root(zeno_obj, rho0, args=(alpha, T_zeno_model[i], dhelmholtz_drho_fun))
        rho_zeno_model[i] = sol_zeno.x

    alpha_zeno = alpha * np.ones(n_zeno)
    pressure_zeno_model = pressure_fun(alpha_zeno, rho_zeno_model, T_zeno_model)

    ###############
    # Boyle Curve #
    ###############
    T_boyle_model = np.linspace(Tr_min_boyle*Tcad_model, Tr_upper_boyle*Tboyle_model, n_boyle)[::-1]
    rho_boyle_model = np.zeros(n_boyle)

    i = 0
    rho0_boyle = 1e-2
    sol_boyle = root(boyle_obj, rho0_boyle, args=(alpha, T_boyle_model[i], d2helmholtz_drho2_fun))
    rho_boyle_model[i] = sol_boyle.x

    for i in range(1, n_boyle):
        rho0 = rho_boyle_model[i-1]
        sol_boyle = root(boyle_obj, rho0, args=(alpha, T_boyle_model[i], d2helmholtz_drho2_fun))
        rho_boyle_model[i] = sol_boyle.x

    alpha_boyle = alpha * np.ones(n_boyle)
    pressure_boyle_model = pressure_fun(alpha_boyle, rho_boyle_model, T_boyle_model)

    #################
    # Charles Curve #
    #################
    T_charles_model = np.linspace(T_lower, Tr_upper_charles*Tcharles_model, n_charles)
    rho_charles_model = np.zeros(n_charles)

    i = 0 
    sol_charles = root(charles_obj, rho0_charles, args=(alpha, T_charles_model[i], thermal_expansion_coeff_fun))
    rho_charles_model[i] = sol_charles.x
    for i in range(1, n_charles):
        rho0 = rho_charles_model[i-1]
        sol_charles = root(charles_obj, rho0, args=(alpha, T_charles_model[i], thermal_expansion_coeff_fun))
        if not sol_charles.success:
            break
        rho_charles_model[i] = sol_charles.x

    T_charles_model = T_charles_model[:i]
    rho_charles_model = rho_charles_model[:i]
    alpha_charles = alpha * np.ones_like(T_charles_model)
    pressure_charles_model = pressure_fun(alpha_charles, rho_charles_model, T_charles_model)
    pressure_charles_model = np.array(pressure_charles_model)

    bool_Tsat = T_charles_model < Tcad_model
    Psat_inter = interp1d(df_vle['T_vle_model'], df_vle['P_vle_model'], bounds_error=False)
    final_bool = np.logical_and(pressure_charles_model < Psat_inter(T_charles_model), bool_Tsat)

    pressure_charles_model[final_bool] = np.nan
    rho_charles_model[np.isnan(pressure_charles_model)] = np.nan
    T_charles_model[np.isnan(pressure_charles_model)] = np.nan

    ################
    # Amagat Curve #
    ################
    T_amagat_model = np.linspace(T_lower, T_upper_amagat, n_amagat)[::-1]
    rho_amagat_model = np.zeros(n_amagat)

    i = 0 
    sol_amagat = root(amagat_obj, rho0_amagat, args=(alpha, T_amagat_model[i], d2helmholtz_fun))
    rho_amagat_model[i] = sol_amagat.x

    for i in range(1, n_amagat):
        rho0 = rho_amagat_model[i-1]
        sol_amagat = root(amagat_obj, rho0, args=(alpha, T_amagat_model[i], d2helmholtz_fun))
        if not sol_amagat.success:
            break
        rho_amagat_model[i] = sol_amagat.x

    T_amagat_model = T_amagat_model[:i]
    rho_amagat_model = rho_amagat_model[:i]  
    alpha_amagat = alpha * np.ones_like(T_amagat_model)
    pressure_amagat_model = pressure_fun(alpha_amagat, rho_amagat_model, T_amagat_model)

    ################################
    # Saving results to dataframes #
    ################################

    # characteristic temperatures
    df_characteristic = pd.DataFrame({'T_boyle': Tboyle_model, 'T_charles': Tcharles_model,
                                    'Tamagat_model': Tamagat_model}, index=[0])

    # Second virial coefficient
    df_model_virial = pd.DataFrame({'T_virial': T_virial,  'B2': B2, 'dB2_dT': dB2_dT, 'B3': B3})

    # Zeno Curve
    df_model_zeno = pd.DataFrame({'T_zeno': T_zeno_model, 'pressure_zeno': pressure_zeno_model, 'rho_zeno': rho_zeno_model})
    df_model_zeno.dropna(how='all', inplace=True)
    df_model_zeno.sort_values('T_zeno', inplace=True)
    # Boyle Curve
    df_model_boyle = pd.DataFrame({'T_boyle': T_boyle_model, 'pressure_boyle': pressure_boyle_model,
                                'rho_boyle': rho_boyle_model})
    df_model_boyle.dropna(how='all', inplace=True)
    df_model_boyle.sort_values('T_boyle', inplace=True)
    # Charles Curve
    df_model_charles = pd.DataFrame({'T_charles': T_charles_model, 'pressure_charles': pressure_charles_model,
                                'rho_charles': rho_charles_model})
    df_model_charles.dropna(how='all', inplace=True)
    df_model_charles.sort_values('T_charles', inplace=True)
    # Amagat Curve
    df_model_amagat = pd.DataFrame({'T_amagat': T_amagat_model, 'pressure_amagat': pressure_amagat_model,
                                'rho_amagat': rho_amagat_model})
    df_model_amagat.dropna(how='all', inplace=True)
    df_model_amagat.sort_values('T_amagat', inplace=True)

    # saving results to the data dictionary
    data_lr['characteristic_temperatures'] = df_characteristic
    data_lr['virial'] = df_model_virial
    data_lr['zeno_curve'] = df_model_zeno
    data_lr['boyle_curve'] = df_model_boyle
    data_lr['charles_curve'] = df_model_charles
    data_lr['amagat_curve'] = df_model_amagat

    return data_lr
