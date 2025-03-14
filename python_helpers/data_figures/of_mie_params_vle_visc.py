import numpy as np
import pandas as pd
from ..helpers import helper_get_alpha
from ..feanneos import vle_solver, critical_point_solver
# Constants
from scipy.constants import Boltzmann, Avogadro
kb = Boltzmann # [J/K] Boltzman's constant
Na = Avogadro  # [mol-1] Avogadro's Number
R = Na * kb    # [J mol-1 K-1] Ideal gas constant


def values_vle_visc_model(inc, expdata_exel, fun_dic, visc_fun, interpd_dict, lambda_a=6.):

    ##################################
    # Reading interpolated functions #
    ##################################
    crit_interp = interpd_dict['crit_interp']
    vle_interp = interpd_dict['vle_interp']

    ########################################
    # Reading functions from the feann EoS #
    ########################################

    # helmholtz_fun = fun_dic['helmholtz_fun']
    # dhelmholtz_drho_fun = fun_dic['dhelmholtz_drho_fun']
    # d2helmholtz_drho2_dT_fun = fun_dic['d2helmholtz_drho2_dT_fun']
    # d2helmholtz_drho2_fun = fun_dic['d2helmholtz_drho2_fun']
    # d2helmholtz_fun = fun_dic['d2helmholtz_fun']

    # pressure_fun = fun_dic['pressure_fun']
    # dpressure_drho_fun = fun_dic['dpressure_drho_fun']
    # d2pressure_drho2_fun = fun_dic['d2pressure_drho2_fun']
    # pressure_and_chempot_fun = fun_dic['pressure_and_chempot_fun']
    enthalpy_residual_fun = fun_dic['enthalpy_residual_fun']
    # thermal_expansion_coeff_fun = fun_dic['thermal_expansion_coeff_fun'] 

    ############################################################
    # Dictionary to store the computed and experimental values #
    ############################################################
    values = {}

    #####################################################
    # Reading the experimental data from the excel file #
    #####################################################
    DataFile = expdata_exel

    # component info
    df_component_info = pd.read_excel(DataFile, sheet_name='info')
    Mw = float(df_component_info['Mw'])  # g/mol
    Tc = float(df_component_info.iloc[0, 1])  # K
    rhoc = float(df_component_info.iloc[0, 2]) * 1000 / Mw  # mol/m3
    Pc = float(df_component_info.iloc[0, 3]) * 1000  # Pa
    Ttriple = float(df_component_info.iloc[0, 4])  # K

    # component VLE
    df_component_vle = pd.read_excel(DataFile, sheet_name='vle')
    Tvle = np.asarray(df_component_vle.iloc[:, 0], dtype=np.float64)  # K
    Pvle = np.asarray(df_component_vle.iloc[:, 1], dtype=np.float64) * 1000.  # Pa
    rhov_vle = np.asarray(df_component_vle.iloc[:, 2], dtype=np.float64) * 1000. / Mw  # mol/m3
    rhol_vle = np.asarray(df_component_vle.iloc[:, 3], dtype=np.float64) * 1000. / Mw  # mol/m3
    Hvap_vle = np.asarray(df_component_vle.iloc[:, 4], dtype=np.float64) * 1000.  # J/mol
    speed_of_sound_vle = np.asarray(df_component_vle.iloc[:, 5], dtype=np.float64)  # m/s
    thermal_conductivity_vle = np.asarray(df_component_vle.iloc[:, 6], dtype=np.float64)  # W/m K
    viscosity_vle = np.asarray(df_component_vle.iloc[:, 7], dtype=np.float64) # Pa s
    """
    # component SLE
    df_component_sle = pd.read_excel(DataFile, sheet_name='sle')
    Tsle = np.asarray(df_component_sle.iloc[:, 0], dtype=np.float64)  # K
    Psle = np.asarray(df_component_sle.iloc[:, 1], dtype=np.float64)*1000.  # Pa

    # component SVE
    df_component_sve = pd.read_excel(DataFile, sheet_name='sve')
    Tsve = np.asarray(df_component_sve.iloc[:, 0], dtype=np.float64)  # K
    Psve = np.asarray(df_component_sve.iloc[:, 1], dtype=np.float64)*1000.  # Pa
    Hsub_sve = np.asarray(df_component_sve.iloc[:, 2], dtype=np.float64) * 1000.  # J/mol
    """
    ################################################
    # Computing VLE at the experimental conditions #
    ################################################
    sigma, eps, lambda_r = inc
    sigma *= 1e-10 # m
    eps *= kb # J

    alpha = helper_get_alpha(lambda_r, lambda_a)

    energy_factor = 1 / (eps * Na)  # J/mol -> dim
    pressure_factor = sigma**3 / eps  # Pa -> adim
    rho_factor = sigma**3 * Na  # mol/m3 -> adim
    T_factor = kb / eps  # K -> adim

    visc_factor = 1. / (np.sqrt(eps * (Mw/1000./Na)) / sigma**2)  # Pa s -> adim

    values['energy_factor'] = energy_factor
    values['pressure_factor'] = pressure_factor
    values['rho_factor'] = rho_factor
    values['T_factor'] = T_factor
    values['visc_factor'] = visc_factor

    ##### computing critical point #######
    # rhocad = rhoc*rho_factor
    # Tcad = Tc*T_factor
    # Pcad = Pc * pressure_factor

    crit0 = crit_interp(alpha)  # rhocad, Tcad, Pcad
    initial_crit_point = [crit0[0], crit0[1]]
    sol_crit = critical_point_solver(alpha, fun_dic, inc0=initial_crit_point)
    rhocad_model, Tcad_model, Pcad_model = sol_crit
    critical_point = [rhocad_model, Tcad_model, Pcad_model]

    # saving experimental values
    values['rhoc_exp'] = rhoc
    values['Tc_exp'] = Tc
    values['Pc_exp'] = Pc
    # saving model values
    values['rhoc_model'] = rhocad_model / rho_factor
    values['Tc_model'] = Tcad_model / T_factor
    values['Pc_model'] = Pcad_model / pressure_factor

    #### Computing VLE ####
    Tvle_ad = Tvle * T_factor
    # Pvle_ad = Pvle * pressure_factor
    # rhov_vle_ad = rhov_vle * rho_factor
    # rhol_vle_ad = rhol_vle * rho_factor
    # Hvap_vle_ad = Hvap_vle * energy_factor

    # VLE calculation should be bounded by critical and triple temperature
    where_vle = Tvle_ad < 0.99 * Tcad_model
    n_vle = len(Tvle_ad)
    alpha_vle = alpha * np.ones(n_vle)
    vle_initial_guesses = vle_interp(alpha_vle, Tvle_ad/Tcad_model)

    rhov_vle_model = np.zeros(n_vle)
    rhol_vle_model = np.zeros(n_vle)
    Pvle_model = np.zeros(n_vle)
    for i in range(n_vle):
        if where_vle[i]:
            sol_vle = vle_solver(alpha, Tvle_ad[i], Pad0=vle_initial_guesses[i, 0], fun_dic=fun_dic,
                                 critical=critical_point, rho0=[vle_initial_guesses[i, 1], vle_initial_guesses[i, 2]])
            Pvle_model[i], rhov_vle_model[i], rhol_vle_model[i] = sol_vle

    enthalpy_vap = enthalpy_residual_fun(alpha_vle, rhov_vle_model, Tvle_ad)
    enthalpy_liq = enthalpy_residual_fun(alpha_vle, rhol_vle_model, Tvle_ad)
    Hvap_vle_model = np.array(enthalpy_vap - enthalpy_liq)
    Hvap_vle_model[~where_vle] = 0.0 # making sure values outside the boundaries

    #### Computing viscosity at the VLE conditions ####
    visc_vle_model = visc_fun(alpha_vle, rhol_vle_model, Tvle_ad)

    # saving experimental values
    values['Tvle_exp'] = Tvle
    values['rhov_vle_exp'] = rhov_vle
    values['rhol_vle_exp'] = rhol_vle
    values['Pvle_exp'] = Pvle
    values['Hvap_vle_exp'] = Hvap_vle
    values['visc_vle_exp'] = viscosity_vle
    # saving computed values
    values['rhov_vle_model'] = rhov_vle_model / rho_factor
    values['rhol_vle_model'] = rhol_vle_model / rho_factor
    values['Pvle_model'] = Pvle_model / pressure_factor
    values['Hvap_vle_model'] = Hvap_vle_model / energy_factor
    values['visc_vle_model'] = visc_vle_model / visc_factor
    values['where_vle'] = where_vle

    return values


def mie_params_of_vle_visc(inc, DataFile, fun_dic, visc_fun, interpd_dict,
                           lambda_a=6.,
                           weight_rhov_vle=1., weight_hvap=1., add_critical_point=True,
                           weight_visc=1.,
                           loss_function=np.nanmean):

    values = values_vle_visc_model(inc, DataFile, fun_dic, visc_fun, interpd_dict, lambda_a=lambda_a)

    # error VLE
    loss = loss_function((values['Pvle_model']/values['Pvle_exp'] - 1.)**2)
    loss += weight_rhov_vle * loss_function((values['rhov_vle_model']/values['rhov_vle_exp'] - 1.)**2)
    loss += loss_function((values['rhol_vle_model']/values['rhol_vle_exp'] - 1.)**2)
    loss += weight_hvap * loss_function((values['Hvap_vle_model']/values['Hvap_vle_exp'] - 1.)**2)

    if add_critical_point:
        # error critical point
        loss += (values['Tc_model']/values['Tc_exp'] - 1.)**2

    # error viscosity
    loss += weight_visc * loss_function((values['visc_vle_model']/values['visc_vle_exp'] - 1.)**2)

    return loss
