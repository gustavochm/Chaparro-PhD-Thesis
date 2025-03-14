import numpy as np
import pandas as pd
from ..helpers import helper_get_alpha
from ..feanneos import vle_solver, sve_solver, sle_solver
from ..feanneos import critical_point_solver, triple_point_solver
# Constants
from scipy.constants import Boltzmann, Avogadro
kb = Boltzmann # [J/K] Boltzman's constant
Na = Avogadro  # [mol-1] Avogadro's Number
R = Na * kb    # [J mol-1 K-1] Ideal gas constant


def values_vle_sle_sve_model(inc, DataFile, fun_dic, interpd_dict, lambda_a=6.):

    ##################################
    # Reading interpolated functions #
    ##################################
    crit_interp = interpd_dict['crit_interp']
    triple_interp = interpd_dict['triple_interp']
    vle_interp = interpd_dict['vle_interp']
    sle_interp = interpd_dict['sle_interp']
    sve_interp = interpd_dict['sve_interp']
    sle_maxT_interp = interpd_dict['sle_maxT_interp']

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

    # component SLE
    df_component_sle = pd.read_excel(DataFile, sheet_name='sle')
    Tsle = np.asarray(df_component_sle.iloc[:, 0], dtype=np.float64)  # K
    Psle = np.asarray(df_component_sle.iloc[:, 1], dtype=np.float64)*1000.  # Pa

    # component SVE
    df_component_sve = pd.read_excel(DataFile, sheet_name='sve')
    Tsve = np.asarray(df_component_sve.iloc[:, 0], dtype=np.float64)  # K
    Psve = np.asarray(df_component_sve.iloc[:, 1], dtype=np.float64)*1000.  # Pa
    Hsub_sve = np.asarray(df_component_sve.iloc[:, 2], dtype=np.float64) * 1000.  # J/mol

    ################################
    # Getting molecular parameters #
    ################################
    sigma, eps, lambda_r = inc
    sigma *= 1e-10 # m
    eps *= kb # J

    alpha = helper_get_alpha(lambda_r, lambda_a)

    energy_factor = 1 / (eps * Na)  # J/mol -> dim
    pressure_factor = sigma**3 / eps  # Pa -> adim
    rho_factor = sigma**3 * Na  # mol/m3 -> adim
    T_factor = kb / eps  # K -> adim

    values['energy_factor'] = energy_factor
    values['pressure_factor'] = pressure_factor
    values['rho_factor'] = rho_factor
    values['T_factor'] = T_factor

    #######################################
    # Computing critical and triple point #
    #######################################

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

    ##### computing triple point #######
    triple0 = triple_interp(alpha) # rhovad, rholad, rhosad, Tad, Pad
    initial_triple_point = [triple0[0], triple0[1], triple0[2], triple0[3]]
    sol_triple = triple_point_solver(alpha, fun_dic, inc0=initial_triple_point)
    rhovad_triple = sol_triple[0]
    rholad_triple = sol_triple[1]
    rhosad_triple = sol_triple[2]
    T_triple = sol_triple[3]
    P_triple = sol_triple[4]

    # saving experimental values
    values['Ttriple_exp'] = Ttriple
    # saving model values
    values['rhov_triple_model'] = rhovad_triple / rho_factor
    values['rhol_triple_model'] = rholad_triple / rho_factor
    values['rhos_triple_model'] = rhosad_triple / rho_factor
    values['Ttriple_model'] = T_triple / T_factor
    values['Ptriple_model'] = P_triple / pressure_factor

    ################################################
    # Computing VLE at the experimental conditions #
    ################################################
    #### Computing VLE ####
    Tvle_ad = Tvle * T_factor
    # Pvle_ad = Pvle * pressure_factor
    # rhov_vle_ad = rhov_vle * rho_factor
    # rhol_vle_ad = rhol_vle * rho_factor
    # Hvap_vle_ad = Hvap_vle * energy_factor

    # VLE calculation should be bounded by critical and triple temperature
    where_vle = np.logical_and(Tvle_ad > T_triple, Tvle_ad < 0.99 * Tcad_model)
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

    # saving experimental values
    values['Tvle_exp'] = Tvle
    values['rhov_vle_exp'] = rhov_vle
    values['rhol_vle_exp'] = rhol_vle
    values['Pvle_exp'] = Pvle
    values['Hvap_vle_exp'] = Hvap_vle
    # saving computed values
    values['rhov_vle_model'] = rhov_vle_model / rho_factor
    values['rhol_vle_model'] = rhol_vle_model / rho_factor
    values['Pvle_model'] = Pvle_model / pressure_factor
    values['Hvap_vle_model'] = Hvap_vle_model / energy_factor
    values['where_vle'] = where_vle

    ################################################
    # Computing SLE at the experimental conditions #
    ################################################

    Tsle_ad = Tsle * T_factor
    # SLE calculation should be bounded by maximum temperature computed by model
    where_sle = Tsle_ad < float(sle_maxT_interp(alpha))

    n_sle = len(Tsle_ad)
    alpha_sle = alpha * np.ones(n_sle)
    sle_initial_guesses = sle_interp(alpha_sle, Tsle_ad)  # P_sle, rhol_sle, rhos_sle
    rhol_sle_model = np.zeros(n_sle)
    rhos_sle_model = np.zeros(n_sle)
    Psle_model = np.zeros(n_sle)
    # Psle_model = np.zeros(n_sle)

    for i in range(n_sle):
        if where_sle[i]:
            rho0 = [sle_initial_guesses[i, 1], sle_initial_guesses[i, 2]]
            sol_sle = sle_solver(alpha, Tsle_ad[i], fun_dic, rho0=rho0)
            rhol_sle_model[i] = sol_sle[1]
            rhos_sle_model[i] = sol_sle[2]
            Psle_model[i] = sol_sle[0]

    enthalpy_liq = enthalpy_residual_fun(alpha_sle, rhol_sle_model, Tsle_ad)
    enthalpy_sol = enthalpy_residual_fun(alpha_sle, rhos_sle_model, Tsle_ad)
    Hmelting_sle_model = np.array(enthalpy_liq - enthalpy_sol)
    Hmelting_sle_model[~where_sle] = 0.0 # making sure values outside the boundaries

    # saving experimental values
    values['Tsle_exp'] = Tsle
    values['Psle_exp'] = Psle
    # saving computed values
    values['where_sle'] = where_sle
    values['rhol_sle_model'] = rhol_sle_model / rho_factor
    values['rhos_sle_model'] = rhos_sle_model / rho_factor
    values['Psle_model'] = Psle_model / pressure_factor
    values['Hmelting_sle_model'] = Hmelting_sle_model / energy_factor

    ################################################
    # Computing SVE at the experimental conditions #
    ################################################
    Tsve_ad = Tsve * T_factor

    where_sve = Tsve_ad < T_triple

    n_sve = len(Tsve_ad)
    alpha_sve = alpha * np.ones(n_sve)
    sve_initial_guesses = sve_interp(alpha_sve, Tsve_ad)  # P_sve, rhov_sve, rhol_sve

    rhov_sve_model = np.zeros(n_sve)
    rhos_sve_model = np.zeros(n_sve)
    Psve_model = np.zeros(n_sve)

    for i in range(n_sve):
        if where_sve[i]:
            rho0 = [sve_initial_guesses[i, 1], sve_initial_guesses[i, 2]]
            sol_sve = sle_solver(alpha, Tsve_ad[i], fun_dic, rho0=rho0)
            rhov_sve_model[i] = sol_sve[1]
            rhos_sve_model[i] = sol_sve[2]
            Psve_model[i] = sol_sve[0]

    enthalpy_vap = enthalpy_residual_fun(alpha_sve, rhov_sve_model, Tsve_ad)
    enthalpy_sol = enthalpy_residual_fun(alpha_sve, rhos_sve_model, Tsve_ad)
    Hsub_sve_model = np.array(enthalpy_vap - enthalpy_sol)
    Hsub_sve_model[~where_sve] = 0.0 # making sure values outside the boundaries

    # saving experimental values
    values['Tsve_exp'] = Tsve
    values['Psve_exp'] = Psve
    values['Hsub_sve_exp'] = Hsub_sve
    # saving computed values
    values['where_sve'] = where_sve
    values['rhov_sve_model'] = rhov_sve_model / rho_factor
    values['rhos_sve_model'] = rhos_sve_model / rho_factor
    values['Psve_model'] = Psve_model / pressure_factor
    values['Hsub_sve_model'] = Hsub_sve_model / energy_factor
    return values


def mie_params_of_vle_sle_sve(inc, DataFile, fun_dic, interpd_dict, 
                                   lambda_a=6., 
                                   weight_rhov_vle=1.,  
                                   weight_sle=1., weight_sve=1.,
                                   weight_enthalpy=1.,
                                   add_critical_point=True,
                                   add_triple_point=True,
                                   add_sle=True, 
                                   add_sve=True,
                                   loss_function=np.nanmean,):

    values = values_vle_sle_sve_model(inc, DataFile, fun_dic, interpd_dict, lambda_a=lambda_a)

    # error VLE
    loss = loss_function((values['Pvle_model']/values['Pvle_exp'] - 1.)**2)
    loss += weight_rhov_vle * loss_function((values['rhov_vle_model']/values['rhov_vle_exp'] - 1.)**2)
    loss += loss_function((values['rhol_vle_model']/values['rhol_vle_exp'] - 1.)**2)
    loss += weight_enthalpy * loss_function((values['Hvap_vle_model']/values['Hvap_vle_exp'] - 1.)**2)

    # error SLE
    if add_sle:
        loss += weight_sle * loss_function((values['Psle_model']/values['Psle_exp'] - 1.)**2)

    # error SVE
    if add_sve:
        loss += weight_sve * loss_function((values['Psve_model']/values['Psve_exp'] - 1.)**2)
        loss += weight_sve * weight_enthalpy * loss_function((values['Hsub_sve_model']/values['Hsub_sve_exp'] - 1.)**2)

    if add_critical_point: 
        # error critical point
        loss += (values['Tc_model']/values['Tc_exp'] - 1.)**2
    if add_triple_point:
        # error triple point
        loss += (values['Ttriple_model']/values['Ttriple_exp'] - 1.)**2
    return loss
