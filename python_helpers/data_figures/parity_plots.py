
import numpy as np
import pandas as pd
from ..helpers import helper_get_alpha
from sklearn.model_selection import train_test_split

import jax
from jax import numpy as jnp
from jax.config import config

import nest_asyncio
nest_asyncio.apply()

PRECISSION = 'float64'
if PRECISSION == 'float64':
    config.update("jax_enable_x64", True)
    type_np = np.float64
    type_jax = jnp.float64
else:
    config.update("jax_enable_x32", True)
    type_np = np.float32
    type_jax = jnp.float32


def parity_plot_computation(df_data, df_virial, fun_dic, seed_split=12, test_size=0.1):

    #######################
    # Reading the databases
    #######################

    # reading PVT data base
    df = df_data
    ms = np.asarray(df['ms'], dtype=type_np)
    lr = np.asarray(df['lr'], dtype=type_np)
    la = np.asarray(df['la'], dtype=type_np)
    rhoad = np.asarray(df['rho*'], dtype=type_np)
    # Min Energy Results
    Tad = np.asarray(df['T_ad'], dtype=type_np)
    Pad = np.asarray(df['P_ad'], dtype=type_np)
    Internal_ad = np.asarray(df['TotEn_ad'], dtype=type_np)
    # NVT results
    Cv_nvt = np.asarray(df['Cv_nvt'], dtype=type_np)
    ThermalPressureCoeff_nvt = np.asarray(df['ThermalPressureCoeff_nvt'], dtype=type_np)
    # NPT results
    Cp_npt = np.asarray(df['Cp_npt'], dtype=type_np)
    ThermalExpansionCoeff_npt = np.asarray(df['ThermalExpansionCoeff_npt'], dtype=type_np)
    IsothermalCompressibility_npt = np.asarray(df['IsothermalCompressibility_npt'], dtype=type_np)
    Cv_npt = np.asarray(df['Cv_npt'], dtype=type_np)
    JouleThomson_npt = np.asarray(df['JouleThomson_npt'], dtype=type_np)
    Gamma_npt = Cp_npt/Cv_npt
    is_stable = np.asarray(df['is_stable'], dtype=type_np)

    alpha = helper_get_alpha(lr, la)
    Z = Pad / (rhoad*Tad)

    Xdata = np.stack([alpha, rhoad, Tad, lr, la]).T

    Ydata = np.stack([Z, Internal_ad, Cv_nvt, Cp_npt, ThermalExpansionCoeff_npt,
                      rhoad*IsothermalCompressibility_npt, JouleThomson_npt, Gamma_npt,
                      ThermalPressureCoeff_nvt]).T

    out = train_test_split(Xdata, Ydata, is_stable, test_size=test_size, shuffle=True, random_state=seed_split)
    X_train, X_test, Y_train, Y_test, is_stable_train, is_stable_test = out

    # reading virial data base

    # Second and third virial coefficient database
    lr_virial = np.asarray(df_virial['lr'], dtype=type_np)
    la_virial = np.asarray(df_virial['la'], dtype=type_np)
    rhoad_virial = np.asarray(df_virial['rho*'], dtype=type_np)
    Tad_virial = np.asarray(df_virial['T*'], dtype=type_np)
    B2_virial = np.asarray(df_virial['B2*'], dtype= type_np)
    dB2_dT_virial = np.asarray(df_virial['dB2*_dT*'], dtype = type_np)
    B3_virial = np.asarray(df_virial['B3*'], dtype= type_np)

    alpha_virial = helper_get_alpha(lr_virial, la_virial)

    Xdata_virial = np.stack([alpha_virial, rhoad_virial, Tad_virial, lr_virial]).T
    Ydata_virial = np.stack([B2_virial, dB2_dT_virial, B3_virial]).T

    out = train_test_split(Xdata_virial, Ydata_virial, test_size=test_size, shuffle=True, random_state=seed_split)
    X_virial_train, X_virial_test, Y_virial_train, Y_virial_test = out

    # unpacking the shuffled PVT data
    alpha_train, rhoad_train, Tad_train, lr_train, la_train = X_train.T
    alpha_test, rhoad_test, Tad_test, lr_test, la_test = X_test.T

    Z_train, internal_train, Cv_train, Cp_train, alphap_train, rho_kappaT_train, muJT_train, Gamma_train, GammaV_train = Y_train.T
    Z_test, internal_test, Cv_test, Cp_test, alphap_test, rho_kappaT_test, muJT_test, Gamma_test, GammaV_test = Y_test.T

    # unpacking the shuffled virial data
    alpha_virial_train, rhoad_virial_train, Tad_virial_train, lr_virial_train = X_virial_train.T
    alpha_virial_test, rhoad_virial_test, Tad_virial_test, lr_virial_test = X_virial_test.T

    B2_train, dB2_dT_train, B3_train = Y_virial_train.T
    B2_test, dB2_dT_test, B3_test = Y_virial_test.T

    # conveting the shuffled data to pandas dataframes
    df_PVT_train = pd.DataFrame({'lr': lr_train, 'alpha': alpha_train, 'density': rhoad_train, 'temperature': Tad_train,
                                 'compressibility_factor': Z_train, 'internal_energy': internal_train, 'isochoric_heat_capacity': Cv_train,
                                 'rho_isothermal_compressibility': rho_kappaT_train, 'thermal_pressure_coefficient': GammaV_train, 
                                 'thermal_expansion_coefficient': alphap_train,
                                 'adiabatic_index': Gamma_train, 'joule_thomson_coefficient': muJT_train, 'isobaric_heat_capacity': Cp_train})

    df_PVT_test = pd.DataFrame({'lr': lr_test, 'alpha': alpha_test, 'density': rhoad_test, 'temperature': Tad_test,
                                'compressibility_factor': Z_test, 'internal_energy': internal_test, 'isochoric_heat_capacity': Cv_test,
                                'rho_isothermal_compressibility': rho_kappaT_test, 'thermal_pressure_coefficient': GammaV_test,
                                'thermal_expansion_coefficient': alphap_test,
                                'adiabatic_index': Gamma_test, 'joule_thomson_coefficient': muJT_test, 'isobaric_heat_capacity': Cp_test})

    df_virials_train = pd.DataFrame({'lr': lr_virial_train, 'alpha': alpha_virial_train, 'density': rhoad_virial_train, 'temperature': Tad_virial_train,
                                    'B2': B2_train, 'dB2_dT': dB2_dT_train, 'B3': B3_train})

    df_virials_test = pd.DataFrame({'lr': lr_virial_test, 'alpha': alpha_virial_test, 'density': rhoad_virial_test, 'temperature': Tad_virial_test,
                                    'B2': B2_test, 'dB2_dT': dB2_dT_test, 'B3': B3_test})

    #################################################################
    # Computing the parity plots using the provided FE-ANN EoS model
    #################################################################

    out_model_train = fun_dic['thermophysical_properties_fun'](alpha_train, rhoad_train, Tad_train)
    out_model_test = fun_dic['thermophysical_properties_fun'](alpha_test, rhoad_test, Tad_test)

    df_model_train = pd.DataFrame(out_model_train)
    df_model_test = pd.DataFrame(out_model_test)

    df_model_train['lr'] = lr_train
    df_model_test['lr'] = lr_test

    df_model_train = df_model_train[df_PVT_train.columns]
    df_model_test = df_model_test[df_PVT_test.columns]

    B2_model_train, dB2_dT_model_train, B3_model_train = fun_dic['virial_coefficients_fun'](alpha_virial_train, Tad_virial_train)
    df_model_virial_train = pd.DataFrame({'lr': lr_virial_train, 'alpha': alpha_virial_train, 'density': rhoad_virial_train, 'temperature': Tad_virial_train,
                                          'B2': B2_model_train, 'dB2_dT': dB2_dT_model_train, 'B3': B3_model_train})

    B2_model_test, dB2_dT_model_test, B3_model_test = fun_dic['virial_coefficients_fun'](alpha_virial_test, Tad_virial_test)
    df_model_virial_test = pd.DataFrame({'lr': lr_virial_test, 'alpha': alpha_virial_test, 'density': rhoad_virial_test, 'temperature': Tad_virial_test,
                                         'B2': B2_model_test, 'dB2_dT': dB2_dT_model_test, 'B3': B3_model_test})

    # saving the results to a single dictionary
    dict_results = {'model_train': df_model_train, 'model_test': df_model_test,
                    'model_virial_train': df_model_virial_train, 'model_virial_test': df_model_virial_test,
                    'PVT_train': df_PVT_train, 'PVT_test': df_PVT_test,
                    'virials_train': df_virials_train, 'virials_test': df_virials_test}
    return dict_results
