import numpy as np
import pandas as pd
from ..helpers import helper_get_alpha
from scipy.constants import Avogadro as Na
from scipy.constants import Boltzmann as kb
import teqp


def data_phase_equilibria_fluid_saft_lr(lambda_r=12., Tlower=0.6, n_vle=100, initial_crit_point=None):

    lambda_a = 6.
    alpha = helper_get_alpha(lambda_r, lambda_a)
    n = n_vle

    z = np.array([1.0])
    m = 1.
    epsilon_over_k = 150  # K^-1
    sigma_m = 3e-10  # m

    # Defyining teqp model
    model = teqp.make_model({
            "kind": 'SAFT-VR-Mie',
            "model": {
                "coeffs": [{
                    "name": "Pseudo",
                    "BibTeXKey": "Lafitte",
                    "m": m,
                    "epsilon_over_k": epsilon_over_k,
                    "sigma_m": sigma_m,
                    "lambda_r": lambda_r,
                    "lambda_a": lambda_a
                }]
            }
        })

    T_factor = 1. / epsilon_over_k
    P_factor = sigma_m**3 / (epsilon_over_k*kb)
    rho_factor = sigma_m**3 * Na
    energy_factor = 1. / ((epsilon_over_k * kb) * Na)

    # solving critical point
    if initial_crit_point is None:
        if lambda_r < 9:
            initial_crit_point = [0.3, 1.8]
        elif lambda_r > 30:
            initial_crit_point = [0.3, 0.9]
        else:
            initial_crit_point = [0.3, 1.3]

    # Getting Critical Point
    Tc0 = initial_crit_point[1] / T_factor
    rhoc0 = initial_crit_point[0] / rho_factor
    Tc, rhoc = model.solve_pure_critical(Tc0, rhoc0)
    pc = rhoc*model.get_R(z)*Tc*(1.+model.get_Ar01(Tc, rhoc, z))

    # Computing VLE
    rhol = np.zeros(n)
    rhov = np.zeros(n)
    Psat = np.zeros(n)
    Uvap = np.zeros(n)
    Hvap = np.zeros(n)

    T = np.linspace(0.99 * Tc, Tlower/T_factor, n)
    i = 0
    rhoL0, rhoV0 = model.extrapolate_from_critical(Tc, rhoc, T[i])
    for i in range(n):
        sol = model.pure_VLE_T(T[i], rhoL0, rhoV0, 20, molefrac=z)
        rhol[i] = sol[0]
        rhov[i] = sol[1]

        pL = rhol[i]*model.get_R(z)*T[i]*(1.+model.get_Ar01(T[i], rhol[i], z))
        pV = rhov[i]*model.get_R(z)*T[i]*(1.+model.get_Ar01(T[i], rhov[i], z))
        Psat[i] = pV

        uL = T[i] * model.get_R(z) * model.get_Ar10(T[i], rhol[i], z)
        uV = T[i] * model.get_R(z) * model.get_Ar10(T[i], rhov[i], z)
        Uvap[i] = uV - uL

        hL = T[i] * model.get_R(z) * (1. + model.get_Ar10(T[i], rhol[i], z) + model.get_Ar01(T[i], rhol[i], z))
        hV = T[i] * model.get_R(z) * (1. + model.get_Ar10(T[i], rhov[i], z) + model.get_Ar01(T[i], rhov[i], z))
        Hvap[i] = hV - hL
        rhoL0, rhoV0 = sol

    # Converting to reduced units
    Tcad = Tc * T_factor
    rhocad = rhoc * rho_factor
    Pcad = pc * P_factor

    Tad = T * T_factor
    Pad = Psat * P_factor
    rholad = rhol * rho_factor
    rhovad = rhov * rho_factor
    Uvap_ad = Uvap * energy_factor
    Hvap_ad = Hvap * energy_factor

    df_info = pd.DataFrame({'lambda_r': [lambda_r], 'lambda_a': [lambda_a], 'alpha': [alpha],
                            'Tcad_model': [Tcad], 'Pcad_model': [Pcad], 'rhocad_model': [rhocad]})

    df_vle = pd.DataFrame({'T_vle_model': Tad, 'P_vle_model': Pad,
                           'rhov_vle_model': rhovad, 'rhol_vle_model': rholad,
                           'Hvap_vle_model': Hvap_ad, 'Uvap_vle_model': Uvap_ad})

    data_df = {'info': df_info, 'vle': df_vle}
    return data_df
