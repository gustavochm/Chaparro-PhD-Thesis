import numpy as np
import pandas as pd
from sgtpy import component, saftvrmie
from sgtpy.constants import Na, kb
np.seterr(all="ignore")


def saft_derivatives(rho, T, eos, T_step=0.1, Xass=None):

    temp_aux = eos.temperature_aux(T)
    rhomolecular = Na * rho
    R = Na * kb

    d2a, Xass = eos.d2ares_drho(rhomolecular, T, Xass)
    beta = temp_aux[0]
    RT = Na/beta

    h = T_step
    a1, Xass1 = eos.dares_drho(rhomolecular, T + h, Xass)
    a2, Xass2 = eos.dares_drho(rhomolecular, T + 2.*h, Xass)
    a_1, Xass_1 = eos.dares_drho(rhomolecular, T - h, Xass)
    a_2, Xass_2 = eos.dares_drho(rhomolecular, T - 2.*h, Xass)

    a = d2a[:2]
    da_drho = a[1] * Na
    d2a_drho = d2a[2] * Na**2

    dFdT = (a_2/12 - 2*a_1/3 + 2*a1/3 - a2/12)/h
    dFdT[1] *= Na

    d2FdT = (-a_2/12 + 4*a_1/3 - 5*a/2 + 4*a1/3 - a2/12) / h**2
    d2FdT[1] *= Na

    ###########################################################
    F = a[0]  # F = A/RT
    Sr_by_R = (- T * dFdT[0] - F)  # S/R = -T dF/dT - A
    Ur_by_RT = F + Sr_by_R  # Ur/RT = F + S/R
    internal = RT * (Ur_by_RT + 1.5)

    P = (rho**2 * da_drho + rho) * RT

    dP_dT = RT*(rho**2 * dFdT[1]) + P/T

    dP_drho = 2*rho*da_drho + 2.
    dP_drho += rho**2 * d2a_drho - 1.
    dP_drho *= RT

    dP_dV = -rho**2 * dP_drho

    # residual isochoric heat capacity
    Cvr = R * (-T**2*d2FdT[0] - 2*T*dFdT[0])
    # residual heat capacity
    Cpr = Cvr - R - T*dP_dT**2/dP_dV

    Cv = Cvr + 1.5 * R

    Cp = Cpr + 2.5 * R

    kappaT = 1./(rho*dP_drho)

    alphaP = dP_dT / (rho*dP_drho)

    gammaV = alphaP/kappaT

    muJT = 1./(rho*Cp) * (T * alphaP - 1.)

    out = dict(P=P, U=internal, Cv=Cv, Cp=Cp, kappaT=kappaT, alphaP=alphaP, gammaV=gammaV, muJT=muJT)

    return out


def data_isotherms_lr_saft(T_list, lambda_r=12., rho_min=1e-3, rho_max=1.25, n=200, eps_kb=150., sigma=3., lambda_a=6.):

    # eps_kb = 150.  # K
    # sigma = 3 # A.

    fluid = component(ms=1, eps=eps_kb, sigma=sigma, lambda_r=lambda_r, lambda_a=lambda_a)
    eos = saftvrmie(fluid)

    alpha = eos.alpha

    # pressure_factor = eos.sigma3/eos.eps
    rho_factor = Na*eos.sigma3
    temperature_factor = kb/eos.eps
    R = Na*kb

    # Isotherms
    rho_array = np.linspace(rho_min, rho_max, n)
    rho = rho_array / rho_factor

    alpha_array = alpha * np.ones_like(rho_array)

    dict_isotherms = dict()
    for Tad in T_list:
        T_array = np.ones_like(rho_array) * Tad
        P_array = np.zeros_like(rho_array)
        U_array = np.zeros_like(rho_array)
        Cv_array = np.zeros_like(rho_array)
        Cp_array = np.zeros_like(rho_array)
        kappaT_array = np.zeros_like(rho_array)
        alphaP_array = np.zeros_like(rho_array)
        gammaV_array = np.zeros_like(rho_array)
        muJT_array = np.zeros_like(rho_array)

        T = Tad / temperature_factor
        for i in range(n):
            out = saft_derivatives(rho[i], T, eos, T_step=1e-2)

            P_array[i] = out['P']
            U_array[i] = out['U']
            Cv_array[i] = out['Cv']
            Cp_array[i] = out['Cp']
            kappaT_array[i] = out['kappaT']
            alphaP_array[i] = out['alphaP']
            gammaV_array[i] = out['gammaV']
            muJT_array[i] = out['muJT']

        P_array *= (eos.sigma3/eos.eps)
        U_array *= (1. / (Na*eos.eps))
        Cv_array /= R
        Cp_array /= R
        kappaT_array *= (eos.eps/eos.sigma3)
        alphaP_array *= (eos.eps/kb)
        gammaV_array *= (eos.sigma3/kb)
        muJT_array *= (kb/eos.sigma3)

        dict_data = dict(alpha=alpha_array, density=rho_array, temperature=T_array, 
                         pressure=P_array, compressibility_factor=(P_array/(rho_array*T_array)), internal_energy=U_array,
                         isochoric_heat_capacity=Cv_array, isothermal_compressibility=kappaT_array, rho_isothermal_compressibility=rho_array*kappaT_array,
                         thermal_expansion_coefficient=alphaP_array, thermal_pressure_coefficient=gammaV_array, 
                         isobaric_heat_capacity=Cp_array, adiabatic_index=(Cp_array/Cv_array), joule_thomson_coefficient=muJT_array)

        dict_isotherms[f'T={Tad:.2f}'] = pd.DataFrame(dict_data)
    return dict_isotherms
