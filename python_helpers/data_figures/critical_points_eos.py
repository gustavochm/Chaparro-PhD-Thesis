import pandas as pd
import numpy as np
from ..feanneos import critical_point_solver
from ..helpers import helper_get_alpha

from jax import numpy as jnp
from jax.config import config

# importing SAFT-VR Mie EoS
from sgtpy import component, saftvrmie
from scipy.constants import Avogadro as Na
from scipy.constants import Boltzmann as kb

# importing Pohl EoS
import teqp

PRECISSION = 'float64'
if PRECISSION == 'float64':
    config.update("jax_enable_x64", True)
    type_np = np.float64
    type_jax = jnp.float64
else:
    config.update("jax_enable_x32", True)
    type_np = np.float32
    type_jax = jnp.float32


#############################
# Critical point FE-ANN EoS #
#############################
def data_critical_point_feanneos(fun_dic, lr_min, lr_max, n, lambda_a=6., inc0=[0.3, 1.3]):
    ### Critical Point
    lr_array = np.linspace(lr_min, lr_max, n)

    alpha_array = helper_get_alpha(lr_array, lambda_a)
    Tcad_critical = np.zeros(n)
    rhocad_critical = np.zeros(n)
    Pcad_critical = np.zeros(n)
    i = 0
    sol_crit = critical_point_solver(alpha_array[i], fun_dic, inc0=inc0, full_output=True)
    Tcad_critical[i] = sol_crit['Tcad']
    rhocad_critical[i] = sol_crit['rhocad']
    Pcad_critical[i] = sol_crit['Pcad']

    for i in range(1, n):
        inc0 = [rhocad_critical[i-1], Tcad_critical[i-1]]
        sol_crit = critical_point_solver(alpha_array[i], fun_dic, inc0=inc0, full_output=True)
        Tcad_critical[i] = sol_crit['Tcad']
        rhocad_critical[i] = sol_crit['rhocad']
        Pcad_critical[i] = sol_crit['Pcad']

    df = pd.DataFrame({'lr': lr_array, 'la': lambda_a * np.ones(n), 'alpha': alpha_array,
                       'Tcad': Tcad_critical, 'rhocad': rhocad_critical, 'Pcad': Pcad_critical})
    return df


##################################
# Critical point SAFT-VR-Mie EoS #
##################################
def data_critical_point_saft(lr_min, lr_max, n, lambda_a=6.):
    ### Critical Point
    lr_array = np.linspace(lr_min, lr_max, n)
    alpha_array = helper_get_alpha(lr_array, lambda_a)
    Tcad_critical = np.zeros(n)
    rhocad_critical = np.zeros(n)
    Pcad_critical = np.zeros(n)

    ##### SAFT-VR-Mie Computation
    for i in range(n):
        lambda_r = lr_array[i]
        eps_kb = 150.  # K
        sigma = 3 # A.

        fluid = component(ms=1, eps=eps_kb, sigma=sigma, lambda_r=lambda_r, lambda_a=lambda_a)
        eos = saftvrmie(fluid)

        pressure_factor = eos.sigma3/eos.eps
        rho_factor = Na*eos.sigma3
        temperature_factor = kb/eos.eps
        # R = Na*kb

        if eos.critical:
            Tcad_critical[i] = eos.Tc * temperature_factor
            rhocad_critical[i] = eos.rhoc * rho_factor
            Pcad_critical[i] = eos.Pc * pressure_factor
        else:
            eos.get_critical(Tc0=Tcad_critical[i-1]/temperature_factor, rhoc0=rhocad_critical[i-1]/rho_factor, overwrite=True)
            Tcad_critical[i] = eos.Tc * temperature_factor
            rhocad_critical[i] = eos.rhoc * rho_factor
            Pcad_critical[i] = eos.Pc * pressure_factor

    df = pd.DataFrame({'lr': lr_array, 'la': lambda_a * np.ones(n), 'alpha': alpha_array,
                       'Tcad': Tcad_critical, 'rhocad': rhocad_critical, 'Pcad': Pcad_critical})
    return df


###########################
# Critical point Pohl EoS #
###########################
def data_critical_point_pohl(lr_min, lr_max, n, lambda_a=6., inc0=[0.3, 1.3]):

    ### Critical Point
    lr_array = np.linspace(lr_min, lr_max, n)
    alpha_array = helper_get_alpha(lr_array, lambda_a)
    Tcad_critical = np.zeros(n)
    rhocad_critical = np.zeros(n)
    Pcad_critical = np.zeros(n)

    rhoc0, Tc0 = inc0
    for i in range(n):
        ##### Pohl EoS
        lambda_r = lr_array[i]
        pohl_model = teqp.make_model({"kind":"Mie_Pohl2023", "model": {"lambda_r": lambda_r}})
        z = np.array([1.0])
        Tc, rhoc = pohl_model.solve_pure_critical(Tc0, rhoc0)
        pc = rhoc*pohl_model.get_R(z)*Tc*(1.+pohl_model.get_Ar01(Tc, rhoc, z))

        Tcad_critical[i] = Tc
        rhocad_critical[i] = rhoc
        Pcad_critical[i] = pc

        Tc0 = Tc
        rhoc0 = rhoc

    df = pd.DataFrame({'lr': lr_array, 'la': lambda_a * np.ones(n), 'alpha': alpha_array, 'Tcad': Tcad_critical, 'rhocad': rhocad_critical, 'Pcad': Pcad_critical})
    return df
