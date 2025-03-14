import numpy as np
from .phase_equilibria_fluid_lr import data_phase_equilibria_fluid_lr
from .phase_equilibria_solid_lr import data_phase_equilibria_solid_lr
from jax.config import config
import jax.numpy as jnp
# Constants
from scipy.constants import Boltzmann, Avogadro
kb = Boltzmann  # [J/K] Boltzman's constant
Na = Avogadro   # [mol-1] Avogadro's Number
R = Na * kb     # [J mol-1 K-1] Ideal gas constant

PRECISSION = 'float64'
if PRECISSION == 'float64':
    config.update("jax_enable_x64", True)
    type_np = np.float64
    type_jax = jnp.float64
else:
    config.update("jax_enable_x32", True)
    type_np = np.float32
    type_jax = jnp.float32


def data_phase_equilibria_mie_solid(fun_dic, diff_fun, visc_fun, tcond_fun,
                                    lambda_r=12., lambda_a=6.,
                                    Tlower=0.4, Tupper=10., n_vle=100, n_sle=1000, n_sve=50,
                                    initial_triple_point=None, initial_crit_point=None):

    data_df = data_phase_equilibria_solid_lr(fun_dic, lambda_r=lambda_r, lambda_a=lambda_a,
                                             Tlower=Tlower, Tupper=Tupper, n_vle=n_vle, n_sle=n_sle, n_sve=n_sve,
                                             initial_crit_point=initial_crit_point, initial_triple_point=initial_triple_point)

    alpha = data_df['info']['alpha'].values
    df_vle = data_df['vle']
    T_vle_model = df_vle['T_vle_model'].to_numpy()
    rhol_vle_model = df_vle['rhol_vle_model'].to_numpy()
    alpha_vle = alpha * np.ones_like(T_vle_model)

    # Second order required for SoS
    cvl_res_vle_model = fun_dic['cv_residual_fun'](alpha_vle, rhol_vle_model, T_vle_model)
    cpl_res_vle_model = fun_dic['cp_residual_fun'](alpha_vle, rhol_vle_model, T_vle_model)
    kappaT_vle_model = fun_dic['isothermal_compressibility_fun'](alpha_vle, rhol_vle_model, T_vle_model)
    # Transport properties
    diffl_vle_model = diff_fun(alpha_vle, rhol_vle_model, T_vle_model)
    viscl_vle_model = visc_fun(alpha_vle, rhol_vle_model, T_vle_model)
    tcondl_vle_model = tcond_fun(alpha_vle, rhol_vle_model, T_vle_model)

    df_vle['cvl_res_vle_model'] = cvl_res_vle_model
    df_vle['cpl_res_vle_model'] = cpl_res_vle_model
    df_vle['kappaT_vle_model'] = kappaT_vle_model
    df_vle['diffl_vle_model'] = diffl_vle_model
    df_vle['viscl_vle_model'] = viscl_vle_model
    df_vle['tcondl_vle_model'] = tcondl_vle_model

    data_df['vle'] = df_vle

    return data_df


def data_phase_equilibria_mie_fluid(fun_dic, diff_fun, visc_fun, tcond_fun,
                                    lambda_r=12.,
                                    Tlower=0.4, n_vle=100, n_sle=1000,
                                    initial_crit_point=None):

    data_df = data_phase_equilibria_fluid_lr(fun_dic, lambda_r=lambda_r,
                                             Tlower=Tlower, n_vle=n_vle, n_sle=n_sle,
                                             initial_crit_point=initial_crit_point)

    alpha = data_df['info']['alpha'].values
    df_vle = data_df['vle']
    T_vle_model = df_vle['T_vle_model'].to_numpy()
    rhol_vle_model = df_vle['rhol_vle_model'].to_numpy()
    alpha_vle = alpha * np.ones_like(T_vle_model)

    # Second order required for SoS
    cvl_res_vle_model = fun_dic['cv_residual_fun'](alpha_vle, rhol_vle_model, T_vle_model)
    cpl_res_vle_model = fun_dic['cp_residual_fun'](alpha_vle, rhol_vle_model, T_vle_model)
    kappaT_vle_model = fun_dic['isothermal_compressibility_fun'](alpha_vle, rhol_vle_model, T_vle_model)
    # Transport properties
    diffl_vle_model = diff_fun(alpha_vle, rhol_vle_model, T_vle_model)
    viscl_vle_model = visc_fun(alpha_vle, rhol_vle_model, T_vle_model)
    tcondl_vle_model = tcond_fun(alpha_vle, rhol_vle_model, T_vle_model)

    df_vle['cvl_res_vle_model'] = cvl_res_vle_model
    df_vle['cpl_res_vle_model'] = cpl_res_vle_model
    df_vle['kappaT_vle_model'] = kappaT_vle_model
    df_vle['diffl_vle_model'] = diffl_vle_model
    df_vle['viscl_vle_model'] = viscl_vle_model
    df_vle['tcondl_vle_model'] = tcondl_vle_model

    data_df['vle'] = df_vle

    return data_df


def mie_data_si_units(sigma, epsilon, Mw, data_df, Cvid_by_R, non_vibrational_deg_freedom=0, include_solid=False):

    sigma_or = 1. * sigma
    epsilon_or = 1. * epsilon

    #####################
    dict_values_mie = {}

    sigma = 1 * sigma # A
    sigma *= 1e-10 # m
    eps = epsilon * kb # J

    energy_factor = 1 / (eps * Na)     # J/mol -> dim
    pressure_factor = sigma**3 / eps   # Pa -> adim
    rho_factor = sigma**3 * Na         # mol/m3 -> adim
    T_factor = kb / eps                # K -> adim

    cv_factor = 1. / R                    # J/mol -> adim
    kappaT_factor = 1 / pressure_factor   # Pa^-1 -> adim

    tcond_factor = 1. / (kb * np.sqrt(eps / (Mw/1000./Na)) / sigma**2)    # W / m K -> adim
    visc_factor = 1. / (np.sqrt(eps * (Mw/1000./Na)) / sigma**2)          # Pa s -> adim
    diff_factor = 1. / (np.sqrt(eps / (Mw/1000./Na)) * sigma)             # m^2 / s -> adim

    # info
    df_info_mie = data_df['info'].copy()
    df_info_mie['Tcad_model'] /= T_factor             # K
    df_info_mie['Pcad_model'] /= pressure_factor      # Pa
    df_info_mie['rhocad_model'] /= rho_factor         # mol/m3
    if include_solid:
        df_info_mie['T_triple'] /= T_factor           # K
        df_info_mie['P_triple'] /= pressure_factor    # Pa
        df_info_mie['rhovad_triple'] /= rho_factor    # mol/m3
        df_info_mie['rholad_triple'] /= rho_factor    # mol/m3
        df_info_mie['rhosad_triple'] /= rho_factor    # mol/m3
        df_info_mie['dUvap_triple'] /= energy_factor  # J/mol
        df_info_mie['dUmel_triple'] /= energy_factor  # J/mol
        df_info_mie['dUsub_triple'] /= energy_factor  # J/mol

    df_info_mie.rename(columns={'Tcad_model': 'Tc_model', 'Pcad_model': 'Pc_model', 'rhocad_model': 'rhoc_model'}, inplace=True)
    if include_solid:
        df_info_mie.rename(columns={'T_triple': 'T_triple_model', 'P_triple': 'P_triple_model',
                                    'rhovad_triple': 'rhov_triple_model', 'rholad_triple': 'rhol_triple_model', 'rhosad_triple': 'rhos_triple_model',
                                    'dUvap_triple': 'dUvap_triple_model', 'dUmel_triple': 'dUmel_triple_model', 'dUsub_triple': 'dUsub_triple_model'}, inplace=True)

    df_info_mie['sigma'] = sigma_or
    df_info_mie['epsilon'] = epsilon_or
    dict_values_mie['info'] = df_info_mie

    # VLE
    df_vle_mie = data_df['vle'].copy()
    df_vle_mie['T_vle_model'] /= T_factor             # K
    df_vle_mie['P_vle_model'] /= pressure_factor      # Pa
    df_vle_mie['rhol_vle_model'] /= rho_factor        # mol/m3
    df_vle_mie['rhov_vle_model'] /= rho_factor        # mol/m3
    df_vle_mie['Hvap_vle_model'] /= energy_factor     # J/mol
    df_vle_mie['Uvap_vle_model'] /= energy_factor     # J/mol
    df_vle_mie['cvl_res_vle_model'] /= cv_factor      # J/mol K
    df_vle_mie['cpl_res_vle_model'] /= cv_factor      # J/mol K
    df_vle_mie['kappaT_vle_model'] /= kappaT_factor   # 1/Pa
    df_vle_mie['diffl_vle_model'] /= diff_factor      # m2/s
    df_vle_mie['viscl_vle_model'] /= visc_factor      # Pa s
    df_vle_mie['tcondl_vle_model'] /= tcond_factor    # W/m K

    # Computing speed of sound
    Cvid_R = Cvid_by_R(df_vle_mie['T_vle_model'].to_numpy())
    Cpid_R = Cvid_R + 1
    cv_liq_vle = df_vle_mie['cvl_res_vle_model'].to_numpy() + Cvid_R * R
    cp_liq_vle = df_vle_mie['cpl_res_vle_model'].to_numpy() + Cpid_R * R
    w2 = 1. / (df_vle_mie['rhol_vle_model'].to_numpy() * df_vle_mie['kappaT_vle_model'].to_numpy() * (cv_liq_vle/cp_liq_vle) * (Mw / 1000.))
    df_vle_mie['speed_of_sound_liq_vle_model'] = np.sqrt(w2)  # m/s

    # Correcting thermal conductivity
    Cvid_R_nonvib = Cvid_R - non_vibrational_deg_freedom/2
    tcond_liq_correction = df_vle_mie['rhol_vle_model'].to_numpy() * df_vle_mie['diffl_vle_model'].to_numpy() * (Cvid_R_nonvib * R)
    df_vle_mie['tcondl_vle_model'] += tcond_liq_correction

    dict_values_mie['vle'] = df_vle_mie

    # SLE and SVE if including solid data
    if include_solid:
        df_sle_mie = data_df['sle'].copy()
        df_sle_mie['T_sle_model'] /= T_factor                 # K
        df_sle_mie['P_sle_model'] /= pressure_factor          # Pa
        df_sle_mie['rhol_sle_model'] /= rho_factor            # mol/m3
        df_sle_mie['rhos_sle_model'] /= rho_factor            # mol/m3
        df_sle_mie['Hmelting_sle_model'] /= energy_factor     # J/mol
        df_sle_mie['Umelting_sle_model'] /= energy_factor     # J/mol
        dict_values_mie['sle'] = df_sle_mie

        df_sve_mie = data_df['sve'].copy()
        df_sve_mie['T_sve_model'] /= T_factor                 # K
        df_sve_mie['P_sve_model'] /= pressure_factor          # Pa
        df_sve_mie['rhov_sve_model'] /= rho_factor            # mol/m3
        df_sve_mie['rhos_sve_model'] /= rho_factor            # mol/m3
        df_sve_mie['Hsub_sve_model'] /= energy_factor         # J/mol
        df_sve_mie['Usub_sve_model'] /= energy_factor         # J/mol
        dict_values_mie['sve'] = df_sve_mie

    return dict_values_mie
