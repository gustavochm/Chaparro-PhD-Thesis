from ..helpers import helper_get_alpha
import numpy as np
import pandas as pd


def latex_description_database(df, df_virials):

    tabular_latex = f"""
    Descriptor & Symbol & \# of data points & Min value & Max value \\\ \hline 
    Repulsive exponent\\textsuperscript{{\emph{{b}}}} & $\\lambda_\\mathrm{{r}}$ & --& {df['lr'].min():.0f} &  {df['lr'].max():.0f} \\\\
    Attractive exponent\\textsuperscript{{\emph{{b}}}} & $\\lambda_\\mathrm{{a}}$ & -- & {df['la'].min():.0f} &  {df['la'].max():.0f} \\\ 
    van der Waals parameter\\textsuperscript{{\emph{{c}}}} & $\\alpha_\\mathrm{{vdw}}$ & -- & {df['alpha'].min():.2f} &  {df['alpha'].max():.2f} \\\ 
    Density & $\\rho^*$ & -- & {df['rho*'].min():.2e} &  {df['rho*'].max():.2f} \\\\
    Temperature\\textsuperscript{{\emph{{d}}}} & $T^*$ & -- & {df['T*'].min():.2f} &  {df['T*'].max():.2f} \\\\
    \\
    Thermophysical Property  \\\ \hline
    Pressure & $P^*$ & {df['P_ad'].count()} & {df['P_ad'].min():.2e} &  {df['P_ad'].max():.2e} \\\\
    Compressibility factor\\textsuperscript{{\emph{{e}}}} & $Z$ &  {df['Z'].count()} & {df['Z'].min():.2e} &  {df['Z'].max():.2f} \\\\
    Internal energy & $U^*$ & {df['TotEn_ad'].count()} & {df['TotEn_ad'].min():.2f} &  {df['TotEn_ad'].max():.2f} \\\\
    Second virial coefficient\\textsuperscript{{\emph{{f}}}} & $ B^*_2$ & {df_virials['B2*'].count()} & {df_virials['B2*'].min():.2f} &  {df_virials['B2*'].max():.2f} \\\\
    Third virial coefficient\\textsuperscript{{\emph{{f}}}} & $ B^*_3$ & {df_virials['B3*'].count()} & {df_virials['B3*'].min():.2f} &  {df_virials['B3*'].max():.2f} \\\\
    \\\\
    Isochoric heat capacity & $C_v^*$ & {df['Cv_nvt'].count()} & {df['Cv_nvt'].min():.2f} &  {df['Cv_nvt'].max():.2f} \\\\
    Isobaric heat capacity & $C_p^*$ & {df['Cp_npt'].count()} & {df['Cp_npt'].min():.2f} &  {df['Cp_npt'].max():.2f} \\\\
    Adiatic index\\textsuperscript{{\emph{{g}}}} & $\gamma$ & {df['gamma'].count()} & {df['gamma'].min():.2f} &  {df['gamma'].max():.2f} \\\\
    Thermal pressure coefficient & $\gamma_V^*$ & {df['ThermalPressureCoeff_nvt'].count()} & {df['ThermalPressureCoeff_nvt'].min():.2e} &  {df['ThermalPressureCoeff_nvt'].max():.2f} \\\\
    Thermal expansion coefficient & $\\alpha_P^*$ & {df['ThermalExpansionCoeff_npt'].count()} & {df['ThermalExpansionCoeff_npt'].min():.2e} &  {df['ThermalExpansionCoeff_npt'].max():.2f} \\\\
    Isothermal compressibility & $\\kappa_T^*$ & {df['IsothermalCompressibility_npt'].count()} & {df['IsothermalCompressibility_npt'].min():.2e} &  {df['IsothermalCompressibility_npt'].max():.2e} \\\\
    Joule-Thomson coefficient & $\\mu_\mathrm{{JT}}^*$ & {df['JouleThomson_npt'].count()} & {df['JouleThomson_npt'].min():.2f} &  {df['JouleThomson_npt'].max():.2f} \\\\
    \hline 
    """
    return tabular_latex


def latex_description_database_tp(dict_md_data):

    df_diff = dict_md_data['self_diffusivity']
    df_visc = dict_md_data['shear_viscosity']
    df_tcond = dict_md_data['thermal_conductivity']

    df_diff['alpha'] = helper_get_alpha(df_diff['lr'], df_diff['la'])
    df_visc['alpha'] = helper_get_alpha(df_visc['lr'], df_visc['la'])
    df_tcond['alpha'] = helper_get_alpha(df_tcond['lr'], df_tcond['la'])

    lr_min = np.min(np.hstack([df_diff['lr'].min(), df_visc['lr'].min(), df_tcond['lr'].min()]))
    lr_max = np.max(np.hstack([df_diff['lr'].max(), df_visc['lr'].max(), df_tcond['lr'].max()]))

    la_min = np.min(np.hstack([df_diff['la'].min(), df_visc['la'].min(), df_tcond['la'].min()]))
    la_max = np.max(np.hstack([df_diff['la'].max(), df_visc['la'].max(), df_tcond['la'].max()]))

    alpha_min = np.min(np.hstack([df_diff['alpha'].min(), df_visc['alpha'].min(), df_tcond['alpha'].min()]))
    alpha_max = np.max(np.hstack([df_diff['alpha'].max(), df_visc['alpha'].max(), df_tcond['alpha'].max()]))

    rho_min = np.min(np.hstack([df_diff['rho*'].min(), df_visc['rho*'].min(), df_tcond['rho*'].min()]))
    rho_max = np.max(np.hstack([df_diff['rho*'].max(), df_visc['rho*'].max(), df_tcond['rho*'].max()]))

    T_min = np.min(np.hstack([df_diff['T*'].min(), df_visc['T*'].min(), df_tcond['T*'].min()]))
    T_max = np.max(np.hstack([df_diff['T*'].max(), df_visc['T*'].max(), df_tcond['T*'].max()]))

    tabular_latex = f"""
    Descriptor & Symbol & \# of data points & Min value & Max value \\\ \hline 
    Repulsive exponent\\textsuperscript{{\emph{{b}}}} & $\\lambda_\\mathrm{{r}}$ & --& {lr_min:.0f} &  {lr_max:.0f} \\\\
    Attractive exponent\\textsuperscript{{\emph{{b}}}} & $\\lambda_\\mathrm{{a}}$ & -- & {la_min:.0f} &  {la_max:.0f} \\\ 
    van der Waals parameter\\textsuperscript{{\emph{{c}}}} & $\\alpha_\\mathrm{{vdw}}$ & -- & {alpha_min:.2f} &  {alpha_max:.2f} \\\ 
    Density & $\\rho^*$ & -- & {rho_min:.2e} &  {rho_max:.2f} \\\\
    Temperature & $T^*$ & -- & {T_min:.2f} &  {T_max:.2f} \\\\
    \\\\
    Transport Property  \\\\
    Self-diffusivity & $D^*$ & {df_diff['self_diffusivity'].count()} & {df_diff['self_diffusivity'].min():.2e} & {df_diff['self_diffusivity'].max():.2f} \\\\
    Shear viscosity & $\\eta^*$ & {df_visc['shear_viscosity'].count()} & {df_visc['shear_viscosity'].min():.2e} & {df_visc['shear_viscosity'].max():.2f} \\\\
    Thermal conductivity & $\\kappa^*$ & {df_tcond['thermal_conductivity'].count()} & {df_tcond['thermal_conductivity'].min():.2f} & {df_tcond['thermal_conductivity'].max():.2f} \\\\
    \hline
    """
    return tabular_latex
