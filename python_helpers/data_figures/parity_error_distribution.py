import numpy as np
import pandas as pd

def parity_error_distribution_latex(excel_parity):
    # reading the data from the excel file
    df_model_train = pd.read_excel(excel_parity, sheet_name='model_train')
    df_model_test = pd.read_excel(excel_parity, sheet_name='model_test')
    df_model_virial_train = pd.read_excel(excel_parity, sheet_name='model_virial_train')
    df_model_virial_test = pd.read_excel(excel_parity, sheet_name='model_virial_test')

    df_train = pd.read_excel(excel_parity, sheet_name='PVT_train')
    df_test = pd.read_excel(excel_parity, sheet_name='PVT_test')
    df_virial_train = pd.read_excel(excel_parity, sheet_name='virials_train')
    df_virial_test = pd.read_excel(excel_parity, sheet_name='virials_test')

    # MD data
    property_dict = dict(compressibility_factor=r'$Z$',
                        internal_energy=r'$U^*$',
                        isochoric_heat_capacity=r'$C_V^*$',
                        rho_isothermal_compressibility=r'$\rho^*\kappa_T^*$',
                        thermal_pressure_coefficient=r'$\gamma_V^*$',
                        thermal_expansion_coefficient=r'$\alpha_P^*$',
                        adiabatic_index=r'$\gamma^*$',
                        joule_thomson_coefficient=r'$\mu_\mathrm{JT}^*$')

    table_latex = r""" """

    for key, value in property_dict.items():
        table_latex += f"\multirow{{2}}{{*}}{{{value}}} & Train & "
        # Train data
        prop_model_train = df_model_train[key].to_numpy()
        prop_db_train = df_train[key].to_numpy()

        mse_train = (prop_model_train - prop_db_train)**2
        mse_min = np.nanmin(mse_train)
        mse_max = np.nanmax(mse_train)
        mse_mean = np.nanmean(mse_train)
        mse_median = np.nanmedian(mse_train)

        for value, end_value in zip([mse_min, mse_max, mse_median, mse_mean], ["&", "&", "&", ""]):
            table_latex += f"{value:.2e} {end_value} "
        table_latex += f"\\\ \n "

        # Test data
        table_latex += f" & Test & "
        prop_model_test = df_model_test[key].to_numpy()
        prop_db_test = df_test[key].to_numpy()

        mse_test = (prop_model_test - prop_db_test)**2
        mse_min = np.nanmin(mse_test)
        mse_max = np.nanmax(mse_test)
        mse_mean = np.nanmean(mse_test)
        mse_median = np.nanmedian(mse_test)

        for value, end_value in zip([mse_min, mse_max, mse_median, mse_mean], ["&", "&", "&", ""]):
            table_latex += f"{value:.2e} {end_value} "

        table_latex += f"\\\ \\hline \n "

    # Virial data
    property_dict_virial = dict(B2=r'$B_2^*$', B3=r'$B_3^*$')

    for key, value in property_dict_virial.items():
        table_latex += f"\multirow{{2}}{{*}}{{{value}}} & Train & "
        # Train data
        prop_model_train = df_model_virial_train[key].to_numpy()
        prop_db_train = df_virial_train[key].to_numpy()

        mse_train = (prop_model_train - prop_db_train)**2
        mse_min = np.nanmin(mse_train)
        mse_max = np.nanmax(mse_train)
        mse_mean = np.nanmean(mse_train)
        mse_median = np.nanmedian(mse_train)

        for value, end_value in zip([mse_min, mse_max, mse_median, mse_mean], ["&", "&", "&", ""]):
            table_latex += f"{value:.2e} {end_value} "
        table_latex += f"\\\ \n "

        # Test data
        table_latex += f" & Test & "
        prop_model_test = df_model_virial_test[key].to_numpy()
        prop_db_test = df_virial_test[key].to_numpy()

        mse_test = (prop_model_test - prop_db_test)**2
        mse_min = np.nanmin(mse_test)
        mse_max = np.nanmax(mse_test)
        mse_mean = np.nanmean(mse_test)
        mse_median = np.nanmedian(mse_test)

        for value, end_value in zip([mse_min, mse_max, mse_median, mse_mean], ["&", "&", "&", ""]):
            table_latex += f"{value:.2e} {end_value} "

        table_latex += f"\\\ \\hline \n "

    return table_latex
