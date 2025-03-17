import pandas as pd
import os

def load_data_function(root_folder_to_read, folder_to_read, expdata_filename):
    # Experimental data from NIST
    ExpDataFile = pd.ExcelFile(os.path.join(root_folder_to_read, folder_to_read, expdata_filename))

    # Model results filenames
    mie_results_vle_fluid_filename = "mie_results_vle_fluid.xlsx"
    mie_results_vle_solid_filename = "mie_results_vle_solid.xlsx"
    mie_results_vle_visc_fluid_filename = "mie_results_vle_visc_fluid.xlsx"
    mie_results_vle_visc_solid_filename = "mie_results_vle_visc_solid.xlsx"
    mie_results_vle_sle_sve_fluid_filename = "mie_results_vle_sle_sve_fluid.xlsx"
    mie_results_vle_sle_sve_solid_filename = "mie_results_vle_sle_sve_solid.xlsx"
    mie_results_vle_sle_sve_visc_fluid_filename = "mie_results_vle_sle_sve_visc_fluid.xlsx"
    mie_results_vle_sle_sve_visc_solid_filename = "mie_results_vle_sle_sve_visc_solid.xlsx"

    # Reading the excel files
    vle_fluid_file = pd.ExcelFile(os.path.join(root_folder_to_read, folder_to_read, mie_results_vle_fluid_filename))
    vle_solid_file = pd.ExcelFile(os.path.join(root_folder_to_read, folder_to_read, mie_results_vle_solid_filename))

    vle_visc_fluid_file = pd.ExcelFile(os.path.join(root_folder_to_read, folder_to_read, mie_results_vle_visc_fluid_filename))
    vle_visc_solid_file = pd.ExcelFile(os.path.join(root_folder_to_read, folder_to_read, mie_results_vle_visc_solid_filename))

    vle_sle_sve_fluid_file = pd.ExcelFile(os.path.join(root_folder_to_read, folder_to_read, mie_results_vle_sle_sve_fluid_filename))
    vle_sle_sve_solid_file = pd.ExcelFile(os.path.join(root_folder_to_read, folder_to_read, mie_results_vle_sle_sve_solid_filename))

    vle_sle_sve_visc_fluid_file = pd.ExcelFile(os.path.join(root_folder_to_read, folder_to_read, mie_results_vle_sle_sve_visc_fluid_filename))
    vle_sle_sve_visc_solid_file = pd.ExcelFile(os.path.join(root_folder_to_read, folder_to_read, mie_results_vle_sle_sve_visc_solid_filename))

    dict_models = dict(vle_fluid_file=vle_fluid_file,
                        vle_solid_file=vle_solid_file,
                        vle_visc_fluid_file=vle_visc_fluid_file,
                        vle_visc_solid_file=vle_visc_solid_file,
                        vle_sle_sve_fluid_file=vle_sle_sve_fluid_file,
                        vle_sle_sve_solid_file=vle_sle_sve_solid_file,
                        vle_sle_sve_visc_fluid_file=vle_sle_sve_visc_fluid_file,
                        vle_sle_sve_visc_solid_file=vle_sle_sve_visc_solid_file)
    return ExpDataFile, dict_models
