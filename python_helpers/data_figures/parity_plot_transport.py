import pandas as pd
import numpy as np
from ..helpers import helper_get_alpha
from sklearn.model_selection import train_test_split


def parity_plot_transport_md(dict_md_data,
                             dict_models,
                             dict_res_models, dict_dilute_functions,
                             dict_entropy_models, dict_entropy_scaling,
                             transport_list=['self_diffusivity', 'shear_viscosity', 'thermal_conductivity']):
    dict_parity = dict()
    for transport_type in transport_list:
        # loading the data
        df = dict_md_data[transport_type]
        lr = df['lr'].to_numpy()
        la = df['la'].to_numpy()
        Tad = df['T*_sim'].to_numpy()
        rhoad = df['rho*_sim'].to_numpy()
        transport_property = df[transport_type].to_numpy()
        Sres = df['Sr'].to_numpy()

        # getting vdw alpha parameter for each Mie fluid
        alpha = helper_get_alpha(lr, la)

        n = len(lr)
        transport_ideal = np.zeros(n)
        for i in range(n):
            transport_ideal[i] = dict_dilute_functions[transport_type](lr[i], Tad[i])

        # data splitting
        data = np.stack([lr, alpha, rhoad, Tad, Sres, transport_property, transport_ideal]).T
        out = train_test_split(data, test_size=0.1, shuffle=True, random_state=12)
        data_train, data_test = out

        lr_train, alpha_train, rhoad_train, Tad_train, Sres_train = data_train[:, :5].T
        transport_property_train, transport_ideal_train = data_train[:, 5:].T

        lr_test, alpha_test, rhoad_test, Tad_test, Sres_test = data_test[:, :5].T
        transport_property_test, transport_ideal_test = data_test[:, 5:].T

        # computing parity plots

        # residual model
        tp_model_res = dict_res_models[transport_type]
        transport_property_res_train = tp_model_res(lr_train, rhoad_train, Tad_train)
        transport_property_res_test = tp_model_res(lr_test, rhoad_test, Tad_test)

        if transport_type == 'self_diffusivity':
            transport_property_res_train = transport_property_res_train + transport_ideal_train
            transport_property_res_train = transport_property_res_train / rhoad_train
            transport_property_res_test = transport_property_res_test + transport_ideal_test
            transport_property_res_test = transport_property_res_test / rhoad_test

        elif transport_type == 'shear_viscosity' or transport_type == 'thermal_conductivity':
            transport_property_res_train = np.exp(transport_property_res_train)
            transport_property_res_train = transport_property_res_train * transport_ideal_train
            transport_property_res_test = np.exp(transport_property_res_test)
            transport_property_res_test = transport_property_res_test * transport_ideal_test

        # full model
        tp_model = dict_models[transport_type]
        transport_property_ann_train = tp_model(alpha_train, rhoad_train, Tad_train)
        transport_property_ann_test = tp_model(alpha_test, rhoad_test, Tad_test)
        if transport_type == 'self_diffusivity':
            transport_property_ann_train = transport_property_ann_train / rhoad_train
            transport_property_ann_test = transport_property_ann_test / rhoad_test
        elif transport_type == 'shear_viscosity' or transport_type == 'thermal_conductivity':
            transport_property_ann_train = np.exp(transport_property_ann_train)
            transport_property_ann_test = np.exp(transport_property_ann_test)

        # Entropy scaling model
        tp_model_entropy = dict_entropy_models[transport_type]
        transport_property_entropy_train = tp_model_entropy(alpha_train, Sres_train)
        transport_property_entropy_test = tp_model_entropy(alpha_test, Sres_test)
        transport_property_entropy_train = dict_entropy_scaling[transport_type](rhoad_train, Tad_train, transport_property_entropy_train, unscale=True)
        transport_property_entropy_test = dict_entropy_scaling[transport_type](rhoad_test, Tad_test, transport_property_entropy_test, unscale=True)

        ##############################################
        # saving results to dataframe and dictionary #
        ##############################################
        df_models_train = pd.DataFrame({'lr': lr_train, 'alpha': alpha_train, 'rhoad': rhoad_train, 'Tad': Tad_train, 'Sres': Sres_train, 
                                        transport_type: transport_property_train, f'{transport_type}_ideal': transport_ideal_train, 
                                        f'{transport_type}_ann': transport_property_ann_train, f'{transport_type}_ann_res': transport_property_res_train,
                                        f'{transport_type}_ann_entropy': transport_property_entropy_train})

        df_models_test = pd.DataFrame({'lr': lr_test, 'alpha': alpha_test, 'rhoad': rhoad_test, 'Tad': Tad_test, 'Sres': Sres_test, 
                                       transport_type: transport_property_test, f'{transport_type}_ideal': transport_ideal_test, 
                                       f'{transport_type}_ann': transport_property_ann_test, f'{transport_type}_ann_res': transport_property_res_test,
                                       f'{transport_type}_ann_entropy': transport_property_entropy_test})

        dict_parity[transport_type] = {'train': df_models_train, 'test': df_models_test}

    return dict_parity


def parity_plot_transport_md_lit(dict_md_lit_data,
                                 dict_models,
                                 transport_list=['self_diffusivity', 'shear_viscosity', 'thermal_conductivity']):

    dict_parity = dict()
    for transport_type in transport_list:

        # loading the data
        df = dict_md_lit_data[transport_type]
        lr = df['lr'].to_numpy()
        la = df['la'].to_numpy()
        Tad = df['T*'].to_numpy()
        rhoad = df['rho*'].to_numpy()
        transport_property = df[transport_type].to_numpy()
        transport_property_std = df[f'{transport_type}_std'].to_numpy()
        author_id = df['author_id'].to_numpy()

        # getting vdw alpha parameter for each Mie fluid
        alpha = helper_get_alpha(lr, la)

        # computing parity plots

        # full model
        tp_model = dict_models[transport_type]
        transport_property_ann = tp_model(alpha, rhoad, Tad)
        if transport_type == 'self_diffusivity':
            transport_property_ann = transport_property_ann / rhoad
        elif transport_type == 'shear_viscosity' or transport_type == 'thermal_conductivity':
            transport_property_ann = np.exp(transport_property_ann)

        ##############################################
        # saving results to dataframe and dictionary #
        ##############################################

        df_model = pd.DataFrame({'lr': lr, 'la': la, 'alpha': alpha, 'rhoad': rhoad, 'Tad': Tad,
                                 transport_type: transport_property,
                                 f'{transport_type}_std': transport_property_std,
                                 f'{transport_type}_ann': transport_property_ann,
                                 'author_id': author_id})

        dict_parity[transport_type] = df_model
    return dict_parity
