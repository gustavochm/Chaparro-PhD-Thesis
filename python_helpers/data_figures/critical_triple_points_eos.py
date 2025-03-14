import pandas as pd
import numpy as np
from ..feanneos import critical_point_solver
from ..feanneos import triple_point_solver
from ..helpers import helper_get_alpha
from jax import numpy as jnp
from jax.config import config
from copy import copy

PRECISSION = 'float64'
if PRECISSION == 'float64':
    config.update("jax_enable_x64", True)
    type_np = np.float64
    type_jax = jnp.float64
else:
    config.update("jax_enable_x32", True)
    type_np = np.float32
    type_jax = jnp.float32


############################################
# Critical and triple points FE-ANN(s) EoS #
############################################
def data_critical_and_triple_point_feanneos(fun_dic, lr_min, lr_max, n=500,
                                            lambda_a=6.,
                                            inc0_crit=[0.3, 2.0],
                                            inc0_triple=[1e-4, 0.91, 0.99, 0.75]):

    #####################
    # setting up arrays #
    #####################
    lr_array = np.linspace(lr_min, lr_max, n)
    alpha_array = helper_get_alpha(lr_array, lambda_a)
    # Critical Point
    Tcad_critical = np.zeros(n)
    rhocad_critical = np.zeros(n)
    Pcad_critical = np.zeros(n)
    success_critical = np.zeros(n, dtype=bool)
    # Triple Point
    rhovad_triple = np.zeros(n)
    rholad_triple = np.zeros(n)
    rhosad_triple = np.zeros(n)
    Tad_triple = np.zeros(n)
    Pad_triple = np.zeros(n)
    success_triple = np.zeros(n, dtype=bool)

    ########################################
    # Computing critical and triple points #
    ########################################
    i = 0
    # Critical Point
    sol_crit = critical_point_solver(alpha_array[i], fun_dic, inc0=inc0_crit, full_output=True)
    Tcad_critical[i] = sol_crit['Tcad']
    rhocad_critical[i] = sol_crit['rhocad']
    Pcad_critical[i] = sol_crit['Pcad']
    success_critical[i] = sol_crit['success']
    # Triple Point
    sol_triple = triple_point_solver(alpha_array[i], fun_dic, inc0=inc0_triple, full_output=True)
    rhovad_triple[i] = sol_triple['rhovad']
    rholad_triple[i] = sol_triple['rholad']
    rhosad_triple[i] = sol_triple['rhosad']
    Tad_triple[i] = sol_triple['Tad']
    Pad_triple[i] = sol_triple['Pad']
    success_triple[i] = sol_triple['success']

    for i in range(1, n):
        # Critical Point
        inc0_crit = [rhocad_critical[i-1], Tcad_critical[i-1]]
        sol_crit = critical_point_solver(alpha_array[i], fun_dic, inc0=inc0_crit, full_output=True)
        Tcad_critical[i] = sol_crit['Tcad']
        rhocad_critical[i] = sol_crit['rhocad']
        Pcad_critical[i] = sol_crit['Pcad']
        success_critical[i] = sol_crit['success']
        # Triple Point
        inc0_triple = [rhovad_triple[i-1], rholad_triple[i-1], rhosad_triple[i-1], Tad_triple[i-1]]
        sol_triple = triple_point_solver(alpha_array[i], fun_dic, inc0=inc0_triple, full_output=True)
        rhovad_triple[i] = sol_triple['rhovad']
        rholad_triple[i] = sol_triple['rholad']
        rhosad_triple[i] = sol_triple['rhosad']
        Tad_triple[i] = sol_triple['Tad']
        Pad_triple[i] = sol_triple['Pad']
        success_triple[i] = sol_triple['success']
        if Tad_triple[i] > Tcad_critical[i]:
            Tcad_critical[i:] = np.nan
            rhocad_critical[i:] = np.nan
            Pcad_critical[i:] = np.nan
            success_critical[i:] = np.nan
            #
            rhovad_triple[i:] = np.nan
            rholad_triple[i:] = np.nan
            rhosad_triple[i:] = np.nan
            Tad_triple[i:] = np.nan
            Pad_triple[i:] = np.nan
            success_triple[i:] = np.nan
            break

    df = pd.DataFrame({'lr': lr_array, 'la': lambda_a * np.ones(n), 'alpha': alpha_array,
                    'Tad_crit': Tcad_critical, 'rhoad_crit': rhocad_critical, 'Pad_crit': Pcad_critical, 'success_crit': success_critical,
                    'Tad_triple': Tad_triple, 'rhovad_triple': rhovad_triple, 'rholad_triple': rholad_triple, 
                    'rhosad_triple': rhosad_triple, 'Pad_triple': Pad_triple, 'success_triple': success_triple, 'Tc/Tt': Tcad_critical/Tad_triple})
    df.dropna(how='any', inplace=True)
    return df


############################################
# Critical and triple points FE-ANN(s) EoS #
############################################
def data_critical_and_triple_point_feanneos_by_parts(fun_dic, lr_min, lr_max, 
                                                     n_min=100,
                                                     n_max=200,
                                                     lambda_a=6.,
                                                     lr0=16,
                                                     inc0_crit=[0.3, 1.3],
                                                     inc0_triple=[1e-3, 0.83, 0.97, 0.67]):
    inc0_crit_lr0 = copy(inc0_crit)
    inc0_triple_lr0 = copy(inc0_triple)


    #####################
    # setting up arrays #
    #####################
    n = n_min
    lr_array_min = np.linspace(lr0, lr_min, n)
    alpha_array_min = helper_get_alpha(lr_array_min, lambda_a)
    # Critical Point
    Tcad_critical_min = np.zeros(n)
    rhocad_critical_min = np.zeros(n)
    Pcad_critical_min = np.zeros(n)
    success_critical_min = np.zeros(n, dtype=bool)
    # Triple Point
    rhovad_triple_min = np.zeros(n)
    rholad_triple_min = np.zeros(n)
    rhosad_triple_min = np.zeros(n)
    Tad_triple_min = np.zeros(n)
    Pad_triple_min = np.zeros(n)
    success_triple_min = np.zeros(n, dtype=bool)

    ########################################
    # Computing critical and triple points #
    ########################################
    i = 0
    # Critical Point
    sol_crit = critical_point_solver(alpha_array_min[i], fun_dic, inc0=inc0_crit_lr0, full_output=True)
    Tcad_critical_min[i] = sol_crit['Tcad']
    rhocad_critical_min[i] = sol_crit['rhocad']
    Pcad_critical_min[i] = sol_crit['Pcad']
    success_critical_min[i] = sol_crit['success']
    # Triple Point
    sol_triple = triple_point_solver(alpha_array_min[i], fun_dic, inc0=inc0_triple_lr0, full_output=True)
    rhovad_triple_min[i] = sol_triple['rhovad']
    rholad_triple_min[i] = sol_triple['rholad']
    rhosad_triple_min[i] = sol_triple['rhosad']
    Tad_triple_min[i] = sol_triple['Tad']
    Pad_triple_min[i] = sol_triple['Pad']
    success_triple_min[i] = sol_triple['success']

    for i in range(1, n):
        # Critical Point
        inc0_crit = [rhocad_critical_min[i-1], Tcad_critical_min[i-1]]
        sol_crit = critical_point_solver(alpha_array_min[i], fun_dic, inc0=inc0_crit, full_output=True)
        Tcad_critical_min[i] = sol_crit['Tcad']
        rhocad_critical_min[i] = sol_crit['rhocad']
        Pcad_critical_min[i] = sol_crit['Pcad']
        success_critical_min[i] = sol_crit['success']
        # Triple Point
        inc0_triple = [rhovad_triple_min[i-1], rholad_triple_min[i-1], rhosad_triple_min[i-1], Tad_triple_min[i-1]]
        sol_triple = triple_point_solver(alpha_array_min[i], fun_dic, inc0=inc0_triple, full_output=True)
        rhovad_triple_min[i] = sol_triple['rhovad']
        rholad_triple_min[i] = sol_triple['rholad']
        rhosad_triple_min[i] = sol_triple['rhosad']
        Tad_triple_min[i] = sol_triple['Tad']
        Pad_triple_min[i] = sol_triple['Pad']
        success_triple_min[i] = sol_triple['success']
        if Tad_triple_min[i] > Tcad_critical_min[i]:
            Tcad_critical_min[i:] = np.nan
            rhocad_critical_min[i:] = np.nan
            Pcad_critical_min[i:] = np.nan
            success_critical_min[i:] = np.nan
            #
            rhovad_triple_min[i:] = np.nan
            rholad_triple_min[i:] = np.nan
            rhosad_triple_min[i:] = np.nan
            Tad_triple_min[i:] = np.nan
            Pad_triple_min[i:] = np.nan
            success_triple_min[i:] = np.nan
            break

    #######

    #####################
    # setting up arrays #
    #####################
    n = n_max
    lr_array_max = np.linspace(lr0, lr_max, n)
    alpha_array_max = helper_get_alpha(lr_array_max, lambda_a)
    # Critical Point
    Tcad_critical_max = np.zeros(n)
    rhocad_critical_max = np.zeros(n)
    Pcad_critical_max = np.zeros(n)
    success_critical_max = np.zeros(n, dtype=bool)
    # Triple Point
    rhovad_triple_max = np.zeros(n)
    rholad_triple_max = np.zeros(n)
    rhosad_triple_max = np.zeros(n)
    Tad_triple_max = np.zeros(n)
    Pad_triple_max = np.zeros(n)
    success_triple_max = np.zeros(n, dtype=bool)

    ########################################
    # Computing critical and triple points #
    ########################################
    i = 0
    # Critical Point
    sol_crit = critical_point_solver(alpha_array_max[i], fun_dic, inc0=inc0_crit_lr0, full_output=True)
    Tcad_critical_max[i] = sol_crit['Tcad']
    rhocad_critical_max[i] = sol_crit['rhocad']
    Pcad_critical_max[i] = sol_crit['Pcad']
    success_critical_max[i] = sol_crit['success']
    # Triple Point
    sol_triple = triple_point_solver(alpha_array_max[i], fun_dic, inc0=inc0_triple_lr0, full_output=True)
    rhovad_triple_max[i] = sol_triple['rhovad']
    rholad_triple_max[i] = sol_triple['rholad']
    rhosad_triple_max[i] = sol_triple['rhosad']
    Tad_triple_max[i] = sol_triple['Tad']
    Pad_triple_max[i] = sol_triple['Pad']
    success_triple_max[i] = sol_triple['success']

    for i in range(1, n):
        # Critical Point
        inc0_crit = [rhocad_critical_max[i-1], Tcad_critical_max[i-1]]
        sol_crit = critical_point_solver(alpha_array_max[i], fun_dic, inc0=inc0_crit, full_output=True)
        Tcad_critical_max[i] = sol_crit['Tcad']
        rhocad_critical_max[i] = sol_crit['rhocad']
        Pcad_critical_max[i] = sol_crit['Pcad']
        success_critical_max[i] = sol_crit['success']
        # Triple Point
        inc0_triple = [rhovad_triple_max[i-1], rholad_triple_max[i-1], rhosad_triple_max[i-1], Tad_triple_max[i-1]]
        sol_triple = triple_point_solver(alpha_array_max[i], fun_dic, inc0=inc0_triple, full_output=True)
        rhovad_triple_max[i] = sol_triple['rhovad']
        rholad_triple_max[i] = sol_triple['rholad']
        rhosad_triple_max[i] = sol_triple['rhosad']
        Tad_triple_max[i] = sol_triple['Tad']
        Pad_triple_max[i] = sol_triple['Pad']
        success_triple_max[i] = sol_triple['success']
        if Tad_triple_max[i] > Tcad_critical_max[i]:
            Tcad_critical_max[i:] = np.nan
            rhocad_critical_max[i:] = np.nan
            Pcad_critical_max[i:] = np.nan
            success_critical_max[i:] = np.nan
            #
            rhovad_triple_max[i:] = np.nan
            rholad_triple_max[i:] = np.nan
            rhosad_triple_max[i:] = np.nan
            Tad_triple_max[i:] = np.nan
            Pad_triple_max[i:] = np.nan
            success_triple_max[i:] = np.nan
            break

    lr_array = np.hstack([lr_array_min, lr_array_max[1:]])
    alpha_array = np.hstack([alpha_array_min, alpha_array_max[1:]])
    Tcad_critical = np.hstack([Tcad_critical_min, Tcad_critical_max[1:]])
    rhocad_critical = np.hstack([rhocad_critical_min, rhocad_critical_max[1:]])
    Pcad_critical = np.hstack([Pcad_critical_min, Pcad_critical_max[1:]])
    success_critical = np.hstack([success_critical_min, success_critical_max[1:]])
    rhovad_triple = np.hstack([rhovad_triple_min, rhovad_triple_max[1:]])
    rholad_triple = np.hstack([rholad_triple_min, rholad_triple_max[1:]])
    rhosad_triple = np.hstack([rhosad_triple_min, rhosad_triple_max[1:]])
    Tad_triple = np.hstack([Tad_triple_min, Tad_triple_max[1:]])
    Pad_triple = np.hstack([Pad_triple_min, Pad_triple_max[1:]])
    success_triple = np.hstack([success_triple_min, success_triple_max[1:]])

    df = pd.DataFrame({'lr': lr_array, 'la': lambda_a * np.ones_like(lr_array), 'alpha': alpha_array,
                       'Tad_crit': Tcad_critical, 'rhoad_crit': rhocad_critical, 'Pad_crit': Pcad_critical, 'success_crit': success_critical,
                       'Tad_triple': Tad_triple, 'rhovad_triple': rhovad_triple, 'rholad_triple': rholad_triple, 
                       'rhosad_triple': rhosad_triple, 'Pad_triple': Pad_triple, 'success_triple': success_triple, 'Tc/Tt': Tcad_critical/Tad_triple})
    df.dropna(how='any', inplace=True)
    df.sort_values('lr', inplace=True)
    return df
