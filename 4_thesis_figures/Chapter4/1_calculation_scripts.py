import os
import sys
import numpy as np
import pandas as pd
import nest_asyncio
nest_asyncio.apply()

import jax
from jax import numpy as jnp
from jax.config import config
from flax.training import checkpoints
from flax import linen as nn

PRECISSION = 'float64'
if PRECISSION == 'float64':
    config.update("jax_enable_x64", True)
    type_np = np.float64
    type_jax = jnp.float64
else:
    config.update("jax_enable_x32", True)
    type_np = np.float32
    type_jax = jnp.float32

sys.path.append("../../")
from python_helpers import helper_get_alpha
from python_helpers import linear_activation

from python_helpers.feanneos import HelmholtzModel
from python_helpers.feanneos import helper_jitted_funs

from python_helpers.transport_properties import TransportModel_PVT_Tinv, TransportModelResidual_PVT_Tinv, TransportModel_entropy
from python_helpers.transport_properties import density_diffusivity_mie6_dilute, viscosity_mie6_dilute, thermal_conductivity_mie6_dilute
from python_helpers.transport_properties import diffusivity_scaling, viscosity_scaling, thermal_conductivity_scaling
# loading computing function file
from python_helpers.data_figures import latex_description_database_tp
from python_helpers.data_figures import parity_plot_transport_md, parity_plot_transport_md_lit
from python_helpers.data_figures import data_tp_isotherms_lr
from python_helpers.data_figures import data_tp_dilute_lrs

#############################
# Compute or not data again #
#############################
compute_data = True

##########################
# Folder to save results #
##########################

folder_to_save = './computed_files'
os.makedirs(folder_to_save, exist_ok=True)

######################
# Loading FE-ANN EoS #
######################

ckpt_folder = '../../3_ann_models/feann_eos'

prefix_params = 'FE-ANN-EoS-params_'

###
Tscale = 'Tinv'
seed = 1
factor = 0.05
EPOCHS = 20000
traind_model_folder = f'models_{Tscale}_factor{factor:.2f}_seed{seed}'
ckpt_folder_model = os.path.join(ckpt_folder, traind_model_folder)
ckpt_Tinv = checkpoints.restore_checkpoint(ckpt_dir=ckpt_folder_model, target=None, prefix=prefix_params)
helmholtz_features = list(ckpt_Tinv['features'].values())
helmholtz_model = HelmholtzModel(features=helmholtz_features)
helmholtz_params = {'params': ckpt_Tinv['params']}
fun_dic = helper_jitted_funs(helmholtz_model, helmholtz_params)

#####################
# Loading TP Models #
#####################

activation_dicts = {'linear': linear_activation, 'softplus': nn.softplus}

########################### 
# Self-diffusivity models #
###########################

folder_diff = '../../3_ann_models/selfdiff_models'
hidden_layers = 2
neurons = 30

# Residual rhodiff model
prefix = 'rhodiff-rho-Tinv-residual-penalty'
ckpt_dir = folder_diff
seed = 0
features = hidden_layers * [neurons]
activation = 'linear'
params_prefix = f'{prefix}-seed{seed}-params_'
state_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix=params_prefix)
params_rhodiff_res = {'params': state_restored['params']}
rhodiff_res_model = TransportModelResidual_PVT_Tinv(features=features, output_activation=activation_dicts[activation])

# Residual rhodiff model
prefix = 'rhodiff-rho-Tinv-penalty'
ckpt_dir = folder_diff
seed = 1
features = hidden_layers * [neurons]
activation = 'softplus'
params_prefix = f'{prefix}-seed{seed}-params_'
state_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix=params_prefix)
params_rhodiff = {'params': state_restored['params']}
rhodiff_model = TransportModel_PVT_Tinv(features=features, output_activation=activation_dicts[activation])

# Diff entropy scaling model
prefix = 'diff-entropy-penalty'
ckpt_dir = folder_diff
seed = 1
features = hidden_layers * [neurons]
activation = 'softplus'
params_prefix = f'{prefix}-seed{seed}-params_'
state_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix=params_prefix)
params_diff_entropy = {'params': state_restored['params']}
diff_entropy_model = TransportModel_entropy(features=features, output_activation=activation_dicts[activation])


########################## 
# Shear viscosity models #
##########################

folder_visc = '../../3_ann_models/visc_models'
hidden_layers = 2
neurons = 30

# Residual logvisc model
prefix = 'logvisc-rho-Tinv-residual-penalty'
ckpt_dir = folder_visc
seed = 42
features = hidden_layers * [neurons]
activation = 'linear'
params_prefix = f'{prefix}-seed{seed}-params_'
state_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix=params_prefix)
params_logvisc_res = {'params': state_restored['params']}
logvisc_res_model = TransportModelResidual_PVT_Tinv(features=features, output_activation=activation_dicts[activation])

# Residual logvisc model
prefix = 'logvisc-rho-Tinv-penalty'
ckpt_dir = folder_visc
seed = 0
features = hidden_layers * [neurons]
activation = 'linear'
params_prefix = f'{prefix}-seed{seed}-params_'
state_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix=params_prefix)
params_logvisc = {'params': state_restored['params']}
logvisc_model = TransportModel_PVT_Tinv(features=features, output_activation=activation_dicts[activation])

# visc entropy scaling
prefix = 'visc-entropy-penalty'
ckpt_dir = folder_visc
seed = 0
features = hidden_layers * [neurons]
activation = 'softplus'
params_prefix = f'{prefix}-seed{seed}-params_'
state_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix=params_prefix)
params_visc_entropy = {'params': state_restored['params']}
visc_entropy_model = TransportModel_entropy(features=features, output_activation=activation_dicts[activation])

##############################
# Thermal conductivity model #
##############################

folder_tcond = '../../3_ann_models/tcond_models'
hidden_layers = 3
neurons = 30

# Residual logtcond model
prefix = 'logtcond-rho-Tinv-residual-penalty'
ckpt_dir = folder_tcond
seed = 42
features = hidden_layers * [neurons]
activation = 'linear'
params_prefix = f'{prefix}-seed{seed}-params_'
state_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix=params_prefix)
params_logtcond_res = {'params': state_restored['params']}
logtcond_res_model = TransportModelResidual_PVT_Tinv(features=features, output_activation=activation_dicts[activation])

# logtcond model
prefix = 'logtcond-rho-Tinv-penalty'
ckpt_dir = folder_tcond
seed = 1
features = hidden_layers * [neurons]
activation = 'linear'
params_prefix = f'{prefix}-seed{seed}-params_'
state_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix=params_prefix)
params_logtcond = {'params': state_restored['params']}
logtcond_model = TransportModel_PVT_Tinv(features=features, output_activation=activation_dicts[activation])

# tcond residual entropy model
prefix = 'tcond-entropy-penalty'
ckpt_dir = folder_tcond
seed = 1337
features = hidden_layers * [neurons]
activation = 'softplus'
params_prefix = f'{prefix}-seed{seed}-params_'
state_restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix=params_prefix)
params_tcond_entropy = {'params': state_restored['params']}
tcond_entropy_model = TransportModel_entropy(features=features, output_activation=activation_dicts[activation])

#######################
# Compiling TP models #
#######################

# Self diffusivity models
rhodiff_res_model_jit = jax.jit(lambda lr, rhoad, Tad: rhodiff_res_model.apply(params_rhodiff_res, jnp.atleast_1d(lr), jnp.atleast_1d(rhoad), jnp.atleast_1d(Tad)))
rhodiff_model_jit = jax.jit(lambda alpha, rhoad, Tad: rhodiff_model.apply(params_rhodiff, jnp.atleast_1d(alpha), jnp.atleast_1d(rhoad), jnp.atleast_1d(Tad)))
diff_entropy_model_jit = jax.jit(lambda alpha, Sres: diff_entropy_model.apply(params_diff_entropy, jnp.atleast_1d(alpha), jnp.atleast_1d(Sres)))
                  
# Shear viscosity models
logvisc_res_model_jit = jax.jit(lambda lr, rhoad, Tad: logvisc_res_model.apply(params_logvisc_res, jnp.atleast_1d(lr), jnp.atleast_1d(rhoad), jnp.atleast_1d(Tad)))
logvisc_model_jit = jax.jit(lambda alpha, rhoad, Tad: logvisc_model.apply(params_logvisc, jnp.atleast_1d(alpha), jnp.atleast_1d(rhoad), jnp.atleast_1d(Tad)))
visc_entropy_model_jit = jax.jit(lambda alpha, Sres: visc_entropy_model.apply(params_visc_entropy, jnp.atleast_1d(alpha), jnp.atleast_1d(Sres)))

# Thermal conductivity models
logtcond_res_model_jit = jax.jit(lambda lr, rhoad, Tad: logtcond_res_model.apply(params_logtcond_res, jnp.atleast_1d(lr), jnp.atleast_1d(rhoad), jnp.atleast_1d(Tad)))
logtcond_model_jit = jax.jit(lambda alpha, rhoad, Tad: logtcond_model.apply(params_logtcond, jnp.atleast_1d(alpha), jnp.atleast_1d(rhoad), jnp.atleast_1d(Tad)))
tcond_entropy_model_jit = jax.jit(lambda alpha, Sres: tcond_entropy_model.apply(params_tcond_entropy, jnp.atleast_1d(alpha), jnp.atleast_1d(Sres)))

dict_res_models = {'self_diffusivity': rhodiff_res_model_jit, 'shear_viscosity': logvisc_res_model_jit, 'thermal_conductivity': logtcond_res_model_jit}
dict_models = {'self_diffusivity': rhodiff_model_jit, 'shear_viscosity': logvisc_model_jit, 'thermal_conductivity': logtcond_model_jit}
dict_entropy_models = {'self_diffusivity': diff_entropy_model_jit, 'shear_viscosity': visc_entropy_model_jit, 'thermal_conductivity': tcond_entropy_model_jit}

### Extra functions
dict_dilute_functions = {'self_diffusivity': density_diffusivity_mie6_dilute, 'shear_viscosity': viscosity_mie6_dilute, 'thermal_conductivity': thermal_conductivity_mie6_dilute}
dict_entropy_scaling = {'self_diffusivity': diffusivity_scaling, 'shear_viscosity': viscosity_scaling, 'thermal_conductivity': thermal_conductivity_scaling}


###################
# Loading TP data #
###################

# reading the data
dbpath = "../../2_databases/mieparticle-diff.csv"
df_diff = pd.read_csv(dbpath)
alpha_diff = helper_get_alpha(df_diff['lr'].to_numpy(), df_diff['la'].to_numpy())
rhoad_diff = df_diff['rho*'].to_numpy()
Tad_diff = df_diff['T*'].to_numpy()
Sres_diff = fun_dic['entropy_residual_fun'](alpha_diff, rhoad_diff, Tad_diff)
df_diff['Sr'] = Sres_diff

dbpath = "../../2_databases/mieparticle-visc.csv"
df_visc = pd.read_csv(dbpath)
alpha_visc = helper_get_alpha(df_visc['lr'].to_numpy(), df_visc['la'].to_numpy())
rhoad_visc = df_visc['rho*'].to_numpy()
Tad_visc = df_visc['T*'].to_numpy()
Sres_visc = fun_dic['entropy_residual_fun'](alpha_visc, rhoad_visc, Tad_visc)
df_visc['Sr'] = Sres_visc

dbpath = "../../2_databases/mieparticle-tcond.csv"
df_tcond = pd.read_csv(dbpath)
alpha_tcond = helper_get_alpha(df_tcond['lr'].to_numpy(), df_tcond['la'].to_numpy())
rhoad_tcond = df_tcond['rho*'].to_numpy()
Tad_tcond = df_tcond['T*'].to_numpy()
Sres_tcond = fun_dic['entropy_residual_fun'](alpha_tcond, rhoad_tcond, Tad_tcond)
df_tcond['Sr'] = Sres_tcond

dict_md_data = {'self_diffusivity': df_diff, 'shear_viscosity': df_visc, 'thermal_conductivity': df_tcond}

# literature data
dbpath = "../../2_databases/MieParticle-TransportProperties-literature/mieparticle-literature-diffusivity.csv"
df_diff_lit = pd.read_csv(dbpath)

dbpath = "../../2_databases/MieParticle-TransportProperties-literature/mieparticle-literature-viscosity.csv"
df_visc_lit = pd.read_csv(dbpath)

dbpath = "../../2_databases/MieParticle-TransportProperties-literature/mieparticle-literature-thermal-conductivity.csv"
df_tcond_lit = pd.read_csv(dbpath)

dict_md_lit_data = {'self_diffusivity': df_diff_lit, 'shear_viscosity': df_visc_lit, 'thermal_conductivity': df_tcond_lit}

###################
# TP calculations #
###################

#######################################
# Summary table to describe database # 
#######################################
if compute_data:
    tabular_latex = latex_description_database_tp(dict_md_data)
    filename = "database_latex_table_tp.md"
    path_to_save = os.path.join(folder_to_save, filename)
    text_file = open(path_to_save, "w")
    text_file.write(tabular_latex)
    text_file.close()

###########################
# Parity plot data for TP # 
###########################
if compute_data:
    transport_list = ['self_diffusivity', 'shear_viscosity', 'thermal_conductivity']
    out_parity = parity_plot_transport_md(dict_md_data,
                                          dict_models,
                                          dict_res_models, dict_dilute_functions,
                                          dict_entropy_models, dict_entropy_scaling,
                                          transport_list=transport_list)
    for transport in transport_list:
        filename = f'parity_data_{transport}.xlsx'
        file_to_save = os.path.join(folder_to_save, filename)
        writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
        dict_transport = out_parity[transport]
        for key, df in dict_transport.items():
            df.to_excel(writer, sheet_name=key, index=False)
        writer.close()

######################################
# Parity plot for TP literature data # 
######################################
if compute_data:
    transport_list = ['self_diffusivity', 'shear_viscosity', 'thermal_conductivity']
    out_parity_lit = parity_plot_transport_md_lit(dict_md_lit_data, dict_models,
                                                  transport_list=transport_list)
    for transport in transport_list:
        filename = f'parity_data_lit_{transport}.xlsx'
        file_to_save = os.path.join(folder_to_save, filename)
        out_parity_lit[transport].to_excel(file_to_save, index=False)

##############################
# Computing isotherms for TP # 
##############################
if compute_data:
    lr_list = [10, 12., 16., 18, 20., 26.]
    T_list = [0.8, 0.9, 1.0, 1.1, 1.3, 2.8, 6.0, 7.2]
    for lambda_r in lr_list:
        isotherms_lr = data_tp_isotherms_lr(dict_models,
                                            dict_res_models, dict_dilute_functions,
                                            T_list,
                                            lambda_r=lambda_r, rho_min=0., rho_max=1.25, n=200)
        filename = f'isotherms_tp_lr{lambda_r:.0f}.xlsx'
        file_to_save = os.path.join(folder_to_save, filename)
        writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
        for key, df in isotherms_lr.items():
            df.to_excel(writer, sheet_name=key, index=False)
        writer.close()

#####################################
# Computing dilute TP for the model #
#####################################
if compute_data:
    lr_list = [12., 16., 20., 24., 28.]
    out_lrs = data_tp_dilute_lrs(dict_models, dict_res_models, dict_dilute_functions, lr_list)
    filename = 'dilute_tp_lrs.xlsx'
    file_to_save = os.path.join(folder_to_save, filename)
    writer = pd.ExcelWriter(file_to_save, engine='xlsxwriter')
    for key, df in out_lrs.items():
        df.to_excel(writer, sheet_name=key, index=False)
    writer.close()
