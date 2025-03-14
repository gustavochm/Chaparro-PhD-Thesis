# Artificial Neural Network models

This folder contains the trained machine learning models in this thesis. Here is a list of available models.

1. [FE-ANN EoS](./feann_eos/). FE-ANN EoS trained exclusively on fluid data, see Chapter 2 for further details. Two models are available that differ on how the temperature is used in ANNs. One uses the temperature itself (T) and the other uses the inverse temperature (1/T).
1. [FE-ANN(s) EoS](./feann_eos/). Trained FE-ANN(s) EoS using fluid and crystal Mie data, see Chapter 3 for further details.
1. [Self-diffusivity models](./selfdiff_models/). Trained models for the self-diffusivity of the Mie fluid. Three models are included: one based $\rho^* D^*$, $\rho^* D^{*, \mathrm{res}}$ and $\tilde{D}^*$ (based on entropy scaling).
1. [Shear viscosity models](./visc_models/). Trained models for the shear viscosity of the Mie fluid. Three models are included: one based $\ln \eta^*$, $\ln \eta^{*, \mathrm{res}}$ and $\tilde{\eta}^*$ (based on entropy scaling).
1. [Thermal conductivity models](./tcond_models/). Trained models for the thermal conductivity of the Mie fluid. Three models are included: one based $\ln \kappa^*$, $\ln \kappa^{*, \mathrm{res}}$ and $\tilde{\kappa}^*$ (based on entropy scaling).

Examples on how to load the models are given in the [loading_models.ipynb](./loading_models.ipynb) jupyter notebook.
