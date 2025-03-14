# Mie particle thermophysical properties data

This folder contains the databases used to train and assess the behavior of the ANN models developed in this thesis. Here is a brief description of each file:

1. ['crit_triple_ms_literature.csv'](./crit_triple_ms_literature.csv): Critical and triple points of the Mie particle from literature (See Chapter 3 of the thesis for a complete list of the references)
1. ['mieparticle-brown.csv'](./mieparticle-brown.csv): Brown characteristic curve data points computed for selected Mie particles. Data obtained from [Stephan and Urschel (2023)](https://doi.org/10.1016/j.molliq.2023.122088).
1. ['mieparticle-data'](./mieparticle-data.csv): Database of thermophysical properties related to the Helmholtz free energy obtained from molecular dynamics simulations. This database explicitly contains the results simulated at different ensembles (NVE/NVT/NPT). (Computed in this work)
1. ['mieparticle-data-training'](./mieparticle-data-training.csv): Database of thermophysical properties related to the Helmholtz free energy. This database is a cleaner version of the ['mieparticle-data'](./mieparticle-data.csv) file and was used to train the FE-ANN and FE-ANN(s) EoS. (Computed in this work)
1. ['mieparticle-virial-coefficients'](./mieparticle-virial-coefficients.csv): Second and third virial coefficients of selected Mie particles. These virial coefficients were obtained directly by numerical integration. See Chapter 2 for further details. (Computed in this work)
1. ['mieparticle-vle'](./mieparticle-vle.csv): Vapor-liquid equilibria data for selected Mie particles obtained by the temperature-quench method. (Computed in this work)
1. ['mieparticle-sle'](./mieparticle-sle.csv): Solid-liquid equilibria data for selected Mie particles obtained by the freeze method. (Computed in this work)
1. ['mieparticle-hvap'](./mieparticle-hvap.csv): Vaporization enthalpy obtained from molecular dynamics results. The equilibrium states were taken from the ['mieparticle-vle'](./mieparticle-vle.csv) file. (Computed in this work)
1. ['mieparticle-melting'](./mieparticle-hmelting.csv): Melting enthalpy obtained from molecular dynamics results. The equilibrium states were taken from the ['mieparticle-sle'](./mieparticle-sle.csv) file. (Computed in this work)
1. ['mieparticle-crit'](./mieparticle-crit.csv): Critical points obtained by fitting a scaling law to vapor-liquid equilibria data obtained from MD simulations (['mieparticle-vle'](./mieparticle-vle.csv) file). (Computed in this work)
1. ['mieparticle-diff'](./mieparticle-diff.csv): Self diffusivity data of selected Mie particles obtained from Equilibrium Molecular Dynamics simulations. See Chapter 4 for further details. (Computed in this work)
1. ['mieparticle-visc'](./mieparticle-diff.csv): Shear viscosity data of selected Mie particles obtained from Equilibrium Molecular Dynamics simulations. See Chapter 4 for further details. (Computed in this work)
1. ['mieparticle-tcond'](./mieparticle-tcond.csv): Thermal conductivity data of selected Mie particles obtained from Equilibrium Molecular Dynamics simulations. See Chapter 4 for further details. (Computed in this work)
