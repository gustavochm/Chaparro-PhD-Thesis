# LAMMPS files

This folder contains [LAMMPS](https://docs.lammps.org/Manual.html) files to reproduce the molecular simulations results obtained from this thesis. Here is a brief description of each file:

1. ['build-lammps-Jun2022'](./build-lammps-Jun2022.sh): This bash script was used to compile LAMMPS (stable version June 2022) with OCTP plugin ([Jamali et al (2019)](https://doi.org/10.1021/acs.jcim.8b00939)).  
1. ['quench-input-vle'](./quench-input-vle.in): This input file is used to compute vapor-liquid equilibria (VLE) using the temperature quench method.
1. Solid-liquid equilibria calculation using the freeze method. (See Chapter 3 for further details)
   1. ['freezing-initial-solid'](./freezing-initial-solid.in): Simulation of an initial crystal phase in a FCC conformartion. This is used the set-up stage of the freeze method.
   1. ['freezing-initial-isobars-steps'](./freezing-initial-solid.in): Input file to perform isobaric heating/cooling needed in the set-up stage of the freeze method.
   1. ['freezing-input-sle'](./freezing-input-sle.in): Input file to perform the coexistence step of the freeze method. This input file requires an initial solid configuration which is obtained from the set-up stage.
1. Files to compute thermophysical properties of the Mie particle.
    1. ['input-nvt'](./input-nvt.in): Input file to perform a simulation in the NVT ensemble. This input file computes thermodynamic statistics of the system. The simulation is initilizated as a FCC crystal.
    1. ['input-npt'](./input-npt.in): Input file to perform a simulation in the NPT ensemble. This input file computes thermodynamic statistics of the system. The simulation is initilizated from the final state of the NVT simulation.
    1. ['input-nve-transport'](./input-nve-transport.in): Input file to perform a simulation in the NVE ensemble. This input file computes thermodynamic statistics of the system and MSDs required to compute transport properties. The simulation is initilizated from the final state of the NVT simulation.
