module purge
module load intel-suite/2019.4
module load mpi/intel-2019.8.254
module load cmake/3.18.2
module load git
module load ffmpeg
module load fftw

# uncomment to download latest version
# git clone https://github.com/lammps/lammps.git
# git clone https://github.com/omoultosEthTuDelft/OCTP.git # this is the original repo
# git clone https://github.com/sondresc/OCTP.git # this is updates for new lammps versions


octp=OCTP
lam=lammps_Jun2022
git clone --branch stable_23Jun2022 https://github.com/lammps/lammps.git ${lam}
# git clone https://github.com/omoultosEthTuDelft/OCTP.git # this is the original repo
git clone https://github.com/sondresc/OCTP.git # this is updates for new lammps versions

# copying modified files
cp ./OCTP/*.cpp ./${lam}/src
cp ./OCTP/*.h ./${lam}/src

cd ~/myprogramms/${lam}
rm -r build
mkdir build
cd build
cmake -DLAMMPS_MACHINE=hpc -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc \
-DCMAKE_Fortran_COMPILER=ifort -DINTEL_ARCH=cpu -DPKG_INTEL=on \
-DPKG_OPT=yes -DPKG_OPENMP=yes -DFFT=mkl \
-DPKG_KSPACE=on -DPKG_RIGID=on -DPKG_EXTRA-PAIR=on -DPKG_MOLECULE=on \
-DPKG_EXTRA-MOLECULE=on ../cmake
# compile lammps
make -j 12
