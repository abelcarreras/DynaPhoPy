##############################################################################
# Example: Si with LAMMPS interface at 900 K                                 #
# -------------------------------------------------------------------------- #
# Calculation of anharmonic properties of Si using LAMMPS python interface.  #
# This mode requires LAMMPS python interface to be installed in your system. #
# Please check LAMMPS website for detailed information:                      #
# http://lammps.sandia.gov/doc/Section_python.html                           #
##############################################################################

1. Prepare LAMMPS input script. The atoms positions are written in an external
file (data_unitcell.si). The unit cell defined in this file should be the same
as the ones defined in the POSCAR_unitcell file.

2. Calculate the harmonic force constants. For this purpose you can use fc_lammps.py
script in scripts folder. Ex:

fc_lammps.py input_dynaphopy in.lammps -o FORCE_CONSTANTS_LAMMPS

3. Run dynaphopy using  "--run_lammps" with arguments. These arguments
are: [lammps input script, MD simulation total time [ps], time step [ps], and
stabilization time [ps]. Also you may want to define "--dim" to set the size of
the supercell respect to the unit cell used in the molecular dynamics (MD)
simulation (by default the unitcell is used). Ex:

dynaphopy input_si --run_lammps in.lammps 50 0.001 50 --dim 2 2 2 -i

Note: The LAMMPS interface allows to perform dynamphopy calculations using LAMMPS MD without
writing data on files. Note that the full MD will be calculated every time
dynaphopy is executed. Use option flags (or interactive mode) to calculate several
properties in one run. Alternatively, the trajectory can be stored in the disk in hdf5 format
for later analysis using "--save_data" option.

Note: The Tersoff potential file included in this example (SiCGe.tersoff) is part of LAMMPS package.
This file is included here only for convenience.