#!/usr/bin/env python

import numpy as np
import phonopy.file_IO as file_IO
import dynaphopy.interface.iofile as io
import dynaphopy.interface.iofile.trajectory_parsers as parsers
import dynaphopy

##################################  STRUCTURE FILES #######################################
# 1. Set the directory in where the FORCE_SETS and structure OUTCAR are placed
# FORCE_SETS : force set file obtained from PHONOPY calculation
# OUTCAR : Single Point calculation of the unit cell structure used in PHONOPY calculation

directory ='/home/abel/VASP/Si/Si-phonon/4x4x4B/'

structure = io.read_from_file_structure_poscar(directory + 'POSCAR')
structure.set_force_set(file_IO.parse_FORCE_SETS(filename=directory+'FORCE_SETS'))


############################### PHONOPY CELL INFORMATION ####################################
# 2. Set primitive matrix, this matrix fulfills that:
#    Primitive_cell = Unit_cell x Primitive_matrix
#    This matrix is the same needed for PHONOPY calculation

structure.set_primitive_matrix([[0.5, 0.0, 0.0],
                                [0.0, 0.5, 0.0],
                                [0.0, 0.0, 0.5]])



# 3. Set super cell phonon, this matrix denotes the super cell used in PHONOPY for creating
# the finite displacements

structure.set_super_cell_phonon([[4, 0, 0],
                                 [0, 4, 0],
                                 [0, 0, 4]])


################################### TRAJECTORY FILES ##########################################
# 4. Set the location of OUTCAR/LAMMPS file containing the Molecular Dynamics trajectory

trajectory = parsers.read_vasp_trajectory('/home/abel/VASP/Si/Si-FINAL3/Si_0.5_400/No1/OUTCAR', structure)
# or
#trajectory = parsers.read_lammps_trajectory('/home/abel/LAMMPS/Si/Si_400.lammpstrj', structure, initial_cut=10000, end_cut=12000)


calculation = dynaphopy.Quasiparticle(trajectory)


############################## DEFINE CALCULATION REQUESTS #####################################
# All this options are totally optional and independent, just comment or uncomment the desired ones
# Other option not yet shown in this example script may be available (take a look at controller,py)
# The requests will be satisfied by request order
# Python scripting features may be used for more complex requests

# 5a. Set the power spectrum algorithm
calculation.select_power_spectra_algorithm(1)
calculation.set_number_of_mem_coefficients(1000)

# 5b. Set wave vector into which is going to be projected the velocity (default: gamma point)
calculation.set_reduced_q_vector([0.5, 0.0, 0.0]) # X Point

# 5c. Define range of frequency to analyze (default: 0-20THz)
calculation.set_frequency_range(np.linspace(2, 15, 2000)) #(range: 2 to 15Thz using 2000 samples)

# 5d. Request Boltzmann distribution trajectory analysis
calculation.show_boltzmann_distribution()

# 5e. Request calculate plot of direct velocity correlation function (without projection)
#calculation.plot_correlation_direct()

# 5f. Request calculate plot of wave vector projected velocity correlation function
#calculation.plot_correlation_wave_vector()

# 5g. Request calculate plot of phonon mode projected velocity correlation function
calculation.plot_power_spectrum_phonon()

# 5h. Request save direct velocity correlation function into file
#calculation.write_correlation_direct('Data Files/correlation_direct.out')

# 5i. Request save wave vector projected velocity correlation function into file
#calculation.write_correlation_wave_vector('Data Files/correlation_wave_vector.out')

# 5j. Request save phonon projected velocity correlation function into file
calculation.write_power_spectrum_phonon('~/mem_phonon.out')

#5k. Request calculation of renormalized force constants
calculation.write_renormalized_constants(filename="FORCE_CONSTANTS")


exit()